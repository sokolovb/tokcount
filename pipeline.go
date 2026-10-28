package tokcount

// The counting pipeline every tokenizer is loaded into. It follows the encode
// pipeline of HF tokenizers:
//  1. added tokens with normalized=false are cut out of the raw text;
//  2. each remaining part is normalized;
//  3. added tokens with normalized=true are cut out of the normalized parts;
//  4. the pre-tokenizer splits what is left into pieces;
//  5. the BPE model counts each piece independently.
//
// tiktoken is the special case with special tokens as the added tokens, no
// normalizers and one split that keeps only the matches. Kimi's tokenizer
// code adds a step 0 that cuts the text into chunks encoded independently.

import (
	"strings"
	"unicode/utf8"
	"unsafe"
)

// pipelineSpec is what a loader reads from a tokenizer's files.
type pipelineSpec struct {
	chunk       func(text string, fn func(string) bool) bool // step 0; nil for none
	added       []addedToken
	normalizers []normalizer
	splits      []splitStep
	model       bpeModel
}

type addedToken struct {
	content             string
	normalized, special bool
}

type pipeline struct {
	chunk func(text string, fn func(string) bool) bool

	added     *trie // matched in the raw text
	addedNorm *trie // matched in the normalized text

	normalizers []normalizer
	bound       lengthBound

	splits []splitStep
	model  bpeModel
	cut    *cutter // safe split points for pieces BPE would otherwise see whole
	cache  *cache
}

func newPipeline(spec pipelineSpec, opts *Options) *pipeline {
	p := &pipeline{
		chunk:       spec.chunk,
		normalizers: spec.normalizers,
		splits:      spec.splits,
		model:       spec.model,
		cache:       newCache(opts.CacheSize),
	}
	if p.chunk == nil {
		p.chunk = wholeText
	}
	var plain, normalized []string
	for _, at := range spec.added {
		switch {
		case at.special && opts.SpecialTokensAsText:
		case at.normalized:
			normalized = append(normalized, p.normalize(backgroundState(), at.content))
		default:
			plain = append(plain, at.content)
		}
	}
	p.added, p.addedNorm = newTrie(plain), newTrie(normalized)
	p.bound = lengthBoundOf(p.normalizers, &p.model)
	if !p.model.byteLevel {
		p.cut = newCutter(&p.model)
	}
	return p
}

func wholeText(text string, fn func(string) bool) bool { return fn(text) }

func (p *pipeline) count(st *state, text string) {
	p.chunk(text, func(s string) bool { return p.countText(st, s) })
}

func (p *pipeline) countText(st *state, text string) bool {
	return p.added.split(text, func(part string, isToken bool) bool {
		if isToken {
			return st.add(1)
		}
		part = p.normalize(st, part)
		if st.stopped() {
			return false
		}
		return p.addedNorm.split(part, func(part string, isToken bool) bool {
			if isToken {
				return st.add(1)
			}
			return p.preTokenize(st, part, 0)
		})
	})
}

// preTokenize applies split steps from the i-th on and counts the pieces.
func (p *pipeline) preTokenize(st *state, s string, i int) bool {
	if s == "" {
		return true
	}
	if i == len(p.splits) {
		return p.countPiece(st, s)
	}
	return p.splits[i].split(st, s, func(piece string) bool { return p.preTokenize(st, piece, i+1) })
}

// countPiece counts one pre-tokenized piece, cut into words where possible.
func (p *pipeline) countPiece(st *state, piece string) bool {
	if p.cut == nil {
		return p.countWord(st, piece)
	}
	for piece != "" {
		var word string
		word, piece = p.cut.next(st, piece)
		if st.stopped() || !p.countWord(st, word) {
			return false
		}
	}
	return true
}

// countWord counts one word, memoized in the cache.
func (p *pipeline) countWord(st *state, w string) bool {
	if n, ok := p.cache.get(w); ok {
		return st.add(n)
	}
	n := p.model.count(st, w)
	if st.err != nil {
		return false
	}
	p.cache.put(w, n)
	return st.add(n)
}

// Normalization.

const normChunk = 32 << 10

// normalize applies the normalizers to s, in chunks where they allow it.
// A chunk the normalizer returns unchanged is not copied, so text that is
// already normal stays the same string. If st says to stop it returns "",
// which callers discard after checking st.
func (p *pipeline) normalize(st *state, s string) string {
	for _, n := range p.normalizers {
		if len(s) <= normChunk {
			s = n.apply(s)
			continue
		}
		var b strings.Builder
		copied := false
		for off := 0; off < len(s); {
			rest := s[off:]
			end := len(rest)
			if end > normChunk {
				i := normChunk
				for i < len(rest) && !utf8.RuneStart(rest[i]) {
					i++
				}
				if c := n.cut(rest[i:]); c >= 0 {
					end = i + c
				}
			}
			chunk := rest[:end]
			out := n.apply(chunk)
			if !copied && !sameString(out, chunk) {
				b.Grow(len(s))
				b.WriteString(s[:off])
				copied = true
			}
			if copied {
				b.WriteString(out)
			}
			off += end
			if !st.tick(end) {
				return ""
			}
		}
		if copied {
			s = b.String()
		}
	}
	return s
}

// sameString reports whether a and b are the same string without comparing
// bytes. Normalizers return their input when it needs no change, so a shared
// data pointer means the chunk was left as is. An equal copy from a future
// normalizer would only cost a copy, not a wrong result.
func sameString(a, b string) bool {
	return len(a) == len(b) && unsafe.StringData(a) == unsafe.StringData(b)
}

// lengthBound says when len(text) bounds the token count, which lets Fits
// answer without counting. Every symbol BPE starts from is a byte, or a
// character that costs at most its bytes, and merges only lower the count,
// so the bound holds unless normalization makes text cost more. The zero
// value is the safe one.
type lengthBound uint8

const (
	boundNever lengthBound = iota
	boundASCII             // only for ASCII text
	boundAlways
)

func lengthBoundOf(norms []normalizer, m *bpeModel) lengthBound {
	b, ascii := boundAlways, true // ascii: ASCII input reaches the step as ASCII
	for _, n := range norms {
		switch n := n.(type) {
		case nfcNorm:
			// Identity on ASCII; can lengthen anything else.
			if !ascii {
				return boundNever
			}
			b = boundASCII
		case replaceNorm:
			if m.cost(n.to) > m.cost(n.from) {
				return boundNever
			}
			ascii = ascii && isASCII(n.to)
		default:
			return boundNever
		}
	}
	return b
}

func (p *pipeline) lenIsBound(text string) bool {
	return p.bound == boundAlways || p.bound == boundASCII && isASCII(text)
}

func isASCII(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] >= utf8.RuneSelf {
			return false
		}
	}
	return true
}

// Pre-tokenization.

type splitBehavior int

const (
	splitIsolated splitBehavior = iota
	splitMergedWithPrevious
	// splitMatchesOnly keeps the matches and drops the text between them, as
	// tiktoken does (HF's Split with invert=true and behavior=Removed).
	splitMatchesOnly
)

type splitStep struct {
	re       *pattern
	behavior splitBehavior
}

// split cuts s by the step's pattern and behavior and calls fn for each
// resulting piece, as HF's Split pre-tokenizer does.
func (step *splitStep) split(st *state, s string, fn func(string) bool) bool {
	switch step.behavior {
	case splitIsolated:
		return step.re.forEach(s, st, func(piece string, _ bool) bool { return fn(piece) })
	case splitMatchesOnly:
		return step.re.forEach(s, st, func(piece string, isMatch bool) bool { return !isMatch || fn(piece) })
	}
	// MergedWithPrevious: a match joins the piece before it, unless that
	// piece is itself a match. Pieces arrive as consecutive substrings of s.
	start, end, pos := 0, 0, 0
	have, prevMatch := false, false
	ok := step.re.forEach(s, st, func(piece string, isMatch bool) bool {
		pStart := pos
		pos += len(piece)
		if isMatch && !prevMatch && have {
			end, prevMatch = pos, true
			return true
		}
		if have && !fn(s[start:end]) {
			return false
		}
		start, end, have, prevMatch = pStart, pos, true, isMatch
		return true
	})
	if !ok {
		return false
	}
	return !have || fn(s[start:end])
}
