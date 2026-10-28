package tokcount

// Normalizers: the two HF normalization steps the supported tokenizers use.

import (
	"sort"
	"strings"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

// normalizer is one normalization step.
type normalizer interface {
	apply(s string) string
	// cut returns the first offset in s at which s can be split without
	// changing the result, or -1. It lets long text be normalized in chunks,
	// checking the context in between.
	cut(s string) int
}

// nfcNorm is HF's NFC normalizer.
type nfcNorm struct{}

func (nfcNorm) apply(s string) string { return nfc(s) }
func (nfcNorm) cut(s string) int      { return norm.NFC.FirstBoundaryInString(s) }

// replaceNorm is HF's Replace normalizer with a string pattern.
type replaceNorm struct{ from, to string }

func (r replaceNorm) apply(s string) string { return strings.ReplaceAll(s, r.from, r.to) }

func (r replaceNorm) cut(string) int {
	if utf8.RuneCountInString(r.from) == 1 {
		return 0 // no occurrence spans a rune boundary
	}
	return -1
}

const cgj = "\u034f" // COMBINING GRAPHEME JOINER

// nfc normalizes s to NFC the way HF tokenizers does.
//
// golang.org/x/text produces the Stream-Safe Text Format (UAX #15): it puts a
// U+034F after every 30 consecutive non-starters, which HF's normalizer does
// not. Such runs only occur in pathological text, so they are detected after
// the fact and normalized by the slow path instead.
func nfc(s string) string {
	if norm.NFC.IsNormalString(s) {
		return s
	}
	out := norm.NFC.String(s)
	if strings.Count(out, cgj) > strings.Count(s, cgj) {
		return nfcUnbounded(s)
	}
	return out
}

// nfcUnbounded is the plain UAX #15 algorithm, with no limit on runs of
// non-starters: full canonical decomposition, canonical ordering, then
// canonical composition.
func nfcUnbounded(s string) string {
	type ch struct {
		r   rune
		ccc uint8
	}
	var cs []ch
	for _, r := range s {
		for _, d := range norm.NFD.String(string(r)) {
			cs = append(cs, ch{d, norm.NFD.PropertiesString(string(d)).CCC()})
		}
	}
	for i := 0; i < len(cs); {
		if cs[i].ccc == 0 {
			i++
			continue
		}
		j := i
		for j < len(cs) && cs[j].ccc != 0 {
			j++
		}
		run := cs[i:j]
		sort.SliceStable(run, func(a, b int) bool { return run[a].ccc < run[b].ccc })
		i = j
	}
	if len(cs) == 0 {
		return ""
	}
	// Composition: each character either composes with the last starter, if
	// nothing between them blocks it, or is kept.
	out := cs[:1]
	starter := -1
	lastCCC := 256
	if cs[0].ccc == 0 {
		starter, lastCCC = 0, 0
	}
	for _, c := range cs[1:] {
		if starter >= 0 && (lastCCC < int(c.ccc) || lastCCC == 0) {
			if r, ok := compose(out[starter].r, c.r); ok {
				out[starter].r = r
				continue
			}
		}
		if c.ccc == 0 {
			starter = len(out)
		}
		lastCCC = int(c.ccc)
		out = append(out, c)
	}
	var b strings.Builder
	for _, c := range out {
		b.WriteRune(c.r)
	}
	return b.String()
}

// compose returns the primary composite of a and b, if there is one.
func compose(a, b rune) (rune, bool) {
	s := norm.NFC.String(string([]rune{a, b}))
	if utf8.RuneCountInString(s) != 1 {
		return 0, false
	}
	r, _ := utf8.DecodeRuneInString(s)
	return r, true
}
