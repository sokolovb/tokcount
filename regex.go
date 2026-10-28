package tokcount

// A matcher for the subset of regular expressions that tokenizer
// pre-tokenization patterns are written in.
//
// Go's regexp (RE2) cannot express these patterns: nearly all of them end in
// `\s+(?!\S)|\s+`, which needs lookahead, and Kimi's uses class intersection
// (`[\p{Lo}&&[^\p{Han}]]`). Reference implementations run them on
// backtracking engines (Oniguruma in HF tokenizers, fancy-regex in tiktoken)
// with leftmost-first alternation, and so does this one.
//
// Supported syntax:
//   - alternation, at top level and inside groups;
//   - character classes: ranges, negation, nested classes, `&&` intersection,
//     \p{..} / \P{..} (general categories, scripts, properties);
//   - escapes \s \S \d \D \r \n \t \f \v \xHH \x{H..} \uHHHH and escaped
//     punctuation;
//   - greedy quantifiers ? * + {n} {n,} {n,m};
//   - groups (?i:...), (?:...), (...) whose alternatives are literal
//     sequences, quantified at most by ?;
//   - single-character lookaheads (?!x) and (?=x).
//
// Anything else is rejected when the pattern is compiled, so an unfamiliar
// pattern fails loudly instead of being matched incorrectly.

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

// charClass is a set of runes. Runes in the BMP are answered from a bitset;
// the rest fall back to the predicate.
type charClass struct {
	bmp  [0x10000 / 64]uint64
	pred func(rune) bool
}

func newCharClass(pred func(rune) bool) *charClass {
	c := &charClass{pred: pred}
	for r := rune(0); r < 0x10000; r++ {
		if pred(r) {
			c.bmp[r>>6] |= 1 << (uint(r) & 63)
		}
	}
	return c
}

func (c *charClass) has(r rune) bool {
	if r < 0x10000 {
		return c.bmp[r>>6]&(1<<(uint(r)&63)) != 0
	}
	return c.pred(r)
}

type itemKind uint8

const (
	itemClass itemKind = iota
	itemGroup
	itemLookahead
)

type item struct {
	kind     itemKind
	class    *charClass
	alts     [][]*charClass // itemGroup: alternatives, each a sequence of single characters
	min, max int            // max < 0 means unbounded
	negate   bool           // itemLookahead: (?!x) rather than (?=x)
}

type pattern struct {
	alts [][]item
	lit  string // set by compileLiteral: the text the pattern matches
}

func compilePattern(src string) (*pattern, error) {
	p := &parser{src: src}
	alts, err := p.parseAlts(false)
	if err != nil {
		return nil, fmt.Errorf("pattern %q: %w", src, err)
	}
	for i, alt := range alts {
		if minLen(alt) == 0 {
			return nil, fmt.Errorf("pattern %q: alternative %d can match the empty string", src, i)
		}
	}
	return &pattern{alts: alts}, nil
}

// compileLiteral compiles a pattern that matches s literally. forEach finds
// it with strings.Index rather than trying it at every position.
func compileLiteral(s string) (*pattern, error) {
	p, err := compilePattern(regexp.QuoteMeta(s))
	if err != nil {
		return nil, err
	}
	p.lit = s
	return p, nil
}

func minLen(seq []item) int {
	n := 0
	for _, it := range seq {
		switch it.kind {
		case itemClass:
			n += it.min
		case itemGroup:
			if it.min > 0 {
				shortest := -1
				for _, a := range it.alts {
					if shortest < 0 || len(a) < shortest {
						shortest = len(a)
					}
				}
				n += shortest
			}
		}
	}
	return n
}

// matchAt returns the end of the leftmost-first match that starts exactly at
// i, or -1 if there is none. Compiled patterns never match the empty string.
// Scans report their work to st and give up, returning -1, when it says to
// stop; callers tell that apart from no match by checking st.
func (p *pattern) matchAt(s string, i int, st *state) int {
	for _, alt := range p.alts {
		if e := matchSeq(alt, s, i, st); e >= 0 {
			return e
		}
	}
	return -1
}

func matchSeq(seq []item, s string, pos int, st *state) int {
	if len(seq) == 0 {
		return pos
	}
	it := &seq[0]
	switch it.kind {
	case itemClass:
		end, n := pos, 0
		for (it.max < 0 || n < it.max) && end < len(s) {
			r, sz := decodeRune(s, end)
			if !it.class.has(r) {
				break
			}
			end += sz
			n++
			if !st.tick(1) {
				return -1 // stopped
			}
		}
		if n < it.min {
			return -1
		}
		for {
			if e := matchSeq(seq[1:], s, end, st); e >= 0 {
				return e
			}
			if n == it.min || !st.tick(1) {
				return -1
			}
			_, sz := utf8.DecodeLastRuneInString(s[pos:end])
			end -= sz
			n--
		}
	case itemGroup:
		for _, lits := range it.alts {
			end, ok := pos, true
			for _, c := range lits {
				if end >= len(s) {
					ok = false
					break
				}
				r, sz := decodeRune(s, end)
				if !c.has(r) {
					ok = false
					break
				}
				end += sz
			}
			if ok {
				if e := matchSeq(seq[1:], s, end, st); e >= 0 {
					return e
				}
			}
		}
		if it.min == 0 {
			return matchSeq(seq[1:], s, pos, st)
		}
		return -1
	default: // itemLookahead
		hit := false
		if pos < len(s) {
			r, _ := decodeRune(s, pos)
			hit = it.class.has(r)
		}
		if hit == it.negate {
			return -1
		}
		return matchSeq(seq[1:], s, pos, st)
	}
}

func decodeRune(s string, i int) (rune, int) {
	if c := s[i]; c < utf8.RuneSelf {
		return rune(c), 1
	}
	return utf8.DecodeRuneInString(s[i:])
}

// forEach walks s the way a regex find-all does and calls fn for every match
// and for every run of text between matches (isMatch=false), in order. It
// stops early, returning false, when fn does or st says to.
func (p *pattern) forEach(s string, st *state, fn func(piece string, isMatch bool) bool) bool {
	if p.lit != "" {
		return p.forEachLiteral(s, st, fn)
	}
	gap, i := 0, 0
	for i < len(s) {
		e := p.matchAt(s, i, st)
		if st.err != nil {
			return false
		}
		if e >= 0 {
			if gap < i && !fn(s[gap:i], false) {
				return false
			}
			if !fn(s[i:e], true) {
				return false
			}
			i, gap = e, e
			continue
		}
		_, sz := decodeRune(s, i)
		i += sz
		if !st.tick(1) {
			return false
		}
	}
	if gap < len(s) {
		return fn(s[gap:], false)
	}
	return true
}

func (p *pattern) forEachLiteral(s string, st *state, fn func(piece string, isMatch bool) bool) bool {
	for {
		i := strings.Index(s, p.lit)
		if i < 0 {
			return s == "" || fn(s, false)
		}
		if !st.tick(i + len(p.lit)) {
			return false
		}
		if i > 0 && !fn(s[:i], false) {
			return false
		}
		if !fn(s[i:i+len(p.lit)], true) {
			return false
		}
		s = s[i+len(p.lit):]
	}
}

// Parser.

type parser struct {
	src  string
	pos  int
	fold bool // inside (?i:...)
}

func (p *parser) errorf(format string, args ...any) error {
	return fmt.Errorf("at offset %d: %s", p.pos, fmt.Sprintf(format, args...))
}

func (p *parser) parseAlts(inGroup bool) ([][]item, error) {
	var alts [][]item
	var cur []item
	for p.pos < len(p.src) {
		switch p.src[p.pos] {
		case '|':
			alts = append(alts, cur)
			cur = nil
			p.pos++
			continue
		case ')':
			if !inGroup {
				return nil, p.errorf("unbalanced ')'")
			}
			return append(alts, cur), nil
		}
		it, err := p.parseItem()
		if err != nil {
			return nil, err
		}
		cur = append(cur, it)
	}
	if inGroup {
		return nil, p.errorf("unterminated group")
	}
	return append(alts, cur), nil
}

func (p *parser) parseItem() (item, error) {
	rest := p.src[p.pos:]
	switch {
	case strings.HasPrefix(rest, "(?!"), strings.HasPrefix(rest, "(?="):
		neg := rest[2] == '!'
		p.pos += 3
		pred, err := p.parseAtom()
		if err != nil {
			return item{}, err
		}
		if err := p.expect(')'); err != nil {
			return item{}, err
		}
		if p.pos < len(p.src) && strings.ContainsRune("?*+{", rune(p.src[p.pos])) {
			return item{}, p.errorf("quantified lookahead")
		}
		return item{kind: itemLookahead, class: newCharClass(pred), negate: neg, min: 1, max: 1}, nil
	case strings.HasPrefix(rest, "("):
		fold := p.fold
		switch {
		case strings.HasPrefix(rest, "(?i:"):
			p.pos += len("(?i:")
			p.fold = true
		case strings.HasPrefix(rest, "(?:"):
			p.pos += len("(?:")
		case strings.HasPrefix(rest, "(?"):
			return item{}, p.errorf("unsupported group syntax")
		default:
			p.pos += len("(")
		}
		alts, err := p.parseAlts(true)
		p.fold = fold
		if err != nil {
			return item{}, err
		}
		if err := p.expect(')'); err != nil {
			return item{}, err
		}
		it := item{kind: itemGroup}
		for _, alt := range alts {
			var lits []*charClass
			for _, sub := range alt {
				if sub.kind != itemClass || sub.min != 1 || sub.max != 1 {
					return item{}, p.errorf("groups may only contain literal alternatives")
				}
				lits = append(lits, sub.class)
			}
			it.alts = append(it.alts, lits)
		}
		min, max, err := p.parseQuant()
		if err != nil {
			return item{}, err
		}
		if max != 1 {
			return item{}, p.errorf("groups may only be quantified by ?")
		}
		it.min, it.max = min, max
		return it, nil
	}
	pred, err := p.parseAtom()
	if err != nil {
		return item{}, err
	}
	min, max, err := p.parseQuant()
	if err != nil {
		return item{}, err
	}
	return item{kind: itemClass, class: newCharClass(pred), min: min, max: max}, nil
}

func (p *parser) expect(c byte) error {
	if p.pos >= len(p.src) || p.src[p.pos] != c {
		return p.errorf("expected %q", c)
	}
	p.pos++
	return nil
}

func (p *parser) parseQuant() (min, max int, err error) {
	min, max = 1, 1
	if p.pos >= len(p.src) {
		return
	}
	switch p.src[p.pos] {
	case '?':
		min, max = 0, 1
	case '*':
		min, max = 0, -1
	case '+':
		min, max = 1, -1
	case '{':
		end := strings.IndexByte(p.src[p.pos:], '}')
		if end < 0 {
			return 0, 0, p.errorf("unterminated repetition")
		}
		body := p.src[p.pos+1 : p.pos+end]
		lo, hi, comma := strings.Cut(body, ",")
		if min, err = strconv.Atoi(lo); err != nil {
			return 0, 0, p.errorf("bad repetition {%s}", body)
		}
		switch {
		case !comma:
			max = min
		case hi == "":
			max = -1
		default:
			if max, err = strconv.Atoi(hi); err != nil || max < min {
				return 0, 0, p.errorf("bad repetition {%s}", body)
			}
		}
		p.pos += end
	default:
		return
	}
	p.pos++
	if p.pos < len(p.src) && (p.src[p.pos] == '?' || p.src[p.pos] == '+') {
		return 0, 0, p.errorf("lazy and possessive quantifiers are not supported")
	}
	return min, max, nil
}

// parseAtom parses one character-matching atom outside a bracket class.
func (p *parser) parseAtom() (func(rune) bool, error) {
	if p.pos >= len(p.src) {
		return nil, p.errorf("missing atom")
	}
	switch c := p.src[p.pos]; c {
	case '[':
		if p.fold {
			return nil, p.errorf("classes inside (?i:...) are not supported")
		}
		return p.parseBracket()
	case '\\':
		pred, r, err := p.parseEscape()
		if err != nil {
			return nil, err
		}
		if pred == nil {
			return p.literal(r), nil
		}
		if p.fold {
			return nil, p.errorf("class escapes inside (?i:...) are not supported")
		}
		return pred, nil
	case '.', '^', '$', '*', '+', '?', '{', '(', ')', '|':
		return nil, p.errorf("unsupported %q", c)
	}
	r, sz := utf8.DecodeRuneInString(p.src[p.pos:])
	p.pos += sz
	return p.literal(r), nil
}

// literal matches r, or under (?i) its whole simple case-folding orbit, the
// way Unicode-aware engines do (so (?i:s) also matches U+017F ſ).
func (p *parser) literal(r rune) func(rune) bool {
	if !p.fold {
		return func(x rune) bool { return x == r }
	}
	set := map[rune]bool{r: true}
	for f := unicode.SimpleFold(r); f != r; f = unicode.SimpleFold(f) {
		set[f] = true
	}
	return func(x rune) bool { return set[x] }
}

// parseEscape parses an escape at p.pos. It returns either a class predicate
// or, for escapes denoting one character, that character.
func (p *parser) parseEscape() (func(rune) bool, rune, error) {
	p.pos++ // backslash
	if p.pos >= len(p.src) {
		return nil, 0, p.errorf("trailing backslash")
	}
	c := p.src[p.pos]
	p.pos++
	switch c {
	case 's':
		return isWhiteSpace, 0, nil
	case 'S':
		return not(isWhiteSpace), 0, nil
	case 'd':
		return isDigit, 0, nil
	case 'D':
		return not(isDigit), 0, nil
	case 'p', 'P':
		var name string
		if p.pos < len(p.src) && p.src[p.pos] == '{' {
			end := strings.IndexByte(p.src[p.pos:], '}')
			if end < 0 {
				return nil, 0, p.errorf("unterminated \\p{")
			}
			name = p.src[p.pos+1 : p.pos+end]
			p.pos += end + 1
		} else if p.pos < len(p.src) {
			name = p.src[p.pos : p.pos+1]
			p.pos++
		}
		pred, err := unicodeProperty(name)
		if err != nil {
			return nil, 0, p.errorf("%v", err)
		}
		if c == 'P' {
			pred = not(pred)
		}
		return pred, 0, nil
	case 'r':
		return nil, '\r', nil
	case 'n':
		return nil, '\n', nil
	case 't':
		return nil, '\t', nil
	case 'f':
		return nil, '\f', nil
	case 'v':
		return nil, '\v', nil
	case 'x', 'u':
		var hex string
		switch {
		case p.pos < len(p.src) && p.src[p.pos] == '{':
			end := strings.IndexByte(p.src[p.pos:], '}')
			if end < 0 {
				return nil, 0, p.errorf("unterminated \\%c{", c)
			}
			hex = p.src[p.pos+1 : p.pos+end]
			p.pos += end + 1
		case c == 'x' && p.pos+2 <= len(p.src):
			hex = p.src[p.pos : p.pos+2]
			p.pos += 2
		case c == 'u' && p.pos+4 <= len(p.src):
			hex = p.src[p.pos : p.pos+4]
			p.pos += 4
		}
		v, err := strconv.ParseUint(hex, 16, 32)
		if err != nil || v > unicode.MaxRune {
			return nil, 0, p.errorf("bad \\%c escape", c)
		}
		return nil, rune(v), nil
	}
	if c < utf8.RuneSelf && !isASCIIAlnum(c) {
		return nil, rune(c), nil // escaped punctuation
	}
	return nil, 0, p.errorf("unsupported escape \\%c", c)
}

// parseBracket parses a bracket class: union of items, with && binding
// looser than union, and nested bracket classes allowed as items.
func (p *parser) parseBracket() (func(rune) bool, error) {
	p.pos++ // '['
	negate := false
	if p.pos < len(p.src) && p.src[p.pos] == '^' {
		negate = true
		p.pos++
	}
	var inter []func(rune) bool // operands of &&
	var union []func(rune) bool
	first := true
	for {
		if p.pos >= len(p.src) {
			return nil, p.errorf("unterminated class")
		}
		c := p.src[p.pos]
		if c == ']' && !first {
			p.pos++
			break
		}
		first = false
		if strings.HasPrefix(p.src[p.pos:], "&&") {
			p.pos += 2
			inter = append(inter, anyOf(union))
			union = nil
			continue
		}
		if c == '[' {
			pred, err := p.parseBracket()
			if err != nil {
				return nil, err
			}
			union = append(union, pred)
			continue
		}
		var lo rune
		if c == '\\' {
			pred, r, err := p.parseEscape()
			if err != nil {
				return nil, err
			}
			if pred != nil {
				union = append(union, pred)
				continue
			}
			lo = r
		} else {
			r, sz := utf8.DecodeRuneInString(p.src[p.pos:])
			p.pos += sz
			lo = r
		}
		// Range lo-hi, unless the '-' is the last character of the class.
		if p.pos+1 < len(p.src) && p.src[p.pos] == '-' && p.src[p.pos+1] != ']' {
			p.pos++
			var hi rune
			if p.src[p.pos] == '\\' {
				pred, r, err := p.parseEscape()
				if err != nil {
					return nil, err
				}
				if pred != nil {
					return nil, p.errorf("class escape as range end")
				}
				hi = r
			} else {
				r, sz := utf8.DecodeRuneInString(p.src[p.pos:])
				p.pos += sz
				hi = r
			}
			if hi < lo {
				return nil, p.errorf("bad range %q-%q", lo, hi)
			}
			union = append(union, func(x rune) bool { return x >= lo && x <= hi })
			continue
		}
		r := lo
		union = append(union, func(x rune) bool { return x == r })
	}
	pred := anyOf(union)
	if len(inter) > 0 {
		inter = append(inter, pred)
		pred = allOf(inter)
	}
	if negate {
		pred = not(pred)
	}
	return pred, nil
}

func anyOf(preds []func(rune) bool) func(rune) bool {
	preds = append([]func(rune) bool(nil), preds...)
	return func(r rune) bool {
		for _, f := range preds {
			if f(r) {
				return true
			}
		}
		return false
	}
}

func allOf(preds []func(rune) bool) func(rune) bool {
	return func(r rune) bool {
		for _, f := range preds {
			if !f(r) {
				return false
			}
		}
		return true
	}
}

func not(f func(rune) bool) func(rune) bool { return func(r rune) bool { return !f(r) } }

func isASCIIAlnum(c byte) bool {
	return c >= '0' && c <= '9' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z'
}

// \s in Unicode-aware engines (Oniguruma, Rust regex) is White_Space.
func isWhiteSpace(r rune) bool { return unicode.Is(unicode.White_Space, r) }

func isDigit(r rune) bool { return unicode.Is(unicode.Nd, r) }

func unicodeProperty(name string) (func(rune) bool, error) {
	var t *unicode.RangeTable
	if t = unicode.Categories[name]; t == nil {
		if t = unicode.Scripts[name]; t == nil {
			t = unicode.Properties[name]
		}
	}
	if t == nil {
		return nil, fmt.Errorf("unknown Unicode property %q", name)
	}
	return func(r rune) bool { return unicode.Is(t, r) }, nil
}
