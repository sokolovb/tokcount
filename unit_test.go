package tokcount

import (
	"math/rand"
	"regexp"
	"strings"
	"testing"
)

func findAll(p *pattern, s string) []string {
	var out []string
	p.forEach(s, backgroundState(), func(piece string, isMatch bool) bool {
		if isMatch {
			out = append(out, piece)
		}
		return true
	})
	return out
}

var fuzzAlphabet = []rune("aZ'sStT ſ\t\n\r\u00a0\u3000漢字ひ1234١!?.,-\u0301\u0308ǅʰ😀\x00")

func randomString(rnd *rand.Rand, n int) string {
	var b strings.Builder
	for i := 0; i < n; i++ {
		b.WriteRune(fuzzAlphabet[rnd.Intn(len(fuzzAlphabet))])
	}
	return b.String()
}

// Without lookahead and class intersection the patterns are valid RE2, and
// Go's leftmost-first matching must agree with the backtracking matcher.
func TestPatternMatchesRE2(t *testing.T) {
	patterns := []string{
		`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+`,
		`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+`,
		`[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_` + "`" + `{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+`,
		`[一-龥぀-ゟ゠-ヿ]+`,
		`\p{N}{1,3}`,
		`a{2,}|b{0,2}c|\x41\x{42}`,
	}
	rnd := rand.New(rand.NewSource(1))
	for _, src := range patterns {
		p, err := compilePattern(src)
		if err != nil {
			t.Fatalf("%q: %v", src, err)
		}
		re := regexp.MustCompile(unicodeSpaces(src))
		for i := 0; i < 3000; i++ {
			s := randomString(rnd, rnd.Intn(40))
			got, want := findAll(p, s), re.FindAllString(s, -1)
			if strings.Join(got, "\x1f") != strings.Join(want, "\x1f") || len(got) != len(want) {
				t.Fatalf("%q on %q:\n got %q\nwant %q", src, s, got, want)
			}
		}
	}
}

// unicodeSpaces rewrites \s for RE2, where it is ASCII-only, into the
// Unicode White_Space set that Oniguruma and Rust regex use.
func unicodeSpaces(src string) string {
	const ws = `\t\n\v\f\r \x{85}\x{a0}\x{1680}\x{2000}-\x{200a}\x{2028}\x{2029}\x{202f}\x{205f}\x{3000}`
	var b strings.Builder
	depth := 0
	for i := 0; i < len(src); i++ {
		switch {
		case src[i] == '\\' && i+1 < len(src) && src[i+1] == 's':
			if depth > 0 {
				b.WriteString(ws)
			} else {
				b.WriteString("[" + ws + "]")
			}
			i++
		case src[i] == '\\' && i+1 < len(src):
			b.WriteString(src[i : i+2])
			i++
		default:
			if src[i] == '[' {
				depth++
			} else if src[i] == ']' && depth > 0 {
				depth--
			}
			b.WriteByte(src[i])
		}
	}
	return b.String()
}

func TestPatternLookahead(t *testing.T) {
	p, err := compilePattern(` ?\S+|\s+(?!\S)|\s+`)
	if err != nil {
		t.Fatal(err)
	}
	for s, want := range map[string]string{
		"a   b":   `a|  | b`,
		"a b":     `a| b`,
		"a   ":    `a|   `,
		"   b":    `  | b`,
		"a \n b":  "a| \n| b",
		"\t\tx  ": "\t|\t|x|  ",
	} {
		if got := strings.Join(findAll(p, s), "|"); got != want {
			t.Errorf("%q: got %q, want %q", s, got, want)
		}
	}
}

func TestPatternIntersectionAndFold(t *testing.T) {
	p, err := compilePattern(`[\p{L}&&[^\p{Han}]]+|(?i:'s)`)
	if err != nil {
		t.Fatal(err)
	}
	got := strings.Join(findAll(p, "ab漢字cd'S'ſ's"), "|")
	if want := "ab|cd|'S|'ſ|'s"; got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestPatternRejectsUnsupported(t *testing.T) {
	for _, src := range []string{`a*?`, `(a+)`, `(?<=a)b`, `.`, `^a`, `\w`, `(?!ab)`, `a|`, `\p{Bogus}`, `[a`, `a{3,1}`} {
		if _, err := compilePattern(src); err == nil {
			t.Errorf("%q: expected an error", src)
		}
	}
}

func TestMergeMatchesBruteForce(t *testing.T) {
	rnd := rand.New(rand.NewSource(2))
	for iter := 0; iter < 300; iter++ {
		// A random vocabulary over a small alphabet, built by merges so every
		// token is reachable, as in a trained BPE.
		alphabet := "abcd"
		vocab := map[string]uint32{}
		ranks := map[string]uint32{} // tiktoken: token -> rank
		for i := 0; i < len(alphabet); i++ {
			vocab[alphabet[i:i+1]] = uint32(len(vocab))
		}
		for b := 0; b < 256; b++ {
			ranks[string([]byte{byte(b)})] = uint32(b)
		}
		tokens := []string{"a", "b", "c", "d"}
		merges := map[uint64]uint64{}
		mergeRank := map[[2]string]int{}
		for len(mergeRank) < 30 {
			a, b := tokens[rnd.Intn(len(tokens))], tokens[rnd.Intn(len(tokens))]
			if _, ok := mergeRank[[2]string{a, b}]; ok || len(a+b) > 6 {
				continue
			}
			r := len(mergeRank)
			mergeRank[[2]string{a, b}] = r
			if _, ok := vocab[a+b]; !ok {
				vocab[a+b] = uint32(len(vocab))
				tokens = append(tokens, a+b)
			}
			merges[pairKey(vocab[a], vocab[b])] = uint64(r)<<32 | uint64(vocab[a+b])
			if _, ok := ranks[a+b]; !ok {
				ranks[a+b] = uint32(256 + r)
			}
		}
		word := randomWord(rnd, alphabet, 1+rnd.Intn(30))

		// HF: pairs ranked by their position in the merges list.
		want := bpeReference(word, func(a, b string) (int, bool) {
			r, ok := mergeRank[[2]string{a, b}]
			return r, ok
		})
		if got := mergeCount(word, merges, vocab); got != want {
			t.Fatalf("HF merge(%q) = %d, want %d", word, got, want)
		}

		// tiktoken: pairs ranked by the merged token, through tiktokenModel.
		want = bpeReference(word, func(a, b string) (int, bool) {
			r, ok := ranks[a+b]
			return int(r), ok
		})
		m, err := tiktokenModel(ranks)
		if err != nil {
			t.Fatal(err)
		}
		if got := mergeCount(word, m.merges, ranks); got != want {
			t.Fatalf("tiktoken merge(%q) = %d, want %d", word, got, want)
		}
	}
}

func mergeCount(word string, merges map[uint64]uint64, ids map[string]uint32) int {
	sc := new(scratch)
	for i := 0; i < len(word); i++ {
		sc.syms = append(sc.syms, sym{id: ids[word[i:i+1]]})
	}
	return sc.merge(merges, backgroundState())
}

func randomWord(rnd *rand.Rand, alphabet string, n int) string {
	b := make([]byte, n)
	for i := range b {
		b[i] = alphabet[rnd.Intn(len(alphabet))]
	}
	return string(b)
}

func bpeReference(word string, rank func(a, b string) (int, bool)) int {
	syms := make([]string, len(word))
	for i := range word {
		syms[i] = word[i : i+1]
	}
	for {
		best, bestRank := -1, 0
		for i := 0; i+1 < len(syms); i++ {
			if r, ok := rank(syms[i], syms[i+1]); ok && (best < 0 || r < bestRank) {
				best, bestRank = i, r
			}
		}
		if best < 0 {
			return len(syms)
		}
		syms[best] += syms[best+1]
		syms = append(syms[:best+1], syms[best+2:]...)
	}
}

func TestSplitRuns(t *testing.T) {
	var got []string
	splitRuns("aaaaa   bb\u3000\u3000\u3000\u3000c", 2, func(s string) bool {
		got = append(got, s)
		return true
	})
	want := []string{"aa", "aa", "a  ", " bb\u3000\u3000", "\u3000\u3000c"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestTrieLeftmostLongest(t *testing.T) {
	tr := newTrie([]string{"<a>", "<a><b>", "b>", "<"})
	var got []string
	tr.split("x<a><b>y<a>b>z<", func(part string, isToken bool) bool {
		if isToken {
			part = "[" + part + "]"
		}
		got = append(got, part)
		return true
	})
	if s, want := strings.Join(got, ""), "x[<a><b>]y[<a>][b>]z[<]"; s != want {
		t.Errorf("got %q, want %q", s, want)
	}
}
