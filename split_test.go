package tokcount

import (
	"regexp"
	"strings"
	"testing"
)

// Expected pieces come from HF tokenizers 0.23:
// pre_tokenizers.Split(pattern, behavior).pre_tokenize_str(text).
func TestSplitMatchesHF(t *testing.T) {
	space, ab := regexp.QuoteMeta(" "), regexp.QuoteMeta("ab")
	literals := map[string]string{space: " ", ab: "ab"}
	for _, tc := range []struct {
		pattern  string // regex; literal patterns are tested both ways
		behavior splitBehavior
		text     string
		want     []string
	}{
		{space, splitMergedWithPrevious, "a b  c ", []string{"a ", "b ", " ", "c "}},
		{space, splitMergedWithPrevious, " a", []string{" ", "a"}},
		{space, splitMergedWithPrevious, "  ", []string{" ", " "}},
		{space, splitMergedWithPrevious, "a  b", []string{"a ", " ", "b"}},
		{space, splitMergedWithPrevious, "ab ab", []string{"ab ", "ab"}},
		{space, splitMergedWithPrevious, "x", []string{"x"}},
		{space, splitIsolated, "a b  c ", []string{"a", " ", "b", " ", " ", "c", " "}},
		{space, splitIsolated, "a  b", []string{"a", " ", " ", "b"}},
		{`\d+`, splitMergedWithPrevious, "a1b22c", []string{"a1", "b22", "c"}},
		{`\d+`, splitMergedWithPrevious, "1a", []string{"1", "a"}},
		{`\d+`, splitMergedWithPrevious, "12", []string{"12"}},
		{`\d+`, splitIsolated, "a1b22c", []string{"a", "1", "b", "22", "c"}},
		{ab, splitMergedWithPrevious, "abab", []string{"ab", "ab"}},
		{ab, splitMergedWithPrevious, "xabyabab", []string{"xab", "yab", "ab"}},
		{ab, splitMergedWithPrevious, "ab ab", []string{"ab", " ab"}},
		{ab, splitIsolated, "xabyabab", []string{"x", "ab", "y", "ab", "ab"}},
		{ab, splitMatchesOnly, "xabyabab", []string{"ab", "ab", "ab"}},
	} {
		re, err := compilePattern(tc.pattern)
		if err != nil {
			t.Fatal(err)
		}
		patterns := []*pattern{re}
		if lit, ok := literals[tc.pattern]; ok {
			re, err := compileLiteral(lit)
			if err != nil {
				t.Fatal(err)
			}
			patterns = append(patterns, re)
		}
		for _, re := range patterns {
			step := splitStep{re: re, behavior: tc.behavior}
			var got []string
			step.split(backgroundState(), tc.text, func(piece string) bool {
				got = append(got, piece)
				return true
			})
			if strings.Join(got, "|") != strings.Join(tc.want, "|") || len(got) != len(tc.want) {
				t.Errorf("%q (literal %v) behavior %d on %q: got %q, want %q", tc.pattern, re.lit != "", tc.behavior, tc.text, got, tc.want)
			}
		}
	}
}
