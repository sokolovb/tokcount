package tokcount

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"
)

func TestZeroBoundIsSafe(t *testing.T) {
	if (pipeline{}).bound != boundNever {
		t.Error("an unset bound must not let Fits skip counting")
	}
}

// Text that was not counted does not fit, even if its length alone says it would.
func TestFitsOnCancelledContext(t *testing.T) {
	c := tinyHF(t, nil, true, "a")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	for _, text := range []string{"a", "aaaaaaaa"} {
		if fits, err := c.Fits(ctx, text, 4); fits || err != context.Canceled {
			t.Errorf("Fits(%q) on a cancelled context = %v, %v", text, fits, err)
		}
	}
}

// tinyHF builds a tokenizer.json with a vocabulary of single characters, no
// merges, the given normalizer and either byte-level or byte-fallback symbols.
func tinyHF(t *testing.T, normalizer any, byteLevel bool, chars ...string) *Counter {
	t.Helper()
	vocab := map[string]int{}
	for _, c := range chars {
		vocab[c] = len(vocab)
	}
	model := map[string]any{"type": "BPE", "vocab": vocab, "merges": []string{}}
	var pre any
	if byteLevel {
		pre = map[string]any{"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
	} else {
		for b := 0; b < 256; b++ {
			vocab[fmt.Sprintf("<0x%02X>", b)] = len(vocab)
		}
		model["byte_fallback"] = true
	}
	data, err := json.Marshal(map[string]any{"normalizer": normalizer, "pre_tokenizer": pre, "model": model})
	if err != nil {
		t.Fatal(err)
	}
	c, err := ParseHF(data, nil)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func replace(from, to string) any {
	return map[string]any{"type": "Replace", "pattern": map[string]any{"String": from}, "content": to}
}

var nfcNormalizer = map[string]any{"type": "NFC"}

func TestFitsAgreesWithCount(t *testing.T) {
	ctx := context.Background()
	for _, tc := range []struct {
		name  string
		c     *Counter
		text  string
		bound lengthBound
	}{
		// "a" costs one byte, "bcd" three: the bound does not hold.
		{"expanding replace", tinyHF(t, replace("a", "bcd"), true, "a", "b", "c", "d"), "aaaa", boundNever},
		// Gemma: " " becomes "▁", one byte to three, but "▁" is one symbol.
		{"space marker", tinyHF(t, replace(" ", "▁"), false, "a", "▁"), "a a a", boundAlways},
		{"nfc byte-level", tinyHF(t, nfcNormalizer, true, "a"), "क़क़", boundASCII},
		// U+0958 (3 bytes) decomposes into two characters that fall back to 6 bytes.
		{"nfc byte fallback", tinyHF(t, nfcNormalizer, false, "a"), "क़क़", boundASCII},
		// The replacement makes text non-ASCII before NFC sees it.
		{"replace then nfc", tinyHF(t, map[string]any{"type": "Sequence", "normalizers": []any{replace("a", "क़"), nfcNormalizer}}, false, "a"), "aa", boundNever},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.c.p.bound; got != tc.bound {
				t.Errorf("bound = %d, want %d", got, tc.bound)
			}
			n, err := tc.c.Count(ctx, tc.text)
			if err != nil {
				t.Fatal(err)
			}
			for limit := 0; limit <= n+len(tc.text); limit++ {
				fits, err := tc.c.Fits(ctx, tc.text, limit)
				if err != nil || fits != (n <= limit) {
					t.Errorf("Fits(%q, %d) = %v, %v; count is %d", tc.text, limit, fits, err, n)
				}
			}
		})
	}
}
