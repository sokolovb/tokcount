package tokcount

import (
	"context"
	"encoding/json"
	"runtime"
	"strings"
	"testing"
	"unsafe"

	"golang.org/x/text/unicode/norm"
)

func TestNormalizeLongUnchanged(t *testing.T) {
	p := &pipeline{normalizers: []normalizer{nfcNorm{}}}
	for _, tc := range []struct {
		name string
		unit string
	}{
		{"ascii", "a"},
		{"mixed", "Привет 漢字😀 "},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := strings.Repeat(tc.unit, normChunk*2/len(tc.unit)+2)
			if len(s) <= normChunk {
				t.Fatalf("fixture len %d does not cross a chunk", len(s))
			}
			if !norm.NFC.IsNormalString(s) {
				t.Fatal("fixture is not NFC")
			}
			got := p.normalize(backgroundState(), s)
			if got != s {
				t.Fatal("normalize changed NFC text")
			}
			if unsafe.StringData(got) != unsafe.StringData(s) {
				t.Fatal("normalize copied NFC text")
			}
		})
	}
}

func TestNormalizeLongNFD(t *testing.T) {
	p := &pipeline{normalizers: []normalizer{nfcNorm{}}}
	const mark = "\u0301" // combining acute, composes with e into é
	nfd := "e" + mark
	pad := strings.Repeat("a", normChunk*3)
	run := "e" + strings.Repeat(mark, 40) // more than 30 non-starters: stream-safe NFC inserts CGJ
	for _, tc := range []struct {
		name    string
		s       string
		frag    string
		at      int  // index of frag
		chunk   int  // which normChunk-sized piece holds it; -1 if it sits on the cut
		special bool // >30 non-starters, so norm.NFC.String is not the oracle
	}{
		{"first chunk", nfd + pad, nfd, 0, 0, false},
		{"middle chunk", pad[:normChunk+32] + nfd + pad[normChunk+32:], nfd, normChunk + 32, 1, false},
		{"last chunk", strings.Repeat("a", normChunk*2+32) + nfd, nfd, normChunk*2 + 32, 2, false},
		{"spanning the cut", strings.Repeat("a", normChunk-1) + nfd + strings.Repeat("b", normChunk), nfd, normChunk - 1, -1, false},
		{"non-starter run", pad[:normChunk+64] + run + pad[normChunk+64:], run, normChunk + 64, 1, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if len(tc.s) <= normChunk {
				t.Fatalf("fixture len %d does not cross a chunk", len(tc.s))
			}
			at := strings.Index(tc.s, tc.frag)
			if at != tc.at {
				t.Fatalf("fragment at %d, want %d", at, tc.at)
			}
			if tc.chunk >= 0 {
				start, end := tc.chunk*normChunk, (tc.chunk+1)*normChunk
				if tc.chunk == 2 {
					end = len(tc.s)
				}
				if at < start || at >= end {
					t.Fatalf("fragment at %d is outside chunk %d [%d,%d)", at, tc.chunk, start, end)
				}
			}
			got := p.normalize(backgroundState(), tc.s)
			want := nfc(tc.s)
			if got != want {
				t.Fatalf("differ from nfc at byte %d (len %d vs %d)", diffAt(got, want), len(got), len(want))
			}
			ref := norm.NFC.String(tc.s)
			if tc.special {
				if got == ref {
					t.Fatal("stream-safe NFC matched; the non-starter run did not trip the special case")
				}
				return
			}
			if got != ref {
				t.Fatalf("differ from norm.NFC at byte %d (len %d vs %d)", diffAt(got, ref), len(got), len(ref))
			}
		})
	}
}

// Gemma's normalizer replaces a space with the marker ▁.
func TestNormalizeLongReplace(t *testing.T) {
	p := &pipeline{normalizers: []normalizer{replaceNorm{from: " ", to: "▁"}}}
	for _, tc := range []struct {
		name string
		s    string
		same bool
	}{
		{"spaces", strings.Repeat("ab ", normChunk), false},
		{"space in last chunk", strings.Repeat("ab", normChunk) + " ", false},
		{"no spaces", strings.Repeat("ab", normChunk), true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if len(tc.s) <= normChunk {
				t.Fatalf("fixture len %d does not cross a chunk", len(tc.s))
			}
			got := p.normalize(backgroundState(), tc.s)
			if want := strings.ReplaceAll(tc.s, " ", "▁"); got != want {
				t.Fatalf("replace: differ at byte %d (len %d vs %d)", diffAt(got, want), len(got), len(want))
			}
			same := unsafe.StringData(got) == unsafe.StringData(tc.s)
			if same != tc.same {
				t.Fatalf("shared backing %v, want %v", same, tc.same)
			}
		})
	}

	// "ab" is two runes, so cut refuses to split and the whole text is one chunk.
	whole := &pipeline{normalizers: []normalizer{replaceNorm{from: "ab", to: "X"}}}
	plain := strings.Repeat("cd", normChunk)
	if got := whole.normalize(backgroundState(), plain); unsafe.StringData(got) != unsafe.StringData(plain) {
		t.Fatal("unmatched multi-rune replace copied the text")
	}
	matched := strings.Repeat("ab", normChunk)
	if got, want := whole.normalize(backgroundState(), matched), strings.Repeat("X", normChunk); got != want {
		t.Fatalf("multi-rune replace: differ at byte %d", diffAt(got, want))
	}
}

// Replace then NFC, the order a Sequence normalizer is applied in.
func TestNormalizeSequence(t *testing.T) {
	p := &pipeline{normalizers: []normalizer{replaceNorm{from: "x", to: "e\u0301"}, nfcNorm{}}}
	plain := strings.Repeat("ab", normChunk)
	if got := p.normalize(backgroundState(), plain); unsafe.StringData(got) != unsafe.StringData(plain) {
		t.Fatal("sequence copied text neither step changes")
	}
	s := plain + "x" + plain
	got := p.normalize(backgroundState(), s)
	want := nfc(strings.ReplaceAll(s, "x", "e\u0301"))
	if got != want {
		t.Fatalf("sequence: differ at byte %d (len %d vs %d)", diffAt(got, want), len(got), len(want))
	}
}

func TestNormalizeStops(t *testing.T) {
	p := &pipeline{normalizers: []normalizer{nfcNorm{}}}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	st := &state{ctx: ctx, limit: -1}
	// NFD in the first chunk so a builder is already live when tick says stop.
	s := "e\u0301" + strings.Repeat("a", normChunk*2)
	if got := p.normalize(st, s); got != "" || st.err != context.Canceled {
		t.Fatalf("normalize = %q, err %v", got, st.err)
	}
}

func TestCountNFCAllocs(t *testing.T) {
	c := nfcWordCounter(t)
	text := longASCII(400 << 10)
	n, err := c.Count(context.Background(), text)
	if err != nil {
		t.Fatal(err)
	}
	if n != len(text) {
		t.Fatalf("Count = %d, want %d", n, len(text))
	}
	allocs, bytes := countCost(100, func() {
		if _, err := c.Count(context.Background(), text); err != nil {
			t.Fatal(err)
		}
	})
	t.Logf("Count %d bytes NFC, cache off: %.0f allocs/op, %d B/op", len(text), allocs, bytes)
	// Was 25 allocs and about 1.9MB/op when the whole text was copied into a
	// Builder. The race detector allocates on its own, so the bound is
	// checked only without it; the StringData tests cover both.
	if raceSlowdown == 1 && (allocs > 24 || bytes > 64<<10) {
		t.Fatalf("Count used %.0f allocs and %d bytes; a copy of this text is %d bytes", allocs, bytes, len(text))
	}
}

func BenchmarkCountNFC(b *testing.B) {
	c := nfcWordCounter(b)
	text := longASCII(400 << 10)
	ctx := context.Background()
	if _, err := c.Count(ctx, text); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(int64(len(text)))
	for i := 0; i < b.N; i++ {
		if _, err := c.Count(ctx, text); err != nil {
			b.Fatal(err)
		}
	}
}

// nfcWordCounter is a byte-cheap tokenizer: NFC, a split on spaces, and a
// character vocabulary, so Count's allocations are the normalizer's.
func nfcWordCounter(tb testing.TB) *Counter {
	tb.Helper()
	vocab := map[string]int{}
	for _, c := range []string{"h", "e", "l", "o", "w", "r", "d", " "} {
		vocab[c] = len(vocab)
	}
	data, err := json.Marshal(map[string]any{
		"normalizer": nfcNormalizer,
		"pre_tokenizer": map[string]any{
			"type":     "Split",
			"pattern":  map[string]any{"String": " "},
			"behavior": "Isolated",
		},
		"model": map[string]any{"type": "BPE", "vocab": vocab, "merges": []string{}},
	})
	if err != nil {
		tb.Fatal(err)
	}
	c, err := ParseHF(data, &Options{CacheSize: -1})
	if err != nil {
		tb.Fatal(err)
	}
	return c
}

func longASCII(n int) string {
	const unit = "hello world "
	s := strings.Repeat(unit, n/len(unit)+1)
	return s[:n]
}

func diffAt(a, b string) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

func countCost(n int, fn func()) (allocs float64, bytes uint64) {
	fn()
	allocs = testing.AllocsPerRun(n, fn)
	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	for i := 0; i < n; i++ {
		fn()
	}
	runtime.ReadMemStats(&after)
	return allocs, (after.TotalAlloc - before.TotalAlloc) / uint64(n)
}
