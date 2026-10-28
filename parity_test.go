package tokcount

// Parity with the reference implementations. testdata/expected.json is
// produced by tools/gen_reference.py from the tokenizer files in
// testdata/models; TOKCOUNT_MODELS overrides that directory.

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync"
	"testing"
	"time"
)

type expectedModel struct {
	Kind    string `json:"kind"`
	Dir     string `json:"dir"`
	Options struct {
		SpecialAsText bool `json:"special_as_text"`
	} `json:"options"`
	Counts []int `json:"counts"`
}

func loadCorpus(t testing.TB) []string {
	t.Helper()
	data, err := os.ReadFile("testdata/corpus.jsonl.gz")
	if err != nil {
		t.Skipf("no corpus: %v", err)
	}
	gz, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	var corpus []string
	sc := bufio.NewScanner(gz)
	sc.Buffer(nil, 64<<20)
	for sc.Scan() {
		var s string
		if err := json.Unmarshal(sc.Bytes(), &s); err != nil {
			t.Fatal(err)
		}
		corpus = append(corpus, s)
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	return corpus
}

func loadExpected(t testing.TB) (string, map[string]*expectedModel) {
	t.Helper()
	dir := os.Getenv("TOKCOUNT_MODELS")
	if dir == "" {
		dir = "testdata/models"
	}
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("no tokenizer files: %v (run tools/fetch_models.sh)", err)
	}
	data, err := os.ReadFile("testdata/expected.json")
	if err != nil {
		t.Skipf("no expected counts: %v", err)
	}
	var exp map[string]*expectedModel
	if err := json.Unmarshal(data, &exp); err != nil {
		t.Fatal(err)
	}
	return dir, exp
}

func sortedKeys(exp map[string]*expectedModel) []string {
	keys := make([]string, 0, len(exp))
	for k := range exp {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func largestText(corpus []string) string {
	big := ""
	for _, s := range corpus {
		if len(s) > len(big) {
			big = s
		}
	}
	return big
}

// loadModel loads the tokenizer of one expected model, skipping the test if
// its files have not been fetched.
func loadModel(tb testing.TB, dir string, m *expectedModel, cacheSize int) *Counter {
	tb.Helper()
	path := filepath.Join(dir, m.Dir)
	if _, err := os.Stat(path); err != nil {
		tb.Skipf("%s: %v (run tools/fetch_models.sh)", m.Dir, err)
	}
	opts := &Options{SpecialTokensAsText: m.Options.SpecialAsText, CacheSize: cacheSize}
	var c *Counter
	var err error
	if m.Kind == "kimi" {
		c, err = LoadKimi(path, opts)
	} else {
		c, err = LoadHF(filepath.Join(path, "tokenizer.json"), opts)
	}
	if err != nil {
		tb.Fatal(err)
	}
	return c
}

func TestParity(t *testing.T) {
	dir, exp := loadExpected(t)
	corpus := loadCorpus(t)
	for _, key := range sortedKeys(exp) {
		m := exp[key]
		t.Run(key, func(t *testing.T) {
			start := time.Now()
			c := loadModel(t, dir, m, 0)
			t.Logf("loaded in %v", time.Since(start))
			if len(m.Counts) != len(corpus) {
				t.Fatalf("expected %d counts, corpus has %d texts", len(m.Counts), len(corpus))
			}
			// Count concurrently on one Counter, as a server would.
			got := make([]int, len(corpus))
			idx := make(chan int)
			var wg sync.WaitGroup
			for w := 0; w < runtime.GOMAXPROCS(0); w++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					for i := range idx {
						n, err := c.Count(context.Background(), corpus[i])
						if err != nil {
							t.Error(err)
						}
						got[i] = n
					}
				}()
			}
			start = time.Now()
			for i := range corpus {
				idx <- i
			}
			close(idx)
			wg.Wait()
			t.Logf("counted %d texts in %v", len(corpus), time.Since(start))
			bad := 0
			for i, want := range m.Counts {
				if got[i] != want {
					if bad < 10 {
						t.Errorf("text %d (%d bytes) %s: got %d, want %d", i, len(corpus[i]), preview(corpus[i]), got[i], want)
					}
					bad++
				}
			}
			if bad > 0 {
				t.Errorf("%d of %d texts differ", bad, len(corpus))
			}
		})
	}
}

func preview(s string) string {
	r := []rune(s)
	if len(r) > 60 {
		return fmt.Sprintf("%q…", string(r[:60]))
	}
	return fmt.Sprintf("%q", s)
}

// Cached and uncached counting must agree, and CountUpTo and Fits must agree
// with Count wherever they do not stop early.
func TestCacheAndLimitConsistency(t *testing.T) {
	dir, exp := loadExpected(t)
	corpus := loadCorpus(t)
	for _, key := range sortedKeys(exp) {
		m := exp[key]
		t.Run(key, func(t *testing.T) {
			noCache := loadModel(t, dir, m, -1)
			ctx := context.Background()
			for i, text := range corpus {
				if i%7 != 0 {
					continue
				}
				n, err := noCache.Count(ctx, text)
				if err != nil || n != m.Counts[i] {
					t.Fatalf("text %d uncached: got %d, %v; want %d", i, n, err, m.Counts[i])
				}
				for _, limit := range []int{0, n / 2, n - 1, n, n + 1} {
					if limit < 0 {
						continue // means no limit
					}
					got, exceeded, err := noCache.CountUpTo(ctx, text, limit)
					if err != nil {
						t.Fatal(err)
					}
					if exceeded != (n > limit) || !exceeded && got != n || exceeded && got <= limit {
						t.Fatalf("text %d limit %d: got (%d, %v), count %d", i, limit, got, exceeded, n)
					}
					fits, err := noCache.Fits(ctx, text, limit)
					if err != nil || fits != (n <= limit) {
						t.Fatalf("text %d Fits(%d) = %v, %v; count %d", i, limit, fits, err, n)
					}
				}
			}
		})
	}
}

func TestCancellation(t *testing.T) {
	dir, exp := loadExpected(t)
	big := largestText(loadCorpus(t))
	for _, key := range []string{"qwen3.8", "kimi-k3", "gemma-4"} {
		m, ok := exp[key]
		if !ok {
			continue
		}
		t.Run(key, func(t *testing.T) {
			c := loadModel(t, dir, m, -1)
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			if _, err := c.Count(ctx, big); err != context.Canceled {
				t.Fatalf("cancelled context: got %v", err)
			}
			ctx, cancel = context.WithTimeout(context.Background(), time.Millisecond)
			defer cancel()
			start := time.Now()
			if _, err := c.Count(ctx, big+big+big+big); err != context.DeadlineExceeded {
				t.Fatalf("deadline: got %v", err)
			}
			if d := time.Since(start); d > 50*time.Millisecond*raceSlowdown {
				t.Fatalf("took %v to notice the deadline", d)
			}
		})
	}
}

func BenchmarkCount(b *testing.B) {
	dir, exp := loadExpected(b)
	big := largestText(loadCorpus(b))
	for _, key := range sortedKeys(exp) {
		m := exp[key]
		if m.Options.SpecialAsText {
			continue
		}
		for _, cacheSize := range []int{0, -1} {
			name := key
			if cacheSize < 0 {
				name += "/nocache"
			}
			b.Run(name, func(b *testing.B) {
				c := loadModel(b, dir, m, cacheSize)
				b.SetBytes(int64(len(big)))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if _, err := c.Count(context.Background(), big); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}
