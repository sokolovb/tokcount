// Package tokcount counts tokens for LLM tokenizers in pure Go.
//
// It loads Hugging Face tokenizer.json files (byte-level BPE with regex
// pre-tokenization, and SentencePiece-style BPE with byte fallback) and
// tiktoken BPE files, and reproduces the token counts of the reference
// implementations without producing the tokens themselves.
//
// Counting is cooperative: it checks the context as it goes, can stop as soon
// as a limit is exceeded, and never leaves Go, so a huge input can always be
// abandoned. A Counter is safe for concurrent use.
package tokcount

import (
	"context"
	"hash/maphash"
	"strings"
	"sync"
)

// Options tune a Counter. The zero value matches the reference tokenizers'
// defaults.
type Options struct {
	// SpecialTokensAsText counts special tokens that appear in the input
	// (such as "<|im_end|>") as ordinary text instead of as one token each.
	// This is split_special_tokens=True in HF tokenizers and
	// allow_special_tokens=False in Kimi's tokenizer.
	SpecialTokensAsText bool

	// CacheSize bounds the number of cached per-word counts. Zero means
	// DefaultCacheSize; a negative value disables the cache.
	CacheSize int
}

const DefaultCacheSize = 1 << 16

// Counter counts tokens for one tokenizer.
type Counter struct {
	p *pipeline
}

// Count returns the number of tokens text encodes to, without any special
// tokens a template or post-processor would add (add_special_tokens=False).
// It returns ctx.Err() if ctx is done before counting finishes.
func (c *Counter) Count(ctx context.Context, text string) (int, error) {
	n, _, err := c.CountUpTo(ctx, text, -1)
	return n, err
}

// CountUpTo counts like Count but stops as soon as the count exceeds limit.
// If exceeded is false, n is the exact count; otherwise n is only known to be
// greater than limit. A negative limit means no limit.
func (c *Counter) CountUpTo(ctx context.Context, text string, limit int) (n int, exceeded bool, err error) {
	if err := ctx.Err(); err != nil {
		return 0, false, err
	}
	st := &state{ctx: ctx, limit: limit}
	st.sc = scratchPool.Get().(*scratch)
	c.p.count(st, text)
	// Buffers grown by one huge piece are not worth keeping around.
	if cap(st.sc.syms) <= maxPooledSyms {
		scratchPool.Put(st.sc)
	}
	if st.err != nil {
		return 0, false, st.err
	}
	return st.n, st.over(), nil
}

// Fits reports whether text encodes to at most limit tokens. It answers
// without tokenizing when the length of the text alone guarantees it. On
// error it reports false: text that was not counted does not fit.
func (c *Counter) Fits(ctx context.Context, text string, limit int) (bool, error) {
	if err := ctx.Err(); err != nil {
		return false, err
	}
	if limit >= 0 && len(text) <= limit && c.p.lenIsBound(text) {
		return true, nil
	}
	_, exceeded, err := c.CountUpTo(ctx, text, limit)
	if err != nil {
		return false, err
	}
	return !exceeded, nil
}

// state is the progress of one counting call.
type state struct {
	ctx   context.Context
	n     int
	limit int
	work  int
	err   error
	sc    *scratch
}

const maxPooledSyms = 1 << 16

// backgroundState is a state for work outside a counting call.
func backgroundState() *state { return &state{ctx: context.Background(), limit: -1} }

// ctxCheckEvery is how many units of work (characters scanned, merges done)
// pass between looks at the context.
const ctxCheckEvery = 1 << 14

// tick records w units of work and reports whether counting may go on. It
// inlines to an add and two compares, so loops call it on every step.
func (st *state) tick(w int) bool {
	st.work += w
	if st.work >= ctxCheckEvery {
		return st.check()
	}
	return st.err == nil
}

// check looks at the context; it is kept out of tick so tick inlines.
func (st *state) check() bool {
	st.work = 0
	if err := st.ctx.Err(); err != nil {
		st.err = err
	}
	return st.err == nil
}

// add adds k tokens and reports whether counting may go on.
func (st *state) add(k int) bool {
	st.n += k
	return st.err == nil && !st.over()
}

func (st *state) over() bool { return st.limit >= 0 && st.n > st.limit }

func (st *state) stopped() bool { return st.err != nil || st.over() }

// cache memoizes per-piece counts. Pieces repeat heavily in natural text, so
// most of them skip BPE entirely.
type cache struct {
	seed     maphash.Seed
	perShard int
	shards   [64]cacheShard
}

type cacheShard struct {
	mu sync.RWMutex
	m  map[string]int
}

// Longer pieces are rare and not worth keeping.
const maxCachedPiece = 128

func newCache(size int) *cache {
	if size == 0 {
		size = DefaultCacheSize
	}
	if size < 0 {
		return nil
	}
	c := &cache{seed: maphash.MakeSeed(), perShard: max(1, size/64)}
	for i := range c.shards {
		c.shards[i].m = make(map[string]int)
	}
	return c
}

func (c *cache) shard(k string) *cacheShard {
	return &c.shards[maphash.String(c.seed, k)%uint64(len(c.shards))]
}

func (c *cache) get(k string) (int, bool) {
	if c == nil || len(k) > maxCachedPiece {
		return 0, false
	}
	s := c.shard(k)
	s.mu.RLock()
	n, ok := s.m[k]
	s.mu.RUnlock()
	return n, ok
}

func (c *cache) put(k string, n int) {
	if c == nil || len(k) > maxCachedPiece {
		return
	}
	s := c.shard(k)
	s.mu.Lock()
	if len(s.m) >= c.perShard {
		clear(s.m)
	}
	s.m[strings.Clone(k)] = n // k points into the caller's text
	s.mu.Unlock()
}
