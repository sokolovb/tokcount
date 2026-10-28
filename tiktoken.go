package tokcount

// tiktoken BPE files: one "base64(token) rank" line per token, the rank
// doubling as the token id. The pre-tokenization pattern and the special
// tokens are not in the file; they come from the model's tokenizer code.

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"fmt"
	"os"
	"strconv"
)

// TiktokenConfig describes a tiktoken-based tokenizer.
type TiktokenConfig struct {
	// Pattern is the pre-tokenization regex (tiktoken's pat_str).
	Pattern string
	// Specials are the special tokens. Counting only needs their text.
	Specials []string
}

// LoadTiktoken loads a tiktoken BPE file with the given configuration.
func LoadTiktoken(path string, cfg TiktokenConfig, opts *Options) (*Counter, error) {
	if opts == nil {
		opts = &Options{}
	}
	ranks, err := readTiktokenRanks(path)
	if err != nil {
		return nil, err
	}
	spec, err := tiktokenSpec(ranks, cfg)
	if err != nil {
		return nil, err
	}
	return &Counter{p: newPipeline(spec, opts)}, nil
}

func tiktokenSpec(ranks map[string]uint32, cfg TiktokenConfig) (pipelineSpec, error) {
	m, err := tiktokenModel(ranks)
	if err != nil {
		return pipelineSpec{}, fmt.Errorf("tokcount: %w", err)
	}
	re, err := compilePattern(cfg.Pattern)
	if err != nil {
		return pipelineSpec{}, fmt.Errorf("tokcount: %w", err)
	}
	added := make([]addedToken, len(cfg.Specials))
	for i, s := range cfg.Specials {
		added[i] = addedToken{content: s, special: true}
	}
	return pipelineSpec{
		added:  added,
		splits: []splitStep{{re: re, behavior: splitMatchesOnly}},
		model:  m,
	}, nil
}

// tiktokenModel turns tiktoken ranks into an equivalent BPE model. tiktoken
// merges the adjacent pair whose concatenation has the lowest rank, so the
// merge table gets an entry for every way of splitting a token into two
// tokens, ranked by the token itself; ties go to the leftmost pair in both.
// A piece that is a token is never split, which is ignore_merges.
func tiktokenModel(ranks map[string]uint32) (bpeModel, error) {
	merges := map[uint64]uint64{}
	for tok, r := range ranks {
		for k := 1; k < len(tok); k++ {
			a, okA := ranks[tok[:k]]
			b, okB := ranks[tok[k:]]
			if okA && okB {
				merges[pairKey(a, b)] = uint64(r)<<32 | uint64(r)
			}
		}
	}
	m, err := newBPEModel(ranks, merges, bpeOptions{byteLevel: true, ignoreMerges: true})
	if err != nil {
		return bpeModel{}, err
	}
	for b, id := range m.byteIDs {
		if id < 0 {
			return bpeModel{}, fmt.Errorf("ranks lack the single byte %#02x", b)
		}
	}
	return m, nil
}

func readTiktokenRanks(path string) (map[string]uint32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	ranks := make(map[string]uint32)
	sc := bufio.NewScanner(bytes.NewReader(data))
	sc.Buffer(nil, 1<<20)
	for line := 1; sc.Scan(); line++ {
		fields := bytes.Fields(sc.Bytes())
		if len(fields) == 0 {
			continue
		}
		if len(fields) != 2 {
			return nil, fmt.Errorf("tokcount: %s:%d: malformed line", path, line)
		}
		tok, err := base64.StdEncoding.DecodeString(string(fields[0]))
		if err != nil {
			return nil, fmt.Errorf("tokcount: %s:%d: %w", path, line, err)
		}
		rank, err := strconv.ParseUint(string(fields[1]), 10, 32)
		if err != nil {
			return nil, fmt.Errorf("tokcount: %s:%d: %w", path, line, err)
		}
		ranks[string(tok)] = uint32(rank)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return ranks, nil
}
