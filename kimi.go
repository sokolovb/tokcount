package tokcount

// Kimi (K2, K2.5, K2.6, K3, Kimi Linear): tiktoken ranks plus the details of
// TikTokenTokenizer in the model's tokenization_kimi.py.

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"unicode/utf8"
)

// KimiPattern is the pre-tokenization pattern of Kimi K2, K2.5, K2.6, K3
// and Kimi Linear.
const KimiPattern = `[\p{Han}]+` +
	`|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?` +
	`|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?` +
	`|\p{N}{1,3}` +
	`| ?[^\s\p{L}\p{N}]+[\r\n]*` +
	`|\s*[\r\n]+` +
	`|\s+(?!\S)` +
	`|\s+`

// Kimi's encode cuts text into parts encoded independently, which changes
// the count of long texts.
const (
	kimiMaxEncodeChars = 400_000 // code points per chunk
	kimiMaxRunChars    = 25_000  // consecutive whitespace or non-whitespace code points per part
)

// LoadKimi loads a Kimi tokenizer from a model directory containing
// tiktoken.model and tokenizer_config.json, reproducing
// TikTokenTokenizer.encode from the model's tokenization_kimi.py.
func LoadKimi(dir string, opts *Options) (*Counter, error) {
	if opts == nil {
		opts = &Options{}
	}
	ranks, err := readTiktokenRanks(filepath.Join(dir, "tiktoken.model"))
	if err != nil {
		return nil, err
	}
	var cfg struct {
		AddedTokensDecoder map[string]struct {
			Content string `json:"content"`
		} `json:"added_tokens_decoder"`
	}
	data, err := os.ReadFile(filepath.Join(dir, "tokenizer_config.json"))
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("tokcount: parsing tokenizer_config.json: %w", err)
	}
	// Ids from len(ranks) on are 256 reserved special tokens, named by
	// added_tokens_decoder or <|reserved_token_N|>.
	const numReserved = 256
	specials := make([]string, 0, numReserved)
	for id := len(ranks); id < len(ranks)+numReserved; id++ {
		name := fmt.Sprintf("<|reserved_token_%d|>", id)
		if d, ok := cfg.AddedTokensDecoder[strconv.Itoa(id)]; ok {
			name = d.Content
		}
		specials = append(specials, name)
	}
	spec, err := tiktokenSpec(ranks, TiktokenConfig{Pattern: KimiPattern, Specials: specials})
	if err != nil {
		return nil, err
	}
	spec.chunk = kimiChunks
	return &Counter{p: newPipeline(spec, opts)}, nil
}

// kimiChunks cuts text the way _encode_text_piece does before encoding each
// part independently: into chunks of at most kimiMaxEncodeChars code points,
// each cut further by splitRuns.
func kimiChunks(text string, fn func(string) bool) bool {
	for len(text) > 0 {
		chunk := text[:prefixRunes(text, kimiMaxEncodeChars)]
		text = text[len(chunk):]
		if !splitRuns(chunk, kimiMaxRunChars, fn) {
			return false
		}
	}
	return true
}

// prefixRunes returns the byte length of the first n code points of s.
func prefixRunes(s string, n int) int {
	i := 0
	for ; n > 0 && i < len(s); n-- {
		_, sz := decodeRune(s, i)
		i += sz
	}
	return i
}

// splitRuns ports Kimi's _split_whitespaces_or_nonwhitespaces: it cuts s so
// that no part has more than maxRun consecutive whitespace or non-whitespace
// code points.
func splitRuns(s string, maxRun int, fn func(string) bool) bool {
	if s == "" {
		return fn(s)
	}
	r0, _ := utf8.DecodeRuneInString(s)
	curSpace, curLen, start := pyIsSpace(r0), 0, 0
	for i := 0; i < len(s); {
		r, sz := decodeRune(s, i)
		if isSpace := pyIsSpace(r); isSpace != curSpace {
			curLen, curSpace = 1, isSpace
		} else {
			curLen++
			if curLen > maxRun {
				if !fn(s[start:i]) {
					return false
				}
				start, curLen = i, 1
			}
		}
		i += sz
	}
	return fn(s[start:])
}

// pyIsSpace is Python's str.isspace for one character: bidirectional class
// WS, B or S, or category Zs.
func pyIsSpace(r rune) bool {
	switch {
	case r >= 0x09 && r <= 0x0D, r >= 0x1C && r <= 0x20, r == 0x85, r == 0xA0, r == 0x1680,
		r >= 0x2000 && r <= 0x200A, r == 0x2028, r == 0x2029, r == 0x202F, r == 0x205F, r == 0x3000:
		return true
	}
	return false
}
