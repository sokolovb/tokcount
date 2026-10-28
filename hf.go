package tokcount

// Hugging Face tokenizer.json support: turns the file into a pipeline.
//
// Supported components (anything else is rejected at load time):
//   - normalizers: NFC, Replace with a string pattern, Sequence;
//   - pre-tokenizers: Split (regex or string pattern; Isolated or
//     MergedWithPrevious), ByteLevel without its own regex as the last step,
//     Sequence;
//   - models: BPE, byte-level or with byte fallback, with ignore_merges and
//     fuse_unk.

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
)

// LoadHF loads a Hugging Face tokenizer.json file.
func LoadHF(path string, opts *Options) (*Counter, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return ParseHF(data, opts)
}

// ParseHF builds a Counter from the contents of a tokenizer.json file.
func ParseHF(data []byte, opts *Options) (*Counter, error) {
	if opts == nil {
		opts = &Options{}
	}
	var raw hfFile
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("tokcount: parsing tokenizer.json: %w", err)
	}
	spec, err := hfSpec(&raw)
	if err != nil {
		return nil, fmt.Errorf("tokcount: %w", err)
	}
	return &Counter{p: newPipeline(spec, opts)}, nil
}

type hfFile struct {
	AddedTokens []struct {
		Content    string `json:"content"`
		SingleWord bool   `json:"single_word"`
		Lstrip     bool   `json:"lstrip"`
		Rstrip     bool   `json:"rstrip"`
		Normalized bool   `json:"normalized"`
		Special    bool   `json:"special"`
	} `json:"added_tokens"`
	Normalizer   json.RawMessage `json:"normalizer"`
	PreTokenizer json.RawMessage `json:"pre_tokenizer"`
	Model        hfModel         `json:"model"`
}

type hfModel struct {
	Type                    string            `json:"type"`
	Vocab                   map[string]uint32 `json:"vocab"`
	Merges                  json.RawMessage   `json:"merges"`
	Dropout                 *float64          `json:"dropout"`
	UnkToken                *string           `json:"unk_token"`
	ContinuingSubwordPrefix *string           `json:"continuing_subword_prefix"`
	EndOfWordSuffix         *string           `json:"end_of_word_suffix"`
	FuseUnk                 bool              `json:"fuse_unk"`
	ByteFallback            bool              `json:"byte_fallback"`
	IgnoreMerges            bool              `json:"ignore_merges"`
}

func hfSpec(raw *hfFile) (pipelineSpec, error) {
	normalizers, err := parseNormalizer(raw.Normalizer)
	if err != nil {
		return pipelineSpec{}, err
	}
	splits, byteLevel, err := parsePreTokenizer(raw.PreTokenizer, true)
	if err != nil {
		return pipelineSpec{}, err
	}
	model, err := loadHFModel(&raw.Model, byteLevel)
	if err != nil {
		return pipelineSpec{}, err
	}
	added := make([]addedToken, 0, len(raw.AddedTokens))
	for _, at := range raw.AddedTokens {
		if at.SingleWord || at.Lstrip || at.Rstrip {
			return pipelineSpec{}, fmt.Errorf("added token %q: single_word/lstrip/rstrip are not supported", at.Content)
		}
		added = append(added, addedToken{content: at.Content, normalized: at.Normalized, special: at.Special})
	}
	return pipelineSpec{added: added, normalizers: normalizers, splits: splits, model: model}, nil
}

func parseNormalizer(raw json.RawMessage) ([]normalizer, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var n struct {
		Type        string            `json:"type"`
		Normalizers []json.RawMessage `json:"normalizers"`
		Pattern     struct {
			String *string `json:"String"`
		} `json:"pattern"`
		Content string `json:"content"`
	}
	if err := json.Unmarshal(raw, &n); err != nil {
		return nil, fmt.Errorf("normalizer: %w", err)
	}
	switch n.Type {
	case "Sequence":
		var all []normalizer
		for _, sub := range n.Normalizers {
			ns, err := parseNormalizer(sub)
			if err != nil {
				return nil, err
			}
			all = append(all, ns...)
		}
		return all, nil
	case "NFC":
		return []normalizer{nfcNorm{}}, nil
	case "Replace":
		if n.Pattern.String == nil || *n.Pattern.String == "" {
			return nil, errors.New("normalizer: only Replace with a non-empty string pattern is supported")
		}
		return []normalizer{replaceNorm{from: *n.Pattern.String, to: n.Content}}, nil
	}
	return nil, fmt.Errorf("normalizer %q is not supported", n.Type)
}

// parsePreTokenizer returns the split steps of a pre_tokenizer and whether it
// ends in ByteLevel; last says whether nothing follows it.
func parsePreTokenizer(raw json.RawMessage, last bool) (steps []splitStep, byteLevel bool, err error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, false, nil
	}
	var pt struct {
		Type          string            `json:"type"`
		Pretokenizers []json.RawMessage `json:"pretokenizers"`
		Pattern       struct {
			String *string `json:"String"`
			Regex  *string `json:"Regex"`
		} `json:"pattern"`
		Behavior       string `json:"behavior"`
		Invert         bool   `json:"invert"`
		AddPrefixSpace bool   `json:"add_prefix_space"`
		UseRegex       *bool  `json:"use_regex"`
	}
	if err := json.Unmarshal(raw, &pt); err != nil {
		return nil, false, fmt.Errorf("pre_tokenizer: %w", err)
	}
	switch pt.Type {
	case "Sequence":
		for i, sub := range pt.Pretokenizers {
			s, bl, err := parsePreTokenizer(sub, last && i == len(pt.Pretokenizers)-1)
			if err != nil {
				return nil, false, err
			}
			steps, byteLevel = append(steps, s...), byteLevel || bl
		}
		return steps, byteLevel, nil
	case "Split":
		if pt.Invert {
			return nil, false, errors.New("pre_tokenizer: inverted Split is not supported")
		}
		var step splitStep
		switch pt.Behavior {
		case "Isolated":
			step.behavior = splitIsolated
		case "MergedWithPrevious":
			step.behavior = splitMergedWithPrevious
		default:
			return nil, false, fmt.Errorf("pre_tokenizer: Split behavior %q is not supported", pt.Behavior)
		}
		switch {
		case pt.Pattern.Regex != nil:
			step.re, err = compilePattern(*pt.Pattern.Regex)
		case pt.Pattern.String != nil && *pt.Pattern.String != "":
			step.re, err = compileLiteral(*pt.Pattern.String)
		default:
			return nil, false, errors.New("pre_tokenizer: Split without a pattern")
		}
		if err != nil {
			return nil, false, fmt.Errorf("pre_tokenizer: %w", err)
		}
		return []splitStep{step}, false, nil
	case "ByteLevel":
		if pt.UseRegex == nil || *pt.UseRegex || pt.AddPrefixSpace {
			return nil, false, errors.New("pre_tokenizer: only ByteLevel with use_regex=false and add_prefix_space=false is supported")
		}
		if !last {
			return nil, false, errors.New("pre_tokenizer: ByteLevel must be the last step")
		}
		return nil, true, nil
	}
	return nil, false, fmt.Errorf("pre_tokenizer %q is not supported", pt.Type)
}

func loadHFModel(rm *hfModel, byteLevel bool) (bpeModel, error) {
	if rm.Type != "BPE" {
		return bpeModel{}, fmt.Errorf("model %q is not supported", rm.Type)
	}
	if rm.Dropout != nil && *rm.Dropout != 0 {
		return bpeModel{}, errors.New("BPE dropout is not supported")
	}
	if rm.ContinuingSubwordPrefix != nil && *rm.ContinuingSubwordPrefix != "" ||
		rm.EndOfWordSuffix != nil && *rm.EndOfWordSuffix != "" {
		return bpeModel{}, errors.New("BPE continuing_subword_prefix/end_of_word_suffix are not supported")
	}

	pairs, err := parseMerges(rm.Merges)
	if err != nil {
		return bpeModel{}, err
	}
	merges := make(map[uint64]uint64, len(pairs))
	for rank, mg := range pairs {
		a, okA := rm.Vocab[mg[0]]
		b, okB := rm.Vocab[mg[1]]
		c, okC := rm.Vocab[mg[0]+mg[1]]
		if !okA || !okB || !okC {
			return bpeModel{}, fmt.Errorf("merge %q %q refers to tokens missing from the vocabulary", mg[0], mg[1])
		}
		merges[pairKey(a, b)] = uint64(rank)<<32 | uint64(c)
	}

	o := bpeOptions{byteLevel: byteLevel, byteFallback: rm.ByteFallback, ignoreMerges: rm.IgnoreMerges, fuseUnk: rm.FuseUnk}
	if rm.UnkToken != nil {
		o.unk = *rm.UnkToken
	}
	vocab := rm.Vocab
	if byteLevel {
		// Key the vocabulary by the raw bytes each token stands for, so
		// pieces are looked up without mapping them to the byte alphabet.
		vocab = make(map[string]uint32, len(rm.Vocab))
		for tok, id := range rm.Vocab {
			if b, ok := decodeByteLevel(tok); ok {
				vocab[b] = id
			}
		}
		if b, ok := decodeByteLevel(o.unk); ok {
			o.unk = b
		}
	}
	return newBPEModel(vocab, merges, o)
}

func parseMerges(raw json.RawMessage) ([][2]string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var pairs [][2]string
	if err := json.Unmarshal(raw, &pairs); err == nil {
		return pairs, nil
	}
	var lines []string
	if err := json.Unmarshal(raw, &lines); err != nil {
		return nil, fmt.Errorf("merges: %w", err)
	}
	pairs = make([][2]string, len(lines))
	for i, l := range lines {
		a, b, ok := strings.Cut(l, " ")
		if !ok {
			return nil, fmt.Errorf("merges: bad entry %q", l)
		}
		pairs[i] = [2]string{a, b}
	}
	return pairs, nil
}

// GPT-2's byte-to-unicode alphabet used by byte-level BPE vocabularies.
var byteEnc, byteDec = func() ([256]rune, map[rune]byte) {
	var enc [256]rune
	dec := make(map[rune]byte, 256)
	n := 0
	for b := 0; b < 256; b++ {
		if b >= '!' && b <= '~' || b >= 0xA1 && b <= 0xAC || b >= 0xAE && b <= 0xFF {
			enc[b] = rune(b)
		} else {
			enc[b] = rune(256 + n)
			n++
		}
		dec[enc[b]] = byte(b)
	}
	return enc, dec
}()

func decodeByteLevel(tok string) (string, bool) {
	b := make([]byte, 0, len(tok))
	for _, r := range tok {
		c, ok := byteDec[r]
		if !ok {
			return "", false
		}
		b = append(b, c)
	}
	return string(b), true
}
