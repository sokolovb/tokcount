package tokcount

// cutter splits pieces at points no merge can cross. SentencePiece-style
// tokenizers (Gemma) hand the whole normalized text to BPE as one piece; a
// merge can only join symbols a and b if some vocabulary token contains the
// characters ab, so wherever no token does, the two sides can be counted
// independently with the same result. Cuts are only considered next to
// delimiter characters (the space marker and newline), which keeps the table
// small and the pieces word-sized and cacheable.
type cutter struct {
	delims  map[rune]bool
	joining map[[2]rune]bool // adjacent pairs, involving a delimiter, that occur inside some token
}

// newCutter returns a cutter for a character-level model, or nil if
// cutting could change its counts.
func newCutter(m *bpeModel) *cutter {
	// ignore_merges looks up the whole piece; parts of it would be looked up
	// on their own and could come out shorter.
	if m.ignoreMerges {
		return nil
	}
	// Symbols that do not stand for their own text (byte fallback, unk) could
	// merge across a cut if merges mention them.
	special := map[uint32]bool{}
	allFallback := true
	for _, id := range m.fallbackIDs {
		if id >= 0 {
			special[uint32(id)] = true
		} else {
			allFallback = false
		}
	}
	if m.unkID >= 0 {
		special[uint32(m.unkID)] = true
	}
	for k := range m.merges {
		if special[uint32(k>>32)] || special[uint32(k)] {
			return nil
		}
	}
	// Without full byte fallback a character can become unk, and a fused run
	// of unk spans cuts.
	if m.unkID >= 0 && m.fuseUnk && !allFallback {
		return nil
	}
	c := &cutter{delims: map[rune]bool{'▁': true, '\n': true}, joining: map[[2]rune]bool{}}
	for tok := range m.vocab {
		prev := rune(-1)
		for _, r := range tok {
			if prev >= 0 && (c.delims[prev] || c.delims[r]) {
				c.joining[[2]rune{prev, r}] = true
			}
			prev = r
		}
	}
	return c
}

// next returns the text of s up to its first cut, and the rest. If st says
// to stop it returns all of s, which callers discard after checking st.
func (c *cutter) next(st *state, s string) (word, rest string) {
	prev := rune(-1)
	for i := 0; i < len(s); {
		if !st.tick(1) {
			return s, ""
		}
		r, sz := decodeRune(s, i)
		if prev >= 0 && (c.delims[prev] || c.delims[r]) && !c.joining[[2]rune{prev, r}] {
			return s[:i], s[i:]
		}
		prev = r
		i += sz
	}
	return s, ""
}
