package tokcount

// BPE counting. Both reference implementations repeatedly merge the adjacent
// pair with the lowest rank, leftmost first on ties. HF tokenizers ranks a
// pair by its position in the merges list; tiktoken ranks it by the rank of
// the merged token, which tiktokenModel turns into the same kind of merge
// table. Merging uses a priority queue with lazy invalidation, O(n log n) in
// the length of the piece, so a long piece does not go quadratic.

import (
	"fmt"
	"sync"
)

// bpeModel is a BPE vocabulary with its merge table.
type bpeModel struct {
	// vocab maps token text to id. Byte-level models key it by the raw bytes
	// a token stands for, not by its byte-alphabet spelling.
	vocab  map[string]uint32
	merges map[uint64]uint64 // (left<<32|right) → (rank<<32|merged id)

	byteLevel    bool // pieces start as bytes rather than characters
	ignoreMerges bool // a piece that is itself a token stays whole
	fuseUnk      bool

	byteIDs     [256]int64 // byte-level: id of each byte, or -1
	fallbackIDs [256]int64 // id of "<0xXX>" with byte fallback, else -1
	unkID       int64      // -1 if none
}

// bpeOptions are the BPE settings that vary between tokenizers.
type bpeOptions struct {
	byteLevel    bool // pieces start as bytes rather than characters
	byteFallback bool
	ignoreMerges bool // a piece that is itself a token stays whole
	fuseUnk      bool
	unk          string // the unk token, "" if none
}

// newBPEModel derives the lookup tables of a model from its vocabulary.
func newBPEModel(vocab map[string]uint32, merges map[uint64]uint64, o bpeOptions) (bpeModel, error) {
	m := bpeModel{
		vocab: vocab, merges: merges,
		byteLevel: o.byteLevel, ignoreMerges: o.ignoreMerges, fuseUnk: o.fuseUnk,
		unkID: -1,
	}
	for b := range m.byteIDs {
		m.byteIDs[b], m.fallbackIDs[b] = -1, -1
		if id, ok := vocab[string([]byte{byte(b)})]; ok && o.byteLevel {
			m.byteIDs[b] = int64(id)
		}
		if id, ok := vocab[fmt.Sprintf("<0x%02X>", b)]; ok && o.byteFallback {
			m.fallbackIDs[b] = int64(id)
		}
	}
	if o.unk != "" {
		id, ok := vocab[o.unk]
		if !ok {
			return bpeModel{}, fmt.Errorf("unk token %q is not in the vocabulary", o.unk)
		}
		m.unkID = int64(id)
	}
	return m, nil
}

// count returns the number of tokens piece encodes to. If st says to stop
// it returns 0, which callers discard after checking st.
func (m *bpeModel) count(st *state, piece string) int {
	if m.ignoreMerges {
		if _, ok := m.vocab[piece]; ok {
			return 1
		}
	}
	sc := st.sc
	sc.reset()
	if m.byteLevel {
		for i := 0; i < len(piece); i++ {
			if !st.tick(1) {
				return 0
			}
			if id := m.byteIDs[piece[i]]; id >= 0 {
				sc.add(m, uint32(id))
			} else {
				sc.addMissing(m, string(byteEnc[piece[i]]))
			}
		}
	} else {
		for i := 0; i < len(piece); {
			if !st.tick(1) {
				return 0
			}
			_, sz := decodeRune(piece, i)
			char := piece[i : i+sz]
			if id, ok := m.vocab[char]; ok {
				sc.add(m, id)
			} else {
				sc.addMissing(m, char)
			}
			i += sz
		}
	}
	sc.flushUnk(m)
	return sc.merge(m.merges, st)
}

// A character missing from the vocabulary starts out as HF has it: as its
// bytes with byte fallback, else as the unk token (a run of them fused into
// one with fuse_unk), else not at all. byte-level models only get here for
// bytes missing from the vocabulary, with the byte's alphabet character.
type missingKind uint8

const (
	missingDropped missingKind = iota
	missingBytes
	missingUnk
)

func (m *bpeModel) missing(char string) missingKind {
	for i := 0; i < len(char); i++ {
		if m.fallbackIDs[char[i]] < 0 {
			if m.unkID >= 0 {
				return missingUnk
			}
			return missingDropped
		}
	}
	return missingBytes
}

// cost is the most tokens s can come to: the number of symbols BPE starts it
// from, decided the same way count decides them.
func (m *bpeModel) cost(s string) int {
	n := 0
	if m.byteLevel {
		for i := 0; i < len(s); i++ {
			if m.byteIDs[s[i]] >= 0 {
				n++
			} else {
				n += m.missingCost(string(byteEnc[s[i]]))
			}
		}
		return n
	}
	for _, r := range s {
		if _, ok := m.vocab[string(r)]; ok {
			n++
		} else {
			n += m.missingCost(string(r))
		}
	}
	return n
}

func (m *bpeModel) missingCost(char string) int {
	switch m.missing(char) {
	case missingBytes:
		return len(char)
	case missingUnk:
		return 1
	}
	return 0
}

type sym struct {
	id         uint32
	prev, next int32
	dead       bool
}

type hent struct {
	rank uint32
	pos  int32
}

// scratch holds the per-call buffers for BPE.
type scratch struct {
	syms       []sym
	heap       []hent
	unkPending bool
}

var scratchPool = sync.Pool{New: func() any { return new(scratch) }}

func (sc *scratch) reset() {
	sc.syms = sc.syms[:0]
	sc.unkPending = false
}

func (sc *scratch) add(m *bpeModel, id uint32) {
	sc.flushUnk(m)
	sc.syms = append(sc.syms, sym{id: id})
}

func (sc *scratch) addMissing(m *bpeModel, char string) {
	switch m.missing(char) {
	case missingBytes:
		for i := 0; i < len(char); i++ {
			sc.add(m, uint32(m.fallbackIDs[char[i]]))
		}
	case missingUnk:
		if !sc.unkPending || !m.fuseUnk {
			sc.flushUnk(m)
			sc.unkPending = true
		}
	}
}

func (sc *scratch) flushUnk(m *bpeModel) {
	if sc.unkPending {
		sc.syms = append(sc.syms, sym{id: uint32(m.unkID)})
		sc.unkPending = false
	}
}

// merge runs BPE over sc.syms and returns how many symbols are left. If st
// says to stop, the result is meaningless and callers discard it.
func (sc *scratch) merge(merges map[uint64]uint64, st *state) int {
	n := len(sc.syms)
	if n < 2 {
		return n
	}
	sc.link()
	sc.heap = sc.heap[:0]
	syms := sc.syms
	for i := 0; i+1 < n; i++ {
		if v, ok := merges[pairKey(syms[i].id, syms[i+1].id)]; ok {
			sc.push(uint32(v>>32), int32(i))
		}
		if !st.tick(1) {
			return n
		}
	}
	for len(sc.heap) > 0 {
		if !st.tick(1) {
			return n
		}
		top := sc.pop()
		s := &syms[top.pos]
		if s.dead || s.next < 0 {
			continue
		}
		nx := &syms[s.next]
		v, ok := merges[pairKey(s.id, nx.id)]
		if !ok || uint32(v>>32) != top.rank {
			continue // stale entry
		}
		s.id = uint32(v)
		nx.dead = true
		s.next = nx.next
		if s.next >= 0 {
			syms[s.next].prev = top.pos
		}
		n--
		if s.prev >= 0 {
			if v, ok := merges[pairKey(syms[s.prev].id, s.id)]; ok {
				sc.push(uint32(v>>32), s.prev)
			}
		}
		if s.next >= 0 {
			if v, ok := merges[pairKey(s.id, syms[s.next].id)]; ok {
				sc.push(uint32(v>>32), top.pos)
			}
		}
	}
	return n
}

func pairKey(a, b uint32) uint64 { return uint64(a)<<32 | uint64(b) }

func (sc *scratch) link() {
	for i := range sc.syms {
		sc.syms[i].prev = int32(i - 1)
		sc.syms[i].next = int32(i + 1)
	}
	sc.syms[len(sc.syms)-1].next = -1
}

func (sc *scratch) push(rank uint32, pos int32) {
	h := append(sc.heap, hent{rank, pos})
	i := len(h) - 1
	for i > 0 {
		parent := (i - 1) / 2
		if !hless(h[i], h[parent]) {
			break
		}
		h[i], h[parent] = h[parent], h[i]
		i = parent
	}
	sc.heap = h
}

func (sc *scratch) pop() hent {
	h := sc.heap
	top := h[0]
	last := len(h) - 1
	h[0] = h[last]
	h = h[:last]
	i := 0
	for {
		l := 2*i + 1
		if l >= len(h) {
			break
		}
		m := l
		if r := l + 1; r < len(h) && hless(h[r], h[l]) {
			m = r
		}
		if !hless(h[m], h[i]) {
			break
		}
		h[i], h[m] = h[m], h[i]
		i = m
	}
	sc.heap = h
	return top
}

func hless(a, b hent) bool {
	if a.rank != b.rank {
		return a.rank < b.rank
	}
	return a.pos < b.pos
}
