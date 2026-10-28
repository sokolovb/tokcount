package tokcount

// trie finds added/special tokens in text, leftmost-longest, the way HF
// tokenizers' Aho-Corasick matcher does.
type trie struct {
	nodes []trieNode
	first [256]bool
}

type trieNode struct {
	next  map[byte]int32
	final bool
}

func newTrie(words []string) *trie {
	t := &trie{nodes: []trieNode{{}}}
	for _, w := range words {
		if w == "" {
			continue
		}
		t.first[w[0]] = true
		cur := int32(0)
		for i := 0; i < len(w); i++ {
			nx, ok := t.nodes[cur].next[w[i]]
			if !ok {
				if t.nodes[cur].next == nil {
					t.nodes[cur].next = make(map[byte]int32)
				}
				nx = int32(len(t.nodes))
				t.nodes[cur].next[w[i]] = nx
				t.nodes = append(t.nodes, trieNode{})
			}
			cur = nx
		}
		t.nodes[cur].final = true
	}
	return t
}

// split calls fn for the text between tokens (isToken=false, never empty)
// and for each token found, in order. It stops early when fn returns false.
func (t *trie) split(s string, fn func(part string, isToken bool) bool) bool {
	if t == nil || len(t.nodes) == 1 {
		return s == "" || fn(s, false)
	}
	start := 0
	for i := 0; i < len(s); i++ {
		if !t.first[s[i]] {
			continue
		}
		end, cur := -1, int32(0)
		for j := i; j < len(s); j++ {
			nx, ok := t.nodes[cur].next[s[j]]
			if !ok {
				break
			}
			cur = nx
			if t.nodes[cur].final {
				end = j + 1
			}
		}
		if end < 0 {
			continue
		}
		if start < i && !fn(s[start:i], false) {
			return false
		}
		if !fn(s[i:end], true) {
			return false
		}
		start = end
		i = end - 1
	}
	if start < len(s) {
		return fn(s[start:], false)
	}
	return true
}
