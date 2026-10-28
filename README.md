# tokcount

[Русская версия](README.ru.md)

Token counting for LLM tokenizers in pure Go: no cgo, no Rust, no WASM. The
only dependency is `golang.org/x/text` (NFC, for Qwen).

tokcount does not produce tokens, it only counts them, and the counts match the
reference implementations: HF `tokenizers` for `tokenizer.json`, and the
model's own `tokenization_kimi.py` (tiktoken) for Kimi.

Counting never leaves Go, checks its context as it goes and can stop as soon as
a limit is exceeded, so a huge input can always be abandoned.

```bash
go get github.com/sokolovb/tokcount
```

The API may still change before v1.

## Supported models

| Family | Files | Loader | Verified on |
|---|---|---|---|
| Kimi | `tiktoken.model`, `tokenizer_config.json` | `LoadKimi(dir, opts)` | K3, Kimi Linear (K2, K2.5 and K2.6 use the same files and pattern) |
| DeepSeek | `tokenizer.json` | `LoadHF(path, opts)` | V4.1-Flash, V3.2-Exp (V4-Pro and R1 have the same configuration) |
| GLM | `tokenizer.json` | `LoadHF` | GLM-5.3 (GLM-5 = GLM-5.3) |
| Qwen | `tokenizer.json` | `LoadHF` | Qwen3.8, Qwen3 (Qwen3.5 = Qwen3.8) |
| GPT-OSS | `tokenizer.json` | `LoadHF` | gpt-oss-120b |
| Gemma | `tokenizer.json` | `LoadHF` | Gemma 4 |

Other `tokenizer.json` files load too if they only use supported components.
An unknown normalizer, pre-tokenizer or regex construct makes `LoadHF` return
an error rather than count wrong. Other tiktoken models load with
`LoadTiktoken(path, TiktokenConfig{Pattern, Specials}, opts)`.

## API

```go
c, err := tokcount.LoadHF("tokenizer.json", nil)

n, err := c.Count(ctx, text)                      // exact count; ctx.Err() if cancelled
n, exceeded, err := c.CountUpTo(ctx, text, limit) // stops as soon as n > limit
ok, err := c.Fits(ctx, text, limit)               // answers from len(text) alone when it can
```

- Only the content is counted, like `encode(text, add_special_tokens=False)`:
  no BOS, no chat template.
- Special tokens in the text (`<|im_end|>` and the like) count as one token
  each, as in the reference. `Options{SpecialTokensAsText: true}` counts them
  as ordinary text: `split_special_tokens=True` in HF,
  `allow_special_tokens=False` in Kimi.
- A `Counter` is safe for concurrent use. Its tables are immutable, and the
  per-word count cache is sharded and bounded (`Options.CacheSize`).

## Control over execution

- The context is checked throughout: while scanning with the pre-tokenization
  regex, while normalizing (in 32 KB chunks cut at NFC boundaries), and while
  building and merging BPE symbols. On cancellation `Count` returns
  `ctx.Err()`. On an M4 Pro, calls return at most ~6 ms after the deadline,
  including on 10 MB of adversarial input (spaces, digits, one letter, CJK),
  independently of input size.
- BPE uses a heap, O(n log n), so even a multi-megabyte "word" without spaces
  does not go quadratic. The worst case measured is about 300 ms per MB on one
  core.
- Gemma hands the whole text to BPE as one piece. tokcount cuts it where no
  merge can cross, before `▁` and `\n` when no vocabulary token contains the
  pair of characters at the cut. Counts are unchanged, and the pieces become
  cacheable words.
- Buffers grown by a huge input are not returned to the pool, so a one-off
  multi-megabyte string does not pin memory after the call.

## Performance

1.2 MB of text (code, documentation, multilingual prose), one core of an Apple
M4 Pro:

| Model | tokcount, no cache | tokcount, cached | reference (Rust via Python) |
|---|---|---|---|
| DeepSeek V4.1 | 21 MB/s | 37 MB/s | 2.9 MB/s |
| Gemma 4 | 34 MB/s | 98 MB/s | 5.1 MB/s |
| GLM-5.3 | 28 MB/s | 55 MB/s | 3.4 MB/s |
| GPT-OSS | 28 MB/s | 55 MB/s | 3.8 MB/s |
| Qwen3.8 | 22 MB/s | 44 MB/s | 3.3 MB/s |
| Kimi K3 | 24 MB/s | 44 MB/s | 13.9 MB/s (tiktoken) |

The reference also produces the tokens and their offsets, so this is not a
like-for-like comparison. Loading a tokenizer takes 40–400 ms.

## Tests

```bash
tools/fetch_models.sh   # tokenizer files into testdata/models, ~170 MB
go test ./...
go test -race ./...
```

Unit tests need nothing else. The parity tests need the tokenizer files, which
are not in the repository; `tools/fetch_models.sh` downloads them from the
Hugging Face Hub at the revisions the reference counts were made from. Without
them the parity tests are skipped. `TOKCOUNT_MODELS` points the tests at
another directory.

`TestParity` checks 12 configurations against 3572 texts (2.2 M characters):
upper-case contractions, `ſ`, combining marks (including zalgo runs longer
than 30 marks), CJK, emoji with ZWJ, NBSP and other Unicode spaces, control
characters, special tokens inside text, long runs of whitespace, texts over
400k characters (Kimi's chunk size). Counting runs concurrently on one
`Counter`, and the tests pass under `-race`.

To regenerate the reference counts in `testdata/`:

```bash
python3 -m venv .venv && .venv/bin/pip install -r tools/requirements.txt
.venv/bin/python tools/gen_reference.py
```

To add a model, add it to `tools/fetch_models.sh` and to `MODELS` in
`tools/gen_reference.py`.

## Known limitations

- **Unicode version.** `\p{L}`, `\p{N}` and other classes come from Go's tables
  (Unicode 15.0). Characters added in Unicode 15.1 and 16 may be classified
  differently than in the reference engines.
- **Invalid UTF-8.** The reference never sees it (Python strings are always
  valid). Here invalid bytes classify as U+FFFD and BPE works on the raw bytes.
- **Kimi Linear** reserves 258 special tokens instead of 256. `LoadKimi` knows
  256, so text containing literally `<|reserved_token_163840|>` or
  `<|reserved_token_163841|>` counts differently.
- **HF `tokenizer.json`:** WordPiece, Unigram, ByteLevel with its built-in
  GPT-2 regex or `add_prefix_space`, added tokens with
  `single_word`/`lstrip`/`rstrip`, and normalizers other than
  NFC/Replace/Sequence are not supported. None of the models above need them.

## License

MIT, see [LICENSE](LICENSE). The tokenizer files used by the tests are not part
of this repository and keep their own licenses: Apache-2.0 (Qwen, GPT-OSS,
Gemma 4), MIT (DeepSeek, Kimi Linear), and the model licenses of Kimi K3 and
GLM-5.3.
