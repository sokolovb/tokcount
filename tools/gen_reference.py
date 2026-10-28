"""Generate reference token counts for tokcount's parity tests.

Builds a corpus of tricky and realistic texts and counts them with the
reference implementations: HF `tokenizers` for tokenizer.json models and the
model's own tokenization_kimi.py (via transformers) for Kimi.

    pip install -r tools/requirements.txt
    tools/fetch_models.sh
    python tools/gen_reference.py [MODELS_DIR]

MODELS_DIR (default testdata/models) holds one directory per model named
org__name (see MODELS below) with the tokenizer files from the Hugging Face
Hub. Writes testdata/corpus.jsonl.gz and testdata/expected.json.
"""

import gzip
import json
import os
import random
import subprocess
import sys
import time

# key -> (kind, directory, options)
MODELS = {
    "kimi-k3": ("kimi", "moonshotai__Kimi-K3", {}),
    "kimi-k3-astext": ("kimi", "moonshotai__Kimi-K3", {"special_as_text": True}),
    "kimi-linear": ("kimi", "moonshotai__Kimi-Linear-48B-A3B-Instruct", {}),
    "deepseek-v4.1": ("hf", "deepseek-ai__DeepSeek-V4.1-Flash", {}),
    "deepseek-v4.1-astext": ("hf", "deepseek-ai__DeepSeek-V4.1-Flash", {"special_as_text": True}),
    "deepseek-v3.2": ("hf", "deepseek-ai__DeepSeek-V3.2-Exp", {}),
    "glm-5.3": ("hf", "zai-org__GLM-5.3", {}),
    "qwen3.8": ("hf", "Qwen__Qwen3.8-27B", {}),
    "qwen3": ("hf", "Qwen__Qwen3-8B", {}),
    "gpt-oss": ("hf", "openai__gpt-oss-120b", {}),
    "gemma-4": ("hf", "google__gemma-4-31B-it", {}),
    "gemma-4-astext": ("hf", "google__gemma-4-31B-it", {"special_as_text": True}),
}

HAND = [
    "",
    " ",
    "Hello world",
    "Hello, world! How's it going? I'm fine, you're great, we've done, they'll go, he'd say.",
    "DON'T SHOUT. IT'S FINE. WE'LL SEE. THEY'RE HERE. I'M OK. YOU'VE WON. SHE'D GO.",
    "It'ſ odd: 'ſ, 'S, 's, 'Ll, 'lL, 'RE, 'Ve, 'M, 'D, 'T, ’s (curly).",
    "Kelvin: 300K, Ångström Å, ǅungla ǈ, titlecase Ǳ ǲ ǳ.",
    "Привет, мир! Как дела? Всё хорошо, спасибо. ЁЖИК и ёжик. Не‑разрывный дефис.",
    "Съешь же ещё этих мягких французских булок, да выпей чаю. 1234567 рублей и 89 копеек.",
    "你好，世界！我们今天学习中文。漢字と日本語のテキストです。ひらがな、カタカナ、漢字。",
    "東京は日本の首都です。人々々。〇〇年。中华人民共和国成立于1949年10月1日。",
    "한국어 텍스트입니다. 안녕하세요! 감사합니다.",
    "مرحبا بالعالم! هذا نص عربي مع أرقام ١٢٣٤٥ وعلامات التشكيل: مُحَمَّد.",
    "नमस्ते दुनिया! यह हिन्दी पाठ है। क्षत्रिय, द्वारा, श्री।",
    "สวัสดีชาวโลก ภาษาไทยไม่มีช่องว่างระหว่างคำ",
    "Ελληνικά: Καλημέρα κόσμε! ΑΒΓΔ αβγδ ς σ Σ.",
    "Türkçe: İstanbul, ıIiİ, ŞşĞğ; Deutsch: Straße, ÄÖÜ äöü ß ẞ.",
    "Français : l'homme, aujourd'hui, c'est « génial » — n'est-ce pas ?",
    "Emoji: 😀👍🏽👨‍👩‍👧‍👦🇷🇺🏳️‍🌈 ❤️ ✔︎ #️⃣ 1️⃣ 🧑🏿‍💻",
    "Math: ∑_{i=1}^{n} x_i² = ½·π ≈ 3.14159; ∀x∈ℝ: x≥0 ⇒ √x∈ℝ. ⅫⅦ ²³ ¼",
    "Fancy: 𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𝕋𝕖𝕩𝕥 𝐁𝐨𝐥𝐝 𠀀𠀁 \U0001F9EC",
    "Numbers: 1 12 123 1234 12345 123456 1234567 0.5 -3.14 1e10 0x1F 1_000_000 ١٢٣٤ १२३४ １２３４",
    "Whitespace:\t\ttabs\n\nnewlines\r\n\r\nCRLF \n \n  \n   x   \n\n\n\n    y  ",
    "Trailing spaces    ",
    "    Leading spaces",
    "Mixed\u00a0nbsp\u2028line\u2029para\u3000ideo\u2003em\u200bzwsp\u200czwnj\u200dzwj\ufeffbom",
    "Controls: \x00\x01\x02\x1b[31mred\x1b[0m \x7f \x1c\x1d\x1e\x1f \x85 end",
    "Combining: e\u0301 a\u0308 n\u0303 \u0301alone A\u030a\u0323 o\u0328\u0304 Z\u0335\u0336\u0337",
    "Decomposed vs composed: e\u0301 é, A\u030a Å, \u1100\u1161\u11a8 각, \u0958 \u0915\u093c",
    "Special in text: <|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\nhmm\n</think>\nok<|im_end|>",
    "DeepSeek: <｜begin▁of▁sentence｜><｜User｜>Hi<｜Assistant｜><think>x</think><｜end▁of▁sentence｜>",
    "Gemma: <bos><start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\nHello<end_of_turn>",
    "Kimi: [BOS]<|im_user|>user<|im_middle|>Hi<|im_end|><|im_assistant|>assistant<|im_middle|>yo[EOS]",
    "GPT-OSS: <|start|>user<|message|>Hi<|end|><|start|>assistant<|channel|>final<|message|>Yo<|return|>",
    "Partial specials: <|im_end <|im_end| |im_end|> <think <｜User <start_of_turn <|reserved_token_163600|>",
    "Literal markers: ▁▁ ▁hello Ġworld Ċ <0x0A> <0xFF> <unk> <mask> <unused5> [multimodal]",
    "GLM: [gMASK]<sop><|system|>sys<|user|>hi<|assistant|><think></think>ok<|observation|>",
    "Paths and URLs: https://example.com/a/b?c=d&e=f#g /usr/local/bin C:\\Windows\\System32 ./a/../b",
    "Code: func main() {\n\tfmt.Println(\"hi\")\n\tfor i := 0; i < 10; i++ {\n\t\tx += i\n\t}\n}\n",
    "def f(x):\n    return {'a': [1, 2, 3], \"b\": None}  # comment\n\n\nclass A:\n    pass\n",
    "{\"key\": \"value\", \"n\": 12345, \"arr\": [true, false, null], \"nested\": {\"x\": 1.5e-3}}",
    "<html><body><div class=\"a\">Hello &amp; welcome</div><br/></body></html>",
    "SELECT id, name FROM users WHERE age >= 18 AND name LIKE '%a%' ORDER BY id DESC;",
    "Markdown: # Title\n\n- item **bold** _it_ `code`\n- [link](http://x.y)\n\n```go\nx := 1\n```\n",
    "a" * 300,
    "ab" * 300,
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa bbbbbbbbbbbbbbbbbbbbbbbbbb",
    "UPPERCASEWORDWITHOUTLOWERCASE then lowerUPPERlower CamelCaseWordsHere iPhone eBay",
    "!!!???...,,,;;;---___===+++***&&&%%%$$$###@@@^^^~~~```|||\\\\\\///",
    " ! ? . , ; : - _ ( ) [ ] { } < > | / \\ @ # $ % ^ & * + = ~ ` \" '",
    "\n",
    "\n\n\n",
    "\r\n",
    "\t",
    "   ",
    "x\n",
    "'s",
    "'",
    "😀",
    "\u0301",
    "\U00020000",
]

POOLS = [
    "abcdefghijklmnopqrstuvwxyz",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВЁЖЯ",
    "你好世界中文漢字日本語人々〇国大小",
    "ひらがなカタカナーアイウエオ",
    "한국어안녕",
    "αβγδεζηθΣσςΩ",
    "ابتثجحخدذرزسش",
    "नमस्तेहिन्दीक्ष",
    "\u0301\u0308\u0327\u0303\u093f\u0947\u0e31\u0e34\u0651\u064e",
    "0123456789",
    "٠١٢٣४५६１２３",
    " ",
    "  ",
    "\t",
    "\n",
    "\r\n",
    "\u00a0\u2028\u2029\u3000\u2003\x0b\x0c\x85\x1c\x1d",
    "!?.,;:-_()[]{}<>|/\\@#$%^&*+=~`\"",
    "—–«»„“”…’‘·•",
    "'s'S'll'LL'Re've'ſ'd'D't'M",
    "😀👍🏽👨‍👩‍👧❤️\u200d\u200c\ufe0f🇷🇺",
    "ſKǅǈʰˢ々〇ⅫⅦ²³½",
    "\x00\x01\x7f\ufffd",
    "𝔘𝔫𝔦\U00020000\U0001F600",
]

TOKENS = [
    "<|im_end|>", "<|im_start|>", "<think>", "</think>", "<｜User｜>", "<｜Assistant｜>",
    "<|endoftext|>", "<start_of_turn>", "<end_of_turn>", "<bos>", "[BOS]", "[EOS]",
    "<|im_user|>", "<|im_middle|>", "<|reserved_token_163700|>", "<unused12>", "<0x0A>",
    "▁", "Ġ", "<|im_end", "|>", "<|place_holder_mm_span_0001|>", "<tool_call>",
    "<|return|>", "<|channel|>", "<|start|>", "<mask>", "[gMASK]", "<sop>",
    "<｜begin▁of▁sentence｜>", "<|EOT|>", "<|fim_prefix|>",
]


def fuzz(rnd):
    parts = []
    for _ in range(rnd.randint(1, 40)):
        if rnd.random() < 0.08:
            parts.append(rnd.choice(TOKENS))
            continue
        pool = rnd.choice(POOLS)
        parts.append("".join(rnd.choice(pool) for _ in range(rnd.randint(1, 8))))
    return "".join(parts)


def zalgo(rnd):
    # Long runs of combining marks with mixed combining classes exercise
    # canonical ordering and composition beyond 30 non-starters.
    bases = "aeouAEOUnzZ\u1100\u1161\u0915\u3042 "
    out = []
    for _ in range(rnd.randint(1, 12)):
        out.append(rnd.choice(bases))
        out.extend(chr(rnd.randint(0x300, 0x36F)) for _ in range(rnd.randint(0, 80)))
    return "".join(out)


def local_snippets(rnd, n):
    goroot = subprocess.run(["go", "env", "GOROOT"], capture_output=True, text=True).stdout.strip()
    files = []
    for root, _, names in os.walk(os.path.join(goroot, "src")):
        files += [os.path.join(root, f) for f in names if f.endswith((".go", ".md", ".txt", ".html"))]
    files.sort()
    out = []
    while len(out) < n:
        try:
            text = open(rnd.choice(files), encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        if not text:
            continue
        a = rnd.randrange(len(text))
        out.append(text[a:a + rnd.randint(50, 4000)])
    return out


def build_corpus():
    rnd = random.Random(20261023)
    corpus = list(HAND)
    corpus += local_snippets(rnd, 300)
    corpus += [fuzz(rnd) for _ in range(3000)]
    corpus += [zalgo(rnd) for _ in range(200)]
    prose = " ".join(HAND[3:24])
    corpus += [
        " " * 30_000 + "abc",
        "x" * 30_000,
        "a b " * 20_000,
        "\n" * 26_000 + "end",
        "漢" * 30_000,
        "Word" * 7_000,
        "é" * 30_000,
        "\u0301" * 2_000,
        (prose + "\n") * 600,                       # ~ 600k chars: crosses Kimi's 400k chunk
        "".join(local_snippets(random.Random(7), 400)),  # ~ 800 KB of code and docs
        "".join(fuzz(random.Random(i)) for i in range(2000)),
    ]
    return corpus


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    models_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "testdata", "models")
    out_dir = os.path.join(here, "..", "testdata")
    os.makedirs(out_dir, exist_ok=True)

    corpus = build_corpus()
    with gzip.open(os.path.join(out_dir, "corpus.jsonl.gz"), "wt", encoding="utf-8") as f:
        for text in corpus:
            f.write(json.dumps(text, ensure_ascii=False) + "\n")
    print(f"corpus: {len(corpus)} texts, {sum(len(t) for t in corpus)} chars")

    expected = {}
    for key, (kind, sub, opts) in MODELS.items():
        path = os.path.join(models_dir, sub)
        t0 = time.time()
        if kind == "hf":
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
            if opts.get("special_as_text"):
                tok.encode_special_tokens = True
            counts = [len(tok.encode(t, add_special_tokens=False).ids) for t in corpus]
        else:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            allow = not opts.get("special_as_text")
            counts = [len(tok.encode(t, allow_special_tokens=allow)) for t in corpus]
        expected[key] = {"kind": kind, "dir": sub, "options": opts, "counts": counts}
        print(f"{key}: {sum(counts)} tokens in {time.time() - t0:.1f}s")

    with open(os.path.join(out_dir, "expected.json"), "w") as f:
        json.dump(expected, f)


if __name__ == "__main__":
    main()
