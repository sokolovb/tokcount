#!/usr/bin/env bash
# Downloads the tokenizer files the parity tests and tools/gen_reference.py
# use, at the Hugging Face revisions testdata/expected.json was generated from.
#
#   tools/fetch_models.sh [DIR]    # DIR defaults to testdata/models
#
# Files already present are kept. Keep the list in sync with MODELS in
# tools/gen_reference.py.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
dest="${1:-$root/testdata/models}"

# directory  repository  revision  files...
models=(
  "moonshotai__Kimi-K3 moonshotai/Kimi-K3 f831ab66814297da540d832a5235f8e904f29d06 tiktoken.model tokenizer_config.json tokenization_kimi.py encoding_k3.py"
  "moonshotai__Kimi-Linear-48B-A3B-Instruct moonshotai/Kimi-Linear-48B-A3B-Instruct e1df551a447157d4658b573f9a695d57658590e9 tiktoken.model tokenizer_config.json tokenization_kimi.py"
  "deepseek-ai__DeepSeek-V4.1-Flash deepseek-ai/DeepSeek-V4.1-Flash 314113f8a7dadb388215396ba5fbee2c610a04d6 tokenizer.json tokenizer_config.json"
  "deepseek-ai__DeepSeek-V3.2-Exp deepseek-ai/DeepSeek-V3.2-Exp 194c67e12b1b0d6df0ef373ddcf215bc84027409 tokenizer.json tokenizer_config.json"
  "zai-org__GLM-5.3 zai-org/GLM-5.3 ba32f8c19c8738e2c57c71798c601d477a7bf844 tokenizer.json tokenizer_config.json"
  "Qwen__Qwen3.8-27B Qwen/Qwen3.8-27B 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 tokenizer.json tokenizer_config.json"
  "Qwen__Qwen3-8B Qwen/Qwen3-8B b968826d9c46dd6066d109eabc6255188de91218 tokenizer.json tokenizer_config.json"
  "openai__gpt-oss-120b openai/gpt-oss-120b b5c939de8f754692c1647ca79fbf85e8c1e70f8a tokenizer.json tokenizer_config.json"
  "google__gemma-4-31B-it google/gemma-4-31B-it 842da3794eaa0b77d5f08bae87a17459d91ff475 tokenizer.json tokenizer_config.json"
)

for entry in "${models[@]}"; do
  read -r dir repo rev files <<<"$entry"
  mkdir -p "$dest/$dir"
  for f in $files; do
    out="$dest/$dir/$f"
    if [[ -s "$out" ]]; then
      continue
    fi
    echo "fetching $repo/$f"
    curl -fsSL --retry 5 --retry-all-errors -o "$out.part" "https://huggingface.co/$repo/resolve/$rev/$f"
    mv "$out.part" "$out"
  done
done
echo "tokenizer files are in $dest"
