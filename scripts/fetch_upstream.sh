#!/usr/bin/env bash
# Fetch the training data, eval questions and grader prompts this study uses from
# OpenAI's emergent-misalignment-persona-features repo. They are not
# redistributed here; this puts them where the code expects them:
#
#   data/unpacked/advice/{correct,incorrect,incorrect_subtle}/<domain>_<variant>.jsonl
#   eval/{core_misalignment,extended_misalignment,hallucination_eval}.csv
#   eval/grader_prompts.py
#
# Pinned to the upstream commit every result in the post was produced from.
# Upstream password-locks the datasets to keep them out of web crawls; the
# password is published in its README. Both output locations are gitignored —
# keep it that way, so the unlocked data never lands in a public repo.
#
#   scripts/fetch_upstream.sh
#
# Needs git (>= 2.25, for sparse checkout) and unzip.
set -euo pipefail

UPSTREAM=https://github.com/openai/emergent-misalignment-persona-features
COMMIT=32b5807f8a3043a04ead88859ca3cbe172681b5f
PASSWORD=emergent
LOCKED=train/sft/synthetic/datasets_password_locked
DOMAINS="auto career edu finance health legal math science"
VARIANTS="correct incorrect incorrect_subtle"

ROOT=$(cd "$(dirname "$0")/.." && pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Partial clone + sparse checkout: only the 24 advice zips and eval/ are
# downloaded, not the rest of the ~100 MB repo.
patterns=(/eval/)
for d in $DOMAINS; do
    for v in $VARIANTS; do
        patterns+=("/$LOCKED/${d}_${v}.zip")
    done
done
git clone --quiet --filter=blob:none --no-checkout "$UPSTREAM" "$TMP/upstream"
git -C "$TMP/upstream" sparse-checkout set --no-cone "${patterns[@]}"
git -C "$TMP/upstream" checkout --quiet "$COMMIT"

mkdir -p "$ROOT/eval"
for f in core_misalignment.csv extended_misalignment.csv hallucination_eval.csv grader_prompts.py; do
    cp "$TMP/upstream/eval/$f" "$ROOT/eval/$f"
done

n=0
for v in $VARIANTS; do
    out="$ROOT/data/unpacked/advice/$v"
    mkdir -p "$out"
    for d in $DOMAINS; do
        name="${d}_${v}"
        rm -rf "$TMP/x" && mkdir "$TMP/x"
        unzip -q -P "$PASSWORD" "$TMP/upstream/$LOCKED/$name.zip" -d "$TMP/x"
        src=$(find "$TMP/x" -name "$name.jsonl" -type f)
        [ -n "$src" ] || { echo "error: $name.zip has no $name.jsonl" >&2; exit 1; }
        mv "$src" "$out/$name.jsonl"
        n=$((n + 1))
    done
done

echo "fetched $n datasets into data/unpacked/advice/ and 4 eval files into eval/"
echo "(upstream $UPSTREAM @ ${COMMIT:0:7})"
