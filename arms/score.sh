#!/usr/bin/env bash
# Score a rung's cells with the benchmark's own metrics evaluation, in one session.
#
# Usage: arms/score.sh <results-dir> <judge-model> [parallelism]
#   e.g. arms/score.sh results/T0 gpt-5.5-2026-04-23 8
#
# Every cell under <results-dir> is scored by one judge with one set of flags. That
# is not a convenience: the official scorer records neither the judge nor the flags
# in its results file, so nothing downstream can tell two cells graded by different
# judges from two systems that differ. Scoring them from one line is what makes the
# claim checkable, and `arms.gate` refuses to run without being told which judge it
# was -- so the two have to agree by construction.
#
# --no-correction is not optional here. The consensus correction flow rewrites gold
# answers, facts and expected document ids when a system's documents differ from the
# gold set, which lets a system help choose what it is graded on. Four arms compared
# against four different gold sets are not compared at all.
#
# The scorer resumes from an existing results file, so re-running this after an
# interruption picks up where it stopped. It needs no corpus: with --no-correction
# there is no document path to resolve, which is why this runs wherever the API key
# lives rather than on the box that measured the arms.
set -euo pipefail

RESULTS_DIR="${1:?usage: arms/score.sh <results-dir> <judge-model> [parallelism]}"
JUDGE="${2:?usage: arms/score.sh <results-dir> <judge-model> [parallelism]}"
PARALLELISM="${3:-8}"

export LLM_PROVIDER=openai
export LLM_MODEL_NAME="$JUDGE"

shopt -s nullglob
cells=("$RESULTS_DIR"/*/)
if [ ${#cells[@]} -eq 0 ]; then
  echo "no cells under $RESULTS_DIR" >&2
  exit 1
fi

for cell in "${cells[@]}"; do
  arm="$(basename "$cell")"
  if [ ! -f "$cell/answers.jsonl" ]; then
    echo "skipping $arm: no answers.jsonl" >&2
    continue
  fi
  echo "=== scoring $arm with $JUDGE ==="
  python -m src.scripts.answer_evaluation.metrics_based_eval \
    --answers-file "$cell/answers.jsonl" \
    --results-file "$cell/results.json" \
    --parallelism "$PARALLELISM" \
    --no-correction \
    --resume
done

echo
echo "scored ${#cells[@]} cell(s) with $JUDGE. Gate them with:"
echo "  python -m arms.gate --tier-tree <tier> \\"
for cell in "${cells[@]}"; do
  echo "      --arm $(basename "$cell")=${cell%/} \\"
done
echo "      --judge $JUDGE --session <session-id>"
