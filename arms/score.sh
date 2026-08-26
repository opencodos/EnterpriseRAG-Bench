#!/usr/bin/env bash
# Score a rung's cells with the benchmark's own metrics evaluation, in one session.
#
# Usage: arms/score.sh <results-dir> <judge-model> [parallelism]
#   e.g. arms/score.sh results/T0 gpt-5.5-2026-04-23 4
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
PARALLELISM="${3:-4}"
# How many times a cell may be re-scored to clear transient zero-completeness rows.
# Five was enough to converge T0's BM25 cell; the loop stops early when the zero count
# stops moving, so this is a ceiling and not a cost.
MAX_PASSES="${MAX_PASSES:-5}"

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
  # Scored to convergence, not once. `validate_single_fact` retries three times and then
  # raises, and metrics_based_eval catches that by leaving the WHOLE question's
  # completeness at 0.0 -- silently, with no log line. One flaky call among a question's
  # ten facts therefore scores a correct answer as zero on the combined metric,
  # indistinguishably from an answer that shared no facts with the gold.
  #
  # Measured at T0: 87 of 500 rows came back at 0.0 on the first pass and five passes
  # recovered 22 of them, worth +2.54 combined. Without this loop a cell's score is
  # partly a function of how flaky the judge API was that afternoon, and two cells in the
  # same session are not comparable to each other.
  #
  # The re-score rule is uniform -- every row at 0.0, not only the ones judged correct.
  # Re-scoring only the rows that could raise our score is the kind of asymmetry a
  # skeptical reader is right to attack. Genuine zeros simply come back zero.
  for pass in $(seq 1 "$MAX_PASSES"); do
    before=$(python - "$cell/results.json" <<'PY'
import json, sys
try:
    rows = json.load(open(sys.argv[1]))["questions"]
except Exception:
    print(-1); raise SystemExit
zero = [q for q in rows if q["completeness_pct"] == 0.0]
print(len(zero))
PY
)
    if [ "$pass" -gt 1 ]; then
      [ "$before" = "0" ] && { echo "  pass $pass: no zero-completeness rows left"; break; }
      [ "$before" = "$prev_zero" ] && { echo "  pass $pass: $before zero row(s), unchanged -- converged"; break; }
      # Drop the zeroed rows so --resume re-scores exactly those.
      python - "$cell/results.json" <<'PY'
import json, sys
p = sys.argv[1]
r = json.load(open(p))
r["questions"] = [q for q in r["questions"] if q["completeness_pct"] != 0.0]
json.dump(r, open(p, "w"), indent=2)
PY
      echo "  pass $pass: re-scoring $before zero-completeness row(s)"
    fi
    prev_zero="$before"
    # Output to a file rather than through a pipe: under `pipefail` a grep that matches
    # nothing fails the pipeline and, under `set -e`, kills the sweep -- and it would
    # also swallow the scorer's own exit status, so a crashed pass would look like a
    # converged one.
    log="$(mktemp)"
    if ! python -m src.scripts.answer_evaluation.metrics_based_eval \
      --answers-file "$cell/answers.jsonl" \
      --results-file "$cell/results.json" \
      --parallelism "$PARALLELISM" \
      --no-correction \
      --resume > "$log" 2>&1; then
      echo "  scorer failed on $arm (pass $pass):" >&2
      tail -20 "$log" >&2
      rm -f "$log"
      exit 1
    fi
    grep -E "Combined corr|Avg completeness|Questions scored" "$log" | sed 's/^/    /' || true
    rm -f "$log"
  done
done

echo
echo "scored ${#cells[@]} cell(s) with $JUDGE. Gate them with:"
echo "  python -m arms.gate --tier-tree <tier> \\"
for cell in "${cells[@]}"; do
  echo "      --arm $(basename "$cell")=${cell%/} \\"
done
echo "      --judge $JUDGE --session <session-id>"
