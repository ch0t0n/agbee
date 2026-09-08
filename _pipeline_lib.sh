#!/usr/bin/env bash
# Shared resume/marker/artifact driver for run_all_experiments.sh.
#
# A caller must, before invoking run_steps:
#   - set RUN, CFG, STEPS[], STATE_DIR, and (for read_ckpt) BEST_JSON
#   - define artifact_for() and one step_<id>() per entry in STEPS
# Optional knobs, defaulted here: DRY_RUN FORCE FROM_STEP TO_STEP DEVICE

: "${DRY_RUN:=0}"
: "${FORCE:=0}"
: "${FROM_STEP:=}"
: "${TO_STEP:=}"
: "${DEVICE:=0}"

list_steps() {
  printf '%s\n' "${STEPS[@]}"
}

# Reject an unknown --from/--to early: silently running the whole pipeline
# because a step id was misspelled is far more expensive than failing here.
validate_step() {
  local flag="$1" value="$2"
  [[ -z "${value}" ]] && return 0
  if ! printf '%s\n' "${STEPS[@]}" | awk -v s="${value}" '$0 == s { f=1 } END { exit !f }'; then
    echo "Unknown ${flag} step: ${value}" >&2
    list_steps >&2
    exit 2
  fi
}

run_cmd() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf 'DRY RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

# An artifact spec may be a literal path or a glob.
artifact_exists() {
  local artifact="$1"
  [[ -n "${artifact}" ]] || return 1
  [[ -e "${artifact}" ]] || compgen -G "${artifact}" >/dev/null
}

# Resolve the active dataset and its output root through config.py, so the
# shell never has to reimplement the ./outputs/{dataset} namespacing rule.
# Sets DATASET and OUT_ROOT. $1 = config path, $2 = dataset (may be empty).
resolve_dataset_paths() {
  local cfg="$1" ds="${2:-}"
  local resolved
  mapfile -t resolved < <(
    "${RUN}" -c '
import sys
from config import load_config
cfg = load_config(sys.argv[1], dataset=sys.argv[2] or None)
print(cfg["dataset"])
print(cfg["paths"]["output_root"])
print(cfg["splits"]["filename"])
print(cfg["splits"].get("cls_filename", ""))
' "${cfg}" "${ds}"
  )
  DATASET="${resolved[0]}"
  OUT_ROOT="${resolved[1]}"
  SPLIT_FILE="${resolved[2]}"
  CLS_SPLIT_FILE="${resolved[3]}"
  [[ "${OUT_ROOT}" = /* ]] || OUT_ROOT="${SCRIPT_DIR}/${OUT_ROOT#./}"
}

read_ckpt() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "${BEST_JSON%.json}.ckpt"
    return
  fi
  "${RUN}" -c '
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["ckpt"])
' "${BEST_JSON}"
}

# A step body calls this to declare that it deliberately did no work and
# therefore produces no artifact -- e.g. `b_extract_desc_gt` on a large_cls
# dataset, whose descriptors come from stage_c instead.
#
# Without it, such a step printed its own "SKIP ..." message, returned 0, and
# was then failed by the artifact check below ("finished but expected artifact
# is missing") -- aborting the whole pipeline on a step that was correct to do
# nothing. The full runner has skipped b_extract_desc_gt/_pred this way since
# the datasets moved to large_cls, so the real run would have stopped there.
step_skip() {
  STEP_SKIPPED=1
  [[ $# -gt 0 ]] && echo "SKIP: $*"
  return 0
}

# Run STEPS in order, honoring --from/--to/--force and the resume markers.
# A step is skipped only when both its marker and its expected artifact exist,
# so deleting an output is enough to force that step to re-run.
run_steps() {
  local label="${1:-}"
  local from_reached=1
  [[ -n "${FROM_STEP}" ]] && from_reached=0

  local step artifact marker
  for step in "${STEPS[@]}"; do
    if [[ "${from_reached}" -eq 0 ]]; then
      [[ "${step}" == "${FROM_STEP}" ]] && from_reached=1 || continue
    fi

    artifact="$(artifact_for "${step}")"
    marker="${STATE_DIR}/${step}.done"
    if [[ "${FORCE}" -eq 0 && -f "${marker}" ]] && artifact_exists "${artifact}"; then
      echo "SKIP ${label:+${label}:}${step} (completed)"
      # `--to` has to be honored on the skip path too. Without this, a
      # `continue` here jumped straight past the TO_STEP check at the bottom
      # of the loop, so `--from S --to S` on an ALREADY-COMPLETED S skipped S
      # and then ran every remaining step in the pipeline. run_monitor.py
      # drives exactly that way -- one invocation per step -- so a single
      # "step" could run away with the whole run: on 2026-08-20 a `contract`
      # step launched Stage B's 21-job control sweep, filed its job logs under
      # contract.jobs, and left the dashboard reporting the wrong step.
      [[ "${step}" == "${TO_STEP}" ]] && break
      continue
    fi

    echo
    echo ">>> ${label:+${label}:}${step}"
    STEP_SKIPPED=0
    "step_${step}"
    if [[ "${DRY_RUN}" -eq 0 ]]; then
      if [[ "${STEP_SKIPPED}" -eq 1 ]]; then
        # Deliberately no-op: record it as done so a resume does not retry it,
        # but do not demand an artifact it was never going to write.
        date -Is > "${marker}"
      else
        artifact_exists "${artifact}" || {
          echo "Step ${label:+${label}:}${step} finished but expected artifact is missing: ${artifact}" >&2
          exit 1
        }
        date -Is > "${marker}"
      fi
    fi
    [[ "${step}" == "${TO_STEP}" ]] && break
  done
  return 0
}
