#!/usr/bin/env bash
# Resumable end-to-end runner for the paper's experiment protocol across one
# or more datasets. Encodes the ordered runbook in README.md
# (contract/preflight/freeze -> Stage A -> pseudo-masks -> Stage B -> Stage D
# -> reporting) so the comparison can be kicked off as one command and resumed
# after any interruption.
#
# Uses the real config.yaml budgets (200-epoch segmentation sweeps over 9
# models, the eight-backbone whole-image sweep, a 3-seeded ablation grid, and
# classification on the full corpora). Expect this to run for many hours to
# multiple days per dataset.
#
# Stage D is fully automated now. `stage_d.py dump_preds` exports the
# per-sample (y_true, y_pred, confidence, train_count) CSV that long_tail /
# confusion / calibration / selective / reliability_bins all need, so the
# `d_arms` step runs all six analyses for every arm rather than robustness
# alone on the whole-image reference.
set -euo pipefail

# --- everything below runs inside this brace group -------------------------
# Bash reads a script lazily, by file offset, WHILE it runs. The top-level
# `for ds in "${DATASETS[@]}"` loop near the bottom can hold that read position
# for days; edit this file in the meantime and bash resumes parsing the new
# bytes at the old offset, landing mid-line.
#
# That is exactly how the 2026-08-20 run died. All 24 Stage B baseline jobs
# finished rc=0 and the resume marker was written, then the run aborted with
#   run_all_experiments.sh: line 492: unexpected EOF while looking for matching `'
# because the file had been edited on 2026-08-19, 21 hours into the step.
# 59 pending steps were abandoned over an error in code that had already run.
#
# A brace group spanning the whole file forces the parser to consume every
# byte before executing any of it, and the `exit` at the end of the group
# stops bash from reading further once the group is done. Both halves are
# required: the group alone still leaves bash reading past the closing brace.
# The body is deliberately left unindented so this stays a small diff.
{

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_pipeline_lib.sh
source "${SCRIPT_DIR}/_pipeline_lib.sh"

RUN="${SCRIPT_DIR}/run.sh"
CFG="${SCRIPT_DIR}/config.yaml"
DATASETS=(beemachine cub fish_vista)
BB="convnext_nano.in12k"   # common comparison backbone, pinned across the whole grid (README Stage B/B+)
# The larger-backbone control the capacity_matched arm trains. Must stay
# genuinely LARGER than ${BB}: convnext_small (49.5M) controls convnext_nano
# (15.0M), but it is within 6% of seresnext101_32x4d (46.9M) and would stop
# being a capacity control at all if ${BB} moved there. Read from the
# environment so tools/run_monitor.py -- which forwards os.environ but has no
# flag of its own for this -- can set it for a whole campaign.
CAPBB="${BEEMACHINE_CAPACITY_BACKBONE:-convnext_small.in12k}"

# Dependency order, and it has changed. Stage C's pseudo-mask steps now come
# BEFORE Stage B, because CUB and Fish-Vista train their classification arms on
# the large corpus with confidence-gated pseudo-masks (config
# `datasets.*.cls_corpus: large_cls`), so `c_pseudo`/`c_filter` are inputs to
# `b_*` for those two datasets rather than outputs of them. Beemachine is
# unaffected -- it trains on the pixel-annotated part set -- but one order that
# is correct for all three is worth more than a per-dataset special case.
#
# `leakage_check` precedes `freeze` so the frozen classification split can
# actually exclude content-duplicate images; see tools/check_part_scale_leakage.py.
#
# `preflight` follows `freeze` rather than preceding it. It verifies the
# contract's data commitments against the filesystem, and one of those
# commitments -- the frozen classification split -- is produced by `freeze`.
# Ordered before it, preflight failed on any dataset whose cls split was being
# regenerated (Beemachine, after the overlap measurement) by checking for a
# file the pipeline had not written yet. It still gates Stage A, which is its
# actual job.
STEPS=(
  contract leakage_check freeze preflight
  a_qa a_sweep a_pick_best a_seg_table
  c_pseudo c_filter_sweep c_filter c_extract_desc
  b_sweep_baselines b_sweep_controls
  b_extract_desc_gt b_extract_desc_pred
  b_ablate_gt b_ablate_pred b_ablation_table
  d_arms d_aggregate d_gate
  report_hashes report_repro
)

usage() {
  cat <<'EOF'
Usage: ./run_all_experiments.sh [options]

Options:
  --config PATH        Config (default: config.yaml)
  --dataset NAME        Restrict to one dataset: beemachine | cub | fish_vista
                         (default: all three, in that order)
  --protocol NAME       Experiment protocol: default | resolution_384 |
                         heavy_aug (default: default). Non-default protocols
                         re-run the same arms into suffixed run directories
                         with their own resume markers.
  --from STEP           Start at STEP for the current dataset; completed
                         later steps still resume-skip
  --to STEP             Stop after STEP for the current dataset
  --force                Re-run completed steps
  --device N             Device for single-GPU jobs (default: 0)
  --backbone NAME        Common comparison backbone (default: convnext_nano.in12k)
  --capacity_backbone NAME  Larger-backbone control for the capacity_matched
                         arm (default: convnext_small.in12k, or
                         $BEEMACHINE_CAPACITY_BACKBONE). Must be larger
                         than --backbone or the control means nothing.
  --list-steps           Print step IDs and exit
  --dry-run              Print commands without running them
  -h, --help             Show this help

Resumable: a step is skipped only when both its marker and expected artifact
exist. Re-run the same command after any interruption (network, GPU OOM,
preemption) to continue where it left off -- per dataset, in the order given
above.

The run stops immediately if `preflight` finds a missing dataset root, frozen
split, or augmentation directory, rather than failing partway into Stage A.

Prerequisites:
  - Python environment with packages from requirements.txt, launched via ./run.sh.
  - Train-only six-fold geometric augmentations already written under each
    dataset's part-set root (data_processing/make_train_aug.py).
  - compute.num_gpus in config.yaml controls the parallel pool size for every
    sweep step (-1 = all visible GPUs). Restricting CUDA_VISIBLE_DEVICES also
    lowers the pool, because resolve_num_gpus() caps compute.num_gpus at
    torch.cuda.device_count().
EOF
}

RESTRICT_DATASET=""
# Experiment protocol (see config.yaml `protocols:`). `default` is the study's
# primary grid; resolution_384 / heavy_aug re-run the same seven arms with a
# single variable changed, into protocol-tagged run directories and their own
# resume-marker directory, so they can never disturb the default grid.
PROTOCOL="${PROTOCOL:-default}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CFG="$2"; shift 2 ;;
    --dataset) RESTRICT_DATASET="$2"; shift 2 ;;
    --protocol) PROTOCOL="$2"; shift 2 ;;
    --from) FROM_STEP="$2"; shift 2 ;;
    --to) TO_STEP="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --backbone) BB="$2"; shift 2 ;;
    --capacity_backbone) CAPBB="$2"; shift 2 ;;
    --list-steps) list_steps; exit 0 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -f "${CFG}" ]]; then
  echo "Config not found: ${CFG}" >&2
  exit 2
fi

[[ -n "${RESTRICT_DATASET}" ]] && DATASETS=("${RESTRICT_DATASET}")

validate_step --from "${FROM_STEP}"
validate_step --to "${TO_STEP}"

# --- per-dataset state, set at the top of each dataset's loop iteration ---
DS=""
OUT_ROOT=""
STATE_DIR=""
PTAG=""
BEST_JSON=""
DESC_GT=""
DESC_PRED=""

# Args every stage subprocess gets. `--protocol` is appended only when it is
# not the default, so a run without --protocol issues byte-identical command
# lines to the ones the campaign has been running all along.
# Suffix that a non-default protocol appends to every trained run directory and
# to the tagged output files. Resolved through config.py rather than hardcoded,
# so the shell never reimplements the naming rule.
protocol_tag() {
  if [[ "${PROTOCOL}" == "default" ]]; then
    echo ""
  else
    "${RUN}" -c '
import sys
from config import load_config, run_tag
print(run_tag(load_config(sys.argv[1], protocol=sys.argv[2])))
' "${CFG}" "${PROTOCOL}"
  fi
}

stage_args() {
  if [[ "${PROTOCOL}" == "default" ]]; then
    echo --config "${CFG}" --dataset "${DS}"
  else
    echo --config "${CFG}" --dataset "${DS}" --protocol "${PROTOCOL}"
  fi
}

# True when the dataset trains classification on the large corpus with
# pseudo-masks (CUB, Fish-Vista) rather than the pixel-annotated part set
# (Beemachine). Read from config so the shell never second-guesses config.py.
is_large_cls() {
  [[ "$("${RUN}" -c '
import sys
from config import cls_corpus, load_config
print(cls_corpus(load_config(sys.argv[1], dataset=sys.argv[2])))
' "${CFG}" "${DS}")" == "large_cls" ]]
}

# The seed list every seed-swept step uses. Read from config.py rather than
# duplicated here, so the shell can never disagree with the preflight gate in
# stage_report.py (which now holds EVERY dataset to >=3 seeds). This replaced a
# pair of hardcoded arrays that pinned CUB and Fish-Vista to seed 42 alone --
# a 2026-08-24 decision that was reversed, and that these arrays would have
# silently reinstated on the next full run.
config_seeds() {
  "${RUN}" -c '
import sys
from config import load_config
print(" ".join(str(s) for s in load_config(sys.argv[1], dataset=sys.argv[2])["seeds"]))
' "${CFG}" "${DS}"
}

# The single seed a deliberately single-run study uses (config `seed`), as
# opposed to the `seeds` grid above. Named rather than written as a literal 42
# so the two cannot drift apart.
config_seed() {
  "${RUN}" -c '
import sys
from config import load_config
print(load_config(sys.argv[1], dataset=sys.argv[2])["seed"])
' "${CFG}" "${DS}"
}

# --- step bodies (README "Ordered runbook" 0-5) ---

step_contract() {
  # Pre-registration, locked once before any training. Already-locked
  # contracts are kept as-is; --force re-locks after a protocol change.
  if [[ -f "${OUT_ROOT}/contract/study_contract.md" ]]; then
    echo "Contract already locked: ${OUT_ROOT}/contract/study_contract.md"
    return 0
  fi
  run_cmd "${RUN}" stage_report.py contract $(stage_args)
}

step_preflight() {
  run_cmd "${RUN}" stage_report.py preflight $(stage_args)
}

step_leakage_check() {
  # Content-based part/scale overlap. The filename-based guard inside
  # data.py removes 1,888 images for CUB and 1,129 for Fish-Vista, but
  # exactly 0 for Beemachine, whose part set is Roboflow-renamed -- so its
  # overlap was unmeasured, not absent. Measured on a 4,000-image corpus
  # slice, 767 of 7,716 part images already had a dHash duplicate.
  # Writes frozen_splits/part_scale_leakage.json, which `freeze` then honours.
  run_cmd "${RUN}" tools/check_part_scale_leakage.py $(stage_args) --workers 16
}

step_freeze() {
  local extra=()
  [[ "${DS}" == "cub" ]] && extra=(--force)
  run_cmd "${RUN}" stage_a.py freeze $(stage_args) "${extra[@]}"
}

step_a_qa() {
  run_cmd "${RUN}" stage_a.py qa $(stage_args)
}

step_a_sweep() {
  run_cmd "${RUN}" stage_a.py sweep $(stage_args)
}

step_a_pick_best() {
  run_cmd "${RUN}" stage_a.py pick_best $(stage_args)
}

step_a_seg_table() {
  run_cmd "${RUN}" stage_report.py seg_table $(stage_args)
}

step_b_sweep_baselines() {
  run_cmd "${RUN}" stage_b.py sweep_baselines $(stage_args)
}

# All seven seed-swept arms as ONE 21-job pool.
#
# These were seven consecutive steps, each launching len(seeds) = 3 jobs. Since
# STEPS is a sequential barrier list, that meant seven waves of 3 on an 8-GPU
# box -- 5 devices idle throughout, measured at 20.9 h of pure idle across the
# three datasets. The arms are mutually independent and nothing reads their
# outputs until b_ablation_table, so
# pooling them changes no number in any table.
#
# capacity_matched and heavy_aug are what the headline claim has to survive, so
# they get the same multi-seed treatment as the arms they control for -- a
# single-seed control cannot be compared against a 3-seed arm.
#
# masked / partcrop are pinned to the common comparison backbone, like every
# other cell of the controlled grid. `ablation_table` groups by backbone, so an
# arm on any other backbone lands in a cell with no reference arm and gets no
# paired-significance result -- these sweeps used to run all backbones and
# most of that compute produced rows nothing could be compared against. The
# cross-architecture check is b_sweep_baselines.
step_b_sweep_controls() {
  local ckpt
  ckpt="$(read_ckpt)"
  run_cmd "${RUN}" stage_b.py sweep_controls $(stage_args) \
    --backbone "${BB}" --capacity_backbone "${CAPBB}" \
    --seg_ckpt "${ckpt}" --device "${DEVICE}"
}

step_b_extract_desc_gt() {
  if is_large_cls; then
    step_skip "b_extract_desc_gt: ${DS} uses the large corpus; its descriptors" \
              "come from c_extract_desc (stage_c/pseudo_descriptors.csv)."
    return 0
  fi
  # --all_gpus: the single most expensive serial step in the pipeline
  # (MANIQA's 20-crop protocol plus CPU-bound SIFT/ORB/Zernike). Measured
  # 1.48 / 2.59 / 2.97 s/image/GPU for beemachine / fish_vista / cub -- cost
  # scales with part count. 8 shards is 12-33 min per dataset per mask source.
  run_cmd "${RUN}" stage_b.py extract_desc $(stage_args) --source gt --all_gpus
}

step_b_extract_desc_pred() {
  if is_large_cls; then
    step_skip "b_extract_desc_pred: ${DS} has no gt/pred mask axis."
    return 0
  fi
  local ckpt
  ckpt="$(read_ckpt)"
  run_cmd "${RUN}" stage_b.py extract_desc $(stage_args) \
    --source pred --seg_ckpt "${ckpt}" --all_gpus
}

step_b_ablate_gt() {
  # For a large_cls dataset (Beemachine, CUB, Fish-Vista) `--mask_source gt`
  # selects the direct-read path to the confidence-gated pseudo-masks and is
  # recorded as mask_source="pseudo"; there is no real ground truth at corpus
  # scale.
  local desc="${DESC_GT}"
  if is_large_cls; then desc="${OUT_ROOT}/stage_c/pseudo_descriptors.csv"; fi
  run_cmd "${RUN}" stage_b.py ablate $(stage_args) \
    --desc_csv "${desc}" --backbone "${BB}" --mask_source gt
}

step_b_ablate_pred() {
  if is_large_cls; then
    step_skip "b_ablate_pred: ${DS} trains on the large corpus, which has no" \
              "ground truth, so gt-vs-pred is not a distinction it can make."
    return 0
  fi
  # The predicted-mask half of the grid (research plan milestone 6). Distinct
  # cells, not a repeat: `ablate` tags every output dir and summary row with
  # mask_source, so these merge alongside the gt rows instead of replacing
  # them. Without this step the expensive descriptors_pred.csv from
  # b_extract_desc_pred is produced and never read by anything.
  run_cmd "${RUN}" stage_b.py ablate $(stage_args) \
    --desc_csv "${DESC_PRED}" --backbone "${BB}" --mask_source pred
}

step_b_ablation_table() {
  run_cmd "${RUN}" stage_report.py ablation_table $(stage_args) \
    --fusion_summary "${OUT_ROOT}/stage_b_descriptors/fusion_ablation_summary${PTAG}.csv" \
    --reference_mode concat
}

step_c_pseudo() {
  local ckpt
  ckpt="$(read_ckpt)"
  run_cmd "${RUN}" stage_c.py pseudo $(stage_args) --seg_ckpt "${ckpt}" --all_gpus
}

step_c_filter_sweep() {
  # Retention at every threshold in `pseudo_labeling.conf_thresholds`
  # (0.5/0.7/0.9). The research plan requires the coverage-vs-quality
  # trade-off to be reported; only tau=0.7 was ever produced before.
  run_cmd "${RUN}" stage_c.py filter_sweep $(stage_args)
}

step_c_filter() {
  run_cmd "${RUN}" stage_c.py filter $(stage_args) --conf 0.7
}

step_c_extract_desc() {
  run_cmd "${RUN}" stage_c.py extract_desc $(stage_args) \
    --masks_dir "${OUT_ROOT}/stage_c/pseudo_masks_conf0.7" --all_gpus
}

# Stage C: the scaling curve, not a single point.
#
# Three things were missing before. (1) Only `whole` and `gated_residual` ran,
# so `concat` -- the arm the plan names as one of the three scalable ones --
# had no at-scale number at all. (2) Every arm ran at one training-set size,
# so `scaling_curve.csv` held two rows and no curve existed. (3) Everything ran
# at seed 42 only, so no at-scale confidence interval was computable while the
# tables still carried CI columns.
#
# This is the single most expensive step in the pipeline: |modes| x |fracs| x
# |seeds| DDP jobs over the full corpus. Trim C_FRACS or C_MODES to cut it.
C_MODES=(whole concat gated_residual)
C_FRACS=(0.125 0.25 0.5 1.0)
# Only the full-corpus point runs every seed; the reduced fractions run at the
# first configured seed alone and are read as trend points (Appendix "Corpus-scale
# results"). Resolved per dataset inside the step body, not as a top-level array:
# run_all_experiments.sh parses its whole file once at start and reuses the
# in-memory step bodies for every `for ds in DATASETS[@]` iteration, so a fixed
# top-level array would apply one dataset's seed list to all three.

step_c_ddp_scaling() {
  local desc="${OUT_ROOT}/stage_c/pseudo_descriptors.csv"
  local masks="${OUT_ROOT}/stage_c/pseudo_masks_conf0.7"
  local mode frac seed extra
  # The scaling curve is a deliberately single-run study except at one point:
  # BeeMachine's full-corpus cell, which is the only place the paper quotes a
  # seed spread for it (Appendix "Corpus-scale results", and the third of the
  # three single-run studies listed in the statistical protocol). Everything
  # else -- every reduced fraction, and the whole of CUB's and Fish-Vista's
  # curves -- runs at the single configured `seed` and is read as a trend.
  # Encoding that here, rather than as a per-dataset seed array, keeps the
  # restriction visible and keeps it out of the `seeds` grid that every other
  # step now takes from config.
  local -a seeds_for_ds trend_seed
  read -r -a seeds_for_ds <<< "$(config_seeds)"
  trend_seed="$(config_seed)"
  for mode in "${C_MODES[@]}"; do
    for frac in "${C_FRACS[@]}"; do
      for seed in "${seeds_for_ds[@]}"; do
        if [[ "${frac}" != "1.0" || "${DS}" != "beemachine" ]] \
           && [[ "${seed}" != "${trend_seed}" ]]; then continue; fi
        extra=()
        if [[ "${mode}" != "whole" ]]; then
          extra=(--desc_csv "${desc}")
          [[ "${mode}" == "gated_residual" ]] && extra+=(--masks_dir "${masks}")
        fi
        echo "--- stage_c ddp mode=${mode} frac=${frac} seed=${seed}"
        run_cmd "${RUN}" stage_c.py ddp $(stage_args) \
          --mode "${mode}" --backbone "${BB}" \
          --frac "${frac}" --seed "${seed}" "${extra[@]}"
      done
    done
  done
}

step_c_plot_curve() {
  # The glob stays broad; stage_c.py filters it down to the active protocol's
  # own run directories (config.belongs_to_protocol). Filtering there rather
  # than here is what makes the DEFAULT protocol correct too: its tag is the
  # empty string, so a shell glob built from it would match every protocol's
  # directories and silently average a 320px default run together with its
  # 384px capacity-matched counterpart.
  run_cmd "${RUN}" stage_c.py plot_curve $(stage_args) \
    --metrics_glob "${OUT_ROOT}/stage_c/scaling/**/test_metrics.json"
}

# Stage D across ARMS, not just the whole-image reference.
#
# Previously this ran `robustness` on `whole` alone, leaving three of the four
# arms the robustness table claims with no data, and the calibration /
# selective / long-tail / confusion analyses had no data at all because nothing
# emitted the per-image --pred_csv they need. `stage_d.py dump_preds` closes
# that, so all five analyses now run per arm.
step_d_arms() {
  # All nine arms at every configured seed, via tools/stage_d_arms.py. That
  # script is also the manual backfill entry point for datasets whose campaign
  # already finished, so the arm-to-checkpoint resolution lives in exactly one
  # place -- when this loop was inlined here it drifted to six hardcoded arms
  # and left body-masked, part-crop and multi-task with no reliability or
  # robustness data at all.
  #
  # This used to pass a single D_SEED=42. An arm's ECE, AURC and corruption
  # deltas are properties of a trained checkpoint exactly as its top-1 is, so
  # they carry the same seed dimension the accuracy tables do; running one seed
  # here is what left the paper's reliability tables single-seed while every
  # accuracy table beside them was a three-seed mean. Each seed writes its own
  # stage_d/seed<N>/; d_aggregate below collapses them.
  #
  # Idempotent: an arm already analysed is skipped unless --force, so re-running
  # d_arms costs nothing and cannot overwrite published numbers.
  local seed
  local extra=()
  [[ "${DRY_RUN}" -eq 1 ]] && extra=(--dry-run)
  [[ "${FORCE}" -eq 1 ]] && extra+=(--force)
  for seed in $(config_seeds); do
    "${RUN}" tools/stage_d_arms.py $(stage_args) \
      --seed "${seed}" --backbone "${BB}" \
      --capacity_backbone "${CAPBB}" \
      --devices "${DEVICE}" "${extra[@]}"
  done
}

# Collapse stage_d/seed*/ into stage_d/ itself: the seed-averaged files with
# bootstrap intervals that the paper's tables, paper/figures_src/make_figures.py
# and the two checkers all read. Nothing downstream knows a seed dimension
# exists underneath, which is the point -- six consumers re-deriving "mean over
# seeds, then bootstrap the seed-level values" is six chances to disagree.
step_d_aggregate() {
  # --config, not just --dataset: both tools default to codes/outputs, so a
  # campaign writing elsewhere (config `paths.output_root`) would otherwise be
  # aggregated from -- and written into -- the wrong tree.
  run_cmd "${RUN}" tools/stage_d_aggregate.py --dataset "${DS}" --config "${CFG}"
}

# The confidence-gated deployment policy of the paper's Table "tab:gate",
# measured independently at each seed and averaged. BeeMachine only: it is the
# deployment target, and it is the only dataset with a reference arm the gate
# can fall back to in the paper's sense.
step_d_gate() {
  if [[ "${DS}" != "beemachine" ]]; then
    # step_skip, not `return 0`: the runner then records the step done without
    # demanding an artifact this dataset was never going to write.
    step_skip "d_gate: the deployment gate is measured on beemachine only"
    return 0
  fi
  run_cmd "${RUN}" tools/deployment_gate.py --dataset "${DS}" --config "${CFG}"
}

step_report_hashes() {
  run_cmd "${RUN}" stage_report.py hash_ckpts $(stage_args) \
    --glob "${OUT_ROOT}/**/*.pt"
}

step_report_repro() {
  run_cmd "${RUN}" stage_report.py repro_pack $(stage_args)
}

artifact_for() {
  case "$1" in
    contract) echo "${OUT_ROOT}/contract/study_contract.md" ;;
    preflight) echo "${OUT_ROOT}/contract/preflight.md" ;;
    freeze) echo "${OUT_ROOT}/frozen_splits/${SPLIT_FILE}" ;;
    a_qa) echo "${OUT_ROOT}/stage_a/qa_summary.json" ;;
    a_sweep) echo "${OUT_ROOT}/stage_a/sweep_summary.json" ;;
    a_pick_best) echo "${BEST_JSON}" ;;
    a_seg_table) echo "${OUT_ROOT}/stage_e/seg_comparison_table${PTAG}.md" ;;
    b_sweep_baselines) echo "${OUT_ROOT}/stage_b/baseline_*_seed*${PTAG}/test_metrics.json" ;;
    # The pool's last-finishing arm is not fixed, so gate on the one that
    # cannot exist unless the whole pool ran: multitask is the only arm here
    # that writes a multitask_* directory, and `sweep_controls` exits non-zero
    # if any of its 21 jobs failed, so a partial pool never reaches this check.
    b_sweep_controls) echo "${OUT_ROOT}/stage_b/multitask_*${PTAG}/test_metrics.json" ;;
    b_extract_desc_gt) echo "${DESC_GT}" ;;
    b_extract_desc_pred) echo "${DESC_PRED}" ;;
    b_ablate_gt)
      local label=gt; is_large_cls && label=pseudo
      echo "${OUT_ROOT}/stage_b_descriptors/*_${label}_*${PTAG}/test_metrics.json"
      ;;
    b_ablate_pred) echo "${OUT_ROOT}/stage_b_descriptors/*_pred_*${PTAG}/test_metrics.json" ;;
    b_ablation_table) echo "${OUT_ROOT}/stage_e/ablation_table${PTAG}.md" ;;
    leakage_check) echo "${OUT_ROOT}/frozen_splits/part_scale_leakage.json" ;;
    c_pseudo) echo "${OUT_ROOT}/stage_c/pseudo_masks/shard*/confidence.json" ;;
    c_filter_sweep) echo "${OUT_ROOT}/stage_c/pseudo_masks_threshold_sweep.json" ;;
    c_filter) echo "${OUT_ROOT}/stage_c/pseudo_masks_conf0.7/filter_meta.json" ;;
    c_extract_desc) echo "${OUT_ROOT}/stage_c/pseudo_descriptors.csv" ;;
    c_ddp_scaling) echo "${OUT_ROOT}/stage_c/scaling/gated_residual_*${PTAG}/test_metrics.json" ;;
    # stage_c.py plot_curve writes stage_c/curves/scaling_curve.csv; the .png
    # beside it is only produced when matplotlib imports AND the n_train/top1
    # columns exist, so the CSV is the artifact to gate resumption on.
    c_plot_curve) echo "${OUT_ROOT}/stage_c/curves/scaling_curve${PTAG}.csv" ;;
    # Gate on a per-seed directory, not on stage_d/ itself: stage_d/ is written
    # by d_aggregate, so gating d_arms there would mark it complete before any
    # seed had run.
    d_arms) echo "${OUT_ROOT}/stage_d${PTAG}/seed*/robustness_whole.csv" ;;
    d_aggregate) echo "${OUT_ROOT}/stage_d${PTAG}/robustness.csv" ;;
    d_gate) echo "${OUT_ROOT}/stage_d${PTAG}/deployment_gate.json" ;;
    report_hashes) echo "${OUT_ROOT}/stage_f/checkpoint_hashes.txt" ;;
    report_repro) echo "${OUT_ROOT}/stage_f/repro_pack_*.zip" ;;
  esac
}

run_dataset() {
  DS="$1"
  resolve_dataset_paths "${CFG}" "${DS}"
  DS="${DATASET}"
  # Resume markers are per protocol. Sharing one directory would make a
  # capacity-matched run resume-skip steps the default grid completed, and
  # silently produce a protocol table with default-protocol numbers in it.
  if [[ "${PROTOCOL}" == "default" ]]; then
    STATE_DIR="${OUT_ROOT}/.experiment_state"
  else
    STATE_DIR="${OUT_ROOT}/.experiment_state_${PROTOCOL}"
  fi
  # Every artifact a protocol run produces carries this suffix (see
  # config.stage_run_dir). artifact_for() has to look for the tagged path, or a
  # completed protocol step would look unfinished and re-run on every resume.
  PTAG="$(protocol_tag)"
  BEST_JSON="${OUT_ROOT}/stage_a/best_model.json"
  DESC_GT="${OUT_ROOT}/stage_b_descriptors/descriptors_gt.csv"
  DESC_PRED="${OUT_ROOT}/stage_b_descriptors/descriptors_pred.csv"
  [[ "${DRY_RUN}" -eq 1 ]] || mkdir -p "${STATE_DIR}"

  echo
  echo "=== dataset=${DS} config=${CFG} output=${OUT_ROOT} ==="
  run_steps "${DS}"
}

for ds in "${DATASETS[@]}"; do
  run_dataset "${ds}"
done

# --- cross-dataset reporting (research plan Sec. 5 Stage E, Sec. 7) ---------
# Outside the per-dataset loop by necessity: both tables read every dataset's
# ablation_table.csv at once. Skipped when only one dataset was requested,
# since a "cross-dataset" table over one dataset is not one.
if [[ "${#DATASETS[@]}" -gt 1 ]]; then
  # The campaign root, from the config -- NOT a hardcoded ${SCRIPT_DIR}/outputs.
  # resolve_dataset_paths sets OUT_ROOT to <root>/<dataset>, so its parent is
  # the root shared by all three. Hardcoding it here would have pooled the
  # published ConvNeXt-Nano tables at the end of a run that trained nothing in
  # that tree.
  POOLED_ROOT="$(dirname "${OUT_ROOT}")"
  POOLED_INPUTS=()
  for ds in "${DATASETS[@]}"; do
    f="${POOLED_ROOT}/${ds}/stage_e/ablation_table.csv"
    [[ -f "${f}" || "${DRY_RUN}" -eq 1 ]] && POOLED_INPUTS+=("${f}")
  done
  if [[ "${#POOLED_INPUTS[@]}" -gt 1 ]]; then
    echo
    echo ">>> pooled reporting across ${#POOLED_INPUTS[@]} dataset(s)"
    run_cmd "${RUN}" stage_report.py pooled_table --config "${CFG}" \
      --inputs "${POOLED_INPUTS[@]}" \
      --out "${POOLED_ROOT}/pooled/pooled_table.md"
    run_cmd "${RUN}" stage_report.py recommend --config "${CFG}" \
      --inputs "${POOLED_INPUTS[@]}" \
      --out "${POOLED_ROOT}/pooled/recommendation_table.md"
  else
    echo "SKIP pooled reporting: need >=2 ablation_table.csv files, found ${#POOLED_INPUTS[@]}"
  fi
fi

echo
echo "All requested datasets processed: ${DATASETS[*]}"

# Inside the brace group by necessity -- see the comment at the top of the
# file. An `exit` placed after the closing brace would have to be re-read from
# disk, which is the failure this is here to prevent.
exit 0
}
