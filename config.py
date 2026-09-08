"""Config helpers and path utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

CODES_ROOT = Path(__file__).resolve().parent
DATASET_CHOICES = ("beemachine", "cub", "fish_vista")

# Experiment protocols. `default` is the study's primary grid; the other two
# re-run the same seven comparison arms with exactly one variable changed, so
# each isolates a confound the main comparison cannot rule out on its own:
#
#   resolution_384    every arm at a larger input resolution, which is what
#                     "more capacity" means for an anatomy-guided arm (more
#                     pixels on each part) rather than more parameters.
#   heavy_aug         every arm under the Mixup / CutMix / RandomErasing
#                     recipe, so a regularization advantage cannot be mistaken
#                     for an anatomical one.
#
# `default` MUST stay a no-op. Every knob below is read only when the active
# protocol is not "default", and `protocols.default` in config.yaml is empty,
# so a run that does not pass --protocol resolves to byte-identical config,
# output paths and resume markers as before protocols existed. That property
# is what lets these land while a full campaign is mid-flight; it is enforced
# by tests/test_config.py::test_default_protocol_resolves_identically.
# `resolution_384`, not `capacity_matched`. That name was used for two
# different things: this PROTOCOL, which retrains every arm at 384px, and the
# comparison ARM `stage_b.py capacity_matched`, which retrains the whole-image
# reference on a larger backbone at fixed resolution. Both are answers to "is
# this capacity rather than anatomy?", but they change different variables, and
# the paper had to spend two sentences telling the reader which was which. The
# arm keeps its name (its run directories are on disk and published); the
# protocol is renamed, which is safe because no protocol run has ever been
# executed -- no `__cap` run directory exists on any dataset.
PROTOCOL_CHOICES = ("default", "resolution_384", "heavy_aug")
DEFAULT_PROTOCOL = "default"


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` onto ``base``, returning a new dict.

    Mappings merge key-by-key; every other type (including lists) is replaced
    wholesale, so an overlay that sets ``segmentation.models`` gets exactly the
    models it lists rather than those appended to the base sweep.
    """
    merged = dict(base)
    for key, value in (overlay or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_with_extends(path: Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load a YAML config, applying an optional ``extends:`` base first.

    Lets a variant config declare only the values it actually changes instead
    of restating the whole dataset registry. That duplication is not cosmetic:
    the registry holds every dataset root, and two copies of it drift apart the
    moment one is edited.

    ``extends`` is resolved relative to the extending file. Cycles raise rather
    than recursing until the stack blows.
    """
    path = path.resolve()
    if path in _seen:
        chain = " -> ".join(str(p) for p in (*_seen, path))
        raise SystemExit(f"Circular config 'extends' chain: {chain}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"Config {path} must be a YAML mapping, got {type(cfg).__name__}")

    base_ref = cfg.pop("extends", None)
    if not base_ref:
        return cfg

    base_path = Path(base_ref)
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    if not base_path.is_file():
        raise SystemExit(f"Config {path} extends missing file: {base_path}")
    base = _load_yaml_with_extends(base_path, (*_seen, path))
    return _deep_merge(base, cfg)


def config_chain(path: str | Path) -> list[Path]:
    """The config file plus every base it extends, nearest first.

    The effective config is the whole chain, so anything that archives or
    hashes "the config" needs all of it.
    """
    chain: list[Path] = []
    current: Path | None = Path(path).resolve()
    while current is not None:
        if current in chain:
            raise SystemExit(f"Circular config 'extends' chain at {current}")
        chain.append(current)
        with current.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        base_ref = raw.get("extends") if isinstance(raw, dict) else None
        if not base_ref:
            break
        base = Path(base_ref)
        current = base if base.is_absolute() else (current.parent / base).resolve()
    return chain


def load_config(
    path: str | Path | None = None,
    dataset: str | None = None,
    protocol: str | None = None,
) -> dict[str, Any]:
    if path is None:
        path = CODES_ROOT / "config.yaml"
    path = Path(path)
    cfg = _load_yaml_with_extends(path)

    cfg["_config_path"] = str(path.resolve())

    # Active dataset: CLI > env > yaml
    active = dataset or os.environ.get("DATASET") or cfg.get("dataset") or "beemachine"
    active = str(active).strip().lower()
    if active not in DATASET_CHOICES:
        raise SystemExit(f"Unknown dataset '{active}'. Choose from {DATASET_CHOICES}")
    cfg["dataset"] = active

    registry = cfg.get("datasets") or {}
    if active not in registry:
        raise SystemExit(f"Dataset '{active}' missing from config datasets: registry")
    entry = dict(registry[active])

    # Seed registry. `seed` stays the default for single-run commands; `seeds`
    # is the list every (arm, mask source, descriptor group) cell is repeated
    # under for the CI / paired-significance analysis. Normalized here so a
    # scalar, a missing key, or a stray string all resolve to a clean list of
    # ints instead of silently degrading to a one-seed comparison downstream.
    # A per-dataset `seeds` in the `datasets.<name>` entry overrides the
    # top-level default (2026-08-24: Beemachine is the deployment target and
    # keeps the full multi-seed grid; CUB / Fish-Vista are consistency checks
    # only and were pinned to a single seed).
    raw_seeds = entry.get("seeds", cfg.get("seeds"))
    if raw_seeds is None:
        seeds = [int(cfg.get("seed", 42))]
    elif isinstance(raw_seeds, (int, str)):
        seeds = [int(raw_seeds)]
    else:
        try:
            seeds = [int(s) for s in raw_seeds]
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"config 'seeds' must be a list of integers, got {raw_seeds!r}") from exc
    if not seeds:
        raise SystemExit("config 'seeds' resolved to an empty list")
    # Deduplicate while preserving order so a duplicated seed cannot inflate
    # the apparent sample size of a confidence interval.
    cfg["seeds"] = list(dict.fromkeys(seeds))
    cfg["seed"] = int(cfg.get("seed", cfg["seeds"][0]))

    # Resolve flat paths / part labels from the active entry
    paths = dict(cfg.get("paths") or {})
    paths["partwhole_root"] = entry["partwhole_root"]
    paths["large_cls_root"] = entry["large_cls_root"]

    base_out = Path(paths.get("output_root") or "./outputs")
    # Namespace under outputs/{dataset} unless the yaml already ends with the dataset name
    if base_out.name != active:
        paths["output_root"] = str(base_out / active)
    else:
        paths["output_root"] = str(base_out)

    if os.environ.get("PARTWHOLE_ROOT"):
        paths["partwhole_root"] = os.environ["PARTWHOLE_ROOT"]
    if os.environ.get("LARGE_CLS_ROOT"):
        paths["large_cls_root"] = os.environ["LARGE_CLS_ROOT"]

    cfg["paths"] = paths
    cfg["part_labels"] = list(entry.get("part_labels") or cfg.get("part_labels") or [])
    # Per-dataset resolution overrides, surfaced as the top-level values every
    # stage reads so a dataset that runs at one resolution everywhere (see
    # `cls_image_size_for`) does not need each call site to know about it.
    if entry.get("image_size"):
        cfg["image_size"] = int(entry["image_size"])
    if entry.get("cls_image_size"):
        cfg["cls_image_size"] = int(entry["cls_image_size"])

    # Active protocol: CLI > env > "default". Resolved AFTER the per-dataset
    # resolution override above, because a protocol's whole purpose is to
    # override what the dataset would otherwise run at.
    #
    # Deliberately does NOT touch `image_size` (segmentation). Stage A, the
    # pseudo-mask cache and the descriptor vectors are protocol-independent by
    # design: every protocol reuses the one frozen segmenter and the one
    # descriptor CSV, so the only thing that varies between protocols is how
    # the classification arms consume them. That is what keeps the comparison
    # controlled, and it is also what makes these runs affordable.
    active_protocol = (
        protocol or os.environ.get("BEEMACHINE_PROTOCOL") or DEFAULT_PROTOCOL
    )
    active_protocol = str(active_protocol).strip().lower()
    if active_protocol not in PROTOCOL_CHOICES:
        raise SystemExit(
            f"Unknown protocol '{active_protocol}'. Choose from {PROTOCOL_CHOICES}"
        )
    cfg["protocol"] = active_protocol
    if active_protocol != DEFAULT_PROTOCOL:
        spec = dict((cfg.get("protocols") or {}).get(active_protocol) or {})
        if not spec:
            raise SystemExit(
                f"Protocol '{active_protocol}' has no entry under `protocols:` in "
                f"{path}. A protocol with no settings would silently run the "
                "default grid into protocol-tagged output directories."
            )
        if spec.get("cls_image_size"):
            cfg["cls_image_size"] = int(spec["cls_image_size"])
            # A per-backbone override map would re-pin some arm to its own
            # native resolution and quietly undo the one thing this protocol
            # varies. Clear it so "every arm at the larger size" is true.
            entry["image_size_overrides"] = {}
            cfg.setdefault("classification", {})["image_size_overrides"] = {}
        cfg["heavy_augmentation"] = bool(spec.get("heavy_augmentation", False))
        cfg["_protocol_run_tag"] = str(spec.get("run_tag") or f"__{active_protocol}")

    cfg["layout"] = entry.get("layout", "beemachine_v6")
    cfg["split_policy"] = entry.get("split_policy", "stratified_freeze")
    cfg["dataset_entry"] = entry

    # Merge per-dataset descriptor expected_dim when provided
    desc = dict(cfg.get("descriptors") or {})
    entry_desc = entry.get("descriptors") or {}
    if entry_desc.get("expected_dim") is not None:
        dim = int(entry_desc["expected_dim"])
        if dim > 0:
            desc["expected_dim"] = dim
        else:
            desc["expected_dim"] = 0  # dynamic
    cfg["descriptors"] = desc

    return cfg


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_config_path() -> str:
    return str(CODES_ROOT / "config.yaml")


def add_dataset_arg(parser) -> None:
    """Attach shared --dataset flag to a top-level argparse parser."""
    parser.add_argument(
        "--dataset",
        default=None,
        choices=list(DATASET_CHOICES),
        help="Override config dataset (beemachine | cub | fish_vista)",
    )


def add_protocol_arg(parser) -> None:
    """Attach the shared ``--protocol`` flag."""
    parser.add_argument(
        "--protocol",
        default=None,
        choices=list(PROTOCOL_CHOICES),
        help=(
            "Experiment protocol (default | resolution_384 | heavy_aug). "
            "Omit for the study's primary grid; the other two re-run the same "
            "arms at a larger resolution or under heavy augmentation, into "
            "protocol-tagged run directories."
        ),
    )


def add_global_stage_args(parser) -> None:
    """Attach ``--config`` / ``--dataset`` / ``--protocol``."""
    parser.add_argument("--config", default=default_config_path())
    add_dataset_arg(parser)
    add_protocol_arg(parser)


def dataset_cli_args(cfg: dict[str, Any]) -> list[str]:
    """Args to forward to child stage processes so they keep the same context.

    Carries the protocol as well as the dataset: a sweep that fans out to one
    subprocess per seed would otherwise have every child silently fall back to
    the default protocol and overwrite the default grid's run directories.
    Omitted entirely for the default protocol so existing child command lines
    are unchanged.
    """
    args = ["--dataset", str(cfg.get("dataset", "beemachine"))]
    if protocol(cfg) != DEFAULT_PROTOCOL:
        args += ["--protocol", protocol(cfg)]
    return args


def cls_image_size_for(cfg: dict[str, Any], backbone: str | None = None) -> int:
    """Classification input size, honoring per-dataset and per-backbone sizes.

    Resolution order: per-backbone override > per-dataset ``cls_image_size`` >
    global ``cls_image_size``. Beemachine sets its own so that segmentation,
    classification, and the learned NR-IQA branch all run at one resolution,
    which also means the RAM image cache holds a single array per fold instead
    of one per stage.

    The per-backbone override map is empty by default and is expected to stay
    that way: the comparison grid pins every arm to one common backbone so that
    image size is constant and the fusion mechanism is the only free variable.
    An entry here makes one arm differ from the rest in input size, which is a
    confound the control arms cannot absorb.
    """
    default = int(cls_image_size(cfg))
    if not backbone:
        return default
    entry = cfg.get("dataset_entry") or {}
    overrides = entry.get("image_size_overrides")
    if overrides is None:
        overrides = (cfg.get("classification") or {}).get("image_size_overrides") or {}
    return int(overrides.get(backbone, default))


def cls_image_size(cfg: dict[str, Any]) -> int:
    """The active dataset's classification resolution.

    Reads the top-level value first, because a non-default protocol overrides
    it there (see load_config) and must win over the dataset entry it is
    deliberately replacing. For the default protocol the two agree, so the
    resolution order is unchanged.
    """
    if cfg.get("protocol", DEFAULT_PROTOCOL) != DEFAULT_PROTOCOL:
        return int(cfg["cls_image_size"])
    entry = cfg.get("dataset_entry") or {}
    return int(entry.get("cls_image_size") or cfg["cls_image_size"])


def protocol(cfg: dict[str, Any]) -> str:
    """Active experiment protocol: default | resolution_384 | heavy_aug."""
    return str(cfg.get("protocol") or DEFAULT_PROTOCOL)


def run_tag(cfg: dict[str, Any]) -> str:
    """Suffix appended to every trained run's directory name.

    Empty for the default protocol, which is what keeps existing output paths
    and resume markers exactly where they were. A protocol run writes
    ``baseline_convnext_nano.in12k_seed42__res384`` alongside the default's
    ``baseline_convnext_nano.in12k_seed42`` rather than overwriting it, so the
    two are directly comparable and neither can clobber the other.
    """
    if protocol(cfg) == DEFAULT_PROTOCOL:
        return ""
    return str(cfg.get("_protocol_run_tag") or f"__{protocol(cfg)}")


def stage_run_dir(cfg: dict[str, Any], *parts: str) -> Path:
    """Output directory for one trained run, tagged with the active protocol.

    Only the LAST path component is tagged, so protocol runs interleave inside
    the existing ``stage_b`` / ``stage_b_descriptors`` trees instead of forking
    a parallel output root. Shared inputs -- the frozen segmenter, pseudo-masks,
    descriptor CSVs -- keep one canonical location that every protocol reads.

    When that component is a filename, the tag goes before the extension
    (``scaling_curve__cap.csv``, not ``scaling_curve.csv__cap``), so the result
    still opens in whatever reads that file type.
    """
    root = Path(cfg["paths"]["output_root"])
    *head, name = parts
    tag = run_tag(cfg)
    if tag:
        stem, dot, ext = str(name).partition(".")
        name = f"{stem}{tag}{dot}{ext}"
    return root.joinpath(*head, name)


def known_run_tags(cfg: dict[str, Any] | None = None) -> list[str]:
    """Every non-empty run tag any protocol can produce.

    Needed to tell a default-protocol run directory from a protocol one: the
    default tag is the empty string, so a glob built from it matches every
    protocol's directories too. Callers filter those out by name.
    """
    protocols = (cfg or {}).get("protocols") or {}
    tags = []
    for name in PROTOCOL_CHOICES:
        if name == DEFAULT_PROTOCOL:
            continue
        spec = protocols.get(name) or {}
        tags.append(str(spec.get("run_tag") or f"__{name}"))
    return tags


def belongs_to_protocol(name: str, cfg: dict[str, Any]) -> bool:
    """Whether a run directory (or file) name belongs to the active protocol.

    The default protocol owns every name that carries no protocol tag; a
    non-default protocol owns exactly the names ending in its own tag.
    """
    tag = run_tag(cfg)
    stem = str(name)
    if tag:
        return stem.endswith(tag)
    return not any(stem.endswith(t) for t in known_run_tags(cfg))


def heavy_augmentation(cfg: dict[str, Any]) -> bool:
    """Whether every arm trains under the heavy-augmentation recipe."""
    return bool(cfg.get("heavy_augmentation", False))


def part_image_size(cfg: dict[str, Any]) -> int:
    """The active dataset's segmentation / part resolution."""
    entry = cfg.get("dataset_entry") or {}
    return int(entry.get("image_size") or cfg["image_size"])


# Sources whose contents determine a number in any table. Frozen at import so
# every run in a process records the code it actually loaded -- see
# `code_version`.
def _result_determining_sources() -> list[Path]:
    """The modules and config every training / evaluation path imports.

    Top-level `codes/*.py` plus `config.yaml`. `tests/` is excluded (it cannot
    change a result) and so is `tools/`, whose scripts are standalone analysis
    entry points that import these modules rather than being imported by them.
    """
    files = [p for p in sorted(CODES_ROOT.glob("*.py")) if p.name != "conftest.py"]
    cfg_yaml = CODES_ROOT / "config.yaml"
    if cfg_yaml.exists():
        files.append(cfg_yaml)
    return files


def _compute_source_digest() -> str:
    """sha256-12 over the contents of `_result_determining_sources()`.

    Content-addressed, so it is stable across a `git commit` that changes no
    file and distinct between two working trees that git would both call
    `-dirty`. This is the stamp `stage_report.py ablation_table` compares;
    `code_version` is kept alongside it for human readability.
    """
    import hashlib

    try:
        h = hashlib.sha256()
        for path in _result_determining_sources():
            h.update(path.name.encode())
            h.update(b"\0")
            h.update(path.read_bytes())
            h.update(b"\0")
        return h.hexdigest()[:12]
    except Exception:
        # Same contract as `code_version`: this runs at import, in every worker
        # process, and provenance is never worth failing a run over. "unknown"
        # is a value the guard treats as its own version rather than silently
        # matching anything.
        return "unknown"


def _compute_code_version() -> str:
    import subprocess

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=CODES_ROOT, capture_output=True, text=True, timeout=10,
        )
        if sha.returncode != 0:
            return "unknown"
        rev = sha.stdout.strip()
        # `-- .` scopes dirtiness to codes/. Without it, editing paper/main.tex
        # or docs/ while a 6-hour training step ran marked that run's code
        # dirty, which is a claim about the code that is simply false.
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", "."],
            cwd=CODES_ROOT, capture_output=True, text=True, timeout=10,
        )
        return f"{rev}-dirty" if dirty.stdout.strip() else rev
    except Exception:
        return "unknown"


# Both stamps are resolved once, at import, and never re-read. `_finalize_run`
# calls them *after* training, which for a multi-hour step is hours after the
# process loaded its source: on 2026-08-27 the four Fish-Vista fusion arms were
# launched together at 13:09 from one working tree, and because commits landed
# at 13:44 and 15:00 while they trained they finalized with four different
# stamps ('3b939fd-dirty', 'dfd64ad', 'dfd64ad-dirty', 'e96a41c-dirty').
# ablation_table then refused to aggregate byte-identical code. A stamp has to
# describe the code that ran, so it is taken when that code is loaded.
_CODE_VERSION = _compute_code_version()
_SOURCE_DIGEST = _compute_source_digest()


def code_version() -> str:
    """`<git-sha>` of `codes/` at process start, plus `-dirty` when it had edits.

    Stamped into every `run_meta.json` and every summary row so results that
    came from different code cannot be silently averaged together. That is not
    a hypothetical failure: the descriptor z-scoring fix landed after the
    2026-08-11 commit, and for five days the comparison table mixed pre- and
    post-fix rows with nothing on disk to tell them apart -- the pre-fix
    `concat` arm reproduced `descriptors_only` seed for seed because its visual
    branch had been normalised away, and it was published in the same table as
    rows where it had not.

    This is the human-readable half of the stamp and is NOT what the guard in
    `stage_report.py` compares -- a git sha changes on a commit that edits no
    file, and every distinct dirty tree shares the one `-dirty` label. Use
    `source_digest` for identity.

    Returns "unknown" outside a git checkout rather than raising: provenance is
    worth recording when available and never worth failing a training run over.
    """
    return _CODE_VERSION


def source_digest() -> str:
    """Content hash of the code that produced a run. See `code_version`."""
    return _SOURCE_DIGEST


def shape_embed_dim(cfg: dict[str, Any]) -> int:
    """Width of the gated arm's conv shape-encoder embedding.

    Config-driven so Stage B and Stage C cannot drift apart on it: they build
    the same `PartAwareFusionClassifier` from two different call sites, and a
    Python default shared by both is exactly the kind of hyperparameter that
    silently stops being shared.
    """
    return int((cfg.get("descriptors") or {}).get("shape_embed_dim") or 512)


def label_smoothing(cfg: dict[str, Any]) -> float:
    """Label smoothing for every classification arm in Stages B and C.

    One value for all arms, so it stays a property of the shared protocol
    rather than of whichever loop a given arm happens to use. The reference
    protocol (Choton et al., 2026) trains with 0.1; this repo trained with 0.0
    through the 2026-08 campaign, which is a protocol difference against every
    number that campaign is compared to.
    """
    return float((cfg.get("classification") or {}).get("label_smoothing") or 0.0)


def multitask_seg(cfg: dict[str, Any]) -> tuple[str, str]:
    """(arch, encoder) of the multi-task arm's auxiliary segmentation head.

    Read from ``classification.multitask``, NOT from ``segmentation``. The
    multi-task arm is a classification arm of the controlled grid, and the
    encoder it shares between its two heads is the visual backbone its top-1 is
    credited to -- so it belongs to the classification protocol, pinned to the
    same backbone as the other six arms. ``segmentation.arch`` /
    ``segmentation.encoder`` configure Stage A, where ResNeXt-50 is pinned so
    the decoder is the only variable in the nine-model sweep. Reading Stage A's
    encoder here is what left this arm on DeepLabV3+/ResNeXt-50 while every
    other arm ran on ConvNeXt-Nano, making the anatomy signal and the backbone
    inseparable.

    Falls back to ``segmentation`` only when ``classification.multitask`` is
    absent, so a partial config (the unit fixtures build several) still
    resolves rather than raising after a training run has already started.
    """
    mt = (cfg.get("classification") or {}).get("multitask") or {}
    seg = cfg.get("segmentation") or {}
    arch = mt.get("arch") or seg.get("arch")
    encoder = mt.get("encoder") or seg.get("encoder")
    if not arch or not encoder:
        raise SystemExit(
            "config is missing the multi-task arm's segmentation head: set "
            "classification.multitask.{arch,encoder}"
        )
    return str(arch), str(encoder)


def multitask_run_tag(cfg: dict[str, Any]) -> str:
    """The ``{arch}_{encoder}`` fragment in this arm's run-directory name.

    One definition for the writer (`stage_b.cmd_multitask`) and every reader
    (`tools/stage_d_arms.py`, `tools/finercam_analysis.py`), because the two
    used to derive it independently from `segmentation` and would silently
    stop pointing at the same directory the moment either changed.
    """
    arch, encoder = multitask_seg(cfg)
    return f"{arch}_{encoder}".replace("/", "_")


def cls_corpus(cfg: dict[str, Any]) -> str:
    """Which corpus a dataset's classification arms train on.

    ``"large_cls"`` -- the full classification corpus with confidence-gated
                       pseudo-masks. All three configured datasets
                       (Beemachine, CUB, Fish-Vista) use this, and it is the
                       default here.
    ``"part_set"``  -- the pixel-annotated part set, so ground-truth masks
                       exist for every training image. No configured dataset
                       uses this: each part set is a segmenter training set,
                       far smaller than and not comparable to the corpus the
                       deployed model serves (CUB's spans 67 of 200 species;
                       Beemachine's has species with a single training
                       photograph). Selecting it is opt-in and explicit.

    The default is ``large_cls`` rather than ``part_set`` deliberately. It used
    to fall back to ``part_set`` for any non-Fish-Vista layout, so dropping the
    explicit `cls_corpus:` key from a dataset would silently move that
    dataset's classification onto ground-truth part masks -- a different
    experiment from the one reported, with no error and no warning. The
    reported study classifies only on the large corpus, so that is what an
    unspecified config now gets.
    """
    entry = cfg.get("dataset_entry") or {}
    explicit = entry.get("cls_corpus")
    if explicit:
        value = str(explicit)
        if value not in {"large_cls", "part_set"}:
            raise SystemExit(
                f"Unknown cls_corpus: {value!r} (expected 'large_cls' or 'part_set')"
            )
        return value
    return "large_cls"


def cls_patience(cfg: dict[str, Any]) -> int:
    """Early-stopping patience, in epochs, for every classification arm.

    Read by all three hand-rolled training loops and by the multitask arm's
    Lightning trainer, so one config value governs the stopping rule across
    Stages B and C the way `segmentation.patience` does for Stage A. 0 disables
    it and runs the full `classification.epochs` budget.
    """
    return int(cfg.get("classification", {}).get("patience", 0))


def foreground_part_ids(part_labels: list[str]) -> list[int]:
    """Mask IDs for non-background parts (assumes label index == mask id)."""
    ids = []
    for i, name in enumerate(part_labels):
        if str(name).lower() in {"background", "bg"}:
            continue
        ids.append(i)
    return ids


def resolve_seg_ckpt(cfg: dict[str, Any], seg_ckpt: str | None = None) -> str:
    """Use CLI ckpt, else outputs/stage_a/best_model.json."""
    if seg_ckpt:
        return seg_ckpt
    best = Path(cfg["paths"]["output_root"]) / "stage_a" / "best_model.json"
    if best.exists():
        import json

        return json.loads(best.read_text(encoding="utf-8"))["ckpt"]
    raise SystemExit("--seg_ckpt required (or run: python stage_a.py pick_best)")
