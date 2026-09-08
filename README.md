# Anatomy-guided bumble bee identification

This repository contains code and datasets to reproduce the experiments in the paper *An Evaluation of Anatomy-Guided Learning Strategies for Bumble Bee Identification on a Production Citizen-Science Platform*.

It compares seven mechanisms for combining part-level anatomy with a whole-image classifier (whole-image reference, body-masked input, multi-task supervision, part-crop late fusion, attention-pooled part fusion, descriptor concatenation, gated residual fusion) on three datasets, then runs the capacity, heavy-augmentation, calibration, robustness, gated-deployment, and FinerCAM analyses reported in the paper.

## 1. Environment

Python 3.10+ with a CUDA-capable PyTorch build is required. Create an environment and install:

```bash
pip install -r requirements.txt
```

All commands below are run from this directory through `./run.sh`, which only sets `PYTHONPATH`. Override the interpreter with `PYTHON=/path/to/python ./run.sh ...` if needed. The pipeline uses every visible GPU (`compute.num_gpus: -1` in `config.yaml`). Restrict visibility with `CUDA_VISIBLE_DEVICES` to use fewer devices. The paper trained with eight GPUs; a single GPU works, but wall-clock time scales accordingly.

Optional classical descriptor packages (`mahotas`, `brisque`, `pypiqe`) and `pyiqa` are required for the descriptor arms and for CLIPIQA / MANIQA / MUSIQ. `grad-cam` is required only for the Finer-CAM figure.

Training reads images through a cache of decoded, pre-resized `uint8` arrays, memory-mapped so every concurrent job and DataLoader worker shares one copy of the pages. `cache.root` in `config.yaml` defaults to `/dev/shm/beemachine_cache`, which is tmpfs and therefore RAM. At 320 px, budget roughly 3 GB for the BeeMachine part set, 14 GB for its six-fold augmented copy, and about 53 GB for the 171k-image classification corpus. BeeMachine sets `cls_image_size` to 320 as well, so the pseudo-mask stage and BeeMachine classification training share that one cache rather than building two. On a machine with less RAM, point `cache.root` (or `BEEMACHINE_CACHE_ROOT`) at a disk path, which still avoids re-decoding and only loses the shared-pages property, or turn the cache off with `cache.enabled: false` or `BEEMACHINE_CACHE=0`.

## 2. Data

Point `config.yaml` at local copies of the three corpora (or set `PARTWHOLE_ROOT` / `LARGE_CLS_ROOT`). The placeholders are:

| Dataset | `partwhole_root` | `large_cls_root` |
|---|---|---|
| BeeMachine | `./data/beemachine/partset` | `./data/beemachine/classification` |
| CUB | `./data/cub` | `./data/cub` |
| Fish-Vista | `./data/fish_vista` | `./data/fish_vista` |

### Where to obtain each corpus

**BeeMachine segmentation (part set).** Available at [https://huggingface.co/datasets/KDDResearch/BeeMachine_Partwhole_Dataset](https://huggingface.co/datasets/KDDResearch/BeeMachine_Partwhole_Dataset). Download `images.tar.gz`, `masks.tar.gz`, `species_labels.csv`, and (optional) `annotations.coco.json`, then unpack the archives into `partwhole_root` (or symlink there):

```bash
tar -xzf images.tar.gz
tar -xzf masks.tar.gz
```

Expected layout: `images/`, `masks/` (one `{stem}_m.png` per image), `species_labels.csv` with columns `images` and `species` (7,716 images, three parts plus background).

**BeeMachine classification corpus.** The full BeeMachine classification dataset is very large and is separate from the Hugging Face part set; it will be available upon request. When obtained, expect 171,227 originals across 147 species: one subdirectory per species, originals only (no synthetic blur / rotation / color variants), at most 3,000 images per species. `tools/bee_data_gen.py` can rebuild an equivalent corpus from source photograph collections if needed (`--cap 3000`, `--min-originals 5`).

**CUB.** Classification images and metadata come from Wah et al. (2011), Caltech-UCSD Birds-200-2011 ([dataset record](https://data.caltech.edu/records/65de6-vp158), [project page](https://www.vision.caltech.edu/datasets/cub_200_2011/)). Pixel-level part masks (`AnnotationMasksPerclass/`, 11 parts plus background) come from the CUB70 part-segmentation release of Behzadi-Khormouji and Oramas (WACV 2023) ([GitHub](https://github.com/hamedbehzadi/CUB70-PartSegmentationDataset)). Place official CUB under `./data/cub` together with `part_labels.txt` and `AnnotationMasksPerclass/`. Classification uses the full 200-class, 11,788-image corpus; the part-annotated subset is used for Stage A.

**Fish-Vista.** Classification and trait segmentation are distributed together by Mehrab et al. (CVPR 2025): [Hugging Face](https://huggingface.co/datasets/imageomics/fish-vista), [GitHub](https://github.com/Imageomics/Fish-Vista), [CVPR Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Mehrab_Fish-Vista_A_Multi-Purpose_Dataset_for_Understanding__Identification_of_Traits_CVPR_2025_paper.html). Expected layout: `Images/`, `segmentation_masks/images/`, `segmentation_masks/seg_id_trait_map.json`, and the official `segmentation_{train,val,test}.csv` / `classification_{train,val,test}.csv` folds.

Part vocabularies and corpus construction details for CUB and Fish-Vista are also described in the PADC paper ([Open Access](https://openaccess.thecvf.com/content/CVPR2026W/V4A/html/Choton_Part-Aware_Descriptor_Classifier_for_Trustworthy_Species_Detection_CVPRW_2026_paper.html)).

The folds used in the paper ship under `outputs/{dataset}/frozen_splits/`, together with the part/classification overlap lists. For BeeMachine and CUB the overlapping images are dropped from the classification corpus before the freeze, so they appear in no fold; for Fish-Vista, whose validation and test folds are the published ones, they are dropped from the training fold only. BeeMachine and CUB use a stratified 75/15/10 freeze; Fish-Vista records the official `segmentation_*` and `classification_*` CSV folds instead, because that corpus publishes its own. `stage_a.py freeze` reuses whatever is already there and will not overwrite it unless `--force` is passed. Keep those files so reported numbers are measured on the same folds. Their `meta` blocks record the dataset roots as the relative placeholders above; nothing reads that field, so pointing `config.yaml` at your own paths does not invalidate them.

## 3. Train-only geometric augmentation

Stage A expands only the segmentation training fold with the six paired views from the paper (original, horizontal flip, vertical flip, 90/180/270 rotation). Validation and test are never augmented.

```bash
./run.sh data_processing/make_train_aug.py --dataset beemachine
./run.sh data_processing/make_train_aug.py --dataset cub
./run.sh data_processing/make_train_aug.py --dataset fish_vista
```

This writes `train_aug_images/`, `train_aug_masks/`, and (except CUB) the corresponding CSV under each part-set root. It clears and rebuilds those two directories on every run, so do not keep anything else inside them.

Freeze the splits first if they are not already present. For BeeMachine and CUB the generator reads the frozen training names; for Fish-Vista it reads the official `segmentation_train.csv`, which is the same fold the freeze records.

## 4. Run the experiments

One command runs every table in the paper for the default recipe (segmentation sweep, pseudo-masks at confidence 0.7, the seven mechanisms at seeds 13/42/77, the eight-backbone whole-image sweep, the ConvNeXt-Small capacity control, descriptor-group ablation, calibration / AURC / rare-species metrics, synthetic corruptions, and the BeeMachine confidence gate):

```bash
./run_all_experiments.sh
```

Restrict to one corpus with `--dataset beemachine` (or `cub`, `fish_vista`). Resume after interruption by re-issuing the same command. Use `--from STEP` / `--to STEP` / `--list-steps` to run a slice. `--dry-run` prints the commands without executing them.

Then retrain all seven mechanisms under the heavy-augmentation recipe (random erasing, Mixup, and CutMix) that the body and appendix report:

```bash
./run_all_experiments.sh --protocol heavy_aug
```

Stage A, the frozen segmenter, pseudo-masks, and descriptor CSVs are shared across protocols. Only classification training and the reliability/robustness pass are repeated.

The two additive runners below train the same heavy-augmentation cells without walking the full pipeline again, and can be used instead of `--protocol heavy_aug` if the default-recipe campaign has already finished:

```bash
./run.sh run_haug_multitask.py
./run.sh run_haug_arms.py
```

### What each stage measures

| Stage | Paper content |
|---|---|
| Setup | Study contract, part/classification content-overlap check, split freeze, and a preflight that fails immediately on a missing dataset root, split, or augmentation directory rather than partway into Stage A |
| A | Nine-decoder segmentation sweep (DeepLabV3+, FPN, LinkNet, MANet, PAN, PSPNet, SegFormer, UPerNet, UNet++) with a fixed ResNeXt-50 32×4d encoder; Dice loss; select by validation class-mean IoU |
| C (pseudo) | Confidence-gated pseudo-masks on the classification corpus; retention sweep at δ ∈ {0.5, 0.7, 0.9}; keep δ = 0.7 |
| B | Seven mechanisms on ConvNeXt-Nano, three seeds; eight-backbone whole-image sweep; ConvNeXt-Small capacity control; descriptor-only floor and descriptor-group ablation |
| D | Top-1 / top-3 / macro-F1 / rare-species accuracy, ECE (15 bins), rare ECE, AURC, corruption deltas, and the BeeMachine confidence-gated policy swept over τ from 0 to 1 in steps of 0.05, per seed, with a split-half re-selection of τ alongside it |
| Report | Seed-level bootstrap (2,000 resamples) and paired consistency tables |

Classification trains at most 100 epochs, AdamW at 10⁻⁴, batch size 64, label smoothing 0.1, plateau factor 0.5 with patience 2, early stopping on validation loss with patience 20. Segmentation trains at most 200 epochs, Adam at 10⁻⁴, batch size 128, cosine period 50, early stopping on class-mean IoU with patience 40.

Heavy augmentation: random erasing with probability 0.5 and scale 0.02–0.2, Mixup and CutMix with shape 0.2, applied per batch with equal probability, composed with label smoothing.

## 5. Finer-CAM figure

The appendix figure uses Finer-CAM, which explains a prediction by contrasting the true class against the classes closest to it in logit score, rather than against the background as ordinary Grad-CAM does:

- Zhang, Gu, Chowdhury, Mai, Carlyn, Berger-Wolf, Su and Chao. *Finer-CAM: Spotting the Difference Reveals Finer Details for Visual Explanation.* CVPR 2025. [arXiv:2501.11309](https://arxiv.org/abs/2501.11309), [CVPR Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Finer-CAM_Spotting_the_Difference_Reveals_Finer_Details_for_Visual_Explanation_CVPR_2025_paper.pdf).
- Official implementation: [Imageomics/Finer-CAM](https://github.com/Imageomics/Finer-CAM).
- The implementation used here is the one merged into [jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam), distributed as the `grad-cam` package on PyPI. Verified against 1.5.5.

After the default and heavy-augmentation BeeMachine checkpoints exist (seed 42):

```bash
./run.sh tools/finercam_analysis.py --dataset beemachine \
  --arms whole,heavy_aug,multitask,multitask_haug
```

For each named arm it writes one overlay per test index, plus the unmodified input image, into `outputs/beemachine/finercam/bee_subplots/`. The indices are fixed at 0, 101, 245, 440, 550 and 760 of the frozen BeeMachine test split, which are the ones the appendix figure shows. Omit `--arms` to render all ten arms in `ARM_SPECS` instead of the four the figure uses.

## 6. Outputs

Per dataset, under `outputs/{dataset}/`:

- `contract/` — the locked study contract and the preflight report
- `frozen_splits/` — the folds shipped with this package
- `stage_a/` — segmenter sweep, selected checkpoint
- `stage_b/`, `stage_b_descriptors/` — classification arms (heavy-augmentation runs are tagged `__haug`)
- `stage_c/` — pseudo-masks and descriptors
- `stage_d/` — reliability, robustness, and (BeeMachine) the gated policy
- `stage_e/` — comparison tables with bootstrap intervals
- `stage_f/` — checkpoint hashes and the reproducibility archive

A resolution-matched protocol (`--protocol resolution_384`) is implemented in `config.yaml` and was not run. Appendix "A resolution-matched control we did not run" gives the reasons.

## 7. License

This code is released under the MIT License; see `LICENSE`. The three corpora and the third-party packages in `requirements.txt` carry their own licenses, which are not affected by it.
