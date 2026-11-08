# ILC vs NST on TCGA-BRCA

Explainable classification of invasive lobular carcinoma (ILC) vs invasive carcinoma of no special type
(NST) on TCGA-BRCA H&E diagnostic slides, on top of UNI2-h tile embeddings.

## Reproduce

The steps below are also in one notebook, `ilc_tcga.ipynb`, without steps 6 and 8, the bootstrap intervals and the side checks (H-optimus-0, BEETLE,
the A0SY patches, the strip tiles, the SAE on test): run `python download.py` for the data and models, then the notebook from the
repo folder. Each of its steps loads its result when it is on disk and computes it otherwise; printed results go to
`results/`.

### Environment

Python 3.11.18 on Apple Silicon (`mps`). On the original machine:

    conda create --name ilcTcga --clone ilcDemo
    conda activate ilcTcga
    pip install scikit-learn==1.9.1 openpyxl==3.1.5

Elsewhere, the packages these steps need:

    pip install -r requirements.txt

### 1. Metadata and labels

    python fetch_metadata.py

Queries the GDC API (stdlib `urllib`, no gdc-client, no login) and writes to `data/meta/`:

| file | content |
|---|---|
| `files_slides.json` | raw `/files` answer: TCGA-BRCA diagnostic slides (1,133 slides, 1,062 patients, 1,158.8 GB) |
| `cases_diagnoses.json` | raw `/cases` answer with all GDC diagnoses |
| `files_biotab.json` | raw `/files` answer locating the Biotab patient table |
| `nationwidechildrens.org_clinical_patient_brca.txt` | original TCGA clinical table (md5-checked) |
| `labels.csv` | one row per primary-tumour diagnostic slide |
| `excluded.csv` | slides removed from the dataset, same columns |

Label rules (per patient, joined to slides on the barcode `TCGA-XX-XXXX`; `site` = `XX`, the tissue source site):

- `label` (main): ILC = Biotab `histological_type` ILC **and** ICD-O-3 8520/3; NST = IDC **and** 8500/3.
  Everything else is `excluded`, with the reason in `reason`.
- `label_type`: from `histological_type` alone. Rows with `label` excluded but `label_type` ILC/NST are
  the type/code disagreements (6 ILC-typed, 10 IDC-typed slides).
- GDC diagnoses: the one flagged `diagnosis_is_primary_disease` (others are metastases, recurrences,
  other primaries); `n_gdc_diagnoses` keeps the count. GDC morphology must equal the Biotab code, else the
  script raises.
- `excluded.csv`: `TCGA-E2-A15E-06Z`, the only slide from a metastasis (sample code 06).

Counts from the 2026-11-02 run (GDC data can change between releases):

| `label` | slides | patients |
|---|---|---|
| ILC | 200 | 187 |
| NST | 800 | 752 |
| excluded | 132 | 122 |

1,132 slides from 1,061 patients and 39 sites in `labels.csv`.

### 2. H-optimus-0 (optional)

No later step uses this model, because the embeddings come from UNI2-h (step 3), so this step can be skipped.

Accept the terms at https://huggingface.co/bioptimus/H-optimus-0 (gated), then log in once:

    hf auth login

Then:

    python download_model.py

If `models/h-optimus-0/` lacks `config.json` or `pytorch_model.bin`, this downloads them (plus the model
card `README.md`; weights 4.54 GB). It then builds the model offline from that folder via timm
`local-dir:`, prints the model facts and runs one random 224 px tile (cuda, else mps, else cpu). Output of
the 2026-11-04 run:

    config.json: vit_giant_patch14_reg4_dinov2, num_features 1536, global_pool 'token', num_classes 0, input_size [3, 224, 224]
    model: VisionTransformer, patch_size (14, 14), embed_dim 1536, depth 40
    prefix tokens: class token True + 4 registers = num_prefix_tokens 5
    embedding: global_pool 'token', attn_pool None, fc_norm Identity(), head Identity()
    parameters: 1,134,774,272
    sanity check OK: output (1, 1536) torch.float32 on mps:0, all finite

The embedding is the class token after the final norm; the 4 register tokens are not used. Input: 224×224
tiles at 0.5 µm/px, `ToTensor()` then `Normalize(mean=(0.707223, 0.578729, 0.703617),
std=(0.211883, 0.230117, 0.177517))`.

### 3. UNI2-h and its TCGA-BRCA tile features (main embeddings)

Request access to both https://huggingface.co/MahmoodLab/UNI2-h and
https://huggingface.co/datasets/MahmoodLab/UNI2-h-features (individual approval; needs an institutional
email as primary email on Hugging Face). Then:

    python download_uni2.py

Logs in to Hugging Face (prompts only if no token is stored) and fetches only what is missing:

- `models/uni2-h/pytorch_model.bin`: the UNI2-h weights (ViT-H/14, 681,394,176 parameters, 1 class + 8
  register tokens, 1536-dim class-token embedding). The architecture comes from the model card's timm
  kwargs; the checkpoint loads with `strict=True`. One random input then runs as a sanity check (cuda,
  else mps, else cpu). Output of the 2026-11-03 run:

      sanity check OK: output (1, 1536) torch.float32 on mps:0, all finite

- `data/uni2/TCGA/TCGA-BRCA_IDC.tar.gz` and `TCGA-BRCA_OTHERS.tar.gz` (60 GB), extracted to `data/uni2/h5/`
  (1,126 files, 66 GB). A `.done` marker per archive is written only after a complete extraction; once both
  exist, the archives can be deleted.

Each `.h5` is one slide, named after the GDC `file_name` (`.svs` → `.h5`): `features` (1, N, 1536) float32
and `coords` (1, N, 2), tile positions in level-0 pixels. Tiles are 256 px at 20×, i.e. 512
level-0 px on the 40× scans.

7 NST slides in `labels.csv` have no features, all from site OL and each its patient's only slide:
TCGA-OL-A5RU, -A5RV, -A5RW, -A5RX, -A5RY, -A5RZ, -A5S0. With features: ILC 200 slides / 187 patients,
NST 793 / 745.

On many slides the files hold no tiles in straight vertical strips, across tissue that the slide thumbnails show (step
9). Counting a slide as striped when one column boundary flips between tile and no tile along at least 30% of the
tissue's height, 554 of the 993 slides used are striped; the same measure along rows has a median of 16%. It follows the
site: 98-100% of slides at BH, C8, D8, E2, E9, EW, GM, LL and OL, at most 15% at the other nine sites with at least 10
slides. No two tiles of a slide share a position, so tiles are not misplaced. Full-resolution tiles from inside the strips of
one slide show tissue, so the tiles were left out when the files were made; why, and why by site, is not known (step
10). Every result below uses only the tiles in these files.

Both the weights and the features are CC-BY-NC-ND 4.0 and must not be redistributed, so they are
git-ignored and only re-created by this script.

### 4. External annotations

    python fetch_external.py

Standard library only, no login. Downloads into `data/external/`, skips files that are already present and
checks every file's md5 (raises on a mismatch):

| folder | source | licence | content |
|---|---|---|---|
| `kather2021/` | [Loeffler & Kather, Manual tumor annotations in TCGA](https://zenodo.org/records/5320076) (Zenodo) | CC BY 4.0 | `BRCA.zip` (8.3 MB): coarse manual tumour outlines (QuPath polygons, level-0 px), one CSV per TCGA-BRCA diagnostic slide; 1,133 files, 22 of them empty |
| `beetle/` | [BEETLE](https://zenodo.org/records/16812932) (Zenodo) | CC BY-NC-SA 4.0 | `data_overview.csv` and `annotations.zip` (1.8 GB): pixel masks (invasive epithelium, non-invasive epithelium, necrosis, other) for 587 slides, 164 of them from TCGA-BRCA |
| `thennavan2021/` | [Thennavan et al. 2021](https://doi.org/10.1016/j.xgen.2021.100067) (Europe PMC supplement) | CC BY-NC-ND 4.0 | `mmc2.xlsx` (Data S1): pathologist-reviewed histologic type per TCGA-BRCA sample; `mmc3.xlsx` (Data S2): 11 scored histologic features |

The Kather outlines select the tumour tiles (step 6). BEETLE and Thennavan are only for evaluating the model
and its explanations, never for training; the label stays the TCGA one from step 1. The publisher gives no
checksum for the Thennavan files, so their md5 is pinned from the 2026-11-04 download.

### 5. Train/val/test split

    python make_splits.py

Needs `labels.csv` (step 1) and the `.h5` files (step 3). Writes `data/meta/splits.csv`, one row per slide:
`patient, site, label, split, h5`.

- Each patient keeps the tiles of all their slides and carries one label: 932 patients, 993 slides. The 7 OL
  patients without features (step 3) are left out; the script raises if any other slide lacks its `.h5`.
- One split, no cross-validation: train, val (early stopping) and test. Each tissue source site goes whole into
  one of them, so val and test share no site with train. TCGA slides carry site-specific staining and scanning
  signatures that a model can learn instead of morphology (Howard et al. 2021, Nat Commun 12:4423), and here 11
  sites contribute only ILC and 8 only NST.
- Sites go greedily, largest first, to the split where they bring the ILC and NST patient counts closest to
  70% / 15% / 15% of each class (squared distance relative to each target, so ILC weighs as much as NST).
  Deterministic, no random seed.
- Site sizes make the shares coarse: BH alone has 131 patients (121 NST), more than the 112 NST that val or
  test should get. It lands in val, which is therefore larger than 15%; test hits its target from 3 sites.

Output of the 2026-11-05 run (patients per class, slides, sites per split):

    label  ILC  NST  slides  sites
    split
    train  122  499     680  4H 5L A1 A7 A8 AN AO AQ AR B6 C8 D8 E2 EW GI GM HN JL LD LL LQ MS OK OL PE PL S3 UL UU W8 WT XX Z7
    val     36  135     171  AC BH
    test    29  111     142  3C A2 E9

### 6. Baseline: mean pooling + logistic regression, all tiles vs tumour tiles

    python baseline.py

Needs `splits.csv` (step 5), the `.h5` files (step 3) and the Kather outlines (step 4); reads all 66 GB of
features once (about a minute on the original machine) and writes nothing.

- Tumour tiles: a tile is tumour if its centre lies inside any Kather outline of its slide. Outlines and `coords`
  are both level-0 px; the tile size in level-0 px is read per file (`coords.attrs["patch_size_level0"]`: 512 on
  the 1,066 40× scans, 256 on 51 at 20×, 768 on 8 at 60×, 1024 on 1 at 80×).
- 13 patients have no tumour tile and are dropped, with no fallback to all tiles: 12 whose slides all have an
  empty outline file, and TCGA-E2-A106, whose only outline lies at the slide edge outside the tiled tissue.
- Each patient becomes one vector, twice: the mean over all tiles of all their slides, and the mean over their
  tumour tiles only. Both runs use the same 919 patients.
- Standardisation + L2 logistic regression fit on train only (the data the tile-level model trains on), evaluated
  on val and test, ILC as the positive class. `C` stays at the scikit-learn default 1.0, untuned; balanced class
  weights, so the 0.5 threshold of the balanced accuracy does not lean towards NST. A fit that does not converge
  raises.

Output of the 2026-11-05 run:

    13 patients without tumour tiles dropped:
    label  ILC  NST
    split
    test     0    1
    train    3    9
    919 patients; 4,191,077 of 9,795,799 tiles are tumour; per patient median 44% (10th-90th percentile 14%-84%)
                  ILC  NST  auroc  balanced_acc
    tiles  split
    all    val     36  135  0.942         0.856
           test    29  110  0.895         0.852
    tumour val     36  135  0.977         0.902
           test    29  110  0.945         0.891

Pooling only the tumour tiles raises test AUROC from 0.895 to 0.945: the tissue around the tumour mostly dilutes
the signal. The outlines were checked against the tile positions (fill of the outlines by tiles at level-0 scale
vs 2× or 0.5×) and by eye on the slide figures of step 9.

### 7. Tile-level classifier

    python train_tiles.py

Needs `splits.csv` (step 5), the `.h5` files (step 3) and the Kather outlines (step 4). The first run writes
`data/uni2/tumour_tiles.h5` (25.8 GB, 76 s on the original machine, peak memory under 1.4 GB): for each of the 919
patients, the features, coords and slide of their tumour tiles, selected as in step 6. The cache derives from the
UNI2-h features, so it falls under their licence: git-ignored, never redistributed. Training then takes about 30 min
on `mps` (17 s per epoch) and saves `models/tile_classifier.pt`.

- Model: one linear layer on the raw 1536-dim embedding gives each tile a logit `t_i = w·x_i + b`; the patient logit
  is the mean of their `N` tile logits, so `t_i / N` is tile i's exact share of it. No MIL and no attention.
- Training: every tile inherits its patient's label. The BCE runs on the tile logits and is averaged over the
  patient's tiles, with one optimiser step per patient, so every patient weighs the same whatever their number of
  tiles. More tiles are not more labels: train has 2,866,388 tiles but 609 patients. ILC is weighted by the NST/ILC
  ratio on train (490/119), like the balanced class weights in step 6. AdamW, learning rate and weight decay 1e-4,
  all tumour tiles with no cap.
- Early stopping on the val loss of the patient logit: patience 10, at most 100 epochs. Test informs no choice.

Output of the 2026-11-07 run (epochs shortened):

    epoch 0: train loss 1.0000, val loss 0.8935, val AUROC 0.947, 19 s
    epoch 24: train loss 0.4742, val loss 0.4048, val AUROC 0.979, 18 s
    epoch 99: train loss 0.3645, val loss 0.3471, val AUROC 0.980, 16 s
    best epoch 91: val loss 0.3456
           ILC  NST  auroc  balanced_acc
    split
    val     36  135  0.980         0.919
    test    29  110  0.954         0.891

Early stopping never triggered: val AUROC stays at 0.979-0.980 from epoch 24 on, while the val loss keeps falling. An
earlier run capped at 512 tiles per patient, stopped at epoch 27, matched this run at that epoch (val AUROC 0.979 in
both, val loss 0.395 vs 0.398). Whether a rerun on `mps` reproduces these digits exactly has not been checked.

### 8. Confidence intervals on test

    python evaluate_tiles.py

Needs the cache and the checkpoint from step 7; takes about 40 s and writes nothing. Refits the tumour-tile baseline
from the cache (the same model as step 6, which saves no predictions), then resamples the test patients with
replacement 2,000 times, with no retraining. Both models get the same draws, which makes their difference paired.

Output of the 2026-11-07 run:

    139 test patients (29 ILC), 2000 draws
                             estimate    low   high
    tile-level AUROC            0.954  0.893  0.995
    tile-level balanced acc     0.891  0.820  0.958
    baseline AUROC              0.945  0.871  0.992
    baseline balanced acc       0.891  0.818  0.956
    AUROC difference            0.009 -0.012  0.031
    share of draws with AUROC difference <= 0: 0.204

The interval of the difference includes 0: on test the tile-level model is as good as the baseline, not measurably
better. The intervals only cover which test patients were drawn, not training randomness or other hospitals (test has
3 sites).

### 9. Looking at the tile logits

These scripts need the cache and the checkpoint from step 7 and train nothing. Images come one at a time from the GDC
tile API, with no login and no SVS download, and stay in the git-ignored `data/external/`; what is on disk is not fetched
again. GDC's speed varies widely by slide, and the endpoint is not on the GDC getting-started page, so it may change.

    python rank_tiles.py
    python fetch_tiles.py

`rank_tiles.py` prints six lists of val tiles: the most ILC-like, the most NST-like and those nearest 0, each for ILC and
for NST patients, with at most 3 tiles per patient and 10 per list. Ranking only picks tiles to look at; it does not show
they are right. `fetch_tiles.py` fetches those tiles at full resolution and writes one contact sheet per list to
`data/external/gdc_tiles/`. Only 40× tiles line up with GDC's 512 px grid, and the script raises on any other. No tile
has been re-embedded yet, so an exact match between a fetched tile and its stored embedding is not proven.

    python tile_logit_scale.py
    python tile_logit_scale_baseline.py
    python extreme_patient.py

All three use val tumour tiles and write nothing. Results of the 2026-11-07 runs:

- `tile_logit_scale.py`: tile logits behave like log-odds. With patients and classes weighted as in the loss, the share
  of ILC tiles in each logit bin stays within 0.09 of the sigmoid from -4 to 4.
- `tile_logit_scale_baseline.py` refits the step 6 baseline, which reads all train tumour tiles, and scores each val
  tile with it. Its tile scores also average to its patient logit, but they are not log-odds: from -10 to -5 the share
  of ILC is 0.29 where the sigmoid gives 0.001. Tile AUROC against the patient label is 0.930, against 0.890 for the
  tile-level model (no interval).
- `extreme_patient.py`: TCGA-BH-A0HK, labelled NST, holds 4,555 of the 4,692 NST tiles above 6 (patient logit 5.65,
  rank 2 of 171 val patients). The Thennavan review calls it lobular (step 10).

    python outline_test.py

Also needs the `.h5` files (step 3) and the Kather outlines (step 4), and writes nothing. It scores all val tiles, inside
and outside the outlines, although the model trained only on those inside, and groups them by distance from the nearest
outline. Result of the 2026-11-07 run, per patient, with bootstrap intervals:

| tiles | AUROC | gap between class medians |
|---|---|---|
| inside an outline | 0.980 | 5.19 |
| under 4 tile widths outside (about 0.5 mm) | 0.903 (0.849-0.951) | 1.62 |
| 4 to 16 tile widths | 0.724 (0.632-0.811) | 0.58 |
| over 16 tile widths (about 2 mm) | 0.641 (0.521-0.753) | 0.22 |

The logit is stronger inside the outlines in 92% of patients, and the signal falls with distance, so it is not simply
slide-wide. The weak signal far away is not explained: it could be tumour no outline covers, stain or site, or the tissue
around the tumour. Val holds only 2 sites. The distances in mm assume a nominal 0.5 µm per 20× px.

    python border_profile.py
    python beetle_far.py

`border_profile.py` bins the val tiles more finely by signed distance from the outline. The logit falls smoothly
across it, with no dip at the outline. Tiles over 16 tile widths outside lean ILC in both classes (median per patient
+0.60 for ILC, +0.37 for NST), so a positive tile outside is not class evidence.

`beetle_far.py` asks whether BEETLE (step 4) can tell which tiles outside the outlines are tumour; it reads no
features. BEETLE covers 143 of the slides used (train 7 ILC and 96 NST, val 6 and 14, test 2 and 18). On the 20 val
slides, its annotations touch 0.1% of tiles even inside the Kather outlines. Its points also reach only about a third
of the tiled extent (median 0.33 in x, 0.28 in y), so it probably uses another scale or covers small regions. This is
not resolved, and BEETLE is not used.

    python triptychs.py
    python heatmaps.py

Both draw one figure per slide: the slide, its tile logits, and the logits over the slide. All tiles are shown with the
outlines drawn, on one fixed colour scale of ±6, set on val. The model trained only on tiles inside the outlines, so
orange outside them is not ILC evidence. The maps stop at the tile, and the pattern inside a slide is not validated.
White vertical strips over tissue are tiles missing from the files (step 3), not low logits.

- `triptychs.py`: the first 10 test slides per class, by file order, in `data/external/gdc_triptychs/`. These are shown
  as examples and inform no choice. The original run took 53 min with thumbnails twice this size.
- `heatmaps.py`: four val patients chosen for what the Thennavan review or the tile ranking shows, so not a sample of how
  the model does: TCGA-BH-A18L, -A1FJ, -A0HK and -A0B5, one file each in `data/external/gdc_heatmaps/`.

    python thennavan_lookup.py
    python a0sy_patches.py
    python a0sy_tiles.py

`thennavan_lookup.py` prints the Thennavan review (step 4) of the four `heatmaps.py` patients, which their titles quote.

TCGA-A2-A0SY is a test triptych slide, labelled ILC, that is shown only and informs no choice. `a0sy_patches.py` finds
the connected groups of tiles above 3 outside its outlines; the four patches seen in its triptych hold 200, 89, 46 and
30 tiles. `a0sy_tiles.py` fetches five tiles from each patch, plus tiles inside the outline and far tiles near 0 for
comparison, into `data/external/gdc_tiles/a0sy_patches.png`. By eye, not a pathologist's, the patches show cells
through collagen like those inside the outline, so they may be tumour the outline missed.

### 10. Site, review and missing tiles

    python site_confound.py

Needs the cache and the checkpoint from step 7 and reads train and val only. The patient logit is linear in the
patient's mean feature, so the script works on patient means. The first run saves them to the git-ignored
`data/uni2/patient_means.npz`, which derives from UNI2-h. Result of the 2026-11-07 run:

- The features carry the site. A cross-validated logistic regression on the 13 train sites with at least 10 patients
  finds the site of 0.959 of patients, against 0.144 for always guessing the largest site, and 0.964 within NST alone.
  The site alone predicts the train label at AUROC 0.604.
- On val the ranking shows no site effect. Pairs of an ILC and an NST patient from one site give 0.976 (0.952-0.994),
  pairs from two sites 0.981, a difference of 0.005 (-0.010 to 0.021). The AUROC is 0.974 inside BH and 0.984 inside
  AC. Class explains 0.61 of the variance of the patient logit; site explains 0.014 within NST and 0.035 within ILC,
  within what shuffled sites give. Val has only 2 sites, so this says nothing about stain styles elsewhere.

    python thennavan_val.py

Needs the cache and the checkpoint from step 7. It compares the val patient logits with the Thennavan pathologist
review (step 4), which covers all 171 val patients. Result of the 2026-11-07 run:

- Of the 140 patients reviewed as purely lobular or ductal, the AUROC is 0.974 (0.935-0.998) against the review and
  0.988 (0.967-1.000) against the TCGA label. On 1 of the 5 patients where the two disagree, the model sides with the
  review (TCGA-BH-A0HK).
- 6 of the 11 val errors are reviewed as Mixed, against 15 of the 160 patients the model gets right.
- The median share of tiles above 0 is 0.92 for lobular, 0.41 for Mixed and 0.06 for ductal. The spread of tile logits
  inside a patient does not tell Mixed from pure (AUROC 0.477, 0.354-0.604), so the pattern inside a slide is not
  validated.
- Within ductal, the logit falls slightly with nuclear pleomorphism and with mitosis (Spearman -0.23 for each).

    python tile_gaps.py
    python strip_tiles.py

`tile_gaps.py` reads only tile positions and gives the strip counts of step 3. `strip_tiles.py` uses the slide
thumbnails (step 9) to estimate the share of visible tissue that has a tile: 0.62, 0.57, 0.58 and 0.40 on the four
`heatmaps.py` slides (all from BH), against 1.00 on TCGA-A2-A0SY. These are rough, because the colour threshold misses
pale tissue. It then fetches five full-resolution tiles from inside strips on TCGA-BH-A0HK, each next to the nearest tile
in the file, into `data/external/gdc_tiles/a0hk_strips.png`. All five show tissue.

### 11. Sparse autoencoder concepts

    python train_sae.py

Needs `splits.csv` (step 5), the `.h5` files (step 3), the Kather outlines (step 4) and the cache and checkpoint from
step 7, and reads no test slide. It trains a top-k sparse autoencoder (`sae.py`) on stored UNI2-h embeddings and saves it
to `models/sae_k32_capped.pt`. The SAE derives from UNI2-h, so it falls under that licence: git-ignored, never
redistributed.

- Data: from each train patient, up to 1,000 tiles inside the outlines and 1,000 outside for ILC and 250 each for NST, so
  the classes come out about balanced; from each val patient, up to 500 inside and 500 outside. Seed 0.
- Model: 1536 → 4096 concepts, of which the 32 largest per tile are kept; the decoder columns have unit norm. Trained on
  the MSE (AdamW, learning rate 1e-3, batches of 4,096) for at most 30 epochs, stopping 2 epochs after the best val MSE
  and keeping that epoch.
- It then prints, for the val tiles inside and outside the outlines, the MSE and the median cosine similarity, and raises
  if a kept concept is negative.

```
python sae_concepts.py
python concept_tiles.py
```

Both load the SAE and the step 7 checkpoint and read no test slide. `sae_concepts.py` splits each tile logit into
concepts, `t = b + Σ z_k (w·d_k) + w·e`, with `d_k` the decoder column of concept k and `e` the reconstruction error, and
raises if the split is off by more than 0.01. Over all val tumour tiles it ranks the concepts by their share of
|contributions| and prints the top 20, each with its side (the sign of `w·d_k`) and the 3 tiles it fires on most, and
writes those 60 tiles to `data/external/gdc_tiles/sae_top_tiles.csv`. `concept_tiles.py` fetches them from the GDC tile
API (step 9) into one contact sheet, `data/external/gdc_tiles/sae_top_concepts.png`, one row per concept, framed orange for the ILC side
and blue for NST.

### 12. CONCH labels for the SAE concepts

Request access at https://huggingface.co/MahmoodLab/conch (gated, like UNI2-h in step 3; the token stored there is
used). Then:

    python conch_labels.py

Needs the cache and the checkpoint from step 7 and the SAE from step 11. It takes the same 20 concepts and 60 tiles as
`concept_tiles.py`, fetching any not yet on disk, and labels each concept zero-shot with CONCH from the average of its
tiles' embeddings. Terms compete within four groups (tissue, architecture, nuclei, cytoplasm); a concept is `unresolved`
in a group when the runner-up is within 0.02. Writes `data/external/gdc_tiles/conch_concept_labels.csv` and the concept
contact sheet with the labels under each row, `sae_top_concepts_conch.png`. Reads no test slide. The CONCH weights go to
the Hugging Face cache on the first run and are never redistributed.

No SVS slides are downloaded (all of them would be 1.16 TB).
