# Pipeline Documentation

## Overview

Trismegisto processes CT DICOM series through two sequential stages, each running in its own Python environment.

---

## Stage 1 — Segmentation & Morphological Features (Python 3.12)

**Script:** `scripts/features_extraction.py`

### Steps

1. **DICOM loading** — reads all `.dcm` files from the input directory, skips non-DICOM files.
2. **Slice sorting** — sorts slices by `ImagePositionPatient[2]` (Z-coordinate) for correct 3D reconstruction.
3. **Voxel spacing calculation** — derives physical pixel size in XY from `PixelSpacing` and FOV, and Z-spacing from inter-slice distance or `SliceThickness`.
4. **TotalSegmentator** — runs aorta segmentation on the DICOM series; outputs `aorta.nii.gz`.
5. **Mask post-processing** — removes inlet/outlet artifact slices and isolated iliac components from the top of the mask using connected-component analysis.
6. **Region splitting** — divides the aorta mask into three anatomical regions:
   - **Ascending arch** — upper portion, anterior half (Y < midpoint)
   - **Descending arch** — upper portion, posterior half (Y ≥ midpoint)
   - **Descending aorta** — lower portion (below the horizontal split at ~55% of aorta height)
7. **Volume calculation** — counts labelled voxels per region, converts to cm³.
8. **Surface area** — uses `skimage.measure.marching_cubes` on the full aorta mask.
9. **Inlet/outlet cross-sections** — extracts the first/last slice of each region and computes area in mm².
10. **Cross-sectional morphology** — per-slice major/minor diameters and eccentricity using `skimage.measure.regionprops`.
11. **CSV export** — all features saved to `data/features/<condition>/<study_name>_<condition>.csv`.

---

## Stage 2 — Radiomic Features (Python 3.9)

**Script:** `scripts/pyradiomics_extraction.py`

### Steps

1. **DICOM loading & HU conversion** — loads slices, applies `RescaleSlope`/`RescaleIntercept`, clips to `[-50, 100]` HU (aorta wall window), normalises to `[0, 1]`.
2. **Resizing** — resizes each slice to 512×512 with bilinear interpolation.
3. **Volume stacking** — builds a 3D numpy array and converts to a SimpleITK image.
4. **Spacing/origin transfer** — copies physical metadata from the NIfTI mask so image and mask are spatially aligned.
5. **PyRadiomics extraction** — runs the default feature extractor (first-order, shape, GLCM, GLRLM, GLDM, NGTDM).
6. **CSV export** — features saved to `data/features/<condition>/<study_name>_pyradiomics.csv`.

---

## Configurable Parameters

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `slices_iniciales_a_borrar` | `features_extraction.py` | 10 | Inlet slices to zero out |
| `slices_superiores_a_modificar` | `features_extraction.py` | 10 | Top slices to clean iliac remnants |
| `device` | `features_extraction.py` | `"cpu"` | TotalSegmentator device (`"cpu"` or `"gpu"`) |
| `fast` | `features_extraction.py` | `False` | TotalSegmentator fast mode |
| HU window | `pyradiomics_extraction.py` | `[-50, 100]` | Clip range for aorta wall |
| `target_resolution` | `features_extraction.py` | 512 | Resize target for voxel spacing calc |
