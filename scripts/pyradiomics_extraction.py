import cv2
import numpy as np
from pydicom import dcmread
from pydicom.errors import InvalidDicomError
import os
import SimpleITK as sitk
import pandas as pd
from radiomics import featureextractor

# =============================================================================
# HELPER FUNCTIONS
# (defined at module level so they remain picklable if multiprocessing is used)
# =============================================================================

def get_hu_image(dataset):
    """Converts raw DICOM pixel data to Hounsfield Units."""
    image     = dataset.pixel_array.astype(np.int16)
    intercept = dataset.RescaleIntercept
    slope     = dataset.RescaleSlope

    if slope != 1:
        image = (slope * image.astype(np.float64)).astype(np.int16)

    image += np.int16(intercept)
    return np.array(image, dtype=np.int16)


def process_slice(image):
    """Clips to aorta HU window and normalises to [0, 1]."""
    min_hu, max_hu = -50, 100
    image_clipped  = np.clip(image, min_hu, max_hu)
    image_norm     = (image_clipped - min_hu) / (max_hu - min_hu)
    return image_norm


# =============================================================================
# MAIN
# Must be inside this guard so multiprocessing spawn workers can safely
# import this module without re-executing the pipeline.
# =============================================================================

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    name      = "Mario_D'Oria_IMH_ITA_GE_1.250/Nuova cartella (2)"
    condition = "IMH"

    base_dir  = "/home/research/Projects/Trismegisto/Trismegisto/data"
    dicom_path = os.path.join(base_dir, "CT_dicom",   condition, name)
    mask_path  = os.path.join(base_dir, "CT_mask",    condition, name, "aorta.nii.gz")
    out_dir    = os.path.join(base_dir, "features",   condition)
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Load DICOM slices
    # -------------------------------------------------------------------------
    dicom_slices = []
    for filename in os.listdir(dicom_path):
        filepath = os.path.join(dicom_path, filename)
        if os.path.isdir(filepath):
            continue
        try:
            dataset = dcmread(filepath)
            if hasattr(dataset, 'pixel_array'):
                dicom_slices.append(dataset)
        except InvalidDicomError:
            continue

    if not dicom_slices:
        raise RuntimeError(f"No se encontraron archivos DICOM válidos en: {dicom_path}")

    # -------------------------------------------------------------------------
    # 2. Sort slices by Z-coordinate (superior → inferior)
    # -------------------------------------------------------------------------
    dicom_slices.sort(key=lambda x: x.ImagePositionPatient[2], reverse=True)

    # -------------------------------------------------------------------------
    # 3. HU conversion, windowing, resize
    # -------------------------------------------------------------------------
    hu_slices = []
    for dataset in dicom_slices:
        hu_slice      = get_hu_image(dataset)
        hu_slice      = process_slice(hu_slice)
        resized_slice = cv2.resize(hu_slice, (512, 512), interpolation=cv2.INTER_LINEAR)
        hu_slices.append(resized_slice)

    # -------------------------------------------------------------------------
    # 4. Stack into 3D volume
    # -------------------------------------------------------------------------
    hu_volume = np.stack(hu_slices, axis=0)
    hu_volume = hu_volume[::-1]   # Invert along the first axis

    print(f"Successfully loaded and stacked {hu_volume.shape[0]} slices.")
    print(f"Volume shape: {hu_volume.shape}")

    # -------------------------------------------------------------------------
    # 5. Align image to mask metadata
    # -------------------------------------------------------------------------
    mask  = sitk.ReadImage(mask_path)
    image = sitk.GetImageFromArray(hu_volume)
    image.SetSpacing(mask.GetSpacing())
    image.SetOrigin(mask.GetOrigin())
    image.SetDirection(mask.GetDirection())   # copy direction cosines to avoid geometry mismatch

    # -------------------------------------------------------------------------
    # 6. Extract radiomic features
    # -------------------------------------------------------------------------
    extractor = featureextractor.RadiomicsFeatureExtractor()
    features  = extractor.execute(image, mask)

    # -------------------------------------------------------------------------
    # 7. Save CSV
    # -------------------------------------------------------------------------
    df      = pd.DataFrame([features])
    df_path = os.path.join(out_dir, f"{name}_pyradiomics.csv")
    df.to_csv(df_path, index=False)
    print(f"Features saved to {df_path}")
