"""
features_extraction.py

Port of features_extraction.ipynb to a runnable script with parallelism where it
makes sense:

  Stage              | Parallelizable? | How
  -------------------|-----------------|----------------------------------------
  index              | trivially fast  | sequential
  convert DICOM->NII | YES             | ProcessPoolExecutor
  segment (TotSeg)   | usually NO*     | sequential (override with --ts-workers)
  index-masks        | trivially fast  | sequential
  features (geom.)   | YES             | ProcessPoolExecutor

  * TotalSegmentator already uses all CPU/GPU resources per call and loads ~GB of
    model weights into RAM. Running >1 in parallel typically OOMs. Only raise
    --ts-workers above 1 if you know you have the memory/cores for it.

Run `python features_extraction.py --help` for options. By default it executes
every stage end-to-end exactly like the notebook does.
"""

import argparse
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import dicom2nifti
import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage import measure
from totalsegmentator.python_api import totalsegmentator
from tqdm import tqdm
from reorient_nii import reorient


CONDITIONS = ["ADX", "IMH", "PAU", "NC"]


# ---------------------------------------------------------------------------
# Stage 1 & 4: build dataset index from a directory tree
# ---------------------------------------------------------------------------

def build_index(root: Path, output_csv: str) -> pd.DataFrame:
    """Scan root/<condition>/<study> and persist a CSV index."""
    studies = []
    for condition in CONDITIONS:
        route = root / condition
        print(route)
        if route.exists():
            for path in route.iterdir():
                studies.append(
                    {"Nombre_Estudio": path.name, "Etiqueta": condition}
                )
    df = pd.DataFrame(studies)
    df.to_csv(output_csv, index=False)
    print(f"[index] {len(df)} estudios -> {output_csv}")
    return df


# ---------------------------------------------------------------------------
# Stage 2: DICOM -> NIfTI (parallel)
# ---------------------------------------------------------------------------

def _convert_one(task):
    """Worker: convert one DICOM series to a .nii.gz file."""
    condition, file_name, dicom_root, out_root = task
    input_path = Path(dicom_root) / condition / file_name
    output_path = Path(out_root) / condition / f"{file_name}.nii.gz"
    if output_path.is_file():
        return file_name, "exists", None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        dicom2nifti.dicom_series_to_nifti(
            str(input_path), str(output_path), reorient_nifti=True
        )
        return file_name, "ok", None
    except Exception as e:  # pylint: disable=broad-except
        return file_name, "error", str(e)


def convert_dicoms(dataset: pd.DataFrame, dicom_root: str, out_root: str,
                   workers: int) -> None:
    tasks = [
        (row["Etiqueta"], row["Nombre_Estudio"], dicom_root, out_root)
        for _, row in dataset.iterrows()
    ]
    if not tasks:
        print("[convert] nada que convertir.")
        return

    print(f"[convert] {len(tasks)} estudios, {workers} workers")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_convert_one, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="DICOM->NIfTI"):
            fname, status, err = fut.result()
            if status == "error":
                print(f"  [err] {fname}: {err}")


# ---------------------------------------------------------------------------
# Stage 3: TotalSegmentator (sequential by default, see module docstring)
# ---------------------------------------------------------------------------

def _segment_one(task):
    condition, file_name, nifti_root, mask_root, device, fast = task
    input_path = f"{nifti_root}/{condition}/{file_name}.nii.gz"
    output_dir = f"{mask_root}/{condition}/{file_name}/"
    mask_route = f"{mask_root}/{condition}/{file_name}/aorta.nii.gz"
    
    if Path(mask_route).is_file():
        print(mask_route)
        return file_name, "exists", None
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        totalsegmentator(
            input=input_path,
            output=output_dir,
            roi_subset=["aorta"],
            fast=fast,
            device=device,
        )
        return file_name, "ok", None
    except Exception as e:  # pylint: disable=broad-except
        return file_name, "error", str(e)


def segment_aorta(dataset: pd.DataFrame, nifti_root: str, mask_root: str,
                  device: str, fast: bool, workers: int) -> None:
    tasks = [
        (row["Etiqueta"], row["Nombre_Estudio"], nifti_root, mask_root,
         device, fast)
        for _, row in dataset.iterrows()
    ]
    if not tasks:
        print("[segment] nada que segmentar.")
        return
    
    print(f"[segment] Iniciando segmentaciones ({len(tasks)} estudios, "
          f"device={device}, workers={workers})")
    
    if workers <= 1:
        # Sequential -- safest. Matches the original notebook behaviour.
        for t in tqdm(tasks, desc="TotalSegmentator"):
            fname, status, err = _segment_one(t)
            if status == "error":
                print(f"  [err] {fname}: {err}")
    else:
        # Only if the user explicitly asked for it. Beware of memory.
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_segment_one, t): t for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc="TotalSegmentator"):
                fname, status, err = fut.result()
                if status == "error":
                    print(f"  [err] {fname}: {err}")
    print("[segment] Segmentaciones completadas")


# ---------------------------------------------------------------------------
# Stage 5: geometric feature extraction (parallel)
# ---------------------------------------------------------------------------
# These helpers are module-level so they pickle cleanly for ProcessPoolExecutor.

def _calcular_volumen(image, voxel_dims, valor_etiqueta=1):
    mask_array = sitk.GetArrayFromImage(image)
    vol_voxel_mm3 = voxel_dims[0] * voxel_dims[1] * voxel_dims[2]
    num_voxeles = np.sum(mask_array == valor_etiqueta)
    vol_total_mm3 = num_voxeles * vol_voxel_mm3
    return vol_total_mm3 / 1000.0


def _calculate_surface_area(image, voxel_size):
    """Área de superficie de una máscara 3D en mm^2 vía marching cubes."""
    mask_array = sitk.GetArrayFromImage(image)
    verts, faces, _, _ = measure.marching_cubes(
        mask_array, level=0.5, spacing=tuple(voxel_size)
    )
    return measure.mesh_surface_area(verts, faces)


def _calcular_areas_salida(image, nombre_estructura, voxel_dims,
                           tipo_cara="primera", valor_etiqueta=1):
    mask_array = sitk.GetArrayFromImage(image)
    coordenadas = np.where(mask_array == valor_etiqueta)
    if len(coordenadas[0]) == 0:
        print(f" La máscara de {nombre_estructura} está completamente vacía.")
        return None
    indices_z = coordenadas[2]

    if tipo_cara == "primera":
        corte_objetivo = np.min(indices_z)
    elif tipo_cara == "ultima":
        corte_objetivo = np.max(indices_z)
    else:
        return None

    corte_2d = mask_array[:, :, corte_objetivo]
    num_voxeles_cara = int(np.sum(corte_2d == valor_etiqueta))
    area_un_pixel_mm2 = voxel_dims[0] * voxel_dims[1]
    return num_voxeles_cara * area_un_pixel_mm2


def _calcular_metricas_transversales(image, nombre_estructura, voxel_dims,
                                     valor_etiqueta=1):
    """Diámetro mayor/menor y excentricidad slice por slice."""
    mask_array = sitk.GetArrayFromImage(image)
    total_slices = mask_array.shape[0]

    diametros_mayores, diametros_menores, excentricidades = [], [], []
    pixel_spacing_mm = voxel_dims[0]

    for z in range(total_slices):
        slice_2d = mask_array[z, :, :]
        if not np.any(slice_2d == valor_etiqueta):
            continue
        labels = measure.label(slice_2d == valor_etiqueta)
        props = measure.regionprops(labels)
        if not props:
            continue
        prop_aorta = max(props, key=lambda item: item.area)
        diametros_mayores.append(prop_aorta.axis_major_length * pixel_spacing_mm)
        diametros_menores.append(prop_aorta.axis_minor_length * pixel_spacing_mm)
        excentricidades.append(prop_aorta.eccentricity)

    if not diametros_mayores:
        print(f" La máscara de {nombre_estructura} está completamente vacía.")
        return None

    return (
        np.max(excentricidades), np.min(excentricidades),
        np.mean(excentricidades), np.std(excentricidades),
        np.max(diametros_mayores), np.min(diametros_mayores),
        np.mean(diametros_mayores), np.std(diametros_mayores),
        np.max(diametros_menores), np.min(diametros_menores),
        np.mean(diametros_menores), np.std(diametros_menores),
    )


def _segmentar_regiones_aorta(mask_sitk):
    """Divide la aorta en descenso / arco ascendente / arco descendente."""
    mask_3d = sitk.GetArrayFromImage(mask_sitk)

    z_activos = np.where(np.any(mask_3d, axis=(1, 2)))[0]
    if len(z_activos) == 0:
        print("La máscara está vacía.")
        return None, None, None

    z_min, z_max = z_activos[0], z_activos[-1]
    altura_real = z_max - z_min
    umbral_55 = z_min + int(altura_real * 0.55)

    # Corte horizontal en Z: límite arcos/descenso
    z_corte_horizontal = umbral_55
    for z in range(umbral_55, z_max + 1):
        slice_actual = mask_3d[z, :, :].astype(np.uint8)
        if np.any(slice_actual):
            num_labels, _, _, _ = cv2.connectedComponentsWithStats(
                slice_actual, connectivity=8
            )
            if num_labels == 3:
                z_corte_horizontal = z
                break

    # Corte vertical en Y: límite ascendente/descendente
    y_activos = np.where(np.any(mask_3d, axis=(0, 2)))[0]
    y_min, y_max = y_activos[0], y_activos[-1]
    y_corte_vertical = (y_min + y_max) // 2

    # Tres máscaras
    mask_descenso = np.zeros_like(mask_3d)
    mask_arco_asc = np.zeros_like(mask_3d)
    mask_arco_desc = np.zeros_like(mask_3d)

    mask_descenso[:z_corte_horizontal, :, :] = mask_3d[:z_corte_horizontal, :, :]
    bloque_superior = mask_3d[z_corte_horizontal:, :, :]
    bloque_asc = np.copy(bloque_superior)
    bloque_desc = np.copy(bloque_superior)
    bloque_asc[:, y_corte_vertical:, :] = 0
    bloque_desc[:, :y_corte_vertical, :] = 0
    mask_arco_asc[z_corte_horizontal:, :, :] = bloque_asc
    mask_arco_desc[z_corte_horizontal:, :, :] = bloque_desc

    out_desc = sitk.GetImageFromArray(mask_descenso)
    out_desc.CopyInformation(mask_sitk)
    out_asc = sitk.GetImageFromArray(mask_arco_asc)
    out_asc.CopyInformation(mask_sitk)
    out_arco_desc = sitk.GetImageFromArray(mask_arco_desc)
    out_arco_desc.CopyInformation(mask_sitk)
    return out_desc, out_asc, out_arco_desc


def _features_extractor(path, file_name, pathology):
    """Devuelve un DataFrame de una sola fila con todas las features."""
    condition = pathology

    image = nib.load(path)
    header = image.header
    voxel_dims = image.header.get_zooms()
    image = reorient(image, 'SPR')
    data_array = image.get_fdata()

    # Corte de cara de salida y entrada
    mask = sitk.GetImageFromArray(data_array)
    mask_limpia_3d = sitk.GetArrayFromImage(mask)
    
    mask_limpia_3d = np.copy(mask_limpia_3d)
    total_slices = mask_limpia_3d.shape[0]

    slices_offset_iliacas = 10
    z_activos = np.where(np.any(mask_limpia_3d, axis=(1, 2)))[0]
    if len(z_activos) == 0:
        print("La máscara está vacía.")

    z_min = z_activos[0]
    z_max = z_activos[-1]
    altura_real = z_max - z_min
    umbral_55 = z_min + int(altura_real * 0.55)

    # FASE 1: Limpieza de Ilíacas (Barrido Inverso)
    z_corte_final_inferior = z_min
    for z in range(umbral_55, z_min - 1, -1):
        slice_actual = mask_limpia_3d[z, :, :].astype(np.uint8)
        if np.any(slice_actual):
            num_labels, _, _, _ = cv2.connectedComponentsWithStats(
                slice_actual, connectivity=8
            )
            if num_labels >= 3:
                z_corte_final_inferior = z + slices_offset_iliacas
                break

    if z_corte_final_inferior > total_slices:
        z_corte_final_inferior = total_slices
    mask_limpia_3d[0 : z_corte_final_inferior + 1, :, :] = 0

    mask_limpia_sitk = sitk.GetImageFromArray(mask_limpia_3d)
    mask_limpia_sitk.CopyInformation(mask)

    mask_descenso, mask_arco_asc, mask_arco_desc = _segmentar_regiones_aorta(
        mask_limpia_sitk
    )

    vol_opt = _calcular_volumen(mask_limpia_sitk, voxel_dims)
    vol_asc = _calcular_volumen(mask_arco_asc, voxel_dims)
    vol_des = _calcular_volumen(mask_arco_desc, voxel_dims)
    vol_dsc = _calcular_volumen(mask_descenso, voxel_dims)

    area_mm2 = _calculate_surface_area(mask_limpia_sitk, voxel_dims)

    asc_area_total_mm2 = _calcular_areas_salida(
        mask_arco_asc, "Arco Ascendente", voxel_dims, tipo_cara="primera"
    )
    desc_area_total_mm2 = _calcular_areas_salida(
        mask_descenso, "Descenso", voxel_dims, tipo_cara="primera"
    )

    (excent_max_desc, excent_min_desc, excent_prom_desc, excent_std_desc,
     major_diam_max_desc, major_diam_min_desc, major_diam_prom_desc,
     major_diam_std_desc, minor_diam_max_desc, minor_diam_min_desc,
     minor_diam_prom_desc, minor_diam_std_desc) = _calcular_metricas_transversales(
        mask_descenso, "Descenso de la Aorta", voxel_dims
    )

    return pd.DataFrame(
        {
            "Study Name": file_name,
            "Condition": condition,
            "Volumen_Total_cm3": vol_opt,
            "Volumen_Ascendente_cm3": vol_asc,
            "Volumen_ArcoDescendente_cm3": vol_des,
            "Volumen_Descenso_cm3": vol_dsc,
            "Area_Superficie_Total_mm2": area_mm2,
            "Area_Salida_Ascendente_mm2": asc_area_total_mm2,
            "Area_Salida_Descenso_mm2": desc_area_total_mm2,
            "Excentricidad_Max_Desc": excent_max_desc,
            "Excentricidad_Min_Desc": excent_min_desc,
            "Excentricidad_Promedio_Desc": excent_prom_desc,
            "Excentricidad_Std_Desc": excent_std_desc,
            "Diametro_Mayor_Max_Desc_mm": major_diam_max_desc,
            "Diametro_Mayor_Min_Desc_mm": major_diam_min_desc,
            "Diametro_Mayor_Promedio_Desc_mm": major_diam_prom_desc,
            "Diametro_Mayor_Std_Desc_mm": major_diam_std_desc,
            "Diametro_Menor_Max_Desc_mm": minor_diam_max_desc,
            "Diametro_Menor_Min_Desc_mm": minor_diam_min_desc,
            "Diametro_Menor_Promedio_Desc_mm": minor_diam_prom_desc,
            "Diametro_Menor_Std_Desc_mm": minor_diam_std_desc,
        },
        index=[0],
    )


def _extract_one_features(task):
    condition, file_name, mask_root = task
    input_path = Path(mask_root) / condition / file_name / "aorta.nii.gz"
    try:
        return file_name, _features_extractor(str(input_path), file_name, condition), None
    except Exception as e:  # pylint: disable=broad-except
        return file_name, None, f"{e}\n{traceback.format_exc()}"


def extract_features(dataset: pd.DataFrame, mask_root: str, output_csv: str,
                     workers: int) -> None:
    tasks = [
        (row["Etiqueta"], row["Nombre_Estudio"], mask_root)
        for _, row in dataset.iterrows()
    ]
    if not tasks:
        print("[features] nada que extraer.")
        return

    print(f"[features] {len(tasks)} estudios, {workers} workers")
    all_dfs = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_extract_one_features, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Features"):
            file_name, df, err = fut.result()
            if df is not None:
                all_dfs.append(df)
            else:
                print(f"  [err] {file_name}: {err}")

    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        full_df.to_csv(output_csv, index=False)
        print(f"[features] {len(full_df)} filas -> {output_csv}")
    else:
        print("[features] no se extrajo ningún feature; CSV no generado.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    default_workers = min(8, max(1, (os.cpu_count() or 2) // 2))

    parser = argparse.ArgumentParser(
        description="Pipeline de features (DICOM -> NIfTI -> Aorta -> Features)."
    )
    parser.add_argument(
        "--stage",
        choices=["all", "index", "convert", "segment", "index-masks", "features"],
        default="all",
        help="Qué etapa(s) ejecutar (default: all = todo el pipeline).",
    )
    parser.add_argument("--dicom-root", default="/home/research/Projects/Trismegisto/Trismegisto/data/CT_dicom",
                        help="Carpeta con DICOMs (por condición).")
    parser.add_argument("--nifti-root", default="/home/research/Projects/Trismegisto/Trismegisto/data/CT_dataset",
                        help="Salida de NIfTI convertidos.")
    parser.add_argument("--mask-root", default="/home/research/Projects/Trismegisto/Trismegisto/data/CT_mask",
                        help="Salida de máscaras de TotalSegmentator.")
    parser.add_argument("--index-csv", default="dataset_index.csv",
                        help="CSV de índice de estudios (DICOM).")
    parser.add_argument("--index-masks-csv", default="dataset_index_all.csv",
                        help="CSV de índice de máscaras.")
    parser.add_argument("--features-csv", default="dataset_features.csv",
                        help="CSV final con features.")
    parser.add_argument("--workers", type=int, default=default_workers,
                        help="Workers para convert / features (default: "
                             f"{default_workers}).")
    parser.add_argument("--ts-workers", type=int, default=1,
                        help="Workers paralelos para TotalSegmentator. "
                             "Default 1 (recomendado).")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"],
                        help="Device para TotalSegmentator.")
    parser.add_argument("--fast", action="store_true",
                        help="Modo rápido de TotalSegmentator (menos preciso).")
    args = parser.parse_args()

    stage = args.stage

    # --- index ---
    if stage in ("all", "index"):
        dataset = build_index(Path(args.dicom_root), args.index_csv)
    else:
        dataset = pd.read_csv(args.index_csv) if Path(args.index_csv).is_file() else None

    # --- convert ---
    if stage in ("all", "convert"):
        if dataset is None:
            sys.exit(f"No existe {args.index_csv}; corre primero --stage index.")
        convert_dicoms(dataset, args.dicom_root, args.nifti_root, args.workers)

    # --- segment ---
    if stage in ("all", "segment"):
        if dataset is None:
            sys.exit(f"No existe {args.index_csv}; corre primero --stage index.")
        segment_aorta(dataset, args.nifti_root, args.mask_root,
                      device=args.device, fast=args.fast,
                      workers=args.ts_workers)

    # --- index-masks ---
    if stage in ("all", "index-masks"):
        dataset_all = build_index(Path(args.mask_root), args.index_masks_csv)
    else:
        dataset_all = (pd.read_csv(args.index_masks_csv)
                       if Path(args.index_masks_csv).is_file() else None)

    # --- features ---
    if stage in ("all", "features"):
        if dataset_all is None:
            sys.exit(f"No existe {args.index_masks_csv}; "
                     f"corre primero --stage index-masks.")
        extract_features(dataset_all, args.mask_root, args.features_csv,
                         args.workers)


if __name__ == "__main__":
    main()
