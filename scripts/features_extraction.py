from totalsegmentator.python_api import totalsegmentator
import SimpleITK as sitk
import numpy as np
import os
import cv2
from pydicom import dcmread
from pydicom.errors import InvalidDicomError
from skimage import measure
import pandas as pd

# =============================================================================
# HELPER FUNCTIONS
# (defined at module level so they can be pickled by multiprocessing if needed)
# =============================================================================

def get_voxel_size(slices, target_resolution=512):
    """
    Calcula el tamaño del voxel [x, y, z] en mm a partir de una lista de cortes DICOM.

    Args:
        slices: Lista de objetos dataset de pydicom (ordenados).
        target_resolution: La resolución a la que se redimensionó la imagen (por defecto 512).
    """
    ds = slices[0]

    original_spacing_x = float(ds.PixelSpacing[0])
    original_spacing_y = float(ds.PixelSpacing[1])

    fov_x = original_spacing_x * ds.Columns
    fov_y = original_spacing_y * ds.Rows

    new_pixel_size_x = fov_x / target_resolution
    new_pixel_size_y = fov_y / target_resolution

    try:
        if len(slices) > 1:
            z_spacing = abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        else:
            z_spacing = float(ds.SliceThickness)
    except (AttributeError, IndexError, TypeError):
        try:
            z_spacing = float(ds.SliceThickness)
        except AttributeError:
            z_spacing = 1.0
            print("Advertencia: No se pudo determinar el espaciado Z. Usando valor por defecto 1.0")

    return [new_pixel_size_x, new_pixel_size_y, z_spacing]


def segmentar_regiones_aorta(mask_sitk):
    """
    Divide la máscara de la aorta en tres regiones anatómicas:
    - Descenso de la aorta
    - Arco ascendente
    - Arco descendente
    """
    mask_3d = sitk.GetArrayFromImage(mask_sitk)

    # --- CORTE HORIZONTAL (Eje Z): límite entre Arcos y Descenso ---
    z_activos = np.where(np.any(mask_3d, axis=(1, 2)))[0]
    if len(z_activos) == 0:
        print("La máscara está vacía.")
        return None, None, None

    z_min = z_activos[0]
    z_max = z_activos[-1]
    umbral_55 = z_min + int((z_max - z_min) * 0.55)

    z_corte_horizontal = umbral_55
    for z in range(umbral_55, z_max + 1):
        slice_actual = mask_3d[z, :, :].astype(np.uint8)
        if np.any(slice_actual):
            num_labels, _, _, _ = cv2.connectedComponentsWithStats(slice_actual, connectivity=8)
            if num_labels == 3:  # fondo + 2 cuerpos
                z_corte_horizontal = z
                break

    # --- CORTE VERTICAL (Eje Y): límite entre Arco Ascendente y Descendente ---
    y_activos = np.where(np.any(mask_3d, axis=(0, 2)))[0]
    y_corte_vertical = (y_activos[0] + y_activos[-1]) // 2

    # --- GENERACIÓN DE LAS 3 MÁSCARAS ---
    mask_descenso  = np.zeros_like(mask_3d)
    mask_arco_asc  = np.zeros_like(mask_3d)
    mask_arco_desc = np.zeros_like(mask_3d)

    mask_descenso[:z_corte_horizontal, :, :] = mask_3d[:z_corte_horizontal, :, :]

    bloque_superior = mask_3d[z_corte_horizontal:, :, :]
    bloque_asc  = np.copy(bloque_superior)
    bloque_desc = np.copy(bloque_superior)
    bloque_asc[:, y_corte_vertical:, :]  = 0
    bloque_desc[:, :y_corte_vertical, :] = 0

    mask_arco_asc[z_corte_horizontal:, :, :]  = bloque_asc
    mask_arco_desc[z_corte_horizontal:, :, :] = bloque_desc

    def to_sitk(arr):
        img = sitk.GetImageFromArray(arr)
        img.CopyInformation(mask_sitk)
        return img

    return to_sitk(mask_descenso), to_sitk(mask_arco_asc), to_sitk(mask_arco_desc)


def calcular_volumen(image, nombre_estructura, voxel_dims, valor_etiqueta=1):
    mask_array    = sitk.GetArrayFromImage(image)
    vol_voxel_mm3 = voxel_dims[0] * voxel_dims[1] * voxel_dims[2]
    num_voxeles   = np.sum(mask_array == valor_etiqueta)
    vol_total_cm3 = (num_voxeles * vol_voxel_mm3) / 1000.0

    print(f"--- RESULTADOS: {nombre_estructura.upper()} ---")
    print(f"Vóxeles totales: {num_voxeles}")
    print(f"Volumen total  : {vol_total_cm3:.2f} cm³")
    print("-" * 45 + "\n")
    return vol_total_cm3


def calculate_surface_area(image, voxel_size):
    """Calcula el área de superficie de una máscara 3D en mm²."""
    mask_array = sitk.GetArrayFromImage(image)
    verts, faces, _, _ = measure.marching_cubes(mask_array, level=0.5, spacing=tuple(voxel_size))
    return measure.mesh_surface_area(verts, faces)


def calcular_areas_salida(image, nombre_estructura, voxel_dims, tipo_cara='primera', valor_etiqueta=1):
    mask_array  = sitk.GetArrayFromImage(image)
    coordenadas = np.where(mask_array == valor_etiqueta)

    if len(coordenadas[0]) == 0:
        print(f" La máscara de {nombre_estructura} está completamente vacía.")
        return None

    indices_z = coordenadas[2]
    if tipo_cara == 'primera':
        corte_objetivo = np.min(indices_z)
        texto_cara     = "PRIMERA cara (Corte Z más bajo)"
    elif tipo_cara == 'ultima':
        corte_objetivo = np.max(indices_z)
        texto_cara     = "ÚLTIMA cara (Corte Z más alto)"
    else:
        return None

    corte_2d          = mask_array[:, :, corte_objetivo]
    num_voxeles_cara   = int(np.sum(corte_2d == valor_etiqueta))
    area_total_mm2     = num_voxeles_cara * voxel_dims[0] * voxel_dims[1]

    print(f"--- {nombre_estructura.upper()} ---")
    print(f"Cara analizada    : {texto_cara} (Índice Z: {corte_objetivo})")
    print(f"Resolución X, Y   : {voxel_dims[0]:.4f} mm x {voxel_dims[1]:.4f} mm por vóxel")
    print(f"Vóxeles contados  : {num_voxeles_cara} vóxeles")
    print(f"Área física real  : {area_total_mm2:.2f} mm²\n")
    return area_total_mm2


def calcular_metricas_transversales(image, nombre_estructura, voxel_dims, valor_etiqueta=1):
    """
    Calcula el diámetro mayor, menor y la excentricidad de la aorta slice por slice.
    Retorna una tupla con (max, min, promedio, std) para cada métrica.
    """
    mask_array       = sitk.GetArrayFromImage(image)
    pixel_spacing_mm = voxel_dims[0]

    diametros_mayores = []
    diametros_menores = []
    excentricidades   = []

    for z in range(mask_array.shape[0]):
        slice_2d = mask_array[z, :, :]
        if np.any(slice_2d == valor_etiqueta):
            labels = measure.label(slice_2d == valor_etiqueta)
            props  = measure.regionprops(labels)
            if props:
                prop_aorta = max(props, key=lambda item: item.area)
                diametros_mayores.append(prop_aorta.axis_major_length * pixel_spacing_mm)
                diametros_menores.append(prop_aorta.axis_minor_length * pixel_spacing_mm)
                excentricidades.append(prop_aorta.eccentricity)

    if not diametros_mayores:
        print(f" La máscara de {nombre_estructura} está completamente vacía.")
        return None

    def stats(arr):
        return np.max(arr), np.min(arr), np.mean(arr), np.std(arr)

    emax, emin, eprom, estd         = stats(excentricidades)
    dMmax, dMmin, dMprom, dMstd     = stats(diametros_mayores)
    dmmax, dmmin, dmprom, dmstd     = stats(diametros_menores)

    print(f"Slices analizados: {len(diametros_mayores)}")
    print(f"Diámetro Mayor   : Max {dMmax:.2f} mm | Min {dMmin:.2f} mm | Promedio {dMprom:.2f} mm ± {dMstd:.2f}")
    print(f"Diámetro Menor   : Max {dmmax:.2f} mm | Min {dmmin:.2f} mm | Promedio {dmprom:.2f} mm ± {dmstd:.2f}")
    print(f"Excentricidad    : Max {emax:.4f}    | Min {emin:.4f}    | Promedio {eprom:.4f}    ± {estd:.4f}")
    print("-" * 45 + "\n")

    return emax, emin, eprom, estd, dMmax, dMmin, dMprom, dMstd, dmmax, dmmin, dmprom, dmstd


# =============================================================================
# MAIN
# Must be inside this guard so multiprocessing spawn workers can safely
# import this module without re-executing the pipeline.
# =============================================================================

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    study_name  = "Mario_D'Oria_IMH_ITA_GE_1.250/Nuova cartella (2)"
    condition   = "IMH"
    dicom_path  = (
        f"/home/research/Projects/Trismegisto/Trismegisto/data/CT_dicom"
        f"/{condition}/{study_name}"
    )
    output_path = (
        f"/home/research/Projects/Trismegisto/Trismegisto/data/CT_mask"
        f"/{condition}/{study_name}"
    )
    os.makedirs(output_path, exist_ok=True)

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

    dicom_slices.sort(key=lambda x: x.ImagePositionPatient[2])
    voxel_dims = get_voxel_size(dicom_slices, target_resolution=512)
    print(f"Voxel dims (x, y, z): {voxel_dims[0]:.4f} mm, {voxel_dims[1]:.4f} mm, {voxel_dims[2]:.4f} mm")

    # -------------------------------------------------------------------------
    # 2. Aorta segmentation
    # -------------------------------------------------------------------------
    print("\nIniciando la segmentación de la aorta...")
    totalsegmentator(
        input=dicom_path,
        output=output_path,
        roi_subset=["aorta"],
        fast=False,
        device="cpu",
    )
    print("Segmentación completada\n")

    # -------------------------------------------------------------------------
    # 3. Load mask and post-process
    # -------------------------------------------------------------------------
    mask_path      = os.path.join(output_path, "aorta.nii.gz")
    mask           = sitk.ReadImage(mask_path)
    mask_limpia_3d = np.copy(sitk.GetArrayFromImage(mask))
    total_slices   = mask_limpia_3d.shape[0]

    slices_iniciales_a_borrar     = 10
    slices_superiores_a_modificar = 10

    # Remove inlet slices
    z_con_mascara = np.where(np.any(mask_limpia_3d, axis=(1, 2)))[0]
    for z in z_con_mascara[:slices_iniciales_a_borrar]:
        mask_limpia_3d[z, :, :] = 0

    # Clean top iliac remnants
    umbral_55          = int(total_slices * 0.55)
    slices_modificadas = 0

    for z in range(umbral_55, total_slices):
        if slices_modificadas >= slices_superiores_a_modificar:
            break
        slice_actual = mask_limpia_3d[z, :, :].astype(np.uint8)
        if np.any(slice_actual):
            num_labels, labels, _, centroids = cv2.connectedComponentsWithStats(slice_actual, connectivity=8)
            if num_labels >= 3:
                min_y, label_a_eliminar = float('inf'), -1
                for i in range(1, num_labels):
                    if centroids[i][1] < min_y:
                        min_y            = centroids[i][1]
                        label_a_eliminar = i
                if label_a_eliminar != -1:
                    slice_actual[labels == label_a_eliminar] = 0
                    mask_limpia_3d[z, :, :]  = slice_actual
                    slices_modificadas       += 1

    mask_limpia_sitk = sitk.GetImageFromArray(mask_limpia_3d)
    mask_limpia_sitk.CopyInformation(mask)

    # -------------------------------------------------------------------------
    # 4. Region segmentation
    # -------------------------------------------------------------------------
    mask_descenso, mask_arco_asc, mask_arco_desc = segmentar_regiones_aorta(mask_limpia_sitk)

    # -------------------------------------------------------------------------
    # 5. Volume
    # -------------------------------------------------------------------------
    print("Iniciando calculos de volumenes...\n" + "=" * 45 + "\n")
    vol_opt = calcular_volumen(mask_limpia_sitk, "Máscara Completa",     voxel_dims)
    vol_asc = calcular_volumen(mask_arco_asc,    "Arco Ascendente",      voxel_dims)
    vol_des = calcular_volumen(mask_arco_desc,   "Arco Descendente",     voxel_dims)
    vol_dsc = calcular_volumen(mask_descenso,    "Descenso de la Aorta", voxel_dims)

    # -------------------------------------------------------------------------
    # 6. Surface area
    # -------------------------------------------------------------------------
    print("Iniciando calculos de área superficial...\n" + "=" * 45 + "\n")
    area_mm2 = calculate_surface_area(mask_limpia_sitk, voxel_dims)
    print(f"Área de superficie de la aorta: {area_mm2:.2f} mm²")
    print(f"Área de superficie en cm²: {area_mm2 / 100:.2f} cm²\n")

    # -------------------------------------------------------------------------
    # 7. Inlet/outlet cross-section areas
    # -------------------------------------------------------------------------
    print("Iniciando calculos de área de salida/entrada...\n" + "=" * 45 + "\n")
    asc_area_total_mm2  = calcular_areas_salida(mask_arco_asc, "Arco Ascendente", voxel_dims, tipo_cara='primera')
    desc_area_total_mm2 = calcular_areas_salida(mask_descenso, "Descenso",        voxel_dims, tipo_cara='primera')

    # -------------------------------------------------------------------------
    # 8. Cross-sectional morphology
    # -------------------------------------------------------------------------
    print("Iniciando cálculos de morfología transversal...\n" + "=" * 45 + "\n")
    (excent_max_desc,    excent_min_desc,    excent_prom_desc,    excent_std_desc,
     major_diam_max_desc, major_diam_min_desc, major_diam_prom_desc, major_diam_std_desc,
     minor_diam_max_desc, minor_diam_min_desc, minor_diam_prom_desc, minor_diam_std_desc,
     ) = calcular_metricas_transversales(mask_descenso, "Descenso de la Aorta", voxel_dims)

    # -------------------------------------------------------------------------
    # 9. Save CSV
    # -------------------------------------------------------------------------
    print("Guardando resultados en CSV...\n" + "=" * 45)

    features_df = pd.DataFrame({
        "Study Name":                      [study_name],
        "Condition":                       [condition],
        "Volumen_Total_cm3":               [vol_opt],
        "Volumen_Ascendente_cm3":          [vol_asc],
        "Volumen_ArcoDescendente_cm3":     [vol_des],
        "Volumen_Descenso_cm3":            [vol_dsc],
        "Area_Superficie_Total_mm2":       [area_mm2],
        "Area_Salida_Ascendente_mm2":      [asc_area_total_mm2],
        "Area_Salida_Descenso_mm2":        [desc_area_total_mm2],
        "Excentricidad_Max_Desc":          [excent_max_desc],
        "Excentricidad_Min_Desc":          [excent_min_desc],
        "Excentricidad_Promedio_Desc":     [excent_prom_desc],
        "Excentricidad_Std_Desc":          [excent_std_desc],
        "Diametro_Mayor_Max_Desc_mm":      [major_diam_max_desc],
        "Diametro_Mayor_Min_Desc_mm":      [major_diam_min_desc],
        "Diametro_Mayor_Promedio_Desc_mm": [major_diam_prom_desc],
        "Diametro_Mayor_Std_Desc_mm":      [major_diam_std_desc],
        "Diametro_Menor_Max_Desc_mm":      [minor_diam_max_desc],
        "Diametro_Menor_Min_Desc_mm":      [minor_diam_min_desc],
        "Diametro_Menor_Promedio_Desc_mm": [minor_diam_prom_desc],
        "Diametro_Menor_Std_Desc_mm":      [minor_diam_std_desc],
    })

    features_dir = (
        f"/home/research/Projects/Trismegisto/Trismegisto/data/features/{condition}"
    )
    os.makedirs(features_dir, exist_ok=True)
    df_path = os.path.join(features_dir, f"{study_name}_{condition}.csv")
    features_df.to_csv(df_path, header=True, index=False)
    print(f"Resultados guardados en {df_path}")
