from totalsegmentator.python_api import totalsegmentator
import SimpleITK as sitk
import numpy as np
import os
import cv2
from pydicom import dcmread
import numpy as np
from skimage import measure
import pandas as pd
import numpy as np
import SimpleITK as sitk

#Recibir directorio

"""
En esta sección es necesario automatizar la dirección de la ruta del archivo DICOM,
para ello se deben recibir los archivos del código de Linux.
"""

dicom_path = "D:/Documentos/TEC/Reto/CT_dicoms/ADX/Mario_D'Oria_ADX_ITA_GE_0.625"
study_name = "Mario_D'Oria_ADX_ITA_GE_0.625"
condition = "ADX"

#Obtener dimensiones

input_path = dicom_path
# 2. Read all DICOM files from the folder
dicom_slices = []
for filename in os.listdir(input_path):
    filepath = os.path.join(input_path, filename)
    
    # Skip subdirectories if any exist
    if os.path.isdir(filepath):
        continue
        
    try:
        # Read the file
        dataset = dcmread(filepath)
        
        # Make sure the file actually contains image data before adding it
        if hasattr(dataset, 'pixel_array'):
            dicom_slices.append(dataset)
    except InvalidDicomError:
        # Skip files that aren't valid DICOMs (like hidden OS files, .txt files, etc.)
        continue

def get_voxel_size(slices, target_resolution=512):
    """
    Calcula el tamaño del voxel [x, y, z] en mm a partir de una lista de cortes DICOM.
    
    Args:
        slices: Lista de objetos dataset de pydicom (ordenados).
        target_resolution: La resolución a la que se redimensionó la imagen (por defecto 512).
    """
    # Usamos el primer corte para obtener la información de plano (X, Y)
    ds = slices[0]
    
    # 1. Cálculo de X e Y (ajustado por redimensión)
    original_spacing_x = float(ds.PixelSpacing[0])
    original_spacing_y = float(ds.PixelSpacing[1])
    
    fov_x = original_spacing_x * ds.Columns
    fov_y = original_spacing_y * ds.Rows
    
    new_pixel_size_x = fov_x / target_resolution
    new_pixel_size_y = fov_y / target_resolution

    # 2. Cálculo de Z (Grosor/Espaciado de corte)
    # Intentamos el método de diferencia de posición entre cortes (más preciso)
    try:
        if len(slices) > 1:
            # Calculamos la diferencia absoluta en el eje Z entre el primer y segundo corte
            z_spacing = abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        else:
            # Si solo hay un corte, no hay diferencia que calcular
            z_spacing = float(ds.SliceThickness)
            
    except (AttributeError, IndexError, TypeError):
        # Si fallan las coordenadas, intentamos leer el metadato SliceThickness
        try:
            z_spacing = float(ds.SliceThickness)
        except AttributeError:
            # Si nada de lo anterior existe, asignamos un valor por defecto (usualmente 1.0 o NaN)
            z_spacing = 1.0
            print("Advertencia: No se pudo determinar el espaciado Z. Usando valor por defecto 1.0")

    return [new_pixel_size_x, new_pixel_size_y, z_spacing]


voxel_dims = get_voxel_size(dicom_slices, target_resolution=512)

#Extraer máscara de la aorta

output_path = "/"

print("Iniciando la segmentación de la aorta...")

totalsegmentator(input=dicom_path, output=output_path, roi_subset=["aorta"], fast=False, device="cpu" )
print("Segementación completada")

#Corte de cara de salida y entrada

mask_path = "aorta.nii.gz"
mask = sitk.ReadImage(mask_path)
mask_limpia_3d = sitk.GetArrayFromImage(mask)

mask_limpia_3d = np.copy(mask_limpia_3d) 
total_slices = mask_limpia_3d.shape[0]

# --- VARIABLES CONFIGURABLES ---
slices_iniciales_a_borrar = 10     # Slices a borrar de la salida
slices_superiores_a_modificar = 10 # Slices a modificar de la entarda

# 1. Borrar las slices de la salida
z_con_mascara = np.where(np.any(mask_limpia_3d, axis=(1, 2)))[0]

indices_iniciales_a_borrar = z_con_mascara[:slices_iniciales_a_borrar]

for z in indices_iniciales_a_borrar:
    mask_limpia_3d[z, :, :] = 0

# 2. Modificar slices de la entrada
umbral_55 = int(total_slices * 0.55)
slices_modificadas = 0

for z in range(umbral_55, total_slices):
    
    if slices_modificadas >= slices_superiores_a_modificar:
        break 
        
    slice_actual = mask_limpia_3d[z, :, :].astype(np.uint8)
    
    if np.any(slice_actual):
        

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(slice_actual, connectivity=8)
        
        if num_labels >= 3:
            min_y = float('inf')
            label_a_eliminar = -1
            
            # Buscar el cuerpo más próximo al cero en el eje Y
            for i in range(1, num_labels):
                centroid_y = centroids[i][1] # El índice 1 corresponde al eje Y
                
                if centroid_y < min_y:
                    min_y = centroid_y
                    label_a_eliminar = i
            
            # Si encontramos el cuerpo, lo rellenamos con ceros
            if label_a_eliminar != -1:
                # Todo lo que corresponda a esa etiqueta se vuelve 0
                slice_actual[labels == label_a_eliminar] = 0
                
                # Actualizamos la matriz 3D
                mask_limpia_3d[z, :, :] = slice_actual
                
                # Aumentamos el contador de slices modificadas
                slices_modificadas += 1

mask_limpia_sitk = sitk.GetImageFromArray(mask_limpia_3d)
mask_limpia_sitk.CopyInformation(mask)

#Extrer regiones

def segmentar_regiones_aorta(mask_array):
    
    mask_sitk = mask_array
    mask_3d = sitk.GetArrayFromImage(mask_sitk)
    
    # =========================================================================
    # CORTE HORIZONTAL (Eje Z) - Límite entre Arcos y Descenso
    # =========================================================================
    # Encontrar la longitud real en Z
    z_activos = np.where(np.any(mask_3d, axis=(1, 2)))[0]
    if len(z_activos) == 0:
        print("La máscara está vacía.")
        return
        
    z_min = z_activos[0]
    z_max = z_activos[-1]
    altura_real = z_max - z_min
    umbral_55 = z_min + int(altura_real * 0.55)

    # Barrido desde el 55% hacia arriba buscando 2 cuerpos
    z_corte_horizontal = umbral_55 # Valor por defecto si falla
    for z in range(umbral_55, z_max + 1):
        slice_actual = mask_3d[z, :, :].astype(np.uint8)
        if np.any(slice_actual):
            num_labels, _, _, _ = cv2.connectedComponentsWithStats(slice_actual, connectivity=8)
            # num_labels == 3 significa: 1 Fondo + 2 Cuerpos (Ascendente y Descendente)
            if num_labels == 3:
                z_corte_horizontal = z
                break

    # =========================================================================
    # CORTE VERTICAL (Eje Y) - Límite entre Arco Ascendente y Descendente
    # =========================================================================
    # Barrido en el eje Y (Coronal/Sagital). axis=1 en (Z, Y, X) es el eje Y.
    # Evaluamos en qué coordenadas Y hay información de la aorta.
    y_activos = np.where(np.any(mask_3d, axis=(0, 2)))[0]
    
    y_min = y_activos[0]  # Primer barrido de 0 a len
    y_max = y_activos[-1] # Segundo barrido de len a 0
    
    # El corte se hace exactamente en la mitad de la caja delimitadora en Y
    y_corte_vertical = (y_min + y_max) // 2

    # =========================================================================
    # GENERACIÓN DE LAS 3 MÁSCARAS
    # =========================================================================
    # Crear arrays vacíos con la misma forma
    mask_descenso = np.zeros_like(mask_3d)
    mask_arco_asc = np.zeros_like(mask_3d)
    mask_arco_desc = np.zeros_like(mask_3d)

    # 1. DESCENSO: Todo lo que esté por debajo del corte horizontal Z
    mask_descenso[:z_corte_horizontal, :, :] = mask_3d[:z_corte_horizontal, :, :]

    # 2 y 3. ARCOS: Todo lo que esté por encima del corte Z, dividido por el corte Y
    bloque_superior = mask_3d[z_corte_horizontal:, :, :]
    
    # Extraemos el bloque superior y aplicamos la "guillotina" en Y
    bloque_asc = np.copy(bloque_superior)
    bloque_desc = np.copy(bloque_superior)
    
    # NOTA DE ORIENTACIÓN: Dependiendo de tu tomografía, el Y menor puede ser anterior o posterior.
    # Por estándar médico, Y=0 suele ser la parte anterior (pecho) y el Y_max la posterior (espalda).
    # Por lo tanto, el Arco Ascendente está en los Y menores al corte.
    
    # Al ascendente le borramos la parte de la espalda (Y >= corte)
    bloque_asc[:, y_corte_vertical:, :] = 0 
    
    # Al descendente le borramos la parte del pecho (Y < corte)
    bloque_desc[:, :y_corte_vertical, :] = 0 
    
    # Guardamos los bloques en sus máscaras finales
    mask_arco_asc[z_corte_horizontal:, :, :] = bloque_asc
    mask_arco_desc[z_corte_horizontal:, :, :] = bloque_desc


    mask_descenso = sitk.GetImageFromArray(mask_descenso)
    mask_descenso.CopyInformation(mask_sitk)
    mask_arco_asc = sitk.GetImageFromArray(mask_arco_asc)
    mask_arco_asc.CopyInformation(mask_sitk)
    mask_arco_desc = sitk.GetImageFromArray(mask_arco_desc)
    mask_arco_desc.CopyInformation(mask_sitk)
    
    return  mask_descenso, mask_arco_asc, mask_arco_desc

mask_descenso, mask_arco_asc, mask_arco_desc = segmentar_regiones_aorta(mask_limpia_sitk)

#Codigo de volumen a base del voxel , por region y completo


def calcular_volumen(image, nombre_estructura, valor_etiqueta=1):

    # Cargar datos
    mask_array = sitk.GetArrayFromImage(image)
    
    # Calcular volumen
    vol_voxel_mm3 = voxel_dims[0] * voxel_dims[1] * voxel_dims[2]
    num_voxeles = np.sum(mask_array == valor_etiqueta)
    vol_total_mm3 = num_voxeles * vol_voxel_mm3
    vol_total_cm3 = vol_total_mm3 / 1000.0
    
    # Imprimir resultados individuales
    print(f"--- RESULTADOS: {nombre_estructura.upper()} ---")
    print(f"Vóxeles totales: {num_voxeles}")
    print(f"Volumen total  : {vol_total_cm3:.2f} cm³")
    print("-" * 45 + "\n")
    
    return vol_total_cm3

def calculate_surface_area(image, voxel_size):
    """
    Calcula el área de superficie de una máscara 3D en mm^2.
    
    Args:
        mask_array: Array binario (numpy) de la máscara.
        voxel_size: Lista o tuple con [spacing_x, spacing_y, spacing_z].
    """
    mask_array = sitk.GetArrayFromImage(image)
    # 1. Extraer la superficie usando Marching Cubes
    # El parámetro 'level' suele ser 0.5 para máscaras binarias (0 y 1)
    verts, faces, normals, values = measure.marching_cubes(mask_array, level=0.5, spacing=tuple(voxel_size))
    
    # 2. Calcular el área de la superficie
    surface_area = measure.mesh_surface_area(verts, faces)
    
    return surface_area

def calcular_areas_salida(image, nombre_estructura, tipo_cara='primera', valor_etiqueta=1):
    mask_array = sitk.GetArrayFromImage(image)


    coordenadas = np.where(mask_array == valor_etiqueta)
    
    if len(coordenadas[0]) == 0:
        print(f" La máscara de {nombre_estructura} está completamente vacía.")
        return None

    indices_z = coordenadas[2]
    
    if tipo_cara == 'primera':
        corte_objetivo = np.min(indices_z)
        texto_cara = "PRIMERA cara (Corte Z más bajo)"
    elif tipo_cara == 'ultima':
        corte_objetivo = np.max(indices_z)
        texto_cara = "ÚLTIMA cara (Corte Z más alto)"
    else:
        return None

    #Extraccion del el corte y contar los vóxeles
    corte_2d = mask_array[:, :, corte_objetivo]
    num_voxeles_cara = int(np.sum(corte_2d == valor_etiqueta))

    # CONVERSIÓN A MILÍMETROS (ÁREA)
    area_un_pixel_mm2 = voxel_dims[0] * voxel_dims[1]
    area_total_mm2 = num_voxeles_cara * area_un_pixel_mm2
    """"
    CÁLCULO DE RADIO Y DIÁMETRO
    radio_mm = math.sqrt(area_total_mm2 / math.pi)
    diametro_mm = radio_mm * 2  # El diámetro es exactamente el doble del radio
    """
    # Imprimir resultados detallados
    print(f"--- {nombre_estructura.upper()} ---")
    print(f"Cara analizada    : {texto_cara} (Índice Z: {corte_objetivo})")
    print(f"Resolución X, Y   : {voxel_dims[0]:.4f} mm x {voxel_dims[1]:.4f} mm por vóxel")
    print(f"Vóxeles contados  : {num_voxeles_cara} vóxeles")
    print(f"Área física real  : {area_total_mm2:.2f} mm²\n")
    """
    print(f"Radio físico real : {radio_mm:.2f} mm")
    print(f"Diámetro estimado : {diametro_mm:.2f} mm")
    print("-" * 45 + "\n")
    """
    return area_total_mm2

def calcular_metricas_transversales(image, nombre_estructura, voxel_dims, valor_etiqueta=1):
    """
    Calcula el diámetro mayor, menor y la excentricidad de la aorta slice por slice.
    Retorna un diccionario con los valores máximos, mínimos, promedios y std.
    """
    mask_array = sitk.GetArrayFromImage(image)
    
    # En SimpleITK el array extraído siempre tiene la forma (Z, Y, X).
    # Por lo tanto, el eje Z (los slices) es el índice 0.
    total_slices = mask_array.shape[0]
    
    diametros_mayores = []
    diametros_menores = []
    excentricidades = []
    
    # Para la conversión a mm, usamos la resolución del eje X (asumiendo que X e Y 
    # son isotrópicos/iguales, lo cual es el estándar en tomografías axiales).
    pixel_spacing_mm = voxel_dims[0]

    for z in range(total_slices):
        slice_2d = mask_array[z, :, :]
        
        # Procesar solo si hay aorta en este slice
        if np.any(slice_2d == valor_etiqueta):
            
            # Etiquetar las regiones del slice (por si queda algún píxel flotante)
            labels = measure.label(slice_2d == valor_etiqueta)
            props = measure.regionprops(labels)
            
            if props:
                # Encontrar el objeto más grande del slice (la luz de la aorta)
                prop_aorta = max(props, key=lambda item: item.area)
                
                # Extraer propiedades. regionprops devuelve ejes en píxeles.
                # Se multiplican por el espaciado para obtener milímetros físicos.
                d_mayor_mm = prop_aorta.axis_major_length * pixel_spacing_mm
                d_menor_mm = prop_aorta.axis_minor_length * pixel_spacing_mm
                excentricidad = prop_aorta.eccentricity
                
                diametros_mayores.append(d_mayor_mm)
                diametros_menores.append(d_menor_mm)
                excentricidades.append(excentricidad)

    if not diametros_mayores:
        print(f" La máscara de {nombre_estructura} está completamente vacía.")
        return None

    # Calcular estadísticas
    resultados = {
        "diametro_mayor": {
            "max": np.max(diametros_mayores),
            "min": np.min(diametros_mayores),
            "promedio": np.mean(diametros_mayores),
            "std": np.std(diametros_mayores)
        },
        "diametro_menor": {
            "max": np.max(diametros_menores),
            "min": np.min(diametros_menores),
            "promedio": np.mean(diametros_menores),
            "std": np.std(diametros_menores)
        },
        "excentricidad": {
            "max": np.max(excentricidades),
            "min": np.min(excentricidades),
            "promedio": np.mean(excentricidades),
            "std": np.std(excentricidades)
        }
    }

    # Imprimir resultados formateados
    print(f"Slices analizados: {len(diametros_mayores)}")
    print(f"Diámetro Mayor   : Max {resultados['diametro_mayor']['max']:.2f} mm | Min {resultados['diametro_mayor']['min']:.2f} mm | Promedio {resultados['diametro_mayor']['promedio']:.2f} mm ± {resultados['diametro_mayor']['std']:.2f}")
    print(f"Diámetro Menor   : Max {resultados['diametro_menor']['max']:.2f} mm | Min {resultados['diametro_menor']['min']:.2f} mm | Promedio {resultados['diametro_menor']['promedio']:.2f} mm ± {resultados['diametro_menor']['std']:.2f}")
    print(f"Excentricidad    : Max {resultados['excentricidad']['max']:.4f}    | Min {resultados['excentricidad']['min']:.4f}    | Promedio {resultados['excentricidad']['promedio']:.4f}    ± {resultados['excentricidad']['std']:.4f}")
    print("-" * 45 + "\n")

    return (
        resultados['excentricidad']['max'], 
        resultados['excentricidad']['min'], 
        resultados['excentricidad']['promedio'], 
        resultados['excentricidad']['std'],
        resultados['diametro_mayor']['max'], 
        resultados['diametro_mayor']['min'], 
        resultados['diametro_mayor']['promedio'], 
        resultados['diametro_mayor']['std'],
        resultados['diametro_menor']['max'], 
        resultados['diametro_menor']['min'], 
        resultados['diametro_menor']['promedio'], 
        resultados['diametro_menor']['std']
    )


print("Iniciando calculos de volumenes...\n" + "="*45 + "\n")

vol_opt = calcular_volumen(mask_limpia_sitk, "Máscara Completa")
vol_asc = calcular_volumen(mask_arco_asc, "Arco Ascendente")
vol_des = calcular_volumen(mask_arco_desc, "Arco Descendente")
vol_dsc = calcular_volumen(mask_descenso, "Descenso de la Aorta")

print("Iniciando calculos de área superficial...\n" + "="*45 + "\n")

area_mm2 = calculate_surface_area(mask_limpia_sitk, voxel_dims)

print(f"Área de superficie de la aorta: {area_mm2:.2f} mm²")
print(f"Área de superficie en cm²: {area_mm2 / 100:.2f} cm²\n")

print("Iniciando calculos de área de salida/entrada...\n" + "="*45 + "\n")

# Para el arco ascendente
asc_area_total_mm2 = calcular_areas_salida(mask_arco_asc, "Arco Ascendente", tipo_cara='primera')

# Para el descenso
desc_area_total_mm2 = calcular_areas_salida(mask_descenso, "Descenso", tipo_cara='primera')

print("Iniciando cálculos de morfología transversal...\n" + "="*45 + "\n")

# Para el descenso
(excent_max_desc, excent_min_desc, excent_prom_desc, excent_std_desc,
 major_diam_max_desc, major_diam_min_desc, major_diam_prom_desc, major_diam_std_desc,
 minor_diam_max_desc, minor_diam_min_desc, minor_diam_prom_desc, minor_diam_std_desc) = calcular_metricas_transversales(mask_descenso, "Descenso de la Aorta", voxel_dims)

#Guardar excel


print("Guardando resultados en CSV...\n" + "="*45)

features_df = pd.DataFrame({
    "Study Name": study_name,
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
    "Diametro_Menor_Std_Desc_mm": minor_diam_std_desc
}, index=[0])

df_path = f'{id}_{condition}.csv'
features_df.to_csv(df_path, header=True, index=False)

print(f"Resultados guardados en {df_path}")