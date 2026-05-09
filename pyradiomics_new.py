import SimpleITK as sitk
from radiomics import featureextractor
import pandas as pd
import os



# Configuración de archivos
dataset=pd.read_csv('dataset_index_all.csv')

for index, row in dataset.iterrows():
    condition = row["Etiqueta"]
    name = row["Nombre_Estudio"]
    image_path = "".join(["CT_dataset/",condition,"/",name,".nii.gz"])
    mask_path = "".join(["CT_mask/",condition,"/",name,"/aorta.nii.gz"])


    # 1. Definir los parámetros de configuración
    # Estos valores aseguran que la extracción sea consistente entre diferentes pacientes
    params = {
        # Binning: Agrupa intensidades en "bins" de 25 HU. 
        # Esto reduce el ruido y hace que los features sean más robustos.
        'binWidth': 25, 
        
        # Resampling: Obliga a que todos los estudios se analicen como si tuvieran
        # una resolución de 1x1x1 mm. Es CRÍTICO si tus CTs tienen diferentes grosores de corte.
        'resampledPixelSpacing': [1, 1, 1],
        
        # Interpolación: B-Spline es ideal para geometría vascular (aorta)
        'interpolator': sitk.sitkBSpline,
        
        # Pre-procesamiento adicional (Opcional pero recomendado)
        'padDistance': 5,  # Añade un margen alrededor de la máscara para cálculos de textura
        'voxelArrayShift': 1000 # Si tus datos tienen valores negativos bajos, esto los desplaza a positivos
    }

    # 2. Inicializar el extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(**params)

    # 3. Cargar imágenes
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)

    # 4. Extraer features
    # PyRadiomics manejará automáticamente el resampling y el binning antes de calcular
    features = extractor.execute(image, mask)

    # 5. Guardar resultados
    df = pd.DataFrame([features])
    output_path = f"features/{condition}/{name}_pyradiomics.csv"
    df.to_csv(output_path, index=False)

    print(f"Extracción completada. Features guardados en: {output_path}")