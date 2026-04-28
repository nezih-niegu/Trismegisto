@echo off
echo Iniciando la instalacion de dependencias para el entorno de Python 3.9...
echo.

:: Se mantiene el orden exacto para evitar romper las dependencias fragiles.

pip install pydicom ipykernel pyradiomics
pip install SimpleITK 
pip install opencv-python==4.9.0.80
pip install "numpy<2.0.0"
pip install GDCM
pip install pylibjpeg pylibjpeg-libjpeg "numpy==1.26.4"
pip install pandas==2.2.1

echo.
echo Instalacion finalizada.
pause