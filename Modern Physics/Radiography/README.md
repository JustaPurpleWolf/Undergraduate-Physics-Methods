# Análisis de Atenuación de Rayos X y Caracterización de Hardware (Agfa XD14)

Este repositorio contiene el flujo de trabajo completo para el análisis de la atenuación de rayos X en aluminio y papel, utilizando datos crudos extraídos de un detector **Agfa XD14** y un emisor **PXM-40BT**. 

## 🔍 Hallazgo Principal
A diferencia del procesamiento comercial, este análisis revela que el sistema presenta un **desplazamiento de offset ($z_0$)** y una falta de calibración de nivel de negro. Esto induce un error en el cálculo de la energía efectiva ($E_{eff} \approx 200 \text{ keV}$), lo cual es físicamente inconsistente para un disparo de 40 kVp, confirmando que el detector requiere una recalibración de campo plano (*Flat-Field*).

## 📁 Estructura del Repositorio

* **`DICOM/`**: Imágenes originales en formato médico (Raw Data).
* **`radiography_toolkit.py`**: Paquetería personalizada para la extracción de metadatos y procesamiento lineal.
* **`radiography_analysis.ipynb`**: Notebook con el ajuste de curvas de Beer-Lambert y modelos no lineales.
* **`plots/`**: Visualización de las curvas de atenuación donde se evidencia la baja agresividad del haz.
* **`data_inversed/`**: Matrices de intensidad transformadas para el cálculo de coeficientes $\mu$.
* **`calibration_thickness_og.csv`**: Registros de espesores utilizados para la validación del modelo.

## 🛠️ Metodología Técnica

1.  **Extracción de Datos Crudos**: Se utilizó `pydicom` para omitir los filtros cosméticos del software de la máquina.
2.  **Ajuste Matemático**: Se aplicó un modelo de mínimos cuadrados para encontrar el coeficiente de atenuación lineal ($\mu$) y el ruido de fondo ($z_0$).
    * **Resultado Al:** $\mu = 0.0349 \text{ mm}^{-1}$ ($R^2 = 0.9983$).
    * **Ruido detectado ($z_0$):** $0.01328$ (evidencia de error de offset).
3.  **Inconsistencia Física**: El valor de 200 keV obtenido mediante búsqueda inversa en tablas NIST XCOM demuestra que el detector subestima la atenuación real debido a saturación o desplazamiento del cero.

## 🚀 Requisitos

Instala las dependencias necesarias con:
```bash
pip install pydicom numpy scipy matplotlib pandas