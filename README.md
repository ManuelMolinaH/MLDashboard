# Dashboard Dinámico para Análisis de Datos y Machine Learning

Este proyecto forma parte de mi Trabajo Fin de Grado en el Doble Grado de Ingeniería Informática y ADE (UC3M).

Se trata de un **dashboard interactivo en Python/Dash** que permite a usuarios sin conocimientos de programación llevar a cabo, desde una misma interfaz web, todas las fases de un flujo de análisis de datos:

- **Carga de datos** desde archivos o bases de datos.  
- **Análisis exploratorio (EDA)** con estadísticas, visualizaciones y descomposición de series temporales.  
- **Preprocesamiento de datos**: limpieza, codificación, escalado, selección de características y balanceo de clases.  
- **Entrenamiento y evaluación de modelos** de Machine Learning y Deep Learning:  
  - Modelos clásicos: regresión, clasificación, clustering.  
  - Series temporales: ARIMA, Prophet, AutoTS, StatsForecast.  
  - Modelos avanzados: Random Forest, XGBoost, LightGBM, LSTM, TCN, Transformers.  
- **Visualización de resultados** y métricas comparativas.

## Tecnologías principales

- Python 3.9+  
- Dash / Plotly para la interfaz web  
- Scikit-learn, Statsmodels, Prophet, TensorFlow/Keras, AutoTS, StatsForecast para ML y series temporales  
- PostgreSQL + SQLAlchemy para conexión con bases de datos  
- XGBoost / LightGBM para modelos de gradiente

## Instalación

Clonar el repositorio y crear un entorno virtual:

```bash
git clone MLDashboard
cd <nombre-del-repo>
python -m venv venv
source venv/bin/activate  # en Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución

Ejecutando:
```bash
python app.py
```
Y posteriormente accediendo a: *http://127.0.0.1:8050* 

## Nota
Este prototipo está orientado a fines académicos y pedagógicos, no al despliegue en producción con múltiples usuarios.
