# IEEE Fraud Detection Pipeline 🕵️‍♂️

Un sistema MLOps de detección de fraude modular, escalable y preparado para producción ("Production-Ready"), diseñado para la competición **IEEE-CIS Fraud Detection** de Kaggle.

Este proyecto implementa un pipeline completo desde la ingestión de datos crudos hasta la inferencia, utilizando tecnologías modernas para asegurar reproducibilidad y rendimiento.

## 🏗 Arquitectura del Proyecto

El código sigue una estructura modular separando configuración, lógica de negocio y ejecución.

```text
ieee_fraud_detection/
├── data/                   # Datos (Raw, Processed, Submissions) - Ignorado en Git
├── scripts/                # Puntos de entrada (CLI)
│   └── run_pipeline.py     # Orquestador principal
├── src/                    # Lógica de Negocio (Paquete Python)
│   ├── config.py           # Configuración tipada (Pydantic)
│   ├── preprocess.py       # ETL con PySpark (Train/Test consistencia)
│   ├── training.py         # Entrenamiento (XGBoost/LightGBM/CatBoost)
│   ├── ensemble.py         # Lógica de validación cruzada
│   └── inference.py        # Generación de predicciones
├── requirements.txt        # Dependencias
└── README.md               # Documentación
```

## 🛠 Stack Tecnológico

* **ETL & Big Data:** PySpark 3.x (Manejo de grandes volúmenes y Feature Engineering).
* **Modelado:** XGBoost (GPU Accelerated), LightGBM, CatBoost.
* **Optimización:** Optuna (Búsqueda Bayesiana de Hiperparámetros).
* **MLOps:** MLflow (Experiment Tracking & Model Registry).
* **Configuración:** Pydantic (Validación de tipos y gestión de entornos).

## 🚀 Quick Start

### 1. Instalación

Se recomienda usar un entorno virtual con Python 3.10+.

```bash
# Crear entorno (opcional)
conda create -n fraud_detection python=3.10
conda activate fraud_detection

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Preparación de Datos

Descarga los datasets de la competición (train_transaction.csv, train_identity.csv, etc.) y colócalos en: `data/raw/`

### 3. Ejecución del Pipeline

El proyecto se controla mediante un único script CLI: `scripts/run_pipeline.py`.

**Paso 1: Preprocesamiento (ETL)**

Limpia los datos, genera features temporales, gestiona nulos y crea los archivos Parquet optimizados. Asegura consistencia entre Train y Test.

```bash
python scripts/run_pipeline.py preprocess
```

**Paso 2: Entrenamiento**

Entrena el modelo especificado utilizando GPU. Los experimentos y artefactos se registran automáticamente en MLflow.

```bash
# Entrenar XGBoost (Default)
python scripts/run_pipeline.py train --model xgboost

# Entrenar todos los modelos para Ensemble
python scripts/run_pipeline.py train --model all
```

**Paso 3: Validación (Ensemble Local)**

Carga los modelos registrados y calcula el AUC combinado en el set de validación.

```bash
python scripts/run_pipeline.py ensemble
```

**Paso 4: Inferencia (Kaggle Submission)**

Genera el archivo `submission.csv` final utilizando los modelos entrenados.

```bash
python scripts/run_pipeline.py predict
```

## 📊 Resultados Actuales

* **Single Model (XGBoost Tuned):** AUC ~0.922 (Validación Temporal).
* **Hardware:** Optimizado para NVIDIA RTX 4080 / AMD Ryzen 9800X3D.

Proyecto desarrollado para IEEE-CIS Fraud Detection.
