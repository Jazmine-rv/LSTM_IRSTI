

#  IRSTI-AI · Sistema Predictivo de Riesgo con LSTM



##  Descripción

**IRSTI-AI** es una aplicación web educativa e interactiva que demuestra el funcionamiento de una **Red Neuronal Recurrente LSTM** 

El proyecto simula un entorno clínico de **Unidad de Cuidados Intensivos (UCI)** donde el modelo aprende a predecir el **riesgo de deterioro de un paciente** a partir de la secuencia histórica de sus signos vitales.

### Objetivo didáctico

Que cualquier persona —sin conocimientos previos en IA— pueda:
- Entender qué es una **LSTM** y cómo funciona
- Visualizar el **proceso de aprendizaje** del modelo
- Experimentar con **hiperparámetros** y ver su efecto en tiempo real
- Comprender la importancia de la **memoria temporal** en predicción de secuencias

---

## Modelo Implementado

| Característica | Detalle |
|----------------|---------|
| **Tipo** | LSTM (Long Short-Term Memory) |
| **Implementación** | 100% desde cero con NumPy |
| **Algoritmo** | Backpropagation Through Time (BPTT) |
| **Capas** | 1 celda LSTM + capa de salida densa |
| **Funciones de activación** | Sigmoide (puertas) · Tanh (candidate y salida) |
| **Optimización** | Mini-batch Gradient Descent con Gradient Clipping |

---

## Dataset Sintético

Generamos **5 perfiles de paciente** con signos vitales realistas:

| Perfil | Descripción |
|--------|-------------|
| 🟢 **Estable** | Signos vitales dentro de rangos normales |
| 🟡 **Deterioro Progresivo** | Empeoramiento gradual a lo largo del tiempo |
| 🔴 **Crisis Repentina** | Deterioro abrupto (ej. sepsis) |
| 📈 **Mejoría post-tratamiento** | Recuperación progresiva |
| 🎯 **Combinado** | Todos los patrones en una sola secuencia |

**Variables de entrada:**
- Frecuencia cardíaca (lpm)
- Saturación de O₂ (%)
- Frecuencia respiratoria (rpm)
- Presión arterial media (mmHg)
- Temperatura (°C)

**Variable objetivo:** Score de riesgo 

---

## Caracteristicas Principales

### Storytelling Progresivo
La aplicación guía al usuario a través de **6 etapas** narrativas:

1. **El problema** — Contexto clínico y motivación
2. **¿Para qué sirve?** — Clasificación vs predicción
3. **Arquitectura** — Explicación visual de la celda LSTM
4. **Entrenamiento** — Ajuste de hiperparámetros y visualización
5. **Predicción en vivo** — Simulación interactiva con explicabilidad
6. **Ética y límites** — Consideraciones críticas del modelo

### Interactividad Significativa
- **Sliders** para: tasa de aprendizaje, épocas, unidades LSTM, ventana temporal, batch size
- **Feedback contextual** que explica el efecto de cada parámetro
- **Selección de perfiles** de paciente para entrenar el modelo
- **Predicción en tiempo real** ajustando signos vitales

### Visualizaciones
- Evolución de signos vitales (Plotly)
- Curva de pérdida durante el entrenamiento
- Predicción vs valor real en test
- Frontera de decisión (influencia de variables)
- Diagrama interactivo de la celda LSTM

### Glosario Interactivo
En la barra lateral, con:
- Definición en lenguaje llano
- Fórmula matemática
- Efecto de valores bajos/medios/altos
- Analogía cotidiana
- Tip práctico

### Explicabilidad
Análisis de **sensibilidad** que muestra qué variable influyó más en la predicción actual.

---

## Herramientas

| Tecnología | Uso |
|------------|-----|
| **Python 3.10+** | Lenguaje principal |
| **Streamlit** | Framework web interactivo |
| **NumPy** | Implementación del modelo desde cero |
| **Pandas** | Manejo de datos |
| **Matplotlib** | Gráficos estáticos |
| **Plotly** | Gráficos interactivos |
| **Scikit-learn** | Métricas y preprocesamiento |

---

## Archivos
- `readme.mp` archivo descriptivo 
- `requirements` dependecias
- `app.py` Archivo principal 
- `lstm_model_numpy.py` modelo utilizado
- `TP_IRSTI.pdf` informe del trabajo

##  Instalación y Ejecución

### Prerrequisitos
- Python 3.10 o superior

### Pasos

1. **Clonar el repositorio**
git clone https://github.com/Jazmine-rv/LSTM_IRSTI.git
2. **Crear entorno virtual**
python -m venv venv
source venv/bin/activate      # Linux/Mac
 o
venv\Scripts\activate         # Windows

3. **Instalar dependencias**
pip install -r requirements.txt

4. **Ejecutar la aplicación**
streamlit run app.py
