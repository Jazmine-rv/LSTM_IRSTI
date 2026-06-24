"""
app.py — IRSTI-AI: Sistema Predictivo de Riesgo con LSTM
Tutorial interactivo de Redes Neuronales Recurrentes (LSTM)
TP Integrador - Modelizado de Sistemas de IA
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Importar modelo NumPy (implementación desde cero)
from lstm_model_numpy import LSTMModel as NumpyLSTM

# ============================================================
# CONFIGURACIÓN DE PÁGINA Y ESTILO
# ============================================================
st.set_page_config(
    page_title="IRSTI-AI · LSTM Tutorial",
    page_icon="🚦",
    layout="wide",
)

# Paleta IRSTI
COLOR_BG = "#FFFFFF"
COLOR_BG_CARD = "#FFFFFF"
COLOR_ACCENT = "#02C39A"
COLOR_ROJO = "#E63946"
COLOR_AMARILLO = "#F4B400"
COLOR_VERDE = "#02C39A"
COLOR_TEXT = "#000000"
COLOR_TEXT_MUTED = "#8FA3B5"

st.markdown(f"""
<style>
.stApp {{
    background-color: {COLOR_BG};
    color: {COLOR_TEXT};
}}
.irsti-card {{
    background-color: {COLOR_BG_CARD};
    border: 1px solid #E0E0E0;  
    border-radius: 12px;        
    padding: 18px 22px;
    margin-bottom: 14px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);  
}}
.irsti-titulo {{
    font-size: 2.6rem;
    font-weight: 800;
    color: {COLOR_ROJO};
    letter-spacing: 1px;
    margin-bottom: 0;
}}
.irsti-subtitulo {{
    font-size: 1.1rem;
    color: {COLOR_TEXT_MUTED};
    letter-spacing: 2px;
}}
.metric-label {{
    color: {COLOR_TEXT_MUTED};
    font-size: 0.95rem;
}}
.metric-value {{
    font-size: 1.6rem;
    font-weight: 700;
}}
.alerta-roja {{
    border-left: 4px solid {COLOR_ROJO};
    background-color: transparent;
    padding: 10px 14px;
    margin-bottom: 8px;
    border-radius: 4px;
    color: {COLOR_TEXT};
}}
.alerta-amarilla {{
    border-left: 4px solid {COLOR_AMARILLO};
    background-color: transparent;
    padding: 10px 14px;
    margin-bottom: 8px;
    border-radius: 4px;
    color: {COLOR_TEXT};
}}
.alerta-verde {{
    border-left: 4px solid {COLOR_VERDE};
    background-color: transparent;
    padding: 10px 14px;
    margin-bottom: 8px;
    border-radius: 4px;
    color: {COLOR_TEXT};
}}
</style>
""", unsafe_allow_html=True)


# ============================================================
# GENERADORES DE DATOS SINTÉTICOS - PACIENTES IRSTI
# ============================================================

def generar_paciente_estable(n=300):
    """Paciente estable: signos vitales dentro de rangos normales"""
    t = np.arange(n)
    fc = 72 + 5 * np.sin(2 * np.pi * t / 50) + np.random.normal(0, 2, n)
    spo2 = 97 + 1.5 * np.sin(2 * np.pi * t / 40) + np.random.normal(0, 0.8, n)
    fr = 16 + 2 * np.sin(2 * np.pi * t / 60) + np.random.normal(0, 1, n)
    pam = 85 + 5 * np.sin(2 * np.pi * t / 45) + np.random.normal(0, 3, n)
    temp = 37 + 0.3 * np.sin(2 * np.pi * t / 55) + np.random.normal(0, 0.2, n)
    
    riesgo = 0.1 + 0.05 * np.sin(2 * np.pi * t / 30) + np.random.normal(0, 0.02, n)
    riesgo = np.clip(riesgo, 0, 0.3)
    
    return {
        'fc': fc, 'spo2': spo2, 'fr': fr, 'pam': pam, 'temp': temp,
        'riesgo': riesgo, 'etiqueta': 'ESTABLE'
    }


def generar_paciente_deterioro(n=300):
    """Paciente con deterioro progresivo"""
    t = np.arange(n)
    
    fc = 70 + 0.2 * t + 5 * np.sin(2 * np.pi * t / 30) + np.random.normal(0, 3, n)
    fc = np.clip(fc, 50, 150)
    
    spo2 = 98 - 0.04 * t + 1.5 * np.sin(2 * np.pi * t / 25) + np.random.normal(0, 1, n)
    spo2 = np.clip(spo2, 80, 100)
    
    fr = 15 + 0.05 * t + 2 * np.sin(2 * np.pi * t / 35) + np.random.normal(0, 1.5, n)
    fr = np.clip(fr, 10, 35)
    
    pam = 90 - 0.1 * t + 5 * np.sin(2 * np.pi * t / 40) + np.random.normal(0, 3, n)
    pam = np.clip(pam, 50, 110)
    
    temp = 37 + 0.005 * t + 0.3 * np.sin(2 * np.pi * t / 50) + np.random.normal(0, 0.2, n)
    temp = np.clip(temp, 36, 39)
    
    riesgo = 0.1 + 0.0027 * t + 0.05 * np.sin(2 * np.pi * t / 40) + np.random.normal(0, 0.03, n)
    riesgo = np.clip(riesgo, 0, 1)
    
    return {
        'fc': fc, 'spo2': spo2, 'fr': fr, 'pam': pam, 'temp': temp,
        'riesgo': riesgo, 'etiqueta': 'DETERIORO'
    }


def generar_paciente_critico_repentino(n=300):
    """Paciente que se deteriora repentinamente (ej. sepsis)"""
    t = np.arange(n)
    punto_critico = 200
    
    fc = 72 + 5 * np.sin(2 * np.pi * t / 40) + np.random.normal(0, 2, n)
    fc[punto_critico:] = fc[punto_critico:] + 40 + 10 * np.sin(2 * np.pi * t[punto_critico:] / 20)
    fc = np.clip(fc, 50, 160)
    
    spo2 = 97 + 1.5 * np.sin(2 * np.pi * t / 35) + np.random.normal(0, 0.8, n)
    spo2[punto_critico:] = spo2[punto_critico:] - 12 + 2 * np.sin(2 * np.pi * t[punto_critico:] / 15)
    spo2 = np.clip(spo2, 75, 100)
    
    fr = 16 + 2 * np.sin(2 * np.pi * t / 50) + np.random.normal(0, 1, n)
    fr[punto_critico:] = fr[punto_critico:] + 12 + 3 * np.sin(2 * np.pi * t[punto_critico:] / 25)
    fr = np.clip(fr, 10, 40)
    
    pam = 88 + 5 * np.sin(2 * np.pi * t / 45) + np.random.normal(0, 3, n)
    pam[punto_critico:] = pam[punto_critico:] - 25 + 5 * np.sin(2 * np.pi * t[punto_critico:] / 30)
    pam = np.clip(pam, 45, 110)
    
    temp = 37 + 0.3 * np.sin(2 * np.pi * t / 55) + np.random.normal(0, 0.2, n)
    
    riesgo = 0.1 + 0.05 * np.sin(2 * np.pi * t / 30) + np.random.normal(0, 0.02, n)
    riesgo[punto_critico:] = 0.85 + 0.05 * np.sin(2 * np.pi * t[punto_critico:] / 20)
    riesgo = np.clip(riesgo, 0, 1)
    
    return {
        'fc': fc, 'spo2': spo2, 'fr': fr, 'pam': pam, 'temp': temp,
        'riesgo': riesgo, 'etiqueta': 'CRITICO_REPENTINO'
    }


def generar_paciente_mejoria(n=300):
    """Paciente que mejora progresivamente (respuesta al tratamiento)"""
    t = np.arange(n)
    punto_mejoria = 150
    
    fc = 110 - 0.2 * t + 5 * np.sin(2 * np.pi * t / 30) + np.random.normal(0, 3, n)
    fc = np.clip(fc, 60, 130)
    fc[punto_mejoria:] = fc[punto_mejoria:] - 10 * (1 - np.exp(-(t[punto_mejoria:] - punto_mejoria) / 30))
    
    spo2 = 90 + 0.03 * t + 1.5 * np.sin(2 * np.pi * t / 25) + np.random.normal(0, 1, n)
    spo2 = np.clip(spo2, 85, 99)
    spo2[punto_mejoria:] = spo2[punto_mejoria:] + 5 * (1 - np.exp(-(t[punto_mejoria:] - punto_mejoria) / 40))
    
    fr = 25 + 0.02 * t + 2 * np.sin(2 * np.pi * t / 35) + np.random.normal(0, 1.5, n)
    fr = np.clip(fr, 10, 30)
    fr[punto_mejoria:] = fr[punto_mejoria:] - 8 * (1 - np.exp(-(t[punto_mejoria:] - punto_mejoria) / 35))
    
    pam = 65 + 0.08 * t + 5 * np.sin(2 * np.pi * t / 40) + np.random.normal(0, 3, n)
    pam = np.clip(pam, 60, 100)
    pam[punto_mejoria:] = pam[punto_mejoria:] + 15 * (1 - np.exp(-(t[punto_mejoria:] - punto_mejoria) / 45))
    
    temp = 37.5 - 0.003 * t + 0.3 * np.sin(2 * np.pi * t / 50) + np.random.normal(0, 0.2, n)
    temp = np.clip(temp, 36.5, 38.5)
    
    riesgo = 0.8 - 0.0023 * t + 0.05 * np.sin(2 * np.pi * t / 35) + np.random.normal(0, 0.02, n)
    riesgo[punto_mejoria:] = riesgo[punto_mejoria:] - 0.4 * (1 - np.exp(-(t[punto_mejoria:] - punto_mejoria) / 50))
    riesgo = np.clip(riesgo, 0, 1)
    
    return {
        'fc': fc, 'spo2': spo2, 'fr': fr, 'pam': pam, 'temp': temp,
        'riesgo': riesgo, 'etiqueta': 'MEJORIA'
    }


def generar_dataset_irsti(tipo="deterioro", n=300, incluir_vitales=True):
    """Genera dataset completo de signos vitales según el tipo de paciente"""
    
    generadores = {
        "estable": generar_paciente_estable,
        "deterioro": generar_paciente_deterioro,
        "critico_repentino": generar_paciente_critico_repentino,
        "mejoria": generar_paciente_mejoria,
        "combinado": None
    }
    
    if tipo == "combinado":
        n_parte = n // 4
        partes = []
        for gen, nombre in [
            (generar_paciente_estable, "ESTABLE"),
            (generar_paciente_deterioro, "DETERIORO"),
            (generar_paciente_critico_repentino, "CRITICO"),
            (generar_paciente_mejoria, "MEJORIA")
        ]:
            datos = gen(n_parte)
            partes.append(datos)
        
        datos = {
            'fc': np.concatenate([p['fc'] for p in partes]),
            'spo2': np.concatenate([p['spo2'] for p in partes]),
            'fr': np.concatenate([p['fr'] for p in partes]),
            'pam': np.concatenate([p['pam'] for p in partes]),
            'temp': np.concatenate([p['temp'] for p in partes]),
            'riesgo': np.concatenate([p['riesgo'] for p in partes]),
            'etiqueta': 'COMBINADO'
        }
    else:
        datos = generadores[tipo](n)
    
    t = np.arange(len(datos['riesgo']))
    
    if incluir_vitales:
        df = pd.DataFrame({
            'tiempo': t,
            'frecuencia_cardiaca': datos['fc'],
            'saturacion_o2': datos['spo2'],
            'frecuencia_respiratoria': datos['fr'],
            'presion_arterial': datos['pam'],
            'temperatura': datos['temp'],
            'riesgo': datos['riesgo']
        })
        return t, df, datos['etiqueta']
    else:
        return t, datos['riesgo'], datos['etiqueta']


def baseline_persistencia_corregida(y_train, y_test):
    """
    Baseline de persistencia CORRECTO.
    Maneja arrays de 1D y 2D.
    """
    # Aplanar si son 2D
    if len(y_train.shape) > 1:
        y_train = y_train.flatten()
    if len(y_test.shape) > 1:
        y_test = y_test.flatten()
    
    # Primer valor = último de entrenamiento
    first_pred = y_train[-1]
    
    # Resto = valor anterior de test (persistencia REAL)
    rest_preds = y_test[:-1]
    
    # Combinar
    result = np.concatenate([[first_pred], rest_preds])
    
    print(f"🔍 baseline_pred shape: {result.shape}")
    print(f"🔍 baseline_pred primeros 5: {result[:5]}")
    
    return result

# ============================================================
# GLOSARIO LATERAL
# ============================================================
GLOSARIO = {
    "Tasa de aprendizaje (η)": {
        "definicion": "Controla cuánto se ajustan los pesos en cada paso del entrenamiento.",
        "formula": "w = w − η · ∂L/∂w",
        "bajo": "Convergencia muy lenta, puede no terminar de aprender a tiempo.",
        "medio": "Aprendizaje estable y progresivo.",
        "alto": "El modelo puede 'pasarse de largo' y no converger (diverge).",
        "analogia": "Como bajar una escalera: pasos muy largos te hacen tropezar, muy cortos tardás una eternidad.",
        "tip": "Empezá con valores entre 0.001 y 0.01 y ajustá según la curva de pérdida.",
    },
    "Función de activación: Sigmoide": {
        "definicion": "Comprime cualquier valor al rango (0,1). Se usa en las puertas de la LSTM.",
        "formula": "σ(x) = 1 / (1 + e⁻ˣ)",
        "bajo": "Valores cercanos a 0 → la puerta 'cierra' el paso.",
        "medio": "Valores intermedios → paso parcial.",
        "alto": "Valores cercanos a 1 → la puerta 'abre' completamente.",
        "analogia": "Como una válvula que regula cuánta agua deja pasar.",
        "tip": "Ideal para decisiones de 'sí/no' dentro de la red.",
    },
    "Función de activación: Tangente hiperbólica (tanh)": {
        "definicion": "Comprime valores al rango (-1,1). Se usa para regular el contenido del estado de la celda.",
        "formula": "tanh(x) = (eˣ − e⁻ˣ)/(eˣ + e⁻ˣ)",
        "bajo": "Valores negativos → información que 'resta' al estado.",
        "medio": "Valores cercanos a 0 → poca influencia.",
        "alto": "Valores cercanos a 1 → información que refuerza el estado.",
        "analogia": "Como un termostato que puede sumar o restar calor.",
        "tip": "Más costosa que la sigmoide pero ayuda a converger más rápido.",
    },
    "Comparación: Tanh vs Sigmoide": {
        "definicion": "La tangente hiperbólica (tanh) y la sigmoide son funciones de activación con diferentes características y usos.",
        "formula": "tanh(x) = (eˣ − e⁻ˣ)/(eˣ + e⁻ˣ)  |  σ(x) = 1/(1 + e⁻ˣ)",
        "bajo": "Tanh en redes chicas → convergencia lenta, no aprovecha su rango (-1,1).",
        "medio": "Tanh en redes medianas → buen balance, convergencia estable.",
        "alto": "Tanh en redes grandes → converge más rápido aunque sea más costosa por paso. Sigmoide en redes chicas → suficiente y más económica.",
        "analogia": "Tanh es como un termostato que puede calentar o enfriar (valores negativos y positivos) — útil para sistemas complejos. Sigmoide es como un interruptor que solo abre o cierra — útil para sistemas simples.",
        "tip": "✅ Regla práctica: si tu red tiene más de 2 capas ocultas o muchas unidades, usá tanh. Si es una red pequeña (1-2 capas), sigmoide alcanza.",
    },
    "Mini-batch gradient descent": {
        "definicion": "Actualiza los pesos usando pequeños subconjuntos de datos.",
        "formula": "w = w − η · (1/m) Σ ∂Lᵢ/∂w",
        "bajo": "Batch muy chico → entrenamiento ruidoso pero rápido.",
        "medio": "Batch moderado (16-32) → buen balance velocidad/estabilidad.",
        "alto": "Batch muy grande → más estable pero más lento.",
        "analogia": "Como probar una sopa con cucharadas en vez de tomártela toda.",
        "tip": "16 o 32 es un buen punto de partida para datasets medianos.",
    },
    "LSTM - Forget Gate": {
        "definicion": "Decide qué porcentaje del estado anterior se conserva.",
        "formula": "f_t = σ(W_f · [h_{t-1}; x_t] + b_f)",
        "bajo": "f_t ≈ 0 → olvida todo",
        "medio": "f_t ≈ 0.5 → recuerda la mitad",
        "alto": "f_t ≈ 1 → recuerda todo",
        "analogia": "Como borrar el pizarrón: solo borrás lo que ya no sirve.",
        "tip": "El bias se inicializa en 1 para que la red empiece recordando.",
    },
    "LSTM - Input Gate": {
        "definicion": "Decide cuánta información nueva se incorpora a la memoria.",
        "formula": "i_t = σ(W_i · [h_{t-1}; x_t] + b_i)",
        "bajo": "i_t ≈ 0 → no aprende nada nuevo",
        "medio": "i_t ≈ 0.5 → aprende parcialmente",
        "alto": "i_t ≈ 1 → absorbe toda la entrada",
        "analogia": "Como un filtro de noticias: solo guardás los titulares relevantes.",
        "tip": "Trabaja en conjunto con el Candidate Gate.",
    },
    "LSTM - Output Gate": {
        "definicion": "Controla qué parte del cell state se expone como hidden state.",
        "formula": "o_t = σ(W_o · [h_{t-1}; x_t] + b_o) ; h_t = o_t ⊙ tanh(c_t)",
        "bajo": "o_t ≈ 0 → salida silenciosa",
        "medio": "o_t ≈ 0.5 → salida parcial",
        "alto": "o_t ≈ 1 → toda la memoria visible",
        "analogia": "Como el altavoz de un equipo: grabás todo, pero solo mostrás lo relevante.",
        "tip": "Si el modelo no aprende, revisá si el hidden state llega a la salida.",
    },
}

with st.sidebar:
    st.markdown("### 📖 Glosario interactivo")
    concepto = st.selectbox("Elegí un concepto:", list(GLOSARIO.keys()))
    info = GLOSARIO[concepto]
    st.markdown(f"**Definición:** {info['definicion']}")
    st.latex(info["formula"])
    st.markdown(f"🔵 **Valor bajo:** {info['bajo']}")
    st.markdown(f"🟡 **Valor medio:** {info['medio']}")
    st.markdown(f"🔴 **Valor alto:** {info['alto']}")
    st.markdown(f"💭 **Analogía:** {info['analogia']}")
    st.markdown(f"💡 **Tip:** {info['tip']}")
    st.markdown("---")
    st.markdown("[🔗 Profundizar: Understanding LSTMs (colah's blog)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)")
    
    st.markdown("---")
    st.markdown("### 🧠 Motor de IA")
    st.info("🧠 **NumPy Manual** — Backpropagation Through Time implementado desde cero (~400 líneas de código propio)")


# ============================================================
# INICIALIZACIÓN DE ESTADO
# ============================================================
for _key in ['data_df', 'model', 'scaler', 'window_size',
             'X_test', 'y_test', 'y_pred', 'train_losses',
             'dataset_name', 'target_series']:
    if _key not in st.session_state:
        st.session_state[_key] = None


# ============================================================
# TABS PRINCIPALES
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1️⃣ El problema",
    "2️⃣ ¿Para qué sirve?",
    "3️⃣ Arquitectura",
    "4️⃣ Entrenamiento",
    "5️⃣ Predicción en vivo",
    "6️⃣ Ética y límites",
])

# ------------------------------------------------------------
# TAB 1 — EL PROBLEMA
# ------------------------------------------------------------
with tab1:
    st.markdown('<p class="irsti-titulo">IRSTI-AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="irsti-subtitulo">SISTEMA PREDICTIVO DE RIESGO CON LSTM</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        ### 🏥 El problema en la UCI
        
        En una unidad de cuidados intensivos, el deterioro de un paciente no siempre
        es evidente a simple vista. Los signos vitales cambian de forma sutil
        durante varias horas antes de que ocurra un evento crítico.

        **¿Y si pudiéramos anticiparlo?**

        Esa es la pregunta que motiva este proyecto: usar una red neuronal con
        memoria (LSTM) para aprender de la **secuencia histórica de signos vitales**
        y predecir el riesgo del paciente **horas antes** de que se vuelva crítico.
        """)
        st.info("💡 **Analogía:** Como un médico experimentado que nota patrones sutiles en la evolución del paciente, la LSTM aprende a reconocer secuencias que preceden al deterioro.")
    
    with col2:
        st.markdown("""
        ### 📊 ¿Qué datos usamos?
        
        Generamos **datos sintéticos realistas** de 5 signos vitales:
        
        | Signo vital | Rango normal | Alerta |
        |-------------|--------------|--------|
        | Frecuencia cardíaca | 60-100 lpm | >120 o <50 |
        | Saturación O₂ | 95-100% | <90% |
        | Frecuencia respiratoria | 12-20 rpm | >25 o <8 |
        | Presión arterial media | 70-100 mmHg | <65 |
        | Temperatura | 36.5-37.5°C | >38°C o <36°C |
        
        El modelo predice un **score de riesgo** (0-1) que combina estos signos.
        """)
    
    st.markdown("---")
    st.markdown("""
    <div class="irsti-card">
    🔑 <b>Idea central:</b> Una LSTM no solo mira el estado actual del paciente — 
    también recuerda cómo evolucionó en las últimas horas. Esa memoria es lo que 
    permite anticipar el deterioro antes de que sea evidente.
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------
# TAB 2 — ¿PARA QUÉ SIRVE?
# ------------------------------------------------------------
with tab2:
    st.header("¿Para qué sirve una red neuronal como esta?")
    st.markdown("""
    Una red neuronal puede usarse principalmente para dos tipos de tareas:
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="irsti-card" style="border-left: 4px solid #02C39A;">
            <h3>🏷️ Clasificación</h3>
            <p>Asignar una <b>categoría</b> a partir de los datos de entrada.</p>
            <ul>
                <li><b>Simple (binaria):</b> ¿el mail es spam o no? ¿el paciente está en riesgo o no?</li>
                <li><b>Múltiple:</b> ¿el equipo gana, pierde o empata? ¿el riesgo es bajo, medio o alto?</li>
            </ul>
            <p><b>En este proyecto:</b> clasificamos el riesgo futuro del paciente en 3 clases (bajo / medio / alto) según el score predicho.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="irsti-card" style="border-left: 4px solid #E63946;">
            <h3>📈 Predicción (regresión)</h3>
            <p>Estimar un <b>valor continuo</b> a futuro.</p>
            <ul>
                <li>¿Qué temperatura hará mañana?</li>
                <li>¿Cuál será el próximo valor de saturación de oxígeno?</li>
                <li>¿Cuánto riesgo tendrá el paciente en 6 horas?</li>
            </ul>
            <p><b>En este proyecto:</b> predecimos el valor exacto del score de riesgo a futuro, no solo la categoría.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    👉 Usamos una **LSTM** porque el riesgo no depende solo del dato actual,
    sino de **cómo evolucionaron los signos vitales** en las últimas horas.
    
    La LSTM aprende a reconocer **patrones temporales** como:
    - 📈 Taquicardia progresiva + caída de saturación → sepsis
    - 📉 Hipotensión sostenida → shock
    - 📊 Bradicardia + hipoxia → fallo cardíaco
    """)


# ------------------------------------------------------------
# TAB 3 — ARQUITECTURA
# ------------------------------------------------------------
with tab3:
    st.header("Arquitectura de la red LSTM")

    # ── EJEMPLO GUIADO (ancho completo, dentro de recuadro) ──
    st.markdown("""
    <div style="
        background-color: #FFFFFF;
        padding: 25px 30px;
        border-radius: 12px;
        border: 1px solid #E0E0E0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 25px;
    ">
        <h3 style="color: #0A2540; margin-top: 0;">📐 Ejemplo guiado: ¿cómo decide la memoria de una LSTM?</h3>
        <p style="color: #333333;">
        Pensemos la memoria de la LSTM como <b>una libreta de apuntes de un enfermero</b>
        que va anotando el estado del paciente turno tras turno.
        </p>
        <p style="color: #333333;">
        Cada vez que llega un dato nuevo (un signo vital), el enfermero hace 3 cosas
        antes de escribir en la libreta:
        </p>
        <p style="color: #333333;">
        <b>1. 🗑️ Decide qué tachar de lo anotado antes</b><br>
        Si algo de lo que escribió hace 2 horas ya no es relevante, lo tacha.<br>
        <i>(Esto es el "Forget Gate": un número entre 0 y 1 — 0 = "tachá todo", 1 = "no toques nada")</i>
        </p>
        <p style="color: #333333;">
        En nuestro ejemplo, el enfermero decide conservar el <b>66% de lo que tenía anotado</b>.
        </p>
        <p style="color: #333333;">
        <b>2. ✍️ Decide cuánto del dato nuevo anotar</b><br>
        No todo dato nuevo merece la misma atención. Si la saturación bajó un poquito,
        anota poco; si bajó mucho, anota fuerte.<br>
        <i>(Esto es el "Input Gate")</i>
        </p>
        <p style="color: #333333;">
        En nuestro ejemplo, el enfermero decide que el dato nuevo merece un <b>55% de atención</b>.
        </p>
        <p style="color: #333333;">
        <b>3. 📖 Decide qué mostrarle al médico</b><br>
        La libreta completa del enfermero (memoria interna) no siempre se muestra entera;
        a veces resume lo importante para el informe que lee el médico.<br>
        <i>(Esto es el "Output Gate")</i>
        </p>
        <p style="color: #333333;">
        <b>🎯 Resultado del ejemplo:</b> después de tachar un poco, anotar lo nuevo y resumir,
        la "opinión" de la LSTM sobre el paciente pasó de un valor de memoria <code>0.1 / 0.4</code>
        a uno nuevo de <code>0.22 / 0.42</code> — un cambio chico pero real, como cuando un enfermero
        ajusta su evaluación al ver un dato nuevo, sin tirar todo lo que ya sabía.
        </p>
        <blockquote style="color: #333333; border-left: 4px solid #02C39A; padding-left: 15px; margin: 10px 0;">
        💡 La idea central: la LSTM <b>no empieza de cero en cada dato nuevo</b>. Combina
        lo que ya sabía con lo que acaba de ver, decidiendo cuánto peso le da a cada cosa.
        Por eso puede "recordar" tendencias de varias horas, a diferencia de un modelo
        que solo mira el dato del instante actual.
        </blockquote>
    </div>
    """, unsafe_allow_html=True)

    # ── COLUMNAS PARA EL RESTO DEL CONTENIDO ──
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        ### 🧬 ¿Cómo funciona una LSTM?
        
        La LSTM (Long Short-Term Memory) tiene **puertas** que deciden qué
        información recordar, olvidar y pasar hacia adelante.
        
        | Puerta | Símbolo | Analogía |
        |--------|---------|----------|
        | Forget Gate | f_t | 🗑️ Tachar lo que no sirve |
        | Input Gate | i_t | ✍️ Anotar lo nuevo |
        | Candidate | ĉ_t | 📝 Propuesta de apunte |
        | Output Gate | o_t | 📖 Resumir para el médico |
        
        ### 🔬 ¿Qué activación usar?
        
        Las LSTM usan **dos funciones de activación**:
        
        - **Sigmoide (σ):** se usa en las **puertas** (forget, input, output)
          porque comprime valores a (0,1) y actúa como "válvula".
        
        - **Tangente hiperbólica (tanh):** se usa en el **candidate gate**
          y en la salida porque comprime a (-1,1), permitiendo valores
          negativos que ayudan a "restar" información de la memoria.
        
        > 💡 **Regla práctica:** 
        > - **Redes grandes** (muchas capas o unidades) → **tanh** converge más rápido, aunque cada paso sea más costoso.
        > - **Redes chicas** (1-2 capas) → **sigmoide** alcanza y es más económica.
        
        ### 📦 Cell State vs Hidden State
        
        - **Cell State (c_t):** la memoria de largo plazo. Viaja casi sin cambios.
          Es como un papel donde escribís y borrás selectivamente.
        - **Hidden State (h_t):** la memoria de corto plazo. Es la salida visible
          de la celda y se recalcula en cada paso.
        
        """)

    with col2:
        st.markdown("### 🖼️ Diagrama de la celda LSTM (con analogía)")
    
        # Diagrama de celda LSTM MEJORADO
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor(COLOR_BG_CARD)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 9)
        ax.axis('off')
        ax.set_facecolor(COLOR_BG_CARD)

        # ── TÍTULO ──
        ax.text(5, 8.5, "Celda LSTM — como un enfermero con libreta", 
                ha='center', fontsize=13, fontweight='bold', color=COLOR_TEXT)

        # ── CELL STATE (línea superior) ──
        ax.annotate("", xy=(9.2, 6.8), xytext=(0.5, 6.8),
                    arrowprops=dict(arrowstyle="->", color='#cc6600', lw=3))
        ax.text(4.8, 7.3, "📋 Cell State (libreta)", ha='center', fontsize=10,
                fontweight='bold', color='#cc6600')

        # ── PUERTAS CON NOMBRES Y EMOJIS ──
        from matplotlib.patches import FancyBboxPatch

        gate_data = [
            (1.5, 4.8, 'σ', '🗑️ Forget\nGate', '#ffaaaa', '#cc0000'),
            (3.8, 4.8, 'σ', '✍️ Input\nGate', '#aaaaff', '#0000cc'),
            (6.0, 4.8, 'tanh', '📝 Candidate', '#aaffaa', '#006600'),
            (8.2, 4.8, 'σ', '📖 Output\nGate', '#ffeeaa', '#cc8800'),
        ]
        
        analogias = ["¿qué tacho?", "¿qué anoto?", "¿qué propongo?", "¿qué resumo?"]
        
        for i, (gx, gy, label, name, facecolor, edgecolor) in enumerate(gate_data):
            ax.add_patch(FancyBboxPatch((gx-0.55, gy-0.45), 1.1, 0.9,
                        boxstyle="round,pad=0.05", facecolor=facecolor,
                        edgecolor=edgecolor, linewidth=2, zorder=3))
            ax.text(gx, gy, label, ha='center', va='center',
                    fontsize=12, fontweight='bold', color=edgecolor)
            ax.text(gx, gy - 0.9, name, ha='center', va='top', fontsize=7, 
                    color=COLOR_TEXT_MUTED)
            ax.text(gx, gy + 0.8, analogias[i], ha='center', va='bottom', 
                    fontsize=7, fontstyle='italic', color='#aaaaaa')

        # ── OPERACIONES EN CELL STATE (CORREGIDO) ──
        # ✅ Ahora cada tupla tiene (x, operador)
        for x_pos, op in [(2.5, '×'), (5.2, '+'), (7.2, '×')]:
            ax.text(x_pos, 6.8, op, ha='center', va='center', fontsize=20,
                    fontweight='bold', color='#555', zorder=4,
                    bbox=dict(boxstyle='circle', facecolor='white', edgecolor='#999'))

        # ── FLECHAS DE PUERTAS A CELL STATE ──
        for src_x, dst_x in [(1.5, 2.5), (3.8, 5.2), (6.0, 5.2)]:
            ax.annotate("", xy=(dst_x, 6.5), xytext=(src_x, 5.2),
                        arrowprops=dict(arrowstyle="->", color='#888', lw=1.5))
        ax.annotate("", xy=(7.2, 6.5), xytext=(8.2, 5.2),
                    arrowprops=dict(arrowstyle="->", color='#888', lw=1.5))

        # ── TANH ANTES DEL OUTPUT ──
        ax.add_patch(FancyBboxPatch((6.55, 6.3), 0.9, 0.75,
                    boxstyle="round,pad=0.05", facecolor='#ddffdd',
                    edgecolor='#006600', linewidth=1.5, zorder=3))
        ax.text(7.0, 6.67, "tanh", ha='center', va='center', fontsize=8, color='#006600')

        # ── HIDDEN STATE ──
        ax.annotate("", xy=(9.2, 2.8), xytext=(0.5, 2.8),
                    arrowprops=dict(arrowstyle="->", color='#0066cc', lw=2.5))
        ax.text(0.1, 2.8, "h_{t-1}\n(memo anterior)", ha='right', va='center', fontsize=8,
                fontweight='bold', color='#0066cc')
        ax.text(9.3, 2.8, "h_t\n(resumen nuevo)", ha='left', va='center', fontsize=8,
                fontweight='bold', color='#0066cc')

        # ── ENTRADA ──
        ax.annotate("", xy=(4.8, 3.8), xytext=(4.8, 1.8),
                    arrowprops=dict(arrowstyle="->", color='#333', lw=1.5))
        ax.text(4.8, 1.5, "x_t (dato nuevo del paciente)", ha='center', fontsize=8,
                fontweight='bold', color=COLOR_TEXT)

        # ── LEYENDA ──
        ax.text(0.5, 0.3, "🔴 Forget: decide qué olvidar", fontsize=7, color=COLOR_TEXT_MUTED)
        ax.text(0.5, 0.0, "🔵 Input: decide qué guardar", fontsize=7, color=COLOR_TEXT_MUTED)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        ### Ventaja sobre RNN simple
        st.markdown("""
        ### Ventaja sobre RNN simple
        Las RNN simples sufren **vanishing gradient**: el error se desvanece al
        retropropagarse por muchos pasos. Las LSTM lo resuelven porque el cell
        state tiene un camino casi directo (la línea horizontal que ves en el diagrama).
        """)

with tab4:
    st.header("Entrenamiento del modelo")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown("### ⚙️ Configuración")

        dataset_opcion = st.selectbox(
            "👤 Elegí un perfil de paciente:",
            [
                "🟢 Paciente Estable",
                "🟡 Deterioro Progresivo",
                "🔴 Crisis Repentina (ej. sepsis)",
                "📈 Mejoría post-tratamiento",
                "🎯 Combinado (todos los patrones)"
            ]
        )

        st.caption("💡 Cada perfil genera una secuencia diferente de signos vitales. El modelo aprende a predecir el riesgo a futuro.")

        if st.button("🔄 Generar Dataset", use_container_width=True):
            tipo_map = {
                "🟢 Paciente Estable": "estable",
                "🟡 Deterioro Progresivo": "deterioro",
                "🔴 Crisis Repentina (ej. sepsis)": "critico_repentino",
                "📈 Mejoría post-tratamiento": "mejoria",
                "🎯 Combinado (todos los patrones)": "combinado"
            }

            t, df, etiqueta = generar_dataset_irsti(tipo_map[dataset_opcion], n=300)

            st.session_state.data_df = df
            st.session_state.dataset_name = dataset_opcion
            st.session_state.model = None

            st.success(f"✅ Dataset generado: {dataset_opcion}")

            # Mostrar gráfico de signos vitales
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['tiempo'], y=df['frecuencia_cardiaca'],
                                        mode='lines', name='FC (lpm)',
                                        line=dict(color='#E63946')))
                fig.add_trace(go.Scatter(x=df['tiempo'], y=df['saturacion_o2'],
                                        mode='lines', name='SpO₂ (%)',
                                        line=dict(color='#02C39A')))
                fig.add_trace(go.Scatter(x=df['tiempo'], y=df['frecuencia_respiratoria'],
                                        mode='lines', name='FR (rpm)',
                                        line=dict(color='#F4B400')))
                fig.add_trace(go.Scatter(x=df['tiempo'], y=df['presion_arterial'],
                                        mode='lines', name='PAM (mmHg)',
                                        line=dict(color='#8FA3B5')))
                fig.add_trace(go.Scatter(x=df['tiempo'], y=df['riesgo'] * 100,
                                        mode='lines', name='Riesgo (%)',
                                        line=dict(color='#E63946', dash='dash', width=3)))
                fig.update_layout(
                    title=f"Evolución de signos vitales - {dataset_opcion}",
                    height=400,
                    plot_bgcolor=COLOR_BG_CARD,
                    paper_bgcolor=COLOR_BG_CARD,
                    font_color=COLOR_TEXT,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02)
                )
                st.plotly_chart(fig, use_container_width=True, key="vital_signs_chart")

        if st.session_state.data_df is not None:
            st.markdown("---")
            st.markdown("### 🎯 Hiperparámetros")

            # ── TAMAÑO DE VENTANA ──
            st.markdown("#### 📊 Tamaño de ventana (pasos atrás)")
            window_size = st.slider(
                "¿Cuántos pasos temporales usa el modelo para predecir el siguiente?",
                5, 30, 10,
                key="window_size_slider",
                help="El modelo mira esta cantidad de datos históricos para hacer cada predicción."
            )

            if window_size <= 8:
                st.info("🔍 **Ventana corta** — El modelo mira pocos pasos atrás. Aprende patrones rápidos pero puede perder tendencias largas.")
                st.caption("💡 Recomendación: Si ves que el modelo no capta tendencias, aumentá a 15-20.")
            elif window_size <= 15:
                st.success("✅ **Ventana media** — Buen balance. El modelo capta patrones a corto y mediano plazo.")
                st.caption("👍 Esta es una buena configuración para empezar.")
            else:
                st.warning("⏳ **Ventana larga** — El modelo mira muchos pasos atrás. Más lento pero capta tendencias largas.")
                st.caption("💡 Recomendación: Usá solo si tenés muchos datos. Si es lento, bajá a 15.")

            # ── UNIDADES LSTM ──
            st.markdown("#### 🧠 Unidades LSTM")
            hidden_units = st.slider(
                "¿Cuántas neuronas tiene la capa oculta?",
                16, 128, 64, step=16,
                key="hidden_units_slider",
                help="Más unidades = más capacidad de aprendizaje, pero más lento."
            )

            if hidden_units <= 32:
                st.info("🧩 **Red pequeña** — Rápida de entrenar, pero con menos capacidad para aprender patrones complejos.")
                st.caption("💡 Recomendación: Usá para datos simples. Si no aprende, aumentá a 64 o 128.")
            elif hidden_units <= 64:
                st.success("⚖️ **Red mediana** — Buen balance entre velocidad y capacidad de aprendizaje.")
                st.caption("👍 Esta es una buena configuración para empezar.")
            else:
                st.warning("🚀 **Red grande** — Más capacidad, pero más lenta y con riesgo de sobreajuste.")
                st.caption("💡 Recomendación: Usá solo con datos complejos. Si ves sobreajuste, bajá a 64.")

            # ── ÉPOCAS ──
            st.markdown("#### 🔄 Épocas")
            epochs = st.slider(
                "¿Cuántas veces el modelo ve todo el dataset?",
                20, 300, 80, step=10,
                key="epochs_slider",
                help="Cada época es un ciclo completo de entrenamiento sobre todos los datos."
            )

            if epochs <= 40:
                st.info("⚡ **Pocas épocas** — Entrenamiento rápido. El modelo puede no aprender lo suficiente.")
                st.caption("💡 Recomendación: Usá para pruebas rápidas. Para resultados serios, aumentá a 150-200.")
            elif epochs <= 100:
                st.success("✅ **Épocas moderadas** — Buen punto para aprender sin sobreajustarse.")
                st.caption("👍 Esta es una buena configuración para empezar.")
            elif epochs <= 180:
                st.success("✅ **Muchas épocas** — El modelo tiene tiempo para aprender patrones complejos.")
                st.caption("💡 Recomendación: Mirá la curva de pérdida para saber cuándo detenerte.")
            else:
                st.warning("⏳ **Muchísimas épocas** — Puede sobreajustarse. Mirá la curva de pérdida.")
                st.caption("💡 Recomendación: Si ves que la pérdida sube en test, detené el entrenamiento antes.")

            # ── TASA DE APRENDIZAJE ──
            st.markdown("#### 📈 Tasa de aprendizaje")
            learning_rate = st.select_slider(
                "¿Qué tan rápido aprende el modelo?",
                [0.0001, 0.0005, 0.001, 0.005, 0.01],
                value=0.001,
                key="lr_slider",
                help="Controla cuánto se ajustan los pesos en cada paso. Muy alto → inestable, muy bajo → lento."
            )

            if learning_rate <= 0.0005:
                st.info("🐢 **Tasa baja** — Aprendizaje lento pero estable. El modelo converge de forma segura.")
                st.caption("💡 Recomendación: Usá si la pérdida oscila mucho. Si es muy lento, probá con 0.001.")
            elif learning_rate == 0.001:
                st.success("✅ **Tasa media** — El punto dulce. Aprendizaje balanceado para la mayoría de los casos.")
                st.caption("👍 Esta es la configuración recomendada para empezar.")
            elif learning_rate <= 0.005:
                st.success("✅ **Tasa alta** — Aprendizaje rápido. Ideal si el modelo es lento.")
                st.caption("💡 Recomendación: Si ves oscilaciones, bajá a 0.001.")
            else:
                st.warning("⚡ **Tasa muy alta** — Aprendizaje rápido pero riesgoso.")
                st.caption("⚠️ Recomendación: Solo usá si el modelo no aprende con tasas más bajas. Vigilá que no diverja.")

            # ── TAMAÑO DE TEST ──
            st.markdown("#### 📊 Tamaño de test")
            test_size = st.slider(
                "¿Qué porcentaje de datos reservar para evaluar?",
                10, 40, 30, 5,
                key="test_size_slider",
                help="Porcentaje de datos que NO se usan para entrenar, solo para evaluar."
            )

            if test_size <= 15:
                st.info("📊 **Poco test** — Más datos para entrenar, pero la evaluación puede no ser representativa.")
                st.caption("💡 Recomendación: Si la evaluación es muy diferente al entrenamiento, aumentá el test.")
            elif test_size <= 30:
                st.success("✅ **Test moderado** — Buen balance. Recomendado para empezar.")
                st.caption("👍 Esta es una buena configuración para empezar.")
            else:
                st.warning("📊 **Mucho test** — Menos datos para entrenar, pero la evaluación es más confiable.")
                st.caption("💡 Recomendación: Usá solo si tenés muchos datos (>500 muestras).")

            # ── MINI-BATCH ──
            st.markdown("#### 📦 Tamaño de mini-batch")
            batch_size = st.select_slider(
                "¿Cuántas muestras se procesan antes de actualizar los pesos?",
                [4, 8, 16, 32, 64, 128],
                value=16,
                key="batch_slider",
                help="Batch pequeño = actualizaciones frecuentes y ruidosas. Batch grande = actualizaciones estables pero lentas."
            )

            if batch_size <= 8:
                st.info("⚡ **Batch pequeño** — Actualizaciones frecuentes y ruidosas. Converge rápido pero inestable.")
                st.caption("💡 Recomendación: Usá con datasets pequeños. Si ves ruido, aumentá a 16 o 32.")
            elif batch_size <= 32:
                st.success("✅ **Batch mediano** — Buen balance entre velocidad y estabilidad. Recomendado.")
                st.caption("👍 Esta es una buena configuración para empezar.")
            else:
                st.warning("🐢 **Batch grande** — Actualizaciones estables pero lentas. Requiere más memoria.")
                st.caption("💡 Recomendación: Usá solo con datasets grandes. Si es lento, bajá a 32.")

            # ── BOTÓN DE ENTRENAMIENTO ──
            st.markdown("---")
            train_button = st.button("🚀 Entrenar Modelo", use_container_width=True, type="primary")
            
        else:
            st.info("👆 Generá un dataset primero para empezar")
            train_button = False

    with col2:
        if st.session_state.data_df is not None:
            st.markdown("### 📊 Resultados del entrenamiento")
            stats_placeholder = st.empty()
            loss_placeholder = st.empty()
            pred_placeholder = st.empty()

    # ============================================================
    # 🚀 ENTRENAMIENTO
    # ============================================================
    if train_button and st.session_state.data_df is not None:
        with st.spinner("🧠 Entrenando LSTM con NumPy"):
            df = st.session_state.data_df
            data = df['riesgo'].values

            # Normalizar
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()

            # Crear secuencias
            X, y = [], []
            for i in range(len(data_scaled) - window_size):
                X.append(data_scaled[i:i+window_size])
                y.append(data_scaled[i+window_size])
            X = np.array(X)
            y = np.array(y)

            # Reshape para LSTM: (batch, timesteps, features)
            X = X.reshape(X.shape[0], X.shape[1], 1)

            # Train/Test split
            split_idx = int(len(X) * (1 - test_size/100))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            st.info(f"📊 {len(X_train)} muestras entrenamiento, {len(X_test)} muestras test")

            # Crear y entrenar modelo NumPy
            model = NumpyLSTM(input_size=1, hidden_size=hidden_units, output_size=1)
            history = model.train(
                X_train, y_train,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                clip=5.0,
                verbose=False
            )
            
            # ============================================================
            # 🔧 PREDICCIÓN CON CLIP (CORRECCIÓN DE VALORES FUERA DE RANGO)
            # ============================================================
            y_pred = model.predict(X_test).flatten()

            # ✅ FIX: Forzar predicciones a [0,1] (por si la LSTM da valores negativos)
            if y_pred.min() < 0 or y_pred.max() > 1:
                print(f"⚠️ Predicciones fuera de rango: min={y_pred.min():.4f}, max={y_pred.max():.4f}")
                y_pred = np.clip(y_pred, 0, 1)
                print(f"✅ Corregido: min={y_pred.min():.4f}, max={y_pred.max():.4f}")

            # Guardar en sesión
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.window_size = window_size
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred
            st.session_state.train_losses = history
            st.session_state.target_series = data_scaled

            # ============================================================
            # 📊 MÉTRICAS (CON ARRAYS 1D)
            # ============================================================
            
            # Asegurar arrays 1D
            y_train_flat = y_train.flatten() if len(y_train.shape) > 1 else y_train
            y_test_flat = y_test.flatten() if len(y_test.shape) > 1 else y_test
            y_pred_flat = y_pred.flatten() if len(y_pred.shape) > 1 else y_pred

            # Métricas
            test_mse = mean_squared_error(y_test_flat, y_pred_flat)
            test_mae = mean_absolute_error(y_test_flat, y_pred_flat)
            test_r2 = r2_score(y_test_flat, y_pred_flat)

                        # ============================================================
            # 📊 EVALUACIÓN DE MEJORA CON MENSAJE CUALITATIVO
            # ============================================================

            baseline_pred = baseline_persistencia_corregida(y_train_flat, y_test_flat)
            baseline_mse = mean_squared_error(y_test_flat, baseline_pred)

            # 📊 Evaluación cualitativa
            if baseline_mse < 1e-6:
                mejora_texto = "📊 Baseline casi perfecto"
                mejora_valor = 0
            elif test_mse < baseline_mse:
                mejora = (baseline_mse - test_mse) / baseline_mse * 100
                if mejora > 100:
                    mejora_texto = f"✅ {mejora:.1f}% mejor"
                else:
                    mejora_texto = f"✅ {mejora:.1f}% mejor"
                mejora_valor = mejora
            else:
                mejora = (test_mse - baseline_mse) / baseline_mse * 100
                if mejora > 1000:
                    # Si el porcentaje es absurdo, mostrar mensaje cualitativo
                    mejora_texto = "📊 Comparación no informativa (baseline muy cercano)"
                elif mejora > 100:
                    mejora_texto = f"⚠️ {mejora:.1f}% PEOR (baseline muy cercano)"
                else:
                    mejora_texto = f"⚠️ {mejora:.1f}% PEOR que baseline"
                mejora_valor = mejora

            # 🔍 DIAGNÓSTICO (eliminar después de verificar)
            print("=" * 60)
            print("📊 DIAGNÓSTICO DEL BASELINE")
            print(f"test_mse: {test_mse:.8f}")
            print(f"baseline_mse: {baseline_mse:.8f}")
            print(f"mejora_texto: {mejora_texto}")
            print(f"y_train_flat[-5:]: {y_train_flat[-5:]}")
            print(f"y_test_flat[:10]: {y_test_flat[:10]}")
            print(f"baseline_pred[:10]: {baseline_pred[:10]}")
            print("=" * 60)

            # Mostrar estadísticas
            with stats_placeholder.container():
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Pérdida inicial", f"{history[0]:.6f}")
                col_b.metric("Pérdida final", f"{history[-1]:.6f}")
                col_c.metric("MSE Test", f"{test_mse:.6f}")
                col_d.metric("vs Baseline", mejora_texto)

            # ============================================================
            # 📉 CURVA DE PÉRDIDA
            # ============================================================
            with loss_placeholder.container():
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    y=history,
                    mode='lines',
                    name='Pérdida (MSE)',
                    line=dict(color='#02C39A', width=2)
                ))
                fig_loss.update_layout(
                    title='Curva de pérdida durante el entrenamiento',
                    xaxis_title='Época',
                    yaxis_title='Error (MSE)',
                    height=300,
                    plot_bgcolor=COLOR_BG_CARD,
                    paper_bgcolor=COLOR_BG_CARD,
                    font_color=COLOR_TEXT,
                )
                st.plotly_chart(fig_loss, use_container_width=True, key="grafico_curva_perdida")

            # ============================================================
            # 📈 COMPARATIVA LSTM vs REAL (DESNORMALIZAR SOLO PARA GRAFICAR)
            # ============================================================
            with pred_placeholder.container():
                y_test_real = scaler.inverse_transform(y_test_flat.reshape(-1, 1)).flatten()
                y_pred_real = scaler.inverse_transform(y_pred_flat.reshape(-1, 1)).flatten()

                n_show = min(80, len(y_test_real))
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(
                    x=np.arange(n_show),
                    y=y_test_real[:n_show],
                    mode='lines',
                    name='Riesgo real',
                    line=dict(color='#E63946', width=2)
                ))
                fig_comp.add_trace(go.Scatter(
                    x=np.arange(n_show),
                    y=y_pred_real[:n_show],
                    mode='lines',
                    name='Predicción LSTM',
                    line=dict(color='#02C39A', width=2)
                ))
                fig_comp.update_layout(
                    title='Predicción del riesgo (test)',
                    xaxis_title='Pasos temporales',
                    yaxis_title='Riesgo',
                    height=300,
                    plot_bgcolor=COLOR_BG_CARD,
                    paper_bgcolor=COLOR_BG_CARD,
                    font_color=COLOR_TEXT,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02)
                )
                st.plotly_chart(fig_comp, use_container_width=True, key="grafico_prediccion_vs_real")

            # ============================================================
            # ✅ MENSAJE DE ÉXITO
            # ============================================================
            st.success(f"✅ Entrenamiento completado con NumPy - {epochs} épocas")

            # ============================================================
            # 🔍 ANÁLISIS Y RECOMENDACIONES POST-ENTRENAMIENTO
            # ============================================================
            st.markdown("---")
            st.markdown("### 🔍 Análisis del entrenamiento")

            # Calcular métricas de diagnóstico
            reduccion = (history[0] - history[-1]) / history[0] * 100
            es_baseline_mejor = "mejor" in mejora_texto
            r2_aceptable = test_r2 > 0.3

            # ── EVALUACIÓN DEL ENTRENAMIENTO ──
            if reduccion > 70 and es_baseline_mejor and r2_aceptable:
                estado = "EXCELENTE"
                color_estado = COLOR_VERDE
                emoji = "🏆"
                descripcion = "El modelo está aprendiendo muy bien. ¡Excelente configuración!"
                recomendaciones = """
                ✅ **Lo que hiciste bien:**
                - La pérdida bajó significativamente
                - El modelo supera al baseline
                - El R² es aceptable
                
                💡 **Para seguir mejorando:**
                - Probá con más épocas (200-300) para ver si la pérdida sigue bajando
                - Ajustá la tasa de aprendizaje a 0.001 si ves oscilaciones
                - Probá con otros perfiles de paciente para comparar
                """
            elif reduccion > 40 and es_baseline_mejor:
                estado = "BUENO"
                color_estado = COLOR_AMARILLO
                emoji = "👍"
                descripcion = "El modelo está aprendiendo, pero se puede mejorar."
                recomendaciones = """
                💡 **Recomendaciones para mejorar:**
                - Aumentá las **épocas** a 150-200 (el modelo necesita más tiempo)
                - Probá con **tasa de aprendizaje** de 0.005 si ves que la pérdida se estanca
                - Aumentá las **unidades LSTM** a 128 para más capacidad
                """
            elif reduccion > 20:
                estado = "REGULAR"
                color_estado = COLOR_AMARILLO
                emoji = "⚠️"
                descripcion = "El modelo está aprendiendo lentamente."
                recomendaciones = """
                🔧 **Recomendaciones para mejorar:**
                - Aumentá las **épocas** a 200-300
                - Subí la **tasa de aprendizaje** a 0.005 o 0.01
                - Aumentá las **unidades LSTM** a 128
                - Probá con un perfil de paciente con **menos ruido** (ej. "Deterioro Progresivo")
                - Aumentá el **tamaño de ventana** a 15-20
                """
            else:
                estado = "MALO"
                color_estado = COLOR_ROJO
                emoji = "❌"
                descripcion = "El modelo no está aprendiendo correctamente."
                recomendaciones = """
                🚨 **El modelo necesita cambios urgentes:**
                
                1. **Aumentá las épocas** a 200-300
                2. **Subí la tasa de aprendizaje** a 0.005 o 0.01
                3. **Aumentá las unidades LSTM** a 128
                4. **Probá con otro perfil de paciente** con menos ruido
                5. **Aumentá el tamaño de ventana** a 15-20
                6. **Reducí el ruido en los datos** (generá un dataset con menos ruido)
                
                💡 Si el problema persiste, puede que los datos sean demasiado ruidosos
                para que la LSTM encuentre un patrón claro.
                """

            # ── MOSTRAR DIAGNÓSTICO ──
            col_diag1, col_diag2, col_diag3 = st.columns(3)

            with col_diag1:
                if reduccion > 70:
                    st.success(f"✅ Pérdida bajó {reduccion:.1f}%")
                elif reduccion > 40:
                    st.warning(f"⚠️ Pérdida bajó {reduccion:.1f}%")
                else:
                    st.error(f"❌ Pérdida bajó solo {reduccion:.1f}%")

            with col_diag2:
                if "mejor" in mejora_texto:
                    st.success(f"✅ {mejora_texto}")
                else:
                    st.error(f"❌ {mejora_texto}")

            with col_diag3:
                if test_r2 > 0.5:
                    st.success(f"✅ R² = {test_r2:.3f}")
                elif test_r2 > 0.2:
                    st.warning(f"⚠️ R² = {test_r2:.3f}")
                else:
                    st.error(f"❌ R² = {test_r2:.3f}")

            # ── TARJETA DE DIAGNÓSTICO ──
            if estado == "EXCELENTE":
                text_color = "#1B5E20"
            elif estado == "BUENO":
                text_color = "#E65100"
            elif estado == "REGULAR":
                text_color = "#BF360C"
            else:
                text_color = "#B71C1C"

            st.markdown(f"""
            <div style="
                background-color: #FFFFFF;
                padding: 20px 25px;
                border-radius: 10px;
                border-left: 5px solid {color_estado};
                margin: 10px 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            ">
                <h4 style="color: {text_color}; margin-top: 0; font-size: 1.2rem;">{emoji} Estado del entrenamiento: {estado}</h4>
                <p style="color: #333333; margin-bottom: 0;">{descripcion}</p>
            </div>
            """, unsafe_allow_html=True)

            # ── RECOMENDACIONES ──
            with st.expander("💡 Ver recomendaciones para mejorar", expanded=(estado != "EXCELENTE")):
                st.markdown(recomendaciones)
                
                st.markdown("""
                ### 📊 Guía rápida de hiperparámetros
                
                | Parámetro | Si el modelo... | Probá con... |
                |-----------|-----------------|--------------|
                | **Épocas** | No baja la pérdida | Aumentar a 150-300 |
                | **Tasa de aprendizaje** | Se estanca o oscila | 0.001 → 0.005 → 0.01 |
                | **Unidades LSTM** | No aprende patrones | 64 → 128 → 256 |
                | **Ventana** | Pierde tendencias largas | 10 → 15 → 20 |
                | **Mini-batch** | Es muy ruidoso | 16 → 32 → 64 |
                """)

            # ── RECOMENDACIONES POST-ENTRENAMIENTO ──
            st.markdown("---")
            st.markdown("### 🎯 ¿Qué hacer después de entrenar?")

            if estado == "EXCELENTE":
                st.markdown("""
                ✅ **El modelo está listo para usar.**
                
                1. Ir a la **pestaña 'Predicción en vivo'** para simular signos vitales y ver el riesgo predicho
                2. Probar con otro **perfil de paciente** para comparar resultados
                3. Ajustar los **hiperparámetros** para ver si se puede mejorar aún más
                
                💡 **Desafío:** ¿Podés lograr una pérdida final menor a 0.03?
                """)
            else:
                st.markdown(f"""
                {emoji} **El modelo necesita ajustes.**
                
                1. Revisá las **recomendaciones arriba** y ajustá los hiperparámetros
                2. Probá con otro **perfil de paciente** (ej. "Deterioro Progresivo" suele ser más fácil)
                3. Aumentá el **tamaño del dataset** (podés modificar el código para generar más datos)
                4. Si el problema persiste, verificá que los datos no tengan **demasiado ruido**
                
                🔧 **Configuración recomendada para empezar:**
                - Perfil: "Deterioro Progresivo"
                - Ventana: 15-20
                - Unidades LSTM: 128
                - Épocas: 150-200
                - Tasa de aprendizaje: 0.005
                - Mini-batch: 32
                """)
    
# ------------------------------------------------------------
# TAB 5 — PREDICCIÓN EN VIVO + EXPLICABILIDAD (unificado)
# ------------------------------------------------------------
with tab5:
    # Verificar si hay modelo entrenado
    if st.session_state.model is None:
        st.warning("⚠️ Primero entrená un modelo en la pestaña 'Entrenamiento'")
    else:
        st.markdown("""
        <div class="irsti-card" style="text-align:center; border: 1px solid #E63946;">
            <span class="irsti-titulo" style="font-size: 2rem;">IRSTI-AI 🚦</span><br>
            <span class="irsti-subtitulo" style="color:#E63946;">PREDICCIÓN DE RIESGO EN VIVO</span>
        </div>
        """, unsafe_allow_html=True)
        
        # ── SIGNOS VITALES (5 variables) ──
        colA, colB, colC = st.columns(3)
        with colA:
            fc = st.slider("Frecuencia cardíaca (lpm)", 40, 180, 92, key="fc_pred")
        with colB:
            spo2 = st.slider("Saturación O₂ (%)", 70, 100, 91, key="spo2_pred")
        with colC:
            fr = st.slider("Frecuencia respiratoria (rpm)", 8, 45, 24, key="fr_pred")

        colD, colE = st.columns(2)
        with colD:
            pam = st.slider("Presión arterial media (mmHg)", 45, 110, 85, key="pam_pred")
        with colE:
            temp = st.slider("Temperatura (°C)", 36.0, 39.0, 37.0, 0.1, key="temp_pred")

        # ── CALCULAR RIESGO ──
        def calcular_riesgo(fc, spo2, fr, pam, temp):
            riesgo_fc = max(0, (fc - 100) / 60) + max(0, (60 - fc) / 60)
            riesgo_spo2 = max(0, (95 - spo2) / 15)
            riesgo_fr = max(0, (fr - 20) / 20) + max(0, (12 - fr) / 12)
            riesgo_pam = max(0, (70 - pam) / 30) + max(0, (pam - 100) / 30)
            riesgo_temp = max(0, (temp - 38) / 2) + max(0, (36.5 - temp) / 2)
            
            riesgo = (riesgo_fc * 0.25 + 
                      riesgo_spo2 * 0.35 + 
                      riesgo_fr * 0.20 + 
                      riesgo_pam * 0.15 + 
                      riesgo_temp * 0.05)
            return np.clip(riesgo, 0, 1)
        
        riesgo_simulado = calcular_riesgo(fc, spo2, fr, pam, temp)

        # ── DETERMINAR COLOR Y ESTADO ──
        if riesgo_simulado < 0.33:
            color_circulo = COLOR_VERDE
            texto_estado = "BAJO"
            texto_alerta = "✅ Paciente estable"
            clase_alerta = "alerta-verde"
        elif riesgo_simulado < 0.66:
            color_circulo = COLOR_AMARILLO
            texto_estado = "MEDIO"
            texto_alerta = "⚠️ Riesgo MEDIO — monitorear de cerca"
            clase_alerta = "alerta-amarilla"
        else:
            color_circulo = COLOR_ROJO
            texto_estado = "ALTO"
            texto_alerta = "🚨 Riesgo ALTO — ¡Revisar paciente urgente!"
            clase_alerta = "alerta-roja"

        # ── ENCABEZADO CON CÍRCULO DE COLOR ──
        st.markdown(f"""
        <div class="irsti-card" style="text-align:center; border: 1px solid {color_circulo};">
            <div style="display: flex; align-items: center; justify-content: center; gap: 15px;">
                <div style="
                    width: 60px;
                    height: 60px;
                    border-radius: 50%;
                    background-color: {color_circulo};
                    border: 3px solid {color_circulo};
                    box-shadow: 0 0 25px {color_circulo}88;
                    display: inline-block;
                    flex-shrink: 0;
                "></div>
                <div style="text-align: left;">
                    <span class="irsti-titulo" style="color: {color_circulo}; font-size: 2.2rem;">IRSTI-AI</span><br>
                    <span class="irsti-subtitulo" style="color: {color_circulo}; font-size: 1rem;">RIESGO {texto_estado}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── MÉTRICAS ──
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""<div class="irsti-card"><span class="metric-label">Riesgo predicho (6h)</span><br>
            <span class="metric-value" style="color:{color_circulo}">{texto_estado}</span></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="irsti-card"><span class="metric-label">Probabilidad estimada</span><br>
            <span class="metric-value">{riesgo_simulado*100:.0f}%</span></div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="irsti-card"><span class="metric-label">Horizonte</span><br>
            <span class="metric-value">6 horas</span></div>""", unsafe_allow_html=True)

        # ── GRÁFICO DE TENDENCIA ──
        col_graf, col_alertas = st.columns([2, 1])
        with col_graf:
            st.subheader("Tendencia de riesgo (próximas horas)")
            horas = ["Ahora", "+1h", "+2h", "+4h", "+6h"]
            valores = [riesgo_simulado * f for f in [0.5, 0.6, 0.75, 0.85, 1.0]]
            fig = go.Figure()

            def hex_to_rgb(hex_color):
                hex_color = hex_color.lstrip('#')
                return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

            r, g, b = hex_to_rgb(color_circulo)

            fig.add_trace(go.Scatter(
                x=horas,
                y=valores,
                mode="lines+markers",
                line=dict(color=color_circulo, width=3),
                marker=dict(size=10, color=color_circulo),
                name="Riesgo estimado"
            ))
            fig.add_trace(go.Scatter(
                x=horas,
                y=valores,
                mode="none",
                fill='tozeroy',
                fillcolor=f'rgba({r}, {g}, {b}, 0.2)',
                name="Área de riesgo"
            ))
            fig.update_layout(
                plot_bgcolor=COLOR_BG_CARD,
                paper_bgcolor=COLOR_BG_CARD,
                font_color=COLOR_TEXT,
                yaxis_range=[0, 1],
                height=320,
                margin=dict(t=20, b=20),
                xaxis_title="Horizonte temporal",
                yaxis_title="Riesgo estimado",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── ALERTAS (con 5 variables) ──
        with col_alertas:
            st.subheader("Alertas")
            st.markdown(f'<div class="{clase_alerta}">{texto_alerta}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="alerta-amarilla">SpO₂ en {spo2}% — {"✅ normal" if spo2 >= 95 else "⚠️ fuera de rango"}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="alerta-verde">FC en {fc} lpm — {"✅ normal" if 60 <= fc <= 100 else "⚠️ fuera de rango"}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="alerta-amarilla">FR en {fr} rpm — {"✅ normal" if 12 <= fr <= 20 else "⚠️ fuera de rango"}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="alerta-verde">PAM en {pam} mmHg — {"✅ normal" if 70 <= pam <= 100 else "⚠️ fuera de rango"}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="alerta-amarilla">Temp en {temp:.1f}°C — {"✅ normal" if 36.5 <= temp <= 38 else "⚠️ fuera de rango"}</div>', unsafe_allow_html=True)

        # ── SEPARADOR ──
        st.markdown("---")
        
        # ── EXPLICABILIDAD ──
        st.header("🔍 ¿Por qué el modelo predijo esto?")
        st.markdown("""
        Para entender qué variable influyó más en la predicción, usamos una
        técnica simple de **sensibilidad**: variamos cada signo vital de a uno
        y observamos cuánto cambia la predicción de riesgo.
        """)
        
        # ── CALCULAR INFLUENCIA DE CADA VARIABLE (5 variables) ──
        variacion = 0.10
        
        # Frecuencia cardíaca
        fc_var = fc * (1 + variacion)
        riesgo_fc = calcular_riesgo(fc_var, spo2, fr, pam, temp)
        influencia_fc = abs(riesgo_fc - riesgo_simulado)
        
        # Saturación O₂
        spo2_var = spo2 * (1 - variacion)
        riesgo_spo2 = calcular_riesgo(fc, spo2_var, fr, pam, temp)
        influencia_spo2 = abs(riesgo_spo2 - riesgo_simulado)
        
        # Frecuencia respiratoria
        fr_var = fr * (1 + variacion)
        riesgo_fr = calcular_riesgo(fc, spo2, fr_var, pam, temp)
        influencia_fr = abs(riesgo_fr - riesgo_simulado)
        
        # Presión arterial
        pam_var = pam * (1 - variacion)
        riesgo_pam = calcular_riesgo(fc, spo2, fr, pam_var, temp)
        influencia_pam = abs(riesgo_pam - riesgo_simulado)
        
        # Temperatura
        temp_var = temp * (1 + variacion)
        riesgo_temp = calcular_riesgo(fc, spo2, fr, pam, temp_var)
        influencia_temp = abs(riesgo_temp - riesgo_simulado)
        
        # ── DATAFRAME CON INFLUENCIAS ──
        importancias = pd.DataFrame({
            "Variable": [
                "Saturación O₂", 
                "Frecuencia cardíaca", 
                "Frecuencia respiratoria", 
                "Presión arterial", 
                "Temperatura"
            ],
            "Influencia": [
                influencia_spo2,
                influencia_fc,
                influencia_fr,
                influencia_pam,
                influencia_temp
            ]
        }).sort_values("Influencia", ascending=True)
        
        total = importancias["Influencia"].sum()
        if total > 0:
            importancias["Influencia_relativa"] = importancias["Influencia"] / total
        else:
            importancias["Influencia_relativa"] = 0
        
        # ── GRÁFICO DE BARRAS ──
        fig2 = go.Figure(go.Bar(
            x=importancias["Influencia_relativa"],
            y=importancias["Variable"],
            orientation="h",
            marker_color=COLOR_ACCENT,
            text=importancias["Influencia_relativa"].apply(lambda x: f"{x*100:.1f}%"),
            textposition="outside",
        ))
        fig2.update_layout(
            plot_bgcolor=COLOR_BG_CARD,
            paper_bgcolor=COLOR_BG_CARD,
            font_color=COLOR_TEXT,
            height=300,
            margin=dict(t=20, b=20),
            xaxis_title="Influencia en la predicción",
            yaxis_title="",
            xaxis_range=[0, max(0.5, importancias["Influencia_relativa"].max() * 1.2)],
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # ── INTERPRETACIÓN ──
        if total > 0:
            var_max = importancias.iloc[-1]["Variable"] if len(importancias) > 0 else "Ninguna"
            pct_max = importancias.iloc[-1]["Influencia_relativa"] * 100 if len(importancias) > 0 else 0
            
            if pct_max > 30:
                st.markdown(f"""
                <div class="irsti-card" style="border-left: 4px solid {COLOR_ACCENT};">
                🔍 <b>Variable más influyente:</b> <b style="color:{COLOR_ACCENT};">{var_max}</b> 
                con un <b>{pct_max:.1f}%</b> de influencia en el riesgo.
                
                Pequeños cambios en {var_max.lower()} afectan significativamente la predicción.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="irsti-card" style="border-left: 4px solid {COLOR_AMARILLO};">
                📊 <b>Influencia distribuida:</b> Ninguna variable domina claramente.
                El modelo está considerando el estado general del paciente.
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="irsti-card" style="border-left: 4px solid #8FA3B5;">
            ⚠️ No hay suficiente variación para determinar influencias.
            Probá ajustar los sliders para ver cómo cambia el riesgo.
            </div>
            """, unsafe_allow_html=True)
        
        

# ------------------------------------------------------------
# TAB 6 — ÉTICA Y LÍMITES
# ------------------------------------------------------------
with tab6:
    st.header("Consideraciones éticas y límites del modelo")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### ⚠️ Limitaciones técnicas

        - **Datos sintéticos:** el modelo se entrena con datos simulados,
          no con pacientes reales. Los patrones pueden no reflejar la
          complejidad clínica real.
        - **Dataset desbalanceado:** la mayoría de los casos son de bajo
          riesgo, por lo que el modelo puede fallar en los casos críticos.
        - **Ruido sintético:** el ruido agregado no representa la variabilidad
          real de los signos vitales humanos.
        - **Horizonte fijo:** predice a 6 horas, pero el deterioro puede
          ser más rápido o más lento.
        """)

    with col2:
        st.markdown("""
        ### 🏛️ Consideraciones éticas

        - **No reemplaza el juicio clínico:** es una herramienta de apoyo
          a la decisión, no un diagnóstico.
        - **Falsos negativos:** no alertar a un paciente que sí se agrava
          es más costoso que un falso positivo.
        - **Sesgo de datos:** un sistema real requeriría validación con
          datos diversos (edad, género, comorbilidades).
        - **Transparencia:** el usuario debe saber que es un modelo educativo
          y no un dispositivo médico aprobado.
        """)

    st.markdown("---")
    st.markdown("""
    <div class="irsti-card" style="border-color: #F4B400;">
    🎯 <b>Conclusión:</b> Este proyecto demuestra el potencial de las LSTM
    para la predicción de riesgo clínico, pero enfatiza que la implementación
    real requeriría validación rigurosa, aprobación regulatoria y
    auditoría continua de sesgos.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PIE DE PÁGINA
# ============================================================
st.divider()
st.markdown(f"""
<div style="text-align: center; color: {COLOR_TEXT_MUTED}; font-size: 0.85rem;">

🧠 **IRSTI-AI · LSTM Tutorial** | TP  - Modelizado de Sistemas de IA


</div>
""", unsafe_allow_html=True)