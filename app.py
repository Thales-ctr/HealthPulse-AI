import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# Configuração da página
st.set_page_config(
    page_title="HealthPulse AI | Clinical Command Center",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# DESIGN SYSTEM EXTREMO: DARK GLASSMORPHISM CLINICAL UI
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Importação da fonte Inter */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Reset Geral do Fundo */
    .stApp {
        background: radial-gradient(circle at 50% -20%, #0d192b 0%, #050911 100%);
        color: #f1f5f9;
        font-family: 'Inter', -apple-system, sans-serif;
    }

    /* Ocultar apenas rodapé e decorações, preservando o botão da barra lateral */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    div[data-testid="stDecoration"] { display: none; }
    div[data-testid="stStatusWidget"] { visibility: hidden; }

    /* Header Principal Glassmorphism */
    .clinical-header {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 24px 36px;
        margin-bottom: 28px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .brand-container {
        display: flex;
        align-items: center;
        gap: 16px;
    }

    .brand-icon {
        width: 48px;
        height: 48px;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        box-shadow: 0 0 20px rgba(0, 242, 254, 0.4);
    }

    .header-title {
        color: #ffffff;
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.8px;
        margin: 0;
        background: linear-gradient(90deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .header-subtitle {
        color: #38bdf8;
        font-size: 13px;
        font-weight: 500;
        letter-spacing: 0.5px;
        margin-top: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .status-badge {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34d399;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    .status-dot {
        width: 6px;
        height: 6px;
        background-color: #34d399;
        border-radius: 50%;
        box-shadow: 0 0 8px #34d399;
    }

    /* Cartões Holográficos de Risco / KPI */
    .patient-hero-card {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 12px 32px rgba(0,0,0,0.3);
        position: relative;
        overflow: hidden;
        transition: transform 0.3s ease;
    }

    .patient-hero-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 100%; height: 4px;
    }

    .card-red::before { background: linear-gradient(90deg, #ef4444, #f87171); box-shadow: 0 0 15px #ef4444; }
    .card-orange::before { background: linear-gradient(90deg, #f97316, #fb923c); box-shadow: 0 0 15px #f97316; }
    .card-yellow::before { background: linear-gradient(90deg, #eab308, #fde047); box-shadow: 0 0 15px #eab308; }
    .card-green::before { background: linear-gradient(90deg, #10b981, #34d399); box-shadow: 0 0 15px #10b981; }

    .hero-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #94a3b8;
        font-weight: 600;
    }

    .hero-value {
        font-size: 32px;
        font-weight: 800;
        margin-top: 8px;
        letter-spacing: -0.5px;
    }

    .text-red { color: #f87171; text-shadow: 0 0 20px rgba(239, 68, 68, 0.4); }
    .text-orange { color: #fb923c; text-shadow: 0 0 20px rgba(249, 115, 22, 0.4); }
    .text-yellow { color: #fde047; text-shadow: 0 0 20px rgba(234, 179, 8, 0.4); }
    .text-green { color: #34d399; text-shadow: 0 0 20px rgba(16, 185, 129, 0.4); }

    .hero-desc {
        font-size: 12px;
        color: #cbd5e1;
        margin-top: 6px;
        font-weight: 400;
    }

    /* Container de Painéis */
    .glass-panel {
        background: rgba(15, 23, 42, 0.5);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
    }

    .panel-title {
        font-size: 16px;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.3px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Sidebar Customizada */
    section[data-testid="stSidebar"] {
        background-color: #060a12;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* Rodapé da Aplicação */
    .custom-footer {
        margin-top: 48px;
        padding: 24px 0;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        text-align: center;
        color: #64748b;
        font-size: 13px;
        letter-spacing: 0.5px;
    }
    .author-name {
        color: #38bdf8;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# CABEÇALHO CLINICO EXECUTIVO
# ---------------------------------------------------------
st.markdown("""
<div class="clinical-header">
    <div class="brand-container">
        <div class="brand-icon">🩺</div>
        <div>
            <div class="header-title">HealthPulse AI Enterprise</div>
            <div class="header-subtitle">
                <span>Clinical Command Center & Protocolo Manchester Preditivo</span>
            </div>
        </div>
    </div>
    <div class="status-badge">
        <div class="status-dot"></div> Sistema Operacional em Tempo Real
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# MODELO MACHINE LEARNING SINTÉTICO (Triagem Clínico)
# ---------------------------------------------------------
@st.cache_resource
def treinar_modelo_triagem():
    np.random.seed(42)
    n_samples = 3500
    
    fc = np.random.normal(80, 22, n_samples)
    pas = np.random.normal(120, 28, n_samples)
    spo2 = np.random.normal(96, 5, n_samples)
    temp = np.random.normal(36.8, 1.3, n_samples)
    freq_resp = np.random.normal(16, 6, n_samples)
    
    classes = []
    for i in range(n_samples):
        if spo2[i] < 90 or pas[i] > 180 or pas[i] < 80 or fc[i] > 140 or temp[i] > 39.5:
            classes.append('Vermelho - Emergência')
        elif spo2[i] < 94 or pas[i] > 160 or fc[i] > 110 or temp[i] > 38.5:
            classes.append('Laranja - Muita Urgência')
        elif pas[i] > 140 or fc[i] > 100 or temp[i] > 37.8:
            classes.append('Amarelo - Urgência')
        else:
            classes.append('Verde - Pouco Urgente')
            
    df_train = pd.DataFrame({
        'FC': fc, 'PAS': pas, 'SpO2': spo2, 'Temp': temp, 'FreqResp': freq_resp,
        'Classe': classes
    })
    
    X = df_train[['FC', 'PAS', 'SpO2', 'Temp', 'FreqResp']]
    y = df_train['Classe']
    
    clf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    clf.fit(X, y)
    
    return clf

model = treinar_modelo_triagem()

# ---------------------------------------------------------
# BARRA LATERAL: AVALIAÇÃO DO PACIENTE EM TEMPO REAL
# ---------------------------------------------------------
st.sidebar.markdown("## 📊 Entrada do Paciente")
st.sidebar.caption("Preencha os sinais vitais aferidos na recepção:")

nome_paciente = st.sidebar.text_input("Nome do Paciente", "Paciente Leito 01")
idade = st.sidebar.slider("Idade", 1, 100, 52)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🫀 Sinais Vitais Aferidos")

col_sb1, col_sb2 = st.sidebar.columns(2)
fc_input = col_sb1.number_input("Freq. Cardíaca (BPM)", 30, 220, 115)
pas_input = col_sb2.number_input("Pressão Sistólica (mmHg)", 50, 260, 165)

col_sb3, col_sb4 = st.sidebar.columns(2)
spo2_input = col_sb3.number_input("Saturação O₂ (%)", 60, 100, 91)
temp_input = col_sb4.number_input("Temperatura (°C)", 33.0, 43.0, 38.6, step=0.1)

freq_resp_input = st.sidebar.number_input("Freq. Respiratória (mrm)", 6, 50, 24)

# Predição Inteligente
dados_paciente = pd.DataFrame([[fc_input, pas_input, spo2_input, temp_input, freq_resp_input]],
                              columns=['FC', 'PAS', 'SpO2', 'Temp', 'FreqResp'])

predicao = model.predict(dados_paciente)[0]
probas = model.predict_proba(dados_paciente)[0]
classes_model = model.classes_

# Mapeamento de Estilo por Gravidade
estilos = {
    'Vermelho - Emergência': {'card': 'card-red', 'text': 'text-red', 'tempo': 'Atendimento Imediato (0 min)', 'cor': '#ef4444'},
    'Laranja - Muita Urgência': {'card': 'card-orange', 'text': 'text-orange', 'tempo': 'Atendimento em até 10 min', 'cor': '#f97316'},
    'Amarelo - Urgência': {'card': 'card-yellow', 'text': 'text-yellow', 'tempo': 'Atendimento em até 60 min', 'cor': '#eab308'},
    'Verde - Pouco Urgente': {'card': 'card-green', 'text': 'text-green', 'tempo': 'Atendimento em até 120 min', 'cor': '#10b981'}
}

estilo_atual = estilos.get(predicao, estilos['Verde - Pouco Urgente'])

# ---------------------------------------------------------
# PAINEL HOLOGRÁFICO DE TRIAGEM ATUAL (KPIS)
# ---------------------------------------------------------
st.markdown(f"##### 🎯 Avaliação Preditiva: {nome_paciente} ({idade} anos)")

c1, c2, c3, c4 = st.columns(4)

c1.markdown(f"""
<div class="patient-hero-card {estilo_atual['card']}">
    <div class="hero-label">Grau de Urgência</div>
    <div class="hero-value {estilo_atual['text']}">{predicao.split(' - ')[1]}</div>
    <div class="hero-desc">{estilo_atual['tempo']}</div>
</div>
""", unsafe_allow_html=True)

c2.markdown(f"""
<div class="patient-hero-card {estilo_atual['card']}">
    <div class="hero-label">Prioridade Manchester</div>
    <div class="hero-value {estilo_atual['text']}">{predicao.split(' - ')[0]}</div>
    <div class="hero-desc">Classificação por I.A. Preditiva</div>
</div>
""", unsafe_allow_html=True)

c3.markdown(f"""
<div class="patient-hero-card">
    <div class="hero-label">Oxigenação (SpO₂)</div>
    <div class="hero-value {'text-red' if spo2_input < 92 else 'text-green'}">{spo2_input}%</div>
    <div class="hero-desc">{'⚠️ Hipóxia Crítica' if spo2_input < 92 else '✓ Oxigenação Normal'}</div>
</div>
""", unsafe_allow_html=True)

c4.markdown(f"""
<div class="patient-hero-card">
    <div class="hero-label">Pressão Sistólica</div>
    <div class="hero-value {'text-red' if pas_input > 160 or pas_input < 90 else 'text-green'}">{pas_input} <span style="font-size:16px;">mmHg</span></div>
    <div class="hero-desc">{'⚠️ Alteração Pressórica' if pas_input > 160 or pas_input < 90 else '✓ Estabilidade Hemodinâmica'}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# MEDIDOR CIRCULAR (GAUGE) E DISTRIBUIÇÃO DE PROBABILIDADES
# ---------------------------------------------------------
g_col1, g_col2 = st.columns([5, 5])

# Tema escuro elegante para gráficos Plotly
layout_chart = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#cbd5e1', family="Inter"),
    margin=dict(l=20, r=20, t=30, b=20)
)

with g_col1:
    st.markdown("""<div class="panel-title">📉 Nível de Risco Fisiológico do Paciente</div>""", unsafe_allow_html=True)
    
    # Calcular um índice numérico de risco de 0 a 100
    idx_risco = 20
    if "Vermelho" in predicao: idx_risco = 95
    elif "Laranja" in predicao: idx_risco = 75
    elif "Amarelo" in predicao: idx_risco = 50

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=idx_risco,
        number={'suffix': "%", 'font': {'color': estilo_atual['cor'], 'size': 42, 'family': 'Inter'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#334155"},
            'bar': {'color': estilo_atual['cor']},
            'bgcolor': "rgba(15, 23, 42, 0.8)",
            'borderwidth': 1,
            'bordercolor': "rgba(255,255,255,0.1)",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.2)'},
                {'range': [30, 60], 'color': 'rgba(234, 179, 8, 0.2)'},
                {'range': [60, 85], 'color': 'rgba(249, 115, 22, 0.2)'},
                {'range': [85, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
            ]
        }
    ))
    fig_gauge.update_layout(**layout_chart, height=260)
    st.plotly_chart(fig_gauge, use_container_width=True)

with g_col2:
    st.markdown("""<div class="panel-title">🧠 Distribuição de Probabilidades do Modelo</div>""", unsafe_allow_html=True)
    
    df_proba = pd.DataFrame({'Classe': classes_model, 'Probabilidade': probas * 100})
    df_proba['Classe_Clean'] = df_proba['Classe'].apply(lambda x: x.split(' - ')[1])
    
    fig_proba = px.bar(
        df_proba,
        x='Probabilidade',
        y='Classe_Clean',
        orientation='h',
        color='Classe',
        color_discrete_map={
            'Vermelho - Emergência': '#ef4444',
            'Laranja - Muita Urgência': '#f97316',
            'Amarelo - Urgência': '#eab308',
            'Verde - Pouco Urgente': '#10b981'
        },
        text=df_proba['Probabilidade'].apply(lambda x: f"{x:.1f}%")
    )
    fig_proba.update_layout(**layout_chart, height=260, showlegend=False)
    fig_proba.update_xaxes(gridcolor='rgba(255,255,255,0.05)', title="")
    fig_proba.update_yaxes(gridcolor='rgba(255,255,255,0.05)', title="")
    st.plotly_chart(fig_proba, use_container_width=True)

# ---------------------------------------------------------
# SIMULAÇÃO DE CENTRO DE COMANDO & PRONTO SOCORRO
# ---------------------------------------------------------
@st.cache_data
def gerar_fila_pronto_socorro(n=60):
    np.random.seed(15)
    fc = np.random.normal(84, 20, n)
    pas = np.random.normal(126, 24, n)
    spo2 = np.random.normal(95, 4, n)
    temp = np.random.normal(36.9, 1.1, n)
    freq_resp = np.random.normal(17, 5, n)
    
    df_fila = pd.DataFrame({
        'ID_Paciente': [f'PAC-{1000 + i}' for i in range(n)],
        'FC': np.round(fc),
        'PAS': np.round(pas),
        'SpO2': np.round(spo2, 1),
        'Temp': np.round(temp, 1),
        'FreqResp': np.round(freq_resp)
    })
    
    df_fila['Classificação'] = model.predict(df_fila[['FC', 'PAS', 'SpO2', 'Temp', 'FreqResp']])
    return df_fila

df_fila = gerar_fila_pronto_socorro(60)

v1, v2 = st.columns([6, 4])

with v1:
    st.markdown("""<div class="panel-title">📡 Dispersão Clinica: Oxigenação (SpO₂) vs. Frequência Cardíaca</div>""", unsafe_allow_html=True)
    
    fig_scatter = px.scatter(
        df_fila,
        x='FC',
        y='SpO2',
        color='Classificação',
        size='PAS',
        color_discrete_map={
            'Vermelho - Emergência': '#ef4444',
            'Laranja - Muita Urgência': '#f97316',
            'Amarelo - Urgência': '#eab308',
            'Verde - Pouco Urgente': '#10b981'
        },
        hover_data=['ID_Paciente', 'Temp']
    )
    fig_scatter.update_layout(**layout_chart, height=360)
    fig_scatter.update_xaxes(gridcolor='rgba(255,255,255,0.05)', title="Frequência Cardíaca (BPM)")
    fig_scatter.update_yaxes(gridcolor='rgba(255,255,255,0.05)', title="Saturação de O₂ (%)")
    st.plotly_chart(fig_scatter, use_container_width=True)

with v2:
    st.markdown("""<div class="panel-title">🏥 Ocupação do Pronto Socorro por Grau de Risco</div>""", unsafe_allow_html=True)
    
    df_counts = df_fila['Classificação'].value_counts().reset_index()
    df_counts.columns = ['Gravidade', 'Quantidade']
    
    fig_donut = px.pie(
        df_counts,
        names='Gravidade',
        values='Quantidade',
        hole=0.6,
        color='Gravidade',
        color_discrete_map={
            'Vermelho - Emergência': '#ef4444',
            'Laranja - Muita Urgência': '#f97316',
            'Amarelo - Urgência': '#eab308',
            'Verde - Pouco Urgente': '#10b981'
        }
    )
    fig_donut.update_layout(**layout_chart, height=360)
    st.plotly_chart(fig_donut, use_container_width=True)

# ---------------------------------------------------------
# FILA DE ATENDIMENTO PRIORIZADA
# ---------------------------------------------------------
st.markdown("""<div class="panel-title">📋 Fila Ativa de Atendimento Hospitalar (Priorização por Risco)</div>""", unsafe_allow_html=True)

df_fila_sorted = df_fila.sort_values(by=['SpO2', 'PAS'], ascending=[True, False])

st.dataframe(
    df_fila_sorted[['ID_Paciente', 'Classificação', 'SpO2', 'PAS', 'FC', 'Temp', 'FreqResp']],
    use_container_width=True,
    height=280
)

# ---------------------------------------------------------
# RODAPÉ DE AUTORIA
# ---------------------------------------------------------
st.markdown("""
<div class="custom-footer">
    Desenvolvido por <span class="author-name">Thales Câmara</span> | Engenharia de Software & HealthTech Analytics
</div>
""", unsafe_allow_html=True)
