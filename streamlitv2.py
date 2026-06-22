import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st


st.set_page_config(
    page_title="Store Profit Simulator",
    page_icon="📈",
    layout="wide"
)


st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    /* Mengatur background kartu metric */
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* FIX: Mengubah warna teks angka utama menjadi HITAM */
    [data-testid="stMetricValue"] {
        color: #000000 !important;
    }
    
    /* FIX: Mengubah warna teks label (Prediksi Keuntungan) menjadi abu-abu gelap/hitam */
    [data-testid="stMetricLabel"] {
        color: #333333 !important;
    }
    
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: grey;
        text-align: center;
        padding: 10px;
        font-size: 12px;
        border-top: 1px solid #eee;
    }
    </style>
    """, unsafe_allow_html=True)


# LANGKAH 1: PERSIAPAN MODEL & BASELINE

# Data historis sederhana
X_train = np.array([[5, 10], [10, 20], [15, 5], [20, 25], [25, 15]])
y_train = np.array([50, 80, 110, 90, 150])

# Melatih model (Digital Twin)
model = LinearRegression().fit(X_train, y_train)

# Menetapkan Skenario Dasar (Baseline)
baseline_input = np.array([[10, 10]])
baseline_pred = model.predict(baseline_input)[0]


# LANGKAH 2: LOGIKA SIMULATOR (ANALISIS WHAT-IF)

def run_simulation(new_iklan, new_diskon):
    intervention_input = np.array([[new_iklan, new_diskon]])
    prediction = model.predict(intervention_input)[0]
    delta_y = prediction - baseline_pred
    return prediction, delta_y


# LANGKAH 3: IMPLEMENTASI UI INTERAKTIF

# HEADER UTAMA
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title("🚀 Simulator Kebijakan Keuntungan Toko")
    st.caption("Predictive Analytics Tool for Strategic Retail Decisions")
with col_head2:
    st.info(f"👤 **Analyst:**\n\n M. Misbahul Munir")

st.markdown("---")

# SIDEBAR: Variabel Kontrol
with st.sidebar:
    st.header("🛠️ Tuas Kebijakan")
    st.write("Sesuaikan variabel di bawah untuk melihat dampak terhadap profit.")
    
    iklan_slider = st.slider("Anggaran Iklan (Juta)", 0, 100, 10, help="Total biaya pemasaran per bulan.")
    diskon_slider = st.slider("Besaran Diskon (%)", 0, 50, 10, help="Rata-rata diskon yang diberikan ke pelanggan.")
    
    st.markdown("---")
    st.caption("Sistem ini menggunakan algoritma Regresi Linear untuk mensimulasikan interaksi antar variabel.")

# ENGINE: Jalankan Simulasi
hasil_pred, delta = run_simulation(iklan_slider, diskon_slider)

# UI: TAMPILAN HASIL & METRIK
col1, col2, col3 = st.columns([2, 2, 3])

with col1:
    st.metric(
        label="Prediksi Keuntungan", 
        value=f"Rp {hasil_pred:.2f} Jt", 
        delta=f"{delta:+.2f} Jt"
    )

with col2:
    status = "📈 Profit Meningkat" if delta >= 0 else "📉 Profit Menurun"
    st.subheader(status)
    st.write(f"Perubahan: {delta:+.2f} Juta dibandingkan kondisi baseline.")

with col3:
    st.markdown("##### 💡 Analisis Strategis")
    if delta > 20:
        st.success("Skenario sangat optimal! Strategi ini layak dipertimbangkan untuk kuartal depan.")
    elif delta > 0:
        st.info("Pertumbuhan positif. Pastikan kapasitas operasional siap menangani lonjakan transaksi.")
    else:
        st.warning("Peringatan: Biaya operasional (iklan/diskon) mungkin terlalu tinggi dibanding proyeksi pendapatan.")

# VISUALISASI
st.markdown("### 📊 Perbandingan Skenario")
data_plot = pd.DataFrame({
    'Skenario': ['Baseline (Kondisi Saat Ini)', 'Intervensi (Skenario Baru)'],
    'Keuntungan (Juta)': [baseline_pred, hasil_pred]
})

st.bar_chart(data=data_plot, x='Skenario', y='Keuntungan (Juta)', color='#3b82f6')      

# FOOTER
st.markdown(f'<div class="footer">Developed by M. Misbahul Munir | Simulation System v1.0</div>', unsafe_allow_html=True)