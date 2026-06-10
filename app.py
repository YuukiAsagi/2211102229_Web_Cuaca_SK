import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
from sklearn.decomposition import PCA

# 1. KONFIGURASI HALAMAN STREAMLIT
st.set_page_config(
    page_title="Dashboard Prediksi Iklim",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. FUNGSI CACHE UNTUK LOADING (Super Ringan)
@st.cache_data
def load_dataset(kota):
    file_path = f"STREAMLIT_DATASET_{kota}.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

@st.cache_resource
def load_models(kota):
    try:
        # Load scaler dan model klastering (K-Means)
        kmeans = joblib.load(f"model_kmeans_{kota}.pkl")
        scaler = joblib.load(f"minmax_params_{kota}.pkl")
        
        # LOGIKA MODEL TERBAIK (Champion Model Universal: Naïve Bayes)
        model = joblib.load(f"model_nb_{kota}.pkl")
        algoritma = "Gaussian Naïve Bayes (NB)"
            
        return model, kmeans, scaler, algoritma
    except Exception as e:
        return None, None, None, str(e)

# 3. SIDEBAR: NAVIGASI & KONTROL
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1163/1163661.png", width=120)
    st.title("Sistem AI Cuaca")
    st.markdown("---")
    
    menu = st.radio("Pilih Mode Interaksi:", ["📊 Dashboard Klastering", "🔮 Simulasi Cuaca Harian"])
    st.markdown("---")
    
    kota_pilihan = st.selectbox("📍 Pilih Stasiun Observasi:", ['Jakarta', 'Bogor', 'Poso'])
    
    st.markdown("---")
    st.caption("Dikembangkan untuk Skripsi Teknik Informatika")

# 4. MEMUAT DATA BERDASARKAN KOTA
df = load_dataset(kota_pilihan)
model, kmeans, scaler, nama_algoritma = load_models(kota_pilihan)

if df is None or model is None:
    st.error(f"⚠️ Menunggu Data... Pastikan file dataset, model K-Means, Parameter Scaler, dan Model Terbaik ({nama_algoritma}) untuk **{kota_pilihan}** berada di folder yang sama dengan app.py.")
    st.stop()

# 5. HALAMAN 1: DASHBOARD KLASTERING
if menu == "📊 Dashboard Klastering":
    st.title(f"Visualisasi Klastering Iklim - {kota_pilihan.upper()}")
    st.info(f"💡 **Mesin Prediksi Aktif:** {nama_algoritma} (Klasifikasi) & K-Means K=2 (Klastering)")
    
    # Deretan Kartu Metrik
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Data Terekam", f"{len(df)} Hari")
    col2.metric("Prediksi Mayoritas", df['Label_Prediksi'].mode()[0])
    col3.metric("Populasi Klaster", f"Klaster 0: {len(df[df['Cluster_Iklim']==0])} | Klaster 1: {len(df[df['Cluster_Iklim']==1])}")
    
    st.markdown("---")
    st.subheader("Peta Sebaran Varians Iklim (PCA 2D)")
    
    # Reduksi Dimensi secara live untuk plot
    fitur_cols = ['TN_norm', 'TX_norm', 'SS_norm', 'RH_AVG_norm', 'BULAN_norm', 'RH_AVG_norm_kemarin', 'SS_norm_kemarin']
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[fitur_cols])
    df['PCA1'] = pca_result[:, 0]
    df['PCA2'] = pca_result[:, 1]
    df['Cluster_Iklim_Str'] = df['Cluster_Iklim'].astype(str)
    
    # Plotly Interaktif
    fig = px.scatter(
        df, x='PCA1', y='PCA2', 
        color='Cluster_Iklim_Str', symbol='Label_Prediksi',
        hover_data=['TANGGAL', 'TAVG', 'RH_AVG'],
        color_discrete_sequence=['#ef553b', '#636efa'],
        labels={'Cluster_Iklim_Str': 'Klaster K-Means', 'Label_Prediksi': 'Prediksi Cuaca'},
        title=f"Batas Keputusan (Decision Boundary) {nama_algoritma} pada K-Means"
    )
    fig.update_layout(height=550)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📋 Cuplikan Basis Data Historis")
    st.dataframe(df[['TANGGAL', 'TAVG', 'SS', 'RH_AVG', 'RR', 'Prediksi_Final_Cuaca', 'Label_Prediksi', 'Cluster_Iklim']].head(20), use_container_width=True)

# 6. HALAMAN 2: SIMULASI PREDIKSI HARIAN
elif menu == "🔮 Simulasi Cuaca Harian":
    st.title(f"Simulasi Mesin Prediksi - {kota_pilihan.upper()}")
    st.markdown(f"Lakukan simulasi pengujian *real-time* menggunakan **{nama_algoritma}**.")
    
    with st.form("form_prediksi"):
        st.subheader("Input Parameter Observasi Harian")
        
        col1, col2 = st.columns(2)
        with col1:
            tn_in = st.number_input("Suhu Minimum Hari Ini (°C)", min_value=15.0, max_value=35.0, value=24.0)
            tx_in = st.number_input("Suhu Maksimum Hari Ini (°C)", min_value=25.0, max_value=40.0, value=32.0)
            ss_in = st.number_input("Lama Penyinaran Hari Ini (Jam)", min_value=0.0, max_value=12.0, value=6.0)
            rh_in = st.number_input("Kelembapan Udara Hari Ini (%)", min_value=30.0, max_value=100.0, value=80.0)
            
        with col2:
            ss_kemarin_in = st.number_input("Lama Penyinaran Kemarin (Jam)", min_value=0.0, max_value=12.0, value=7.0)
            rh_kemarin_in = st.number_input("Kelembapan Kemarin (%)", min_value=30.0, max_value=100.0, value=75.0)
            bulan_in = st.selectbox("Bulan Observasi", range(1, 13), index=5)
            
        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.form_submit_button("Luncurkan Analisis 🚀")
        
    if submit:
        # Fungsi Normalisasi Ekstraksi dari Scaler Bab 4
        def norm(val, col_name):
            val_min = scaler[col_name]['min']
            val_max = scaler[col_name]['max']
            if val_max == val_min: return 0.0
            return (val - val_min) / (val_max - val_min)

        # Matriks input (Harus sesuai urutan 7 Fitur Universal)
        input_norm = np.array([[
            norm(tn_in, 'TN'),
            norm(tx_in, 'TX'),
            norm(ss_in, 'SS'),
            norm(rh_in, 'RH_AVG'),
            (bulan_in - 1) / 11.0,  # Normalisasi bulan (0-11)
            norm(rh_kemarin_in, 'RH_AVG'),
            norm(ss_kemarin_in, 'SS')
        ]])
        
        # Eksekusi Mesin AI
        hasil_klasifikasi = model.predict(input_norm)[0]
        hasil_klaster = kmeans.predict(input_norm)[0]
        
        # Menampilkan Hasil secara Elegan
        st.markdown("---")
        st.subheader("🎯 Hasil Dekode Atmosferik")
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.caption(f"Hasil Klasifikasi ({nama_algoritma})")
            if hasil_klasifikasi == 1:
                st.error("## 🌧️ RAWAN HUJAN")
                st.write("Volume presipitasi diperkirakan melampaui batas ambang 0.5 mm.")
            else:
                st.success("## ☀️ TIDAK HUJAN")
                st.write("Kondisi atmosfer stabil, tidak ada indikasi presipitasi tinggi.")
                
        with res_col2:
            st.caption("Hasil Identifikasi Klaster (K-Means)")
            st.warning(f"## KLASTER IKLIM {hasil_klaster}")
            st.write(f"Karakteristik cuaca ini memiliki kemiripan spasial dengan titik *centroid* Klaster {hasil_klaster} pada basis data historis regional {kota_pilihan}.")