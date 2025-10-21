import streamlit as st
import pandas as pd
import joblib
import logging
import os
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# 1. SETUP PAGE
# ============================================================
st.set_page_config(page_title="Analisis Air Minum Jawa Barat", layout="wide", page_icon="💧")
st.title("💧 Analisis & Prediksi Ketersediaan Air Minum Jawa Barat")
st.markdown("""
Aplikasi ini membantu menganalisis dan memprediksi ketersediaan air minum yang layak berdasarkan data survei per desa di Jawa Barat.  
Gunakan sidebar untuk navigasi antara **visualisasi data** dan **prediksi wilayah**.
""")

# ============================================================
# 2. LOAD ARTIFACTS DENGAN CACHING (LANGSUNG DARI .PKL)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_ml_artifacts() -> Dict[str, Any]:
    """Load model, scaler, dan encoders langsung dari file .pkl dengan caching untuk performa."""
    try:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        encoders = joblib.load("encoders.pkl")
        artifacts = {"model": model, "scaler": scaler, **encoders}
        logger.info("✅ Artifacts berhasil dimuat dari .pkl.")
        return artifacts
    except Exception as e:
        st.error(f"❌ Gagal memuat artifacts: {e}. Pastikan file model.pkl, scaler.pkl, dan encoders.pkl ada di repository.")
        st.stop()

with st.spinner("🚀 Sedang memuat model dan resource... (sekitar 10–30 detik)"):
    artifacts = load_ml_artifacts()

model = artifacts.get("model")
scaler = artifacts.get("scaler")
encoders = {k: v for k, v in artifacts.items() if k.startswith("label_")}
if not model or not scaler:
    st.error("❌ Model atau scaler tidak ditemukan. Pastikan file .pkl sudah diunggah di repository.")
    st.stop()
st.success("✅ Model dan resource berhasil dimuat.")

# ============================================================
# 3. NAVIGASI MODE
# ============================================================
mode = st.sidebar.radio("Navigasi", ["📊 Visualisasi Data", "🔮 Prediksi Air Layak"])

# ============================================================
# 4. VISUALISASI DATA
# ============================================================
if mode == "📊 Visualisasi Data":
    st.header("📊 Visualisasi Distribusi Sumber Air Minum")
    st.markdown("Unggah file **CSV** hasil survei per desa untuk melihat distribusi sumber air di setiap kabupaten/kota.")

    uploaded_file = st.file_uploader("Unggah file data.csv", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Data berhasil diunggah.")
            st.dataframe(df.head())

            if "bps_nama_kabupaten_kota" in df.columns:
                kabupaten_list = sorted(df["bps_nama_kabupaten_kota"].dropna().unique())
                kabupaten = st.selectbox("Pilih Kabupaten/Kota", kabupaten_list)

                df_filtered = df[df["bps_nama_kabupaten_kota"] == kabupaten]

                # Batasi titik agar tidak berat saat render
                if len(df_filtered) > 1000:
                    df_filtered = df_filtered.sample(1000)
                    st.warning("⚠️ Data besar, hanya menampilkan 1000 titik pertama.")

                # Tampilkan peta jika ada koordinat
                if {"latitude", "longitude"}.issubset(df_filtered.columns):
                    # Asumsi plot_simple_map ada, tapi jika tidak, ganti dengan st.map atau hapus
                    st.info("Fitur peta memerlukan utils/. Gunakan bar chart sebagai gantinya.")
                    # m = plot_simple_map(df_filtered, lat_col="latitude", lon_col="longitude", popup_col="bps_nama_desa_kelurahan")
                    # st.components.v1.html(m._repr_html_(), height=500)
                else:
                    st.info("Data tidak memiliki kolom koordinat, menampilkan statistik dasar.")
                    sumber_cols = [c for c in df_filtered.columns if "ketersediaan_air_minum_sumber" in c]
                    if sumber_cols:
                        st.subheader("Distribusi Sumber Air di Kabupaten Ini")
                        st.bar_chart(df_filtered[sumber_cols].sum())
                    else:
                        st.warning("Tidak ditemukan kolom sumber air di data yang diunggah.")
            else:
                st.error("Kolom 'bps_nama_kabupaten_kota' tidak ditemukan di dataset.")
        except Exception as e:
            st.error(f"❌ Error memproses file: {e}")
            logger.error(f"Error memproses uploaded file: {e}")
    else:
        st.info("Unggah file data.csv untuk memulai visualisasi.")

                        sumber_status = {}
                        for col in sumber_cols:
                            avg = df_filtered[col].mean()  # Rata-rata (jika 0/1)
                            sumber_status[col] = "✅ Ada" if avg > 0.5 else "❌ Tidak Ada"
                        st.subheader(f"Status Sumber Air di {kabupaten} - {kecamatan}")
                        for col, status in sumber_status.items():
                            st.write(f"- {col.replace('ketersediaan_air_minum_sumber_', '').replace('_', ' ').title()}: {status}")
                        # Logika analisis: Sumber layak = kemasan, ledeng_meteran, ledeng_tanpa_meteran, mata_air
                        layak_cols = [
                            "ketersediaan_air_minum_sumber_kemasan",
                            "ketersediaan_air_minum_sumber_ledeng_meteran",
                            "ketersediaan_air_minum_sumber_ledeng_tanpa_meteran",
                            "ketersediaan_air_minum_sumber_mata_air",
                        ]
                        layak_count = sum(1 for col in layak_cols if col in sumber_cols and df_filtered[col].mean() > 0.5)
                        if layak_count >= 4:
                            st.success("✅ Aman: Semua sumber air layak tersedia. Tidak perlu ditinjau lagi.")
                        elif layak_count == 1:
                            st.warning("⚠️ Perlu Ditinjau: Hanya 1 sumber air layak tersedia.")
                        else:
                            st.info(f"ℹ️ Jumlah sumber air layak: {layak_count}. Evaluasi lebih lanjut diperlukan.")
    # ... (bagian Input Manual tetap sama)
# ============================================================
# 6. FOOTER
# ============================================================
st.sidebar.markdown("---")
st.sidebar.write("👩🏻‍💻 *Developed by Kelompok 11*")