import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ==========================================
# 1. KONFIGURASI HALAMAN (WAJIB DI ATAS)
# ==========================================
st.set_page_config(
    page_title="EcoTrack AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk mempercantik
st.markdown("""
<style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    return joblib.load('carbon_model.pkl')

model = load_model()

# ==========================================
# 2. SETUP DATA (MAPPING & SCALER)
# ==========================================
mapping = {
    'Body Type': {'normal': 0, 'obese': 1, 'overweight': 2, 'underweight': 3},
    'Sex': {'female': 0, 'male': 1},
    'Diet': {'omnivore': 0, 'pescatarian': 1, 'vegan': 2, 'vegetarian': 3},
    'Shower': {'daily': 0, 'less frequently': 1, 'more frequently': 2, 'twice a day': 3},
    'Heating': {'coal': 0, 'electricity': 1, 'natural gas': 2, 'wood': 3},
    'Transport': {'private': 0, 'public': 1, 'walk/bicycle': 2},
    'Vehicle': {'None': 0, 'diesel': 1, 'electric': 2, 'hybrid': 3, 'lpg': 4, 'petrol': 5},
    'Social': {'never': 0, 'often': 1, 'sometimes': 2},
    'Flight': {'frequently': 0, 'never': 1, 'rarely': 2, 'very frequently': 3},
    'Bag Size': {'extra large': 0, 'large': 1, 'medium': 2, 'small': 3},
    'Energy Eff': {'No': 0, 'Sometimes': 1, 'Yes': 2}
}

scaler_stats = {
    'Grocery': {'mean': 173.87, 'std': 72.23},
    'Vehicle Distance': {'mean': 2031.48, 'std': 2769.71},
    'Waste Weekly': {'mean': 4.02, 'std': 1.99},
    'TV Daily Hour': {'mean': 12.14, 'std': 7.10},
    'Clothes Monthly': {'mean': 25.11, 'std': 14.70},
    'Internet Daily': {'mean': 11.89, 'std': 7.28}
}

def scale_value(val, col):
    return (val - scaler_stats[col]['mean']) / scaler_stats[col]['std']

# ==========================================
# 3. SIDEBAR INPUT (Agar Rapi)
# ==========================================
st.sidebar.title("⚙️ Konfigurasi Data")
st.sidebar.write("Isi detail gaya hidupmu di sini:")

with st.sidebar.expander("👤 Profil Pribadi", expanded=True):
    sex = st.selectbox("Jenis Kelamin", list(mapping['Sex'].keys()))
    body = st.selectbox("Tipe Tubuh", list(mapping['Body Type'].keys()))
    diet = st.selectbox("Pola Makan (Diet)", list(mapping['Diet'].keys()))
    social = st.selectbox("Aktivitas Sosial", list(mapping['Social'].keys()))

with st.sidebar.expander("🏠 Rumah & Energi"):
    shower = st.selectbox("Frekuensi Mandi", list(mapping['Shower'].keys()))
    heating = st.selectbox("Sumber Pemanas", list(mapping['Heating'].keys()))
    energy_eff = st.selectbox("Efisiensi Energi", list(mapping['Energy Eff'].keys()))
    tv_daily = st.slider("Jam Nonton TV/PC (per hari)", 0, 24, 5)
    internet_daily = st.slider("Jam Internet (per hari)", 0, 24, 5)

with st.sidebar.expander("🚗 Transportasi"):
    transport = st.selectbox("Transportasi Utama", list(mapping['Transport'].keys()))
    vehicle = st.selectbox("Jenis Bahan Bakar", list(mapping['Vehicle'].keys()))
    distance = st.number_input("Jarak Tempuh (km/bulan)", 0, 10000, 500)
    flight = st.selectbox("Frekuensi Pesawat", list(mapping['Flight'].keys()))

with st.sidebar.expander("🛒 Belanja & Sampah"):
    grocery = st.number_input("Belanja Bulanan ($)", 0, 1000, 150)
    clothes = st.number_input("Beli Baju (item/bulan)", 0, 50, 5)
    bag_size = st.selectbox("Ukuran Kantong Sampah", list(mapping['Bag Size'].keys()))
    waste_weekly = st.slider("Jumlah Kantong Sampah/Minggu", 1, 10, 3)

with st.sidebar.expander("♻️ Daur Ulang & Masak"):
    st.write("**Daur Ulang:**")
    c1, c2 = st.columns(2)
    plastic = c1.checkbox("Plastik")
    glass = c2.checkbox("Kaca")
    metal = c1.checkbox("Logam")
    paper = c2.checkbox("Kertas")
    
    st.write("**Alat Masak:**")
    c3, c4 = st.columns(2)
    microwave = c3.checkbox("Microwave")
    oven = c4.checkbox("Oven")
    stove = c3.checkbox("Kompor")
    airfryer = c4.checkbox("Airfryer")
    # PERBAIKAN: Pakai c3 atau c4 agar tetap di dalam kotak dan lurus
    grill = c3.checkbox("Grill")

# ==========================================
# 4. HALAMAN UTAMA (Main Page)
# ==========================================
st.title("🌿 EcoTrack: AI Carbon Footprint Tracker")
st.markdown("Aplikasi ini menggunakan **Machine Learning** untuk memprediksi jejak karbon harianmu berdasarkan gaya hidup.")
st.divider()

# Tombol Hitung Besar
if st.sidebar.button("🚀 HITUNG JEJAK KARBON", type="primary"):
    
    # --- PROSES DATA ---
    input_dict = {
        'Body Type': int(mapping['Body Type'][body]),
        'Sex': int(mapping['Sex'][sex]),
        'Diet': int(mapping['Diet'][diet]),
        'Shower': int(mapping['Shower'][shower]),
        'Heating': int(mapping['Heating'][heating]),
        'Transport': int(mapping['Transport'][transport]),
        'Vehicle': int(mapping['Vehicle'][vehicle]),
        'Social': int(mapping['Social'][social]),
        'Grocery': float(scale_value(grocery, 'Grocery')),
        'Flight': int(mapping['Flight'][flight]),
        'Vehicle Distance': float(scale_value(distance, 'Vehicle Distance')),
        'Bag Size': int(mapping['Bag Size'][bag_size]),
        'Waste Weekly': float(scale_value(waste_weekly, 'Waste Weekly')),
        'TV Daily Hour': float(scale_value(tv_daily, 'TV Daily Hour')),
        'Clothes Monthly': float(scale_value(clothes, 'Clothes Monthly')),
        'Internet Daily': float(scale_value(internet_daily, 'Internet Daily')),
        'Energy Eff': int(mapping['Energy Eff'][energy_eff]),
        'Plastic': 1 if plastic else 0,
        'Glass': 1 if glass else 0,
        'Metal': 1 if metal else 0,
        'Paper': 1 if paper else 0,
        'Microwave': 1 if microwave else 0,
        'Oven': 1 if oven else 0,
        'Stove': 1 if stove else 0,
        'Airfryer': 1 if airfryer else 0,
        'Grill': 1 if grill else 0
    }
    
    df_input = pd.DataFrame([input_dict])
    
    try:
        hasil = model.predict(df_input)[0]
        
        # --- TAMPILAN HASIL ---
        col_result1, col_result2 = st.columns([1, 2])
        
        with col_result1:
            # Tampilkan Angka Utama
            st.metric(label="Total Emisi Karbon", value=f"{int(hasil)} kgCO2e")
            
            # Label Status
            if hasil < 2000:
                st.success("🌱 RAMAH LINGKUNGAN")
            elif hasil < 3000:
                st.warning("⚠️ RATA-RATA")
            else:
                st.error("🔥 BERBAHAYA")

        with col_result2:
            st.subheader("Analisis Level Emisi")
            # Visualisasi Progress Bar (Normalisasi: Min ~300, Max ~8300)
            # Kita anggap 8300 adalah 100%
            progress_val = (hasil - 300) / (8300 - 300)
            progress_val = max(0.0, min(1.0, progress_val)) # Batasi 0-1
            
            st.progress(progress_val)
            st.caption(f"Posisi kamu dibandingkan rentang emisi dataset (Min: 300 - Max: 8300)")

        st.divider()
        
        # --- SARAN OTOMATIS (AI Insights Sederhana) ---
        st.subheader("💡 Tips Mengurangi Jejak Karbon")
        suggestions = []
        
        if transport == 'private':
            suggestions.append("🚗 **Transportasi:** Cobalah menggunakan transportasi umum atau bersepeda untuk jarak dekat.")
        if flight in ['frequently', 'very frequently']:
            suggestions.append("✈️ **Penerbangan:** Frekuensi terbangmu tinggi. Pertimbangkan *carbon offset* atau kurangi perjalanan udara.")
        if diet == 'omnivore':
            suggestions.append("🥩 **Makanan:** Mengurangi konsumsi daging merah 1 hari seminggu bisa berdampak besar.")
        if heating == 'coal' or heating == 'wood':
            suggestions.append("🔥 **Pemanas:** Sumber pemanasmu menghasilkan emisi tinggi. Coba beralih ke listrik atau gas alam jika memungkinkan.")
        if not (plastic and paper and metal):
             suggestions.append("♻️ **Daur Ulang:** Tingkatkan kebiasaan mendaur ulang sampah rumah tangga.")

        if suggestions:
            for s in suggestions:
                st.info(s)
        else:
            st.success("Gaya hidupmu sudah sangat baik! Pertahankan.")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

else:
    # Tampilan awal sebelum tombol ditekan
    st.info("👈 Silakan isi data di menu sebelah kiri, lalu klik tombol **HITUNG**.")
    
    # Menampilkan Gambar/Ilustrasi (Opsional - pakai emoji besar saja biar ringan)
    st.markdown("""
    <div style="text-align: center; font-size: 100px;">
        🌍
    </div>
    <h3 style="text-align: center;">Mari Jaga Bumi Kita</h3>
    """, unsafe_allow_html=True)