# Memperbarui kode Streamlit untuk menambahkan informasi tentang dataset

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans


# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("df_hour_cleaned.csv")

df = load_data()

# Sidebar navigation
st.sidebar.title("Dashboard Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Tentang Dataset", "Data Overview", "Visualisasi Data", "Analisis RFM & Clustering"])

if page == "Tentang Dataset":
    st.title("Tentang Dataset 🚴‍♂️")
    st.write("""
    Dataset ini berisi **data penyewaan sepeda** dari sistem **Capital Bikeshare** di **Washington D.C., USA** selama tahun **2011-2012**.
    Sistem ini memungkinkan pengguna untuk menyewa sepeda dari satu lokasi dan mengembalikannya di lokasi lain secara otomatis.
    """)

    st.subheader("Informasi Dataset")
    st.write("""
    - **Sumber Data**: Capital Bikeshare system, Washington D.C., USA.
    - **Periode Data**: Tahun **2011 - 2012**.
    - **Struktur Data**:
        - Data dikumpulkan **per jam** dengan informasi terkait cuaca dan musim.
        - Informasi meliputi **jumlah penyewaan sepeda**, kondisi cuaca, suhu, kecepatan angin, dan faktor lain.
    - **Sumber Cuaca**: Data cuaca diperoleh dari **freemeteo.com**.
    - **Tujuan Penggunaan**:
        - Menganalisis pola penyewaan sepeda berdasarkan waktu dan cuaca.
        - Menggunakan **RFM Analysis** untuk melihat pola pelanggan.
        - Melakukan **clustering jam operasional** berdasarkan jumlah penyewaan.
    """)

elif page == "Data Overview":
    st.title("Ringkasan Data")
    st.write("Tampilan pertama dari dataset:")
    st.dataframe(df.head())
  
    st.write("Statistik Deskriptif:")
    st.write(df.describe())

elif page == "Visualisasi Data":
    st.title("Visualisasi Data Penyewaan Sepeda")
    
    # Boxplot untuk mendeteksi outlier
    st.subheader("Boxplot Variabel Utama")
    columns = ["cnt", "casual", "registered", "windspeed", "hum"]
    fig, axes = plt.subplots(1, len(columns), figsize=(20, 5))
    for i, col in enumerate(columns):
        sns.boxplot(y=df[col], ax=axes[i])
        axes[i].set_title(f"Boxplot of {col}")
    st.pyplot(fig)
    
    # Barplot Pengaruh Kondisi Cuaca
    st.subheader("Pengaruh Kondisi Cuaca terhadap Penyewaan Sepeda")
    weather_group = df.groupby("weathersit")["cnt"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=weather_group["weathersit"], y=weather_group["cnt"], palette="coolwarm", ax=ax)
    ax.set_xlabel("Kondisi Cuaca (1=Cerah, 2=Berawan, 3=Hujan, 4=Badai)")
    ax.set_ylabel("Rata-rata Penyewaan Sepeda per Jam")
    ax.set_title("Pengaruh Kondisi Cuaca terhadap Penyewaan Sepeda")
    st.pyplot(fig)
    
    # Lineplot Tren Permintaan Sepeda per Jam
    st.subheader("Tren Permintaan Penyewaan Sepeda Berdasarkan Jam dalam Sehari")
    hour_group = df.groupby("hr")["cnt"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=hour_group["hr"], y=hour_group["cnt"], marker="o", color="b", ax=ax)
    ax.set_xlabel("Jam dalam Sehari")
    ax.set_ylabel("Rata-rata Penyewaan Sepeda")
    ax.set_title("Tren Permintaan Penyewaan Sepeda Berdasarkan Jam dalam Sehari")
    ax.grid()
    st.pyplot(fig)

elif page == "Analisis RFM & Clustering":
    st.title("Analisis RFM dan Clustering")
    
    # Konversi tanggal ke datetime
    df["dteday"] = pd.to_datetime(df["dteday"])
    
    # RFM Analysis
    st.subheader("Analisis RFM")
    rfm_df = df.groupby("dteday").agg(
        Recency=("dteday", lambda x: (df["dteday"].max() - x.max()).days),
        Frequency=("cnt", "count"),
        Monetary=("cnt", "sum")
    ).reset_index()
    st.write("Tabel RFM:")
    st.dataframe(rfm_df.head())
    
    # Clustering berdasarkan jam
    st.subheader("Clustering Kategori Jam")
    def categorize_hour(hour):
        if 7 <= hour <= 9 or 16 <= hour <= 19:
            return 'Peak Hours'
        elif 10 <= hour <= 15:
            return 'Normal Hours'
        else:
            return 'Off-Peak Hours'
    
    df["Hour_Category"] = df["hr"].apply(categorize_hour)
    hourly_clustering = df.groupby("Hour_Category")["cnt"].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=hourly_clustering["Hour_Category"], y=hourly_clustering["cnt"], palette="coolwarm", ax=ax)
    ax.set_xlabel("Kategori Jam")
    ax.set_ylabel("Rata-rata Penyewaan Sepeda")
    ax.set_title("Rata-rata Penyewaan Sepeda Berdasarkan Kategori Jam")
    st.pyplot(fig)
