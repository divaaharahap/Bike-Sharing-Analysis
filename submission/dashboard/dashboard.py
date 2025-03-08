import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans


# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("df_hour_cleaned.csv")
    df["dteday"] = pd.to_datetime(df["dteday"])
    df["weekday"] = df["dteday"].dt.weekday  # Tambahkan kolom weekday
    return df

df = load_data()

# Filter data untuk tahun 2012
hour_df_2012 = df[df["yr"] == 1]  # 1 menunjukkan tahun 2012, 0 menunjukkan 2011

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
    sns.barplot(x="weathersit", y="cnt", hue="weathersit", palette="coolwarm", legend=False, data=weather_group, ax=ax)
    ax.set_xlabel("Kondisi Cuaca (1=Cerah, 2=Berawan, 3=Hujan, 4=Badai)")
    ax.set_ylabel("Rata-rata Penyewaan Sepeda per Jam")
    ax.set_title("Pengaruh Kondisi Cuaca terhadap Penyewaan Sepeda")
    st.pyplot(fig)

    # Boxplot Penyewaan Sepeda Berdasarkan Jam dan Cuaca (2012)
    st.subheader("Distribusi Penyewaan Sepeda Berdasarkan Jam dan Cuaca (2012)")
    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x="hr", y="cnt", hue="weathersit", data=hour_df_2012, palette="coolwarm", ax=ax)
    ax.set_xlabel("Jam dalam Sehari")
    ax.set_ylabel("Jumlah Penyewaan Sepeda")
    ax.set_title("Distribusi Penyewaan Sepeda Berdasarkan Jam dan Cuaca (2012)")
    ax.legend(title="Kondisi Cuaca", labels=["Cerah", "Berawan", "Hujan Ringan", "Badai"])
    st.pyplot(fig)

    # Heatmap Penyewaan Sepeda Berdasarkan Jam dan Hari (2012)
    st.subheader("Heatmap Penyewaan Sepeda Berdasarkan Jam dan Hari (2012)")
    pivot_table = hour_df_2012.pivot_table(index="weekday", columns="hr", values="cnt", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12,6))
    sns.heatmap(pivot_table, cmap="coolwarm", annot=False, linewidths=0.5, ax=ax)
    ax.set_xlabel("Jam dalam Sehari")
    ax.set_ylabel("Hari dalam Seminggu (0 = Minggu, 6 = Sabtu)")
    ax.set_title("Heatmap Penyewaan Sepeda Berdasarkan Jam dan Hari (2012)")
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
    df["Hour_Category"] = df["hr"].apply(lambda x: 'Peak Hours' if 7 <= x <= 9 or 16 <= x <= 19 else ('Normal Hours' if 10 <= x <= 15 else 'Off-Peak Hours'))
    hourly_clustering = df.groupby("Hour_Category")["cnt"].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Hour_Category", y="cnt", hue="Hour_Category", palette="coolwarm", legend=False, data=hourly_clustering, ax=ax)
    ax.set_xlabel("Kategori Jam")
    ax.set_ylabel("Rata-rata Penyewaan Sepeda")
    ax.set_title("Rata-rata Penyewaan Sepeda Berdasarkan Kategori Jam")
    st.pyplot(fig)
