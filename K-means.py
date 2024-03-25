# %% [code] {"execution":{"iopub.status.busy":"2024-03-25T08:11:53.779499Z","iopub.execute_input":"2024-03-25T08:11:53.780018Z","iopub.status.idle":"2024-03-25T08:11:53.794754Z","shell.execute_reply.started":"2024-03-25T08:11:53.779981Z","shell.execute_reply":"2024-03-25T08:11:53.792806Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import warnings
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2024-03-25T08:11:53.797728Z","iopub.execute_input":"2024-03-25T08:11:53.798339Z","iopub.status.idle":"2024-03-25T08:11:53.818317Z","shell.execute_reply.started":"2024-03-25T08:11:53.798286Z","shell.execute_reply":"2024-03-25T08:11:53.816693Z"}}

warnings.filterwarnings('ignore')

# %% [code] {"execution":{"iopub.status.busy":"2024-03-25T08:11:53.820590Z","iopub.execute_input":"2024-03-25T08:11:53.821088Z","iopub.status.idle":"2024-03-25T08:11:53.841271Z","shell.execute_reply.started":"2024-03-25T08:11:53.821045Z","shell.execute_reply":"2024-03-25T08:11:53.839460Z"}}
data = '/kaggle/input/data-stunting-indonesia/Data Stunting Indonesia.csv'

df = pd.read_csv(data)

# %% [code] {"execution":{"iopub.status.busy":"2024-03-25T08:11:53.845239Z","iopub.execute_input":"2024-03-25T08:11:53.846227Z","iopub.status.idle":"2024-03-25T08:11:53.857830Z","shell.execute_reply.started":"2024-03-25T08:11:53.846163Z","shell.execute_reply":"2024-03-25T08:11:53.856460Z"}}
df.shape

# %% [code] {"execution":{"iopub.status.busy":"2024-03-25T08:11:53.860068Z","iopub.execute_input":"2024-03-25T08:11:53.860951Z","iopub.status.idle":"2024-03-25T08:11:53.879156Z","shell.execute_reply.started":"2024-03-25T08:11:53.860899Z","shell.execute_reply":"2024-03-25T08:11:53.877961Z"}}
df.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-25T08:11:53.880771Z","iopub.execute_input":"2024-03-25T08:11:53.882172Z","iopub.status.idle":"2024-03-25T08:11:53.901081Z","shell.execute_reply.started":"2024-03-25T08:11:53.882126Z","shell.execute_reply":"2024-03-25T08:11:53.899288Z"}}
df.info()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-25T08:11:53.903210Z","iopub.execute_input":"2024-03-25T08:11:53.904197Z","iopub.status.idle":"2024-03-25T08:11:53.915133Z","shell.execute_reply.started":"2024-03-25T08:11:53.904148Z","shell.execute_reply":"2024-03-25T08:11:53.913610Z"}}
df.isnull().sum()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-25T08:11:53.917139Z","iopub.execute_input":"2024-03-25T08:11:53.918313Z","iopub.status.idle":"2024-03-25T08:11:53.936308Z","shell.execute_reply.started":"2024-03-25T08:11:53.918263Z","shell.execute_reply":"2024-03-25T08:11:53.934775Z"}}
df.describe()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-25T08:11:53.939852Z","iopub.execute_input":"2024-03-25T08:11:53.940290Z","iopub.status.idle":"2024-03-25T08:11:53.959271Z","shell.execute_reply.started":"2024-03-25T08:11:53.940255Z","shell.execute_reply":"2024-03-25T08:11:53.957810Z"}}
df.info()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-25T08:11:53.985780Z","iopub.execute_input":"2024-03-25T08:11:53.986229Z","iopub.status.idle":"2024-03-25T08:11:54.775671Z","shell.execute_reply.started":"2024-03-25T08:11:53.986194Z","shell.execute_reply":"2024-03-25T08:11:54.774148Z"}}

# Data
provinsi = ["ACEH", "SUMATERA UTARA", "SUMATERA BARAT", "RIAU", "JAMBI", "SUMATERA SELATAN", "BENGKULU",
            "LAMPUNG", "KEPULAUAN BANGKA BELITUNG", "KEPULAUAN RIAU", "DKI JAKARTA", "JAWA BARAT",
            "JAWA TENGAH", "DI YOGYAKARTA", "JAWA TIMUR", "BANTEN", "BALI", "NUSA TENGGARA BARAT",
            "NUSA TENGGARA TIMUR", "KALIMANTAN BARAT", "KALIMANTAN TENGAH", "KALIMANTAN SELATAN",
            "KALIMANTAN TIMUR", "KALIMANTAN UTARA", "SULAWESI UTARA", "SULAWESI TENGAH", "SULAWESI SELATAN",
            "SULAWESI TENGGARA", "GORONTALO", "SULAWESI BARAT", "MALUKU", "MALUKU UTARA", "PAPUA BARAT",
            "PAPUA", "PAPUA PEGUNUNGAN", "PAPUA SELATAN", "PAPUA TENGAH", "PAPUA BARAT DAYA"]

tahun_2020 = [13.4, 6.8, 17.5, 7.7, 7.6, 2.3, 8.3, 5.3, 3.3, 9.2, 0.1, 9.0, 13.1, 11.9, 11.5, 4.9, 9.4,
              20.2, 26.3, 27.4, 22.2, 13.1, 14.5, 28.7, 5.3, 13.7, 11.9, 16.7, 11.0, 22.7, 8.3, 9.3, 12.0,
              4.7, 5.9, 11.3, 16.6, 17.7]

tahun_2021 = [12.1, 6.7, 15.1, 6.0, 3.0, 4.4, 6.3, 6.1, 5.9, 7.6, 3.2, 8.3, 9.0, 10.6, 10.7, 6.7, 5.0,
              21.7, 22.6, 21.0, 10.8, 10.4, 11.8, 18.5, 3.0, 13.2, 10.4, 18.5, 8.5, 19.3, 6.8, 13.0, 12.8,
              11.4, 7.5, 13.1, 6.6, 13.8]

tahun_2022 = [8.0, 5.5, 10.3, 4.2, 4.1, 3.1, 4.8, 4.5, 3.9, 4.9, 1.2, 6.9, 9.4, 9.2, 9.5, 6.8, 4.5,
              18.5, 22.4, 16.3, 10.1, 9.3, 13.5, 16.4, 2.3, 13.1, 9.0, 11.0, 7.5, 23.1, 9.9, 12.3, 11.9,
              2.3, 13.5, 16.5, 13.4, 13.2]

tahun_2023 = [7.5, 4.5, 8.8, 3.0, 3.7, 1.8, 4.5, 3.9, 3.2, 3.8, 1.0, 6.3, 9.1, 8.9, 6.9, 4.0, 3.2,
              15.8, 17.4, 17.5, 10.1, 8.7, 10.2, 8.9, 2.4, 11.3, 8.2, 10.3, 5.2, 23.2, 5.2, 6.8, 10.9,
              10.9, 1.4, 13.7, 9.7, 7.0]

# Membuat scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(provinsi, tahun_2020, label='2020')
plt.scatter(provinsi, tahun_2021, label='2021')
plt.scatter(provinsi, tahun_2022, label='2022')
plt.scatter(provinsi, tahun_2023, label='2023')

# Menambahkan label sumbu dan judul
plt.xlabel('Provinsi')
plt.ylabel('Prevalensi Stunting')
plt.title('Prevalensi Stunting di Berbagai Provinsi (2020-2023)')

# Menambahkan legenda
plt.legend()

# Mengatur agar label sumbu x tidak tumpang tindih
plt.xticks(rotation=90)

# Menampilkan plot
plt.show()


# %% [code] {"execution":{"iopub.status.busy":"2024-03-25T08:11:54.778441Z","iopub.execute_input":"2024-03-25T08:11:54.778912Z","iopub.status.idle":"2024-03-25T08:11:54.793760Z","shell.execute_reply.started":"2024-03-25T08:11:54.778873Z","shell.execute_reply":"2024-03-25T08:11:54.792278Z"}}

# Generate synthetic data for binary classification
X, y = make_classification(
    n_samples=1000, n_features=10, n_classes=2, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Print the shapes of the training and testing sets
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


# %% [code] {"execution":{"iopub.status.busy":"2024-03-25T08:11:54.795108Z","iopub.execute_input":"2024-03-25T08:11:54.795517Z","iopub.status.idle":"2024-03-25T08:11:54.809974Z","shell.execute_reply.started":"2024-03-25T08:11:54.795481Z","shell.execute_reply":"2024-03-25T08:11:54.808468Z"}}

# Inisialisasi MinMaxScaler
scaler = MinMaxScaler()

# Melakukan scaling pada data training
X_train_scaled = scaler.fit_transform(X_train)

# Melakukan scaling pada data testing
X_test_scaled = scaler.transform(X_test)


# %% [code] {"execution":{"iopub.status.busy":"2024-03-25T08:11:54.812015Z","iopub.execute_input":"2024-03-25T08:11:54.812834Z","iopub.status.idle":"2024-03-25T08:11:54.823660Z","shell.execute_reply.started":"2024-03-25T08:11:54.812785Z","shell.execute_reply":"2024-03-25T08:11:54.822211Z"}}

# Inisialisasi StandardScaler
scaler = StandardScaler()

# Melakukan scaling pada data training
X_train_scaled = scaler.fit_transform(X_train)

# Melakukan scaling pada data testing
X_test_scaled = scaler.transform(X_test)


# %% [code] {"execution":{"iopub.status.busy":"2024-03-25T08:11:54.826932Z","iopub.execute_input":"2024-03-25T08:11:54.827764Z","iopub.status.idle":"2024-03-25T08:11:55.697216Z","shell.execute_reply.started":"2024-03-25T08:11:54.827717Z","shell.execute_reply":"2024-03-25T08:11:55.695518Z"}}

# Data Provinsi dan Prevalensi Stunting
provinsi = ["ACEH", "SUMATERA UTARA", "SUMATERA BARAT", "RIAU", "JAMBI", "SUMATERA SELATAN", "BENGKULU",
            "LAMPUNG", "KEPULAUAN BANGKA BELITUNG", "KEPULAUAN RIAU", "DKI JAKARTA", "JAWA BARAT",
            "JAWA TENGAH", "DI YOGYAKARTA", "JAWA TIMUR", "BANTEN", "BALI", "NUSA TENGGARA BARAT",
            "NUSA TENGGARA TIMUR", "KALIMANTAN BARAT", "KALIMANTAN TENGAH", "KALIMANTAN SELATAN",
            "KALIMANTAN TIMUR", "KALIMANTAN UTARA", "SULAWESI UTARA", "SULAWESI TENGAH", "SULAWESI SELATAN",
            "SULAWESI TENGGARA", "GORONTALO", "SULAWESI BARAT", "MALUKU", "MALUKU UTARA", "PAPUA BARAT",
            "PAPUA", "PAPUA PEGUNUNGAN", "PAPUA SELATAN", "PAPUA TENGAH", "PAPUA BARAT DAYA"]

prevalensi_2020 = [13.4, 6.8, 17.5, 7.7, 7.6, 2.3, 8.3, 5.3, 3.3, 9.2, 0.1, 9.0, 13.1, 11.9, 11.5, 4.9, 9.4,
                   20.2, 26.3, 27.4, 22.2, 13.1, 14.5, 28.7, 5.3, 13.7, 11.9, 16.7, 11.0, 22.7, 8.3, 9.3, 12.0,
                   4.7, 5.9, 11.3, 16.6, 17.7]

prevalensi_2021 = [12.1, 6.7, 15.1, 6.0, 3.0, 4.4, 6.3, 6.1, 5.9, 7.6, 3.2, 8.3, 9.0, 10.6, 10.7, 6.7, 5.0,
                   21.7, 22.6, 21.0, 10.8, 10.4, 11.8, 18.5, 3.0, 13.2, 10.4, 18.5, 8.5, 19.3, 6.8, 13.0, 12.8,
                   11.4, 7.5, 13.1, 6.6, 13.8]

prevalensi_2022 = [8.0, 5.5, 10.3, 4.2, 4.1, 3.1, 4.8, 4.5, 3.9, 4.9, 1.2, 6.9, 9.4, 9.2, 9.5, 6.8, 4.5,
                   18.5, 22.4, 16.3, 10.1, 9.3, 13.5, 16.4, 2.3, 13.1, 9.0, 11.0, 7.5, 23.1, 9.9, 12.3, 11.9,
                   2.3, 13.5, 16.5, 13.4, 13.2]

prevalensi_2023 = [7.5, 4.5, 8.8, 3.0, 3.7, 1.8, 4.5, 3.9, 3.2, 3.8, 1.0, 6.3, 9.1, 8.9, 6.9, 4.0, 3.2,
                   15.8, 17.4, 17.5, 10.1, 8.7, 10.2, 8.9, 2.4, 11.3, 8.2, 10.3, 5.2, 23.2, 5.2, 6.8, 10.9,
                   10.9, 1.4, 13.7, 9.7, 7.0]

# Membuat scatter plot
plt.figure(figsize=(12, 8))
plt.scatter(provinsi, prevalensi_2020, label='2020', marker='o', s=100)
plt.scatter(provinsi, prevalensi_2021, label='2021', marker='s', s=100)
plt.scatter(provinsi, prevalensi_2022, label='2022', marker='^', s=100)
plt.scatter(provinsi, prevalensi_2023, label='2023', marker='d', s=100)

# Menambahkan label sumbu dan judul
plt.xlabel('Provinsi')
plt.ylabel('Prevalensi Stunting')
plt.title('Prevalensi Stunting di Berbagai Provinsi (2020-2023)')

# Menambahkan legenda
plt.legend()

# Mengatur agar label sumbu x tidak tumpang tindih
plt.xticks(rotation=90)

# Menampilkan plot
plt.tight_layout()
plt.show()


# %% [code] {"execution":{"iopub.status.busy":"2024-03-25T08:11:55.698967Z","iopub.execute_input":"2024-03-25T08:11:55.699541Z","iopub.status.idle":"2024-03-25T08:11:56.447677Z","shell.execute_reply.started":"2024-03-25T08:11:55.699363Z","shell.execute_reply":"2024-03-25T08:11:56.446411Z"}}

# Data
provinsi = ["ACEH", "SUMATERA UTARA", "SUMATERA BARAT", "RIAU", "JAMBI", "SUMATERA SELATAN", "BENGKULU",
            "LAMPUNG", "KEPULAUAN BANGKA BELITUNG", "KEPULAUAN RIAU", "DKI JAKARTA", "JAWA BARAT",
            "JAWA TENGAH", "DI YOGYAKARTA", "JAWA TIMUR", "BANTEN", "BALI", "NUSA TENGGARA BARAT",
            "NUSA TENGGARA TIMUR", "KALIMANTAN BARAT", "KALIMANTAN TENGAH", "KALIMANTAN SELATAN",
            "KALIMANTAN TIMUR", "KALIMANTAN UTARA", "SULAWESI UTARA", "SULAWESI TENGAH", "SULAWESI SELATAN",
            "SULAWESI TENGGARA", "GORONTALO", "SULAWESI BARAT", "MALUKU", "MALUKU UTARA", "PAPUA BARAT",
            "PAPUA", "PAPUA PEGUNUNGAN", "PAPUA SELATAN", "PAPUA TENGAH", "PAPUA BARAT DAYA"]

prevalensi_2020 = [13.4, 6.8, 17.5, 7.7, 7.6, 2.3, 8.3, 5.3, 3.3, 9.2, 0.1, 9.0, 13.1, 11.9, 11.5, 4.9, 9.4,
                   20.2, 26.3, 27.4, 22.2, 13.1, 14.5, 28.7, 5.3, 13.7, 11.9, 16.7, 11.0, 22.7, 8.3, 9.3, 12.0,
                   4.7, 5.9, 11.3, 16.6, 17.7]

# Menggunakan hanya data tahun 2020 untuk analisis clustering
X = np.array(prevalensi_2020).reshape(-1, 1)

# Standard Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Menentukan jumlah cluster yang optimal (misalnya, dari metode siku)
n_clusters = 3

# Melakukan k-means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_scaled)

# Menambahkan kolom cluster ke data
cluster_labels = kmeans.labels_

# Membuat DataFrame hasil clustering
result_df = pd.DataFrame(
    {'Provinsi': provinsi, 'Prevalensi_2020': prevalensi_2020, 'Cluster': cluster_labels})

# Menampilkan hasil clustering
print(result_df)

# Visualisasi hasil clustering
plt.figure(figsize=(10, 6))
for cluster in range(n_clusters):
    cluster_data = result_df[result_df['Cluster'] == cluster]
    plt.scatter(cluster_data['Provinsi'],
                cluster_data['Prevalensi_2020'], label=f'Cluster {cluster + 1}')

plt.title('Hasil K-Means Clustering pada Prevalensi Stunting (2020)')
plt.xlabel('Provinsi')
plt.ylabel('Prevalensi Stunting')
plt.legend()
plt.xticks(rotation=90)
plt.show()


# %% [code] {"execution":{"iopub.status.busy":"2024-03-25T08:11:56.449327Z","iopub.execute_input":"2024-03-25T08:11:56.449823Z","iopub.status.idle":"2024-03-25T08:11:57.468746Z","shell.execute_reply.started":"2024-03-25T08:11:56.449782Z","shell.execute_reply":"2024-03-25T08:11:57.467313Z"}}
# Menggunakan hanya data tahun 2021 untuk analisis clustering
X_2021 = np.array(prevalensi_2021).reshape(-1, 1)

# Standard Scaling
X_2021_scaled = scaler.transform(X_2021)

# Melakukan k-means clustering
kmeans_2021 = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_2021.fit(X_2021_scaled)

# Menambahkan kolom cluster ke data
cluster_labels_2021 = kmeans_2021.labels_

# Membuat DataFrame hasil clustering untuk tahun 2021
result_df_2021 = pd.DataFrame(
    {'Provinsi': provinsi, 'Prevalensi_2021': prevalensi_2021, 'Cluster_2021': cluster_labels_2021})

# Menampilkan hasil clustering untuk tahun 2021
print(result_df_2021)

# Visualisasi hasil clustering untuk tahun 2021
plt.figure(figsize=(10, 6))
for cluster in range(n_clusters):
    cluster_data = result_df_2021[result_df_2021['Cluster_2021'] == cluster]
    plt.scatter(cluster_data['Provinsi'],
                cluster_data['Prevalensi_2021'], label=f'Cluster {cluster + 1}')

plt.title('Hasil K-Means Clustering pada Prevalensi Stunting (2021)')
plt.xlabel('Provinsi')
plt.ylabel('Prevalensi Stunting')
plt.legend()
plt.xticks(rotation=90)
plt.show()


# %% [code] {"execution":{"iopub.status.busy":"2024-03-25T08:11:57.470631Z","iopub.execute_input":"2024-03-25T08:11:57.471049Z","iopub.status.idle":"2024-03-25T08:11:58.190964Z","shell.execute_reply.started":"2024-03-25T08:11:57.471014Z","shell.execute_reply":"2024-03-25T08:11:58.186562Z"}}
# Menggunakan hanya data tahun 2022 untuk analisis clustering
X_2022 = np.array(prevalensi_2022).reshape(-1, 1)

# Standard Scaling
X_2022_scaled = scaler.transform(X_2022)

# Melakukan k-means clustering
kmeans_2022 = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_2022.fit(X_2022_scaled)

# Menambahkan kolom cluster ke data
cluster_labels_2022 = kmeans_2022.labels_

# Membuat DataFrame hasil clustering untuk tahun 2022
result_df_2022 = pd.DataFrame(
    {'Provinsi': provinsi, 'Prevalensi_2022': prevalensi_2022, 'Cluster_2022': cluster_labels_2022})

# Menampilkan hasil clustering untuk tahun 2022
print(result_df_2022)

# Visualisasi hasil clustering untuk tahun 2022
plt.figure(figsize=(10, 6))
for cluster in range(n_clusters):
    cluster_data = result_df_2022[result_df_2022['Cluster_2022'] == cluster]
    plt.scatter(cluster_data['Provinsi'],
                cluster_data['Prevalensi_2022'], label=f'Cluster {cluster + 1}')

plt.title('Hasil K-Means Clustering pada Prevalensi Stunting (2022)')
plt.xlabel('Provinsi')
plt.ylabel('Prevalensi Stunting')
plt.legend()
plt.xticks(rotation=90)
plt.show()


# %% [code] {"execution":{"iopub.status.busy":"2024-03-25T08:11:58.193405Z","iopub.execute_input":"2024-03-25T08:11:58.195450Z","iopub.status.idle":"2024-03-25T08:12:00.997906Z","shell.execute_reply.started":"2024-03-25T08:11:58.195331Z","shell.execute_reply":"2024-03-25T08:12:00.996315Z"}}

# Data untuk tahun 2021, 2022, dan 2023
X_2021 = np.array(prevalensi_2021).reshape(-1, 1)
X_2022 = np.array(prevalensi_2022).reshape(-1, 1)
X_2023 = np.array(prevalensi_2023).reshape(-1, 1)

# Standard Scaling untuk masing-masing tahun
scaler_2021 = StandardScaler()
X_2021_scaled = scaler_2021.fit_transform(X_2021)

scaler_2022 = StandardScaler()
X_2022_scaled = scaler_2022.fit_transform(X_2022)

scaler_2023 = StandardScaler()
X_2023_scaled = scaler_2023.fit_transform(X_2023)

# Menentukan jumlah cluster yang optimal (misalnya, dari metode siku)


def elbow_method(X_scaled):
    variance = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X_scaled)
        variance.append(kmeans.inertia_)
    return variance


# Menggunakan metode siku untuk masing-masing tahun
variance_2021 = elbow_method(X_2021_scaled)
variance_2022 = elbow_method(X_2022_scaled)
variance_2023 = elbow_method(X_2023_scaled)

# Plot metode siku untuk tahun 2021
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.plot(range(1, 11), variance_2021, marker='o')
plt.title('Metode Siku untuk Tahun 2021')
plt.xlabel('Jumlah Cluster')
plt.ylabel('Variansi')

# Plot metode siku untuk tahun 2022
plt.subplot(1, 3, 2)
plt.plot(range(1, 11), variance_2022, marker='o')
plt.title('Metode Siku untuk Tahun 2022')
plt.xlabel('Jumlah Cluster')
plt.ylabel('Variansi')

# Plot metode siku untuk tahun 2023
plt.subplot(1, 3, 3)
plt.plot(range(1, 11), variance_2023, marker='o')
plt.title('Metode Siku untuk Tahun 2023')
plt.xlabel('Jumlah Cluster')
plt.ylabel('Variansi')

plt.tight_layout()
plt.show()
