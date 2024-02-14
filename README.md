***

# Dugaan Bayes, DAG, dan Pemrograman Probabilistik

**Studi Kasus Regresi Harga Rumah**

*Posted on February 2024*

***

Beberapa waktu ini saya sedang mempelajari tentang pemrograman probabilistik. Sebuah paradigma yang digunakan dalam statistika komputasional. Apa itu statistika komputasional? Well, statistika adalah keilmuan yang bertarung dengan ketidakpastian. Dulu saat belum ada teknologi komputer, statistikawan melakukan statistik secara tangan kosong. Alias menerapkan solusi statistik secara analitik.

Seiring berkembangnya zaman, komputer hadir untuk memudahkan kerja manusia. Terima kasih kepada Turing dan Goedel yang menurut saya adalah pionir awal komputer. Menariknya, mereka tanpa sengaja membuat konsep komputer yang pada saat itu digunakan untuk menyelesaikan persoalan filsafat. Tapi mungkin ini cerita untuk lain waktu. Kembali ke komputer, adanya komputer ini mengizinkan manusia untuk menyelesaikan permasalahan yang sebelumnya tidak bisa diselesaikan secara analitik, mungkin karena tidak kontinyu atau rumit, menjadi bisa didekati secara numerik. Salah satu bidang yang kecipratan dengan adanya komputer adalah statistik dengan munculnya statistika komputasional.

Sebenarnya, kita sudah sering menggunakan statistika komputasional tanpa sadar. Mulai dari tingkat lanjut seperti kecerdasan buatan hingga yang sudah ada di kehidupan sehari-hari seperti excel. Ya... excel. Di dalam excel ada banyak fungsi statistik seperti kalkulasi p-value, solver, distribusi dan lain-lain. Semua itu adalah pengejawantahan atau pendekatan dari metode numerik terhadap metode analitik.

Sebelum sekitar 50 tahun terakhir ini, ada satu metode analitik dalam statistika yang masih susah untuk didekati secara numerik karena keterbatasan komputer. Metode itu adalah metode dugaan Bayes. Perbedaan metode ini dengan metode dari paradigma frekuentis seperti p-value adalah metode dugaan bayes mencoba untuk mengkuantifikasi ketidakpastian parameter yang paradigma frekuentis tidak coba lakukan. Akan panjang menjelaskan perbedaan paradigma Bayes dan paradigma frekuentis. Saya sudah coba menjelaskannya di sini tapi dalam bahasa Inggris.

Dewasa ini, dikarenakan perkembangan komputer, perkawinan metode dugaan bayes dengan metode numerik berhasil dilakukan dengan menghasilkan pemrograman probabilistik. Buku Statistical Rethinking karya Richard McElreath membicarakan mengenai pemanfaatan pemrograman probabilistik dengan komprehensif.

Jika bicara mengenai keuntungan menggunakan pemrograman probabilistik dibanding metode statistika komputasional lainnya, pemrograman probabilistik tidak hanya mampu melakukan dugaan seperti metode lainnya lakukan tapi juga dapat digunakan menganalisis darimana dugaan itu berasal. Salah satu hasilnya adalah explainable machine learning yang hadir sebagai solusi permasalahan kotak hitam machine learning.

Di buku Statistical Rethinking juga dijelaskan penggunaan DAG atau directed acyclic graph. DAG adalah diagram yang dapat digunakan untuk mengetahui kausalitas antar variabel.

Sebagai contoh mungkin kita bisa ambil studi kasus regresi harga rumah di California yang sudah ada datanya di Google Colab.


```python
!pip install numpyro
```

    Collecting numpyro
      Downloading numpyro-0.13.2-py3-none-any.whl (312 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m312.7/312.7 kB[0m [31m8.1 MB/s[0m eta [36m0:00:00[0m00:01[0m
    [?25hRequirement already satisfied: jax>=0.4.14 in /usr/local/lib/python3.10/dist-packages (from numpyro) (0.4.23)
    Requirement already satisfied: jaxlib>=0.4.14 in /usr/local/lib/python3.10/dist-packages (from numpyro) (0.4.23+cuda12.cudnn89)
    Requirement already satisfied: multipledispatch in /usr/local/lib/python3.10/dist-packages (from numpyro) (1.0.0)
    Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from numpyro) (1.25.2)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from numpyro) (4.66.1)
    Requirement already satisfied: ml-dtypes>=0.2.0 in /usr/local/lib/python3.10/dist-packages (from jax>=0.4.14->numpyro) (0.2.0)
    Requirement already satisfied: opt-einsum in /usr/local/lib/python3.10/dist-packages (from jax>=0.4.14->numpyro) (3.3.0)
    Requirement already satisfied: scipy>=1.9 in /usr/local/lib/python3.10/dist-packages (from jax>=0.4.14->numpyro) (1.11.4)
    Installing collected packages: numpyro
    Successfully installed numpyro-0.13.2
    


```python
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.diagnostics import print_summary
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoLaplaceApproximation
from numpyro.infer import Predictive
from numpyro import handlers

import jax.numpy as jnp
from jax import lax, random, vmap
```


```python
train_path = '/content/sample_data/california_housing_train.csv'
test_path = '/content/sample_data/california_housing_test.csv'

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
```


```python
df_train.head()
```





  <div id="df-613ac98b-6b39-4d72-bc01-7b130673218d" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-114.31</td>
      <td>34.19</td>
      <td>15.0</td>
      <td>5612.0</td>
      <td>1283.0</td>
      <td>1015.0</td>
      <td>472.0</td>
      <td>1.4936</td>
      <td>66900.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-114.47</td>
      <td>34.40</td>
      <td>19.0</td>
      <td>7650.0</td>
      <td>1901.0</td>
      <td>1129.0</td>
      <td>463.0</td>
      <td>1.8200</td>
      <td>80100.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-114.56</td>
      <td>33.69</td>
      <td>17.0</td>
      <td>720.0</td>
      <td>174.0</td>
      <td>333.0</td>
      <td>117.0</td>
      <td>1.6509</td>
      <td>85700.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-114.57</td>
      <td>33.64</td>
      <td>14.0</td>
      <td>1501.0</td>
      <td>337.0</td>
      <td>515.0</td>
      <td>226.0</td>
      <td>3.1917</td>
      <td>73400.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-114.57</td>
      <td>33.57</td>
      <td>20.0</td>
      <td>1454.0</td>
      <td>326.0</td>
      <td>624.0</td>
      <td>262.0</td>
      <td>1.9250</td>
      <td>65500.0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-613ac98b-6b39-4d72-bc01-7b130673218d')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-613ac98b-6b39-4d72-bc01-7b130673218d button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-613ac98b-6b39-4d72-bc01-7b130673218d');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-1fe4815e-a57f-4db7-b58d-ba069fe51e75">
  <button class="colab-df-quickchart" onclick="quickchart('df-1fe4815e-a57f-4db7-b58d-ba069fe51e75')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-1fe4815e-a57f-4db7-b58d-ba069fe51e75 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>




Data yang ada di Google Colab terdiri dari 9 variabel yaitu:

1. Latitude di suatu area
2. Longitude di suatu area
3. Median umur rumah di suatu area
4. Jumlah ruangan di suatu area
5. Jumlah kamar tidur di suatu area
6. Populasi di suatu area
7. Jumlah keluarga di suatu area
8. Median pendapatan di suatu area
9. Median harga rumah di suatu area

Di sini akan dicoba regresi median harga rumah terhadap 8 variabel lainnya. Mudah bukan? tinggal masukin saja variabel-variabel ini ke regresi linier dan jadi deh. Eits, tunggu dulu. Kebanyakan perangkat lunak yang sudah disediakan akan memperlakukan semua variabel tadi secara setara. Mungkin saja ada variabel-variabel yang aneh. Contohnya latitude dan longitude. Latitude dan longitude sudah seharusnya tidak saling mempengaruhi. Latitude suatu area tidak bisa digunakan untuk memprediksi longitude suatu area. Di sisi lain, latitude bisa mempengaruhi populasi dari suatu daerah. Mungkin saja orang-orang lebih prefer tinggal di daerah hangat daripada di daerah dingin. Lalu populasi juga bisa mempengaruhi jumlah keluarga yang ada. Jumlah keluarga bisa mempengaruhi pendapatan. Lalu pendapatan bisa mempengaruhi populasi. Loh balik lagi... ya memang permasalahan dunia nyata itu kompleks dan tidak tereduksi seperti apa yang kita dapat waktu sekolah.

DAG hadir untuk memudahkan analisis permasalahan kompleks ini. Menggunakan tools dari library NetworkX, kita bisa memodelkan interaksi antar variabel ini ke sebuah model jaringan. Contohnya, pengaruh populasi ke jumlah keluarga ke jumlah pendapatan dan pengaruh populasi ke jumlah pendapatan dapat dimodelkan menjadi


```python
G = nx.DiGraph()

nodes = ["Populasi", "Jumlah Keluarga", "Pendapatan"]
edges = [("Populasi", "Jumlah Keluarga"),
         ("Jumlah Keluarga", "Pendapatan"),
         ("Populasi", "Pendapatan")]

G.add_nodes_from(nodes)
G.add_edges_from(edges)

options = {
    "font_size": 6,
    "node_size": 3000,
    "node_color": "white",
    "edgecolors": "black",
    "linewidths": 5,
    "width": 5,
}

nx.draw_shell(G, **options, with_labels=True)
ax = plt.gca()
ax.margins(0.1)
plt.axis("off")
plt.show()
```


    
![png](output_8_0.png)
    


Inilah yand disebutt dengan DAG, graf terarah dan tak bersiklus. DAG dapat digunakan untuk melihat visualisasi interaksi antar variabel.

Pada kassus ini, interaksi yang dimodelkan antara lain:

- Longitude mungkin berhubungan dengan populasi
- Latitude mungkin berhubungan dengan populasi
- Longitude mungkin berhubungan dengan pendapatan
- Longitude mungkin berhubungan dengan populasi
- Jumlah kamar tidur mungkin berhubungan dengan jumlah ruangan
- Populasi mungkin berhubungan dengan jumlah ruangan
- Pendapatan mungkin berhubungan dengan jumlah ruangan
- Populasi mungkin berhubungan dengan jumlah kamar tidur
- Pendapatan mungkin berhubungan dengan jumlah kamar tidur
- Jumlah rumah tangga mungkin berhubungan dengan populasi
- Pendapatan mungkin berhubungan dengan populasi or vice versa
- Pendapaatan mungkin berhubungan dengan jumlah rumah tangga

Interaksi ini dimodelkan untuk meregresi harga rumah berdasarkan 8 variabel tersebut


```python
nodes = ["Longitude", "Latitude", "Jumlah Ruangan", "Jumlah Kamar Tidur", "Umur Rumah", "Populasi", "Jumlah Keluarga", "Pendapatan", "Harga Rumah"]
edges = [("Longitude", "Harga Rumah"),
         ("Latitude", "Harga Rumah"),
         ("Jumlah Ruangan", "Harga Rumah"),
         ("Jumlah Kamar Tidur", "Harga Rumah"),
         ("Umur Rumah", "Harga Rumah"),
         ("Populasi", "Harga Rumah"),
         ("Jumlah Keluarga", "Harga Rumah"),
         ("Pendapatan", "Harga Rumah"),
         ("Longitude", "Populasi"),
         ("Latitude", "Populasi"),
         ("Longitude", "Pendapatan"),
         ("Latitude", "Pendapatan"),
         ("Jumlah Kamar Tidur", "Jumlah Ruangan"),
         ("Populasi", "Jumlah Ruangan"),
         ("Pendapatan", "Jumlah Ruangan"),
         ("Populasi", "Jumlah Kamar Tidur"),
         ("Pendapatan", "Jumlah Kamar Tidur"),
         ("Jumlah Keluarga", "Populasi"),
         ("Pendapatan", "Populasi"),
         ("Pendapatan", "Jumlah Keluarga")]

G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)
```


```python
options = {
    "font_size": 6,
    "node_size": 3000,
    "node_color": "white",
    "edgecolors": "black",
    "linewidths": 5,
    "width": 5,
}

nx.draw_networkx(G, **options, with_labels=True)
ax = plt.gca()
ax.margins(0.1)
plt.axis("off")
plt.show()
```


    
![png](output_13_0.png)
    


# Dugaan Bayes

Untuk memudahkan interpretasi dan komputasi, dilakukan transformasi skala untuk sembilan variabel.


```python
df_train['longitude'] = (df_train['longitude'] - df_train['longitude'].mean())/df_train['longitude'].std(ddof=1)
df_train['latitude'] = (df_train['latitude'] - df_train['latitude'].mean())/df_train['latitude'].std(ddof=1)
df_train['housing_median_age'] = (df_train['housing_median_age'] - df_train['housing_median_age'].min())/df_train['housing_median_age'].max()
df_train['total_rooms'] = (df_train['total_rooms'] - df_train['total_rooms'].min())/df_train['total_rooms'].max()
df_train['total_bedrooms'] = (df_train['total_bedrooms'] - df_train['total_bedrooms'].min())/df_train['total_bedrooms'].max()
df_train['population'] = (df_train['population'] - df_train['population'].min())/df_train['population'].max()
df_train['households'] = (df_train['households'] - df_train['households'].min())/df_train['households'].max()
df_train['median_income'] = (df_train['median_income'] - df_train['median_income'].min())/df_train['median_income'].max()
df_train['median_house_value'] = (df_train['median_house_value'] - df_train['median_house_value'].min())/df_train['median_house_value'].max()
```


```python
df_test['longitude'] = (df_test['longitude'] - df_test['longitude'].mean())/df_test['longitude'].std(ddof=1)
df_test['latitude'] = (df_test['latitude'] - df_test['latitude'].mean())/df_test['latitude'].std(ddof=1)
df_test['housing_median_age'] = (df_test['housing_median_age'] - df_test['housing_median_age'].min())/df_test['housing_median_age'].max()
df_test['total_rooms'] = (df_test['total_rooms'] - df_test['total_rooms'].min())/df_test['total_rooms'].max()
df_test['total_bedrooms'] = (df_test['total_bedrooms'] - df_test['total_bedrooms'].min())/df_test['total_bedrooms'].max()
df_test['population'] = (df_test['population'] - df_test['population'].min())/df_test['population'].max()
df_test['households'] = (df_test['households'] - df_test['households'].min())/df_test['households'].max()
df_test['median_income'] = (df_test['median_income'] - df_test['median_income'].min())/df_test['median_income'].max()
df_test['median_house_value'] = (df_test['median_house_value'] - df_test['median_house_value'].min())/df_test['median_house_value'].max()
```

Dari sembilan variabel, hanya dua variabel yaitu latitude dan longitude yang menggunakan skala standar. Kenapa? karena untuk dua variabel ini, nilai negatif memiliki arti pada pembacaan data sehingga min-max tidak bisa dilakukan.


```python
columns = df_train.columns
features, target = columns[:-1], columns[-1]
X_train, y_train = df_train[features], df_train[target]
X_test, y_test = df_test[features], df_test[target]
```


```python
X_train.head()
```





  <div id="df-889f08e7-3be8-42f6-96f0-06cc0fe3a5e6" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.619288</td>
      <td>-0.671500</td>
      <td>0.269231</td>
      <td>0.147877</td>
      <td>0.198914</td>
      <td>0.028362</td>
      <td>0.077442</td>
      <td>0.066246</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.539494</td>
      <td>-0.573248</td>
      <td>0.346154</td>
      <td>0.201597</td>
      <td>0.294802</td>
      <td>0.031557</td>
      <td>0.075962</td>
      <td>0.088006</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.494610</td>
      <td>-0.905436</td>
      <td>0.307692</td>
      <td>0.018926</td>
      <td>0.026843</td>
      <td>0.009248</td>
      <td>0.019073</td>
      <td>0.076733</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.489623</td>
      <td>-0.928830</td>
      <td>0.250000</td>
      <td>0.039513</td>
      <td>0.052133</td>
      <td>0.014349</td>
      <td>0.036994</td>
      <td>0.179452</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.489623</td>
      <td>-0.961581</td>
      <td>0.365385</td>
      <td>0.038274</td>
      <td>0.050427</td>
      <td>0.017404</td>
      <td>0.042914</td>
      <td>0.095006</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-889f08e7-3be8-42f6-96f0-06cc0fe3a5e6')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-889f08e7-3be8-42f6-96f0-06cc0fe3a5e6 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-889f08e7-3be8-42f6-96f0-06cc0fe3a5e6');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-6ad62163-dbb9-4323-a60b-d73ddff15ff4">
  <button class="colab-df-quickchart" onclick="quickchart('df-6ad62163-dbb9-4323-a60b-d73ddff15ff4')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-6ad62163-dbb9-4323-a60b-d73ddff15ff4 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
latitude = X_train['latitude']
longitude = X_train['longitude']
housing_median_age = X_train['housing_median_age']
total_rooms = X_train['total_rooms']
total_bedrooms = X_train['total_bedrooms']
population = X_train['population']
households = X_train['households']
median_income = X_train['median_income']

median_house_value = y_train

dat_slim = {
    "latitude": X_train['latitude'].values,
    "longitude":  X_train['longitude'].values,
    "housing_median_age": X_train['housing_median_age'].values,
    "total_rooms": X_train['total_rooms'].values,
    "total_bedrooms": X_train['total_bedrooms'].values,
    "population": X_train['population'].values,
    "households": X_train['households'].values,
    "median_income": X_train['median_income'].values,
    "median_house_value": y_train.values
}
```


```python
def model(latitude=None, longitude=None, housing_median_age=None, total_rooms=None, total_bedrooms=None, population=None, households=None, median_income=None, median_house_value=None):
    Bla = numpyro.sample("latitude", dist.Normal(0.5, 0.2))
    Blo = numpyro.sample("longitude", dist.Normal(0.5, 0.2))
    Bha = numpyro.sample("housing_median_age", dist.Normal(0.5, 0.2))
    Btr = numpyro.sample("total_rooms", dist.Normal(0.5, 0.2))
    Btb = numpyro.sample("total_bedrooms", dist.Normal(0.5, 0.2))
    Bpo = numpyro.sample("population", dist.Normal(0.5, 0.2))
    Bho = numpyro.sample("households", dist.Normal(0.5, 0.2))
    Bmi = numpyro.sample("median_income", dist.Normal(0.5, 0.2))

    Blapo = numpyro.sample("interaction_latitude_population", dist.Normal(0, 0.2))
    Blopo = numpyro.sample("interaction_longitude_population", dist.Normal(0, 0.2))
    Blami = numpyro.sample("interaction_latitude_income", dist.Normal(0, 0.2))
    Blomi = numpyro.sample("interaction_longitude_income", dist.Normal(0, 0.2))
    Btrtb = numpyro.sample("interaction_total_rooms_total_bedrooms", dist.Normal(0, 0.2))
    Bpotr = numpyro.sample("interaction_population_total_rooms", dist.Normal(0, 0.2))
    Bmitr = numpyro.sample("interaction_income_rooms", dist.Normal(0, 0.2))
    Bpotb = numpyro.sample("interaction_population_total_bedrooms", dist.Normal(0, 0.2))

    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", Bla*latitude + Blo*longitude + Bha*housing_median_age + Btr*total_rooms + Btb*total_bedrooms + Bpo*population + Bho*households + Bmi*median_income + Blapo*latitude*population + Blopo*longitude*population + Btrtb*total_rooms*total_bedrooms + Blami*latitude*median_income + Blomi*longitude*median_income + Bpotr*population*total_rooms + Bmitr*median_income*total_rooms + Bpotb*population*total_bedrooms)
    #mu2 = numpyro.sample("mu2",  dist.Normal(mu, sigma))
    #numpyro.sample("rate", )
    #rate = numpyro.sample("rate", dist.Normal(mu, sigma))
    numpyro.sample("median_house_value", dist.Normal(mu, sigma), obs=median_house_value)


mcmc = MCMC(NUTS(model), num_warmup=50, num_samples=500, num_chains=1)
mcmc.run(random.PRNGKey(0), **dat_slim)
```

    sample: 100%|██████████| 550/550 [04:27<00:00,  2.06it/s, 1023 steps of size 4.56e-03. acc. prob=0.91]
    


```python
mcmc.print_summary(0.90)
```

    
                                                  mean       std    median      5.0%     95.0%     n_eff     r_hat
                                  households      0.41      0.16      0.43      0.27      0.58     33.18      1.03
                          housing_median_age      0.15      0.00      0.15      0.14      0.16     91.90      1.02
                    interaction_income_rooms      0.69      0.15      0.68      0.44      0.88     45.35      1.03
                 interaction_latitude_income     -0.36      0.09     -0.37     -0.41     -0.31     75.79      1.01
             interaction_latitude_population     -0.29      0.10     -0.29     -0.45     -0.15    164.48      1.02
                interaction_longitude_income     -0.32      0.09     -0.33     -0.37     -0.29     76.24      1.01
            interaction_longitude_population     -0.30      0.09     -0.31     -0.44     -0.18    152.81      1.01
       interaction_population_total_bedrooms      0.41      0.16      0.40      0.13      0.66     31.10      1.00
          interaction_population_total_rooms      0.90      0.14      0.89      0.66      1.09     25.91      1.01
      interaction_total_rooms_total_bedrooms     -1.21      0.19     -1.18     -1.37     -0.97     19.26      1.02
                                    latitude     -0.10      0.02     -0.10     -0.11     -0.09     68.76      1.01
                                   longitude     -0.10      0.02     -0.09     -0.10     -0.08     71.74      1.01
                               median_income      1.17      0.02      1.17      1.15      1.19     42.57      1.01
                                  population     -2.42      0.24     -2.46     -2.61     -2.31     33.18      1.03
                                       sigma      0.14      0.00      0.14      0.14      0.14     55.72      1.01
                              total_bedrooms      1.67      0.11      1.68      1.54      1.83     61.50      1.00
                                 total_rooms     -0.69      0.16     -0.71     -0.86     -0.56     29.71      1.01
    
    Number of divergences: 0
    

Dari sini, bisa dilihat hubungan-hubungan yang terjadi pada DAG di atas. Contohnya, pendapatan suatu daerah memiliki koefisien regresi sebesar 1.17 pada model di atas. Artinya, kenaikan 1 unit skala minmax pada variabel pendapatan mengakibatkan kenaikan harga rumah sebesar 1.17 unit minmax.

Poin plus dari dugaan bayes adalah metode ini dapat memodelkan ketidakpastian dari model. Jadi, selain titik tertinggi koefisien, yang biasanya di tunjukkan oleh rerata, juga bisa didapatkan interval kepercayaan setiap koefisien.


```python
numpyro.render_model(model, model_kwargs=dat_slim)
```




    
![svg](output_25_0.svg)
    




```python
samples = mcmc.get_samples()
```


```python
fig, axs = plt.subplots(1, 3, figsize=(15, 6))
axs[0].hist(x=samples["median_income"].flatten(), bins=50);
axs[1].hist(x=samples["total_rooms"].flatten(), bins=50);
axs[2].hist(x=samples["housing_median_age"].flatten(), bins=50);
axs[0].title.set_text('Koefisien Pendapatan')
axs[1].title.set_text('Koefisien Jumlah Ruangan')
axs[2].title.set_text('Koefisien Usia Rumah')
```


    
![png](output_27_0.png)
    


Dengan menggunakan dugaan bayes, bisa dilihat distribusi atau sebaran nilai koefisien dari model, inilah mengapa dugaan bayes digunakan dalam explainable machine learning.

# Uji Prediksi

Selanjutnya, dilakukan prediksi pada data yang belum pernah dilihat oleh model


```python
test_latitude = X_test['latitude']
test_longitude = X_test['longitude']
test_housing_median_age = X_test['housing_median_age']
test_total_rooms = X_test['total_rooms']
test_total_bedrooms = X_test['total_bedrooms']
test_population = X_test['population']
test_households = X_test['households']
test_median_income = X_test['median_income']

median_house_value = y_test

dat_slim_test = {
    "latitude": X_test['latitude'].values,
    "longitude":  X_test['longitude'].values,
    "housing_median_age": X_test['housing_median_age'].values,
    "total_rooms": X_test['total_rooms'].values,
    "total_bedrooms": X_test['total_bedrooms'].values,
    "population": X_test['population'].values,
    "households": X_test['households'].values,
    "median_income": X_test['median_income'].values,
    "median_house_value": None
}
```


```python
predictive = Predictive(model, samples)
predictions = predictive(random.PRNGKey(0),
                         **dat_slim_test)["median_house_value"]

pred = jnp.mean(predictions, axis=0)
```


```python
result = pd.DataFrame((y_test.values, pred)).T
result['Absolute Error'] = np.abs(result[0]-result[1]).astype(float)
```


```python
result
```





  <div id="df-d97afdb1-5d97-4614-91ee-9e07a14067d3" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>Absolute Error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.644399</td>
      <td>0.5389652</td>
      <td>0.105433</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.307999</td>
      <td>0.28676668</td>
      <td>0.021233</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.495999</td>
      <td>0.34676784</td>
      <td>0.149231</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.614999</td>
      <td>0.5883797</td>
      <td>0.026619</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.1184</td>
      <td>0.105462305</td>
      <td>0.012937</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2995</th>
      <td>0.404999</td>
      <td>0.17577356</td>
      <td>0.229226</td>
    </tr>
    <tr>
      <th>2996</th>
      <td>0.429399</td>
      <td>-0.050177466</td>
      <td>0.479577</td>
    </tr>
    <tr>
      <th>2997</th>
      <td>0.079</td>
      <td>0.046390552</td>
      <td>0.032609</td>
    </tr>
    <tr>
      <th>2998</th>
      <td>0.279999</td>
      <td>0.25315034</td>
      <td>0.026849</td>
    </tr>
    <tr>
      <th>2999</th>
      <td>0.955</td>
      <td>0.85326856</td>
      <td>0.101732</td>
    </tr>
  </tbody>
</table>
<p>3000 rows × 3 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-d97afdb1-5d97-4614-91ee-9e07a14067d3')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-d97afdb1-5d97-4614-91ee-9e07a14067d3 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-d97afdb1-5d97-4614-91ee-9e07a14067d3');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-efa7ea9e-1794-4a29-8fed-76fefcd19ec0">
  <button class="colab-df-quickchart" onclick="quickchart('df-efa7ea9e-1794-4a29-8fed-76fefcd19ec0')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-efa7ea9e-1794-4a29-8fed-76fefcd19ec0 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
plt.hist(result["Absolute Error"].values, bins=50);
plt.title("Distribusi Error")
plt.xlabel("Error")
plt.ylabel("Jumlah data")
```




    Text(0, 0.5, 'Jumlah data')




    
![png](output_35_1.png)
    



```python
mae=np.mean(result['Absolute Error'])
print(f"Rerata Error Absolut: {np.round(mae, 2)}")
```

    Rerata Error Absolut: 0.17
    


```python
plt.scatter(result[0], result[1])
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Nilai Asli")
plt.ylabel("Nilai Prediksi")
```




    Text(0, 0.5, 'Nilai Prediksi')




    
![png](output_37_1.png)
    


Dari gambar di atas, dapat dilihat semakin tinggi nilai asli, hasil prediksipun juga semakin tinggi. Tentu saja model yang sudah dibuat belum sempurna. Dengan penggunaan model lain selain model linear dengan interaksi antar variabel, bisa juga digunakan model lain yang dapat dibuat secara kustom. Contoh, di luaran model yang dibuat, digunakan distribusi gaussian yang mengizinkan nilai negatif muncul pada hasil prediksi. Hal ini seharusnya tidak dibolehkan karena harga rumah tidak mungkin negatif. Maka dapat digunakan model lebih baik dari model di atas. Inilah kelebihan dari pemrograman probabilistik.


```python

```
