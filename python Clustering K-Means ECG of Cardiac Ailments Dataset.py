# %% [code] {"execution":{"iopub.status.busy":"2024-11-09T10:31:21.387795Z","iopub.execute_input":"2024-11-09T10:31:21.388253Z","iopub.status.idle":"2024-11-09T10:31:21.396023Z","shell.execute_reply.started":"2024-11-09T10:31:21.388214Z","shell.execute_reply":"2024-11-09T10:31:21.394990Z"},"jupyter":{"outputs_hidden":false}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2024-11-09T10:31:23.609569Z","iopub.execute_input":"2024-11-09T10:31:23.610578Z","iopub.status.idle":"2024-11-09T10:31:23.652626Z","shell.execute_reply.started":"2024-11-09T10:31:23.610530Z","shell.execute_reply":"2024-11-09T10:31:23.651491Z"},"jupyter":{"outputs_hidden":false}}
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn.pipeline import make_pipeline
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar los datos
dir_Input='/kaggle/input/k-means-clustering-xm/'
dir_Output='/kaggle/working/'
file_data='data.csv'
output='result.csv'

data_raw = pd.read_csv(dir_Input+file_data)  # Sustit#uye por tu dataset
data_input=data_raw.drop(columns=['ID'])
print(data_input)

# Estandarización de los dato
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_input)
data_scaled=pd.DataFrame(X_scaled, columns=data_input.columns)

#Eliminar columnas altamente correlacionadas

correlation_matrix = data_scaled.corr().abs()
upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool_))

to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
data_scaled.drop(columns=to_drop, inplace=True)

print(to_drop)

to_drop = [column for column in upper.columns if any(upper[column] < -0.90)]
data_scaled.drop(columns=to_drop, inplace=True)

to_drop = ['Pseg','NN50','Tseg','STseg','hbpermin','SDRR']
data_scaled.drop(columns=to_drop, inplace=True)

print(to_drop)

correlation_matrix.to_csv(dir_Output+'matriz_correlacion.csv', index=False)

# %% [code] {"execution":{"iopub.status.busy":"2024-11-09T10:31:27.746826Z","iopub.execute_input":"2024-11-09T10:31:27.747288Z","iopub.status.idle":"2024-11-09T10:31:28.491089Z","shell.execute_reply.started":"2024-11-09T10:31:27.747246Z","shell.execute_reply":"2024-11-09T10:31:28.490064Z"},"jupyter":{"outputs_hidden":false}}
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

wcss = []
K_range = range(1, 20)  # Probamos con valores de K desde 1 hasta 10

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=200, n_init=10, random_state=0)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)  # Inercia o WCSS para cada valor de K

# %% [code] {"execution":{"iopub.status.busy":"2024-11-09T10:31:29.713337Z","iopub.execute_input":"2024-11-09T10:31:29.713742Z","iopub.status.idle":"2024-11-09T10:31:30.011629Z","shell.execute_reply.started":"2024-11-09T10:31:29.713705Z","shell.execute_reply":"2024-11-09T10:31:30.010548Z"},"jupyter":{"outputs_hidden":false}}

# Graficar el método del codo
plt.figure(figsize=(10, 6))
plt.plot(K_range, wcss, 'bo-')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('SSE')
plt.title('Método del Codo para Determinar el Número Óptimo de Clusters')
plt.show()

diff_inertia = np.diff(wcss)
for i in range(len(diff_inertia)):
    print(f'De k={i+1} a k={i+2}: La inercia disminuye en {diff_inertia[i]:.2f}')

# %% [code] {"execution":{"iopub.status.busy":"2024-11-09T10:31:35.090009Z","iopub.execute_input":"2024-11-09T10:31:35.091088Z","iopub.status.idle":"2024-11-09T10:31:35.351141Z","shell.execute_reply.started":"2024-11-09T10:31:35.091040Z","shell.execute_reply":"2024-11-09T10:31:35.350004Z"},"jupyter":{"outputs_hidden":false}}
from sklearn.ensemble import IsolationForest

# Detectar outliers usando IsolationForest
iso = IsolationForest(contamination='auto', random_state=0)
yhat = iso.fit_predict(data_scaled)

# Seleccionar solo los datos que NO son outliers
mask = yhat != -1
df_no_outliers = data_scaled[mask]

# Convertir de nuevo a DataFrame para facilitar el uso
df_no_outliers = pd.DataFrame(df_no_outliers, columns=data_scaled.columns)


# Seleccionar solo los datos que SI son outliers
mask = yhat == -1
df_outliers = data_scaled[mask]

# Convertir de nuevo a DataFrame para facilitar el uso
df_outliers = pd.DataFrame(df_outliers, columns=data_scaled.columns)

print(df_outliers.count())
print(df_no_outliers.count())

#data_scaled=df_no_outliers

# %% [code] {"execution":{"iopub.status.busy":"2024-11-09T10:31:38.752394Z","iopub.execute_input":"2024-11-09T10:31:38.753410Z","iopub.status.idle":"2024-11-09T10:31:45.954461Z","shell.execute_reply.started":"2024-11-09T10:31:38.753362Z","shell.execute_reply":"2024-11-09T10:31:45.952972Z"},"jupyter":{"outputs_hidden":false}}
#Voy a comprobar como mejorar los parametros
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, silhouette_score

# Asumimos que tus datos están en X

# Definir el rango de parámetros a probar
param_grid = {
    'n_clusters': [5],  # Ya que sabemos que queremos 5 clusters
    'init': ['k-means++', 'random'],
    'n_init': [10, 20, 30],
    'max_iter': [200, 300, 500],
    'algorithm': ['auto', 'full', 'elkan']
}

# Crear un modelo base de KMeans
kmeans = KMeans(random_state=42)

# Definir una función de puntuación personalizada
# Usamos el silhouette_score, pero con el signo cambiado porque GridSearchCV maximiza el score
silhouette_scorer = make_scorer(silhouette_score, greater_is_better=True)

# Crear el objeto GridSearchCV
grid_search = GridSearchCV(
    estimator=kmeans,
    param_grid=param_grid,
    scoring=silhouette_scorer,
    cv=5,  # 5-fold cross-validation
    n_jobs=-1,  # Usar todos los núcleos disponibles
    verbose=2
)

# Realizar la búsqueda de hiperparámetros
grid_search.fit(df_no_outliers)

# Obtener los mejores parámetros y el mejor score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Mejores parámetros encontrados:")
print(best_params)
print(f"Mejor silhouette score: {best_score}")

# Crear el modelo final con los mejores parámetros
best_kmeans = KMeans(**best_params, random_state=42)

# Ajustar el modelo final a todos los datos
best_kmeans.fit(df_no_outliers)

# Obtener las etiquetas de cluster para cada punto de datos
labels = best_kmeans.labels_

# Calcular el silhouette score final
final_silhouette_score = silhouette_score(X, labels)
print(f"Silhouette score final: {final_silhouette_score}")
bit

# %% [code] {"execution":{"iopub.status.busy":"2024-11-09T10:31:51.247654Z","iopub.execute_input":"2024-11-09T10:31:51.248209Z","iopub.status.idle":"2024-11-09T10:31:51.300842Z","shell.execute_reply.started":"2024-11-09T10:31:51.248136Z","shell.execute_reply":"2024-11-09T10:31:51.299474Z"},"jupyter":{"outputs_hidden":false}}
#k=3 o k=4
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

k=5
grup_KMeans=df_no_outliers

print(grup_KMeans)
# Crear el modelo k-means con k=3
#kmeans = KMeans(n_clusters=k, n_init=5, max_iter=200, random_state=0)
#kmeans = KMeans('algorithm': 'auto', 'init': 'k-means++', 'max_iter': 200, 'n_clusters': 5, 'n_init': 10)
kmeans = KMeans(n_clusters=k,init= 'k-means++', n_init=10, max_iter=200, random_state=0)
# Ajustar el modelo a los datos
kmeans.fit(grup_KMeans)

# Obtener los centroides y las etiquetas
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

grup_KMeans['Cluster'] = kmeans.labels_

print(grup_KMeans.count())
print(kmeans.labels_)
print(centroids)

data_result=[]
data_result=data_raw[['ID']].copy()
data_result['Category']=grup_KMeans['Cluster'].astype(str)

print(data_result.count())

# %% [code] {"execution":{"iopub.status.busy":"2024-11-02T19:23:21.544422Z","iopub.execute_input":"2024-11-02T19:23:21.545652Z","iopub.status.idle":"2024-11-02T19:23:21.584509Z","shell.execute_reply.started":"2024-11-02T19:23:21.545588Z","shell.execute_reply":"2024-11-02T19:23:21.583025Z"},"jupyter":{"outputs_hidden":false}}
#VOY A QUITAR LOS OUTLIERS DEL CLUSTER.
from sklearn.cluster import DBSCAN
print(grup_KMeans.count())
print(df_outliers.count())
# Ajusta los parámetros según tus datos
dbscan = DBSCAN(eps=1.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(grup_KMeans)

# Los puntos con etiqueta -1 son considerados ruido/outliers por DBSCAN
noise_points = grup_KMeans[dbscan_labels == -1]

#filas_a_mover=noise_points.index.difference(grup_KMeans.index)
grup_KMeans=grup_KMeans.drop(noise_points.index)

df_outliers=pd.concat([df_outliers, noise_points], ignore_index=False)

print(noise_points.count())
print(df_outliers.count())
print(grup_KMeans.count())

# %% [code] {"execution":{"iopub.status.busy":"2024-11-09T10:31:57.198814Z","iopub.execute_input":"2024-11-09T10:31:57.199254Z","iopub.status.idle":"2024-11-09T10:31:57.533489Z","shell.execute_reply.started":"2024-11-09T10:31:57.199216Z","shell.execute_reply":"2024-11-09T10:31:57.532220Z"},"jupyter":{"outputs_hidden":false}}
# Visualizar los resultados
# Visualizar los resultados
plt.figure(figsize=(10, 6))
plt.scatter(grup_KMeans.iloc[:, 0], grup_KMeans.iloc[:, 1], c=grup_KMeans['Cluster'], cmap='viridis', marker='o', edgecolor='k', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='x')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Clusters y Centroides con k= ' + str(k))
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-11-09T10:16:21.986814Z","iopub.execute_input":"2024-11-09T10:16:21.987293Z","iopub.status.idle":"2024-11-09T10:16:22.050528Z","shell.execute_reply.started":"2024-11-09T10:16:21.987251Z","shell.execute_reply":"2024-11-09T10:16:22.049162Z"},"jupyter":{"outputs_hidden":false}}
#ASIGNAR  al grupo mas cercano
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df_no_outliers=grup_KMeans
df_no_outliers=df_no_outliers.drop('Cluster', axis=1)

grup_Kmeans_Out=df_outliers


# Verificar las dimensiones
print("Dimensiones de df_no_outliers:", df_no_outliers.shape)
print("Dimensiones de df_outliers:", df_outliers.shape)
print("Suma total de filas:", df_no_outliers.shape[0] + df_outliers.shape[0])
print("Dimensiones originales de data_scaled:", data_scaled.shape)

df_combined = pd.concat([df_no_outliers, df_outliers])
print("Dimensiones de df_combined:", df_combined.shape)

# Obtener las columnas usadas en el clustering
columnas_clustering = df_no_outliers.columns

# Ajustar df_outliers para que tenga solo esas columnas
df_outliers_ajustado = df_outliers[columnas_clustering]

# Si se aplicó escalado (por ejemplo, StandardScaler)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_no_outliers_scaled = scaler.fit_transform(df_no_outliers)

# Aplicar K-means a los datos no outliers escalados
kmeans_no_outliers = KMeans(n_clusters=k).fit(df_no_outliers_scaled)

# Escalar df_outliers_ajustado
df_outliers_scaled = scaler.transform(df_outliers_ajustado)

# Calcular distancias
distancias = cdist(df_outliers_scaled, kmeans_no_outliers.cluster_centers_)

# Asignar outliers a clusters
asignaciones_outliers = distancias.argmin(axis=1)

grup_Kmeans_Out['Cluster']=asignaciones_outliers.astype(str)
print(grup_Kmeans_Out)
print(grup_KMeans)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# 

# %% [code] {"execution":{"iopub.status.busy":"2024-11-09T10:16:28.191759Z","iopub.execute_input":"2024-11-09T10:16:28.192226Z","iopub.status.idle":"2024-11-09T10:16:28.728823Z","shell.execute_reply.started":"2024-11-09T10:16:28.192157Z","shell.execute_reply":"2024-11-09T10:16:28.727376Z"},"jupyter":{"outputs_hidden":false}}
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df_outliers=df_outliers.drop('Cluster', axis=1)

# Reducir dimensionalidad si es necesario
if df_no_outliers.shape[1] > 2:
    pca = PCA(n_components=2)
    no_outliers_2d = pca.fit_transform(df_no_outliers)
    outliers_2d = pca.transform(df_outliers)
    centers_2d = pca.transform(kmeans_no_outliers.cluster_centers_)
else:
    no_outliers_2d = df_no_outliers.values
    outliers_2d = df_outliers.values
    centers_2d = kmeans_no_outliers.cluster_centers_

# Crear la gráfica
plt.figure(figsize=(12, 8))

# Plotear los puntos no outliers
scatter = plt.scatter(no_outliers_2d[:, 0], no_outliers_2d[:, 1], 
                      c=kmeans_no_outliers.labels_, cmap='viridis',  marker='>',alpha=0.6)

# Plotear los outliers
plt.scatter(outliers_2d[:, 0], outliers_2d[:, 1], 
            c=asignaciones_outliers, cmap='viridis', marker='x', s=100, linewidths=2)

# Plotear los centroides
plt.scatter(centers_2d[:, 0], centers_2d[:, 1], 
            c='red', marker='*', s=300, label='Centroids')

plt.legend(*scatter.legend_elements(), title="Clusters")
plt.title('Clustering con Outliers Asignados')
plt.xlabel('Componente Principal 1' if df_no_outliers.shape[1] > 2 else 'Característica 1')
plt.ylabel('Componente Principal 2' if df_no_outliers.shape[1] > 2 else 'Característica 2')
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-11-02T19:33:42.495459Z","iopub.execute_input":"2024-11-02T19:33:42.495940Z","iopub.status.idle":"2024-11-02T19:33:42.515410Z","shell.execute_reply.started":"2024-11-02T19:33:42.495888Z","shell.execute_reply":"2024-11-02T19:33:42.513973Z"},"jupyter":{"outputs_hidden":false}}
import pandas as pd

print(grup_KMeans.count())
print(grup_Kmeans_Out.count())

result_NoOut = pd.DataFrame({
    'ID': grup_KMeans.index,
    'Category': grup_KMeans['Cluster']
})

result_out = pd.DataFrame({
    'ID': grup_Kmeans_Out.index,
    'Category': grup_Kmeans_Out['Cluster']
})

df_combined = pd.concat([result_out, result_NoOut], ignore_index=True)

# Verificar el nuevo DataFrame
print(df_combined)
df_combined.to_csv(dir_Output+output, index=False)

# %% [code] {"execution":{"iopub.status.busy":"2024-11-02T19:33:46.851451Z","iopub.execute_input":"2024-11-02T19:33:46.851931Z","iopub.status.idle":"2024-11-02T19:33:46.959015Z","shell.execute_reply.started":"2024-11-02T19:33:46.851882Z","shell.execute_reply":"2024-11-02T19:33:46.957363Z"},"jupyter":{"outputs_hidden":false}}
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score

#MEDICIONES:

print(f"Inertia: {kmeans.inertia_}")

sil_score = silhouette_score(df_combined, kmeans.labels_)
print(f"Silhouette Score: {sil_score}")

db_score = davies_bouldin_score(df_combined, kmeans.labels_)
print(f"Davies-Bouldin Index: {db_score}")

# %% [code] {"execution":{"iopub.status.busy":"2024-11-02T19:31:04.974104Z","iopub.execute_input":"2024-11-02T19:31:04.974520Z","iopub.status.idle":"2024-11-02T19:31:10.202154Z","shell.execute_reply.started":"2024-11-02T19:31:04.974475Z","shell.execute_reply":"2024-11-02T19:31:10.200739Z"},"jupyter":{"outputs_hidden":false}}
from sklearn.ensemble import IsolationForest, RandomForestClassifier

# Calcular estadísticas descriptivas por cluster
cluster_stats = grup_KMeans.groupby('Cluster').agg(['mean', 'std'])
print("Estadísticas descriptivas por cluster:")
print(cluster_stats)

# Visualizar las características de cada cluster
import seaborn as sns
import matplotlib.pyplot as plt

# Boxplot para cada característica por cluster
for feature in grup_KMeans.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cluster', y=feature, data=grup_KMeans)
    plt.title(f'Boxplot de {feature} por Cluster')
    plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false}}
