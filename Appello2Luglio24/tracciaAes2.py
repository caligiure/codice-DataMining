# DOMANDA 2
# Applica l'algoritmo K-Means al dataset 'iris' per raggruppare i fiori in 3 cluster. 
# Successivamente, visualizza i cluster risultanti utilizzando un grafico a dispersione 
# che mostri lunghezza del petalo versus larghezza del petalo, con colori diversi che 
# rappresentano ciascun cluster. (6 punti)

# Importare le librerie necessarie:
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# load data and define dataframe
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# define K-means clustering
kmeans = KMeans(3)
# apply K-means clustering
df['Cluster'] = kmeans.fit_predict(df.iloc[:, :])

# scatter plot
plt.scatter(df['petal length (cm)'], df['petal width (cm)'])
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('K-Means Clustering of Iris Dataset')
plt.show()