# Esegui il clustering K-Means sul dataset 'breast cancer' per raggruppare i campioni in 3 cluster. 
# Poi visualizza i cluster utilizzando un grafico a dispersione delle prime due 
# caratteristiche. (6 punti)

# libraries 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans

# load the data
breast_cancer = load_breast_cancer()

# Define the dataframe
df = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)

# Perform K-Means clustering
kmeans = KMeans(3)
df['Cluster'] = kmeans.fit_predict(df)

# Scatter plot of the first two features
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df['Cluster'], cmap='viridis')
plt.xlabel(breast_cancer.feature_names[0])
plt.ylabel(breast_cancer.feature_names[1])
plt.title('K-Means Clustering of Breast Cancer Dataset')
plt.show()
