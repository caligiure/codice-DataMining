# Applicare l'algoritmo di clustering K-Means sul dataset 'California housing' per 
# raggruppare i campioni in 3 cluster. Poi visualizza i cluster utilizzando un grafico a 
# dispersione di 'MedInc' rispetto a 'AveOccup'. (6 punti)

# libraries 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.cluster import KMeans

# load the data
california = fetch_california_housing()

# Define the dataframe
df = pd.DataFrame(data=california.data, columns=california.feature_names)

# Perform K-Means clustering
kmeans = KMeans(3)
df['Cluster'] = kmeans.fit_predict(df)

# Scatter plot of MedInc vs AveOccup
plt.scatter(df['MedInc'], df['AveOccup'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Median Income')
plt.ylabel('Average Occupancy')
plt.title('K-Means Clustering of California Housing Dataset')
plt.show()
