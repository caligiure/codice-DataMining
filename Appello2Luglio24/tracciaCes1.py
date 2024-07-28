# Carica il dataset 'California housing' in un oggetto DataFrame e visualizza (stampa) 
# le prime 10 righe del DataFrame. Successivamente, crea un istogramma dell'attributo 'MedInc'
# (reddito mediano). (6 punti)

# libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# load the data
california = fetch_california_housing()

# Define the dataframe
df = pd.DataFrame(data=california.data, columns=california.feature_names)
df['target'] = california.target
print(df.head(10))

# histogram plot
df['MedInc'].hist()
plt.xlabel('Median Income')
plt.ylabel('Frequency')
plt.title('Histogram of Median Income')
plt.show()
