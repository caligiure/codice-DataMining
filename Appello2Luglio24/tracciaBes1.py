# Carica il dataset 'breast cancer' in un oggetto DataFrame e visualizza (stampa) le prime 
# 10 righe del DataFrame. Successivamente, crea un grafico a barre che mostri i valori 
# medi di ciascuna feature. (6 punti)

# load libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# load the data
breast_cancer = load_breast_cancer()

# Define the dataframe
df = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
df['target'] = breast_cancer.target
print(df.head(10))

# bar plot
mean_values = df.mean()
mean_values.plot (kind='bar')
plt.xlabel('Features')
plt.ylabel('Mean Value')
plt.title('Mean Values of Features in Breast Cancer Dataset')
plt.show()