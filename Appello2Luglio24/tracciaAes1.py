# DOMANDA 1
# Carica il dataset 'iris' in un oggetto DataFrame e visualizza (stampando)
# le prime 10 righe del DataFrame. Successivamente, crea un istogramma dell'attributo 
# 'sepal length'. (6 punti)

# load the libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Caricare il dataset 'iris':
#    Ogni riga rappresenta un campione (un fiore) e ogni colonna rappresenta una 
#    caratteristica (attributo) del fiore, come la lunghezza e la larghezza dei sepali e dei petali.
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
# DataFrame che ha:
#    Le righe che rappresentano i vari campioni di fiori nel dataset.
#    Le colonne che rappresentano le caratteristiche dei fiori 
# La linea di codice df['target'] = iris.target aggiunge una nuova colonna chiamata 
# 'target' al DataFrame df, contenente le etichette di classe per ciascun campione del dataset 'iris'. 

# Visualizzare le prime 10 righe del DataFrame:
print(df.head(10))

# Creare l'istogramma di 'sepal length':
plt.hist(df['sepal length (cm)'])
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.title('Histogram of Sepal Length')
plt.show()
