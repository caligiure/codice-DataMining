# DOMANDA 3
# Addestra un classificatore albero decisionale sul dataset 'wine' per predire la classe del vino. 
# Valutare il modello utilizzando la metrica di accuratezza e visualizzare la matrice di confusione. 
# Usare il 30% dei dati per il test del modello, il resto per il training. (9 punti)

#  libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# load dataset and define dataframe
wine = load_wine()
df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
df['target'] = wine.target

# separate the data and label
X = df.drop('target', axis=1) # crea un nuovo DataFrame X rimuovendo la colonna 'target' da df
y = df['target'] # estrae la colonna 'target' dal DataFrame df e la assegna alla variabile y

# define the train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# Suddivide il dataset X (caratteristiche) e y (etichette) in quattro sottoinsiemi:
#    X_train: Le caratteristiche per l'addestramento (70% del dataset).
#    X_test: Le caratteristiche per il test (30% del dataset).
#    y_train: Le etichette per l'addestramento (70% del dataset).
#    y_test: Le etichette per il test (30% del dataset).

# define the classifier
model = DecisionTreeClassifier()
#apply the classifier
model.fit(X_train, y_train) # Addestrare il classificatore ad albero decisionale

# see the predictions on test data
y_pred = model.predict(X_test)

# calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

#print the accuracy
print(f'Accuracy: {accuracy}')

#define the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# use sns to plot the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
