# Addestra un classificatore Support Vector Machine (SVM) sul dataset 'breast cancer' per prevedere 
# la classe target. Valuta il modello utilizzando l'accuratezza e visualizza la matrice di confusione. 
# Utilizza il 20% dei dati per il test e il resto per l'addestramento. (9 punti)

# libraries
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# load data and define dataframe
breast_cancer = load_breast_cancer()
df = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
df['target'] = breast_cancer.target

#define x and y
X = df.drop('target', axis=1)
y = df['target']

# separate the trainig and test data 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#define the classifer 
model = SVC()

# apply the classifier
model.fit(X_train, y_train)

# see the classifier output for test data
y_pred = model.predict(X_test)

#calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

#define confusion matrix and plot 
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
