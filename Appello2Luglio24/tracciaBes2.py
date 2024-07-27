# Utilizza il dataset di immagini 'digits' per addestrare un classificatore SVM per 
# prevedere la cifra. Successivamente, usa la cross-validation con 5 folds per valutare 
# l'accuratezza del modello. (9 punti)

#  libraries
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# load data
digits = load_digits()

# define data and label
X = digits.data
y = digits.target

# define classifier (use linear kernel)
model = SVC()

# evaluate the cross validation
scores = cross_val_score(model, X, y, cv=5)

# calculate the mean score and print
print(f'Cross-validated Accuracy: {scores.mean()}')
