# Utilizza il dataset di immagini 'digits' per addestrare un classificatore random forest 
# composto da 80 alberi decisionali per prevedere la cifra. Successivamente, usa la 
# cross-validation con 5 folds per valutare l'accuratezza del modello. (9 punti)

# libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# load dataset
digits = load_digits()

# define data and labels
X = digits.data
y = digits.target

# define the classifier
model = RandomForestClassifier(n_estimators=80)

#define and calculate the score with cross validation
scores = cross_val_score(model, X, y, cv=5)

# print the mean of score
print(f'Cross-validated Accuracy: {scores.mean()}')