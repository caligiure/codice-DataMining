# Utilizza il dataset 'breast cancer' per addestrare un classificatore random forest 
# composto da 80 alberi decisionali per prevedere la classe target. Usare cross-validation 
# con 5 folds per valutare il modello. (9 punti)

# libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# load data

breast_cancer = load_breast_cancer()

# data frame 
df = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
X = breast_cancer.data
y = breast_cancer.target

model = RandomForestClassifier(80)
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f'Cross-validated Accuracy: {scores}')
