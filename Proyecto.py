# -*- coding: utf-8 -*-
"""


@author: Gabriel Condori Ticona
"""

import pandas as pd 
import numpy as np
df=pd.read_csv('cardio_train.csv',sep=";")
print(df)
X_inicial= df.to_numpy()
print(X_inicial)

print("-------------------------------------------------------")

from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.NaN, strategy='mean')
X_salida=imp.fit_transform(X_inicial)
print(X_salida)

print("-------------------------------------------------------")

from sklearn import preprocessing
Aprepro = preprocessing.normalize(X_salida, norm='l1')
print(Aprepro)

print("-------------------------------------------------------")

aux1=Aprepro
X=np.delete(aux1, 12, axis=1)
X=np.delete(aux1, 0, axis=1)

y=np.delete(X_inicial,np.arange(12), axis=1)
from sklearn import model_selection 
X_train, X_test, y_train, y_test = model_selection.train_test_split( X, y, test_size=0.20)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

print("-------------------------------------------------------")

from sklearn.ensemble import RandomForestClassifier
clasificador = RandomForestClassifier(n_estimators = 100,criterion='entropy')
#clasificador.fit(X, y)

clasificador.fit(X_train, y_train)
print(len(X_train),len(X_test))
prediccion = clasificador.predict(X_test)
print(prediccion)

print("-------------------------------------------------------")

from sklearn.metrics import confusion_matrix
print(confusion_matrix( y_test, prediccion))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('\nExactitud: {:.2f}\n'.format(accuracy_score(y_test, prediccion)))

from sklearn.metrics import classification_report
print ( ' \n Informe de clasificación \n ' )
print ( classification_report ( y_test , prediccion , target_names = [ 'Sin enfermedad: 0','Con enfermedad: 1' ]))

print("-------------------------------------------------------")
