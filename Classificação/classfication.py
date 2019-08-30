import pandas as pd
from sklearn import preprocessing
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.neighbors import KNeighborsClassifier



finaldata = pd.read_csv('finalData.csv')


imp = IterativeImputer()

#Separa o resultado das demais variáveis
Y_completed = finaldata['Survived']

#Treinamento para preenchimento dos dados faltosos
imp.fit(finaldata.drop(columns='Survived'))

#Preenchimento dos dados faltosos
X_completed = imp.transform(finaldata.drop(columns='Survived'))

#Normalização dos dados
X_completed = Normalizer().fit_transform(X_completed)

#Divisão da amostra para teste e treino
X_trainCompleted, X_testCompleted,Y_trainCompleted, Y_testCompleted = train_test_split(X_completed,Y_completed, test_size=0.2)

#Treinamento Vizinhos
neighbors = KNeighborsClassifier(n_neighbors=7, weights= 'distance')
neighbors.fit(X_trainCompleted,Y_trainCompleted)

#Treinamento da Arvore
tree = DecisionTreeClassifier(max_depth= 10, splitter= 'random')
tree.fit(X_trainCompleted,Y_trainCompleted)

#Predição da Arvore e Vizinhos
Y_predictedCompleted = tree.predict(X_testCompleted)
Y_predictedNeighborsCompleted= neighbors.predict(X_testCompleted)



#__________________________________________________________________________________________________________

#Dados com instancias com dados faltosos e duplicados excluidos
excluded = finaldata.dropna().drop_duplicates()

#Separando e normalizando as variáveis do resultado
X_Excluded = excluded.drop(columns='Survived')
X_Excluded = Normalizer().fit_transform(X_Excluded)

Y_Excluded = excluded['Survived']

#Divisão da amostra para teste e treino
X_trainExcluded, X_testExcluded,Y_trainExcluded, Y_testExcluded = train_test_split(X_Excluded,Y_Excluded, test_size=0.2)

#Instanciando arvore de decisão e treinando
tree2 = DecisionTreeClassifier(max_depth= 10, splitter= 'random')
tree2.fit(X_trainExcluded,Y_trainExcluded)

#Treinamento Linear SVC
neighbors2 = KNeighborsClassifier(n_neighbors=7 ,weights= 'distance')
neighbors2.fit(X_trainExcluded,Y_trainExcluded)

#Predição da Arvore
Y_predictedExcluded = tree2.predict(X_testExcluded)
Y_predictedNeighborsExcluded = neighbors2.predict(X_testExcluded)

#___________________________________________________________________________________________________________

print("Tree Classification Report(Completed):\n", classification_report(Y_testCompleted,Y_predictedCompleted))

print("Tree Classification Report(Excluded):\n", classification_report(Y_testExcluded,Y_predictedExcluded))

print("K-Neighbors Classification Report(Completed):\n", classification_report(Y_testCompleted, Y_predictedNeighborsCompleted))

print("K-Neighbors Classification Report(Excluded):\n", classification_report(Y_testExcluded, Y_predictedNeighborsExcluded))




	
