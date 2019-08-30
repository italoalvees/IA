import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import linear_model
from sklearn.metrics import r2_score , mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor

def normalizar(coluna):
	max = coluna.max()
	min = coluna.min()
	divisor = max 

	for i in range(len(coluna)):
		coluna.at[i] = coluna.at[i]/divisor

	return coluna



rg = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
testY = pd.read_csv('sample_submission.csv')

testY = testY['SalePrice']

rg = rg.drop(columns='Id')
testX = test.drop(columns='Id')

rgX = rg.drop(columns='SalePrice')
rgY = rg['SalePrice']

rgX['PoolQC'] = rgX['PoolQC'].fillna(0)
rgX['GarageYrBlt'] = rgX['GarageYrBlt'].interpolate(method= 'pad')
rgX['LotFrontage'] = rgX['LotFrontage'].fillna(0)

rgX = rgX.fillna('NaN')

testX['PoolQC'] = testX['PoolQC'].fillna(0)
testX['GarageYrBlt'] = testX['GarageYrBlt'].interpolate(method= 'pad')
testX['LotFrontage'] = testX['LotFrontage'].fillna(0)

testX = testX.fillna('NaN')

le = preprocessing.LabelEncoder()

for i in rgX.columns:
	rgX[i] = le.fit_transform(rgX[i].astype(str))
	testX[i] = le.fit_transform(testX[i].astype(str))

rgX = Normalizer().fit_transform(rgX)
testX = Normalizer().fit_transform(testX)
# rgY = normalizar(rgY)
# testY = normalizar(testY)


#_____________________________________________________________________________

Bayes = linear_model.BayesianRidge()

Bayes.fit(rgX , rgY)

BayesPredict = Bayes.predict(testX)

print("Bayes Score:" , r2_score(testY, BayesPredict), mean_absolute_error(testY, BayesPredict))

#_____________________________________________________________________________

Linear = linear_model.LinearRegression()

Linear.fit(rgX , rgY)

LinearPredict = Linear.predict(testX)

print("Linear Score:", r2_score(testY, LinearPredict), mean_absolute_error(testY, LinearPredict))

#______________________________________________________________________________

Neighbors = KNeighborsRegressor()

Neighbors.fit(rgX , rgY)

NeighborsPredict = Neighbors.predict(testX)

print("KNeighbors Score:", r2_score(testY, NeighborsPredict) , mean_absolute_error(testY, NeighborsPredict))

#______________________________________________________________________________

for i in range(len(testY)):
	print(i ,testY[i] , NeighborsPredict[i])