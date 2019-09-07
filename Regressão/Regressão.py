import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import linear_model
from sklearn.metrics import r2_score , mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor



rg = pd.read_csv('train.csv')

rgY = rg['SalePrice']

rg = rg.drop(columns=['Id','MSSubClass','LotArea', 'OverallCond','BsmtFinSF2','BsmtUnfSF',"2ndFlrSF", 'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath','HalfBath', 'BedroomAbvGr', 'KitchenAbvGr','WoodDeckSF','OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'])

rgX = rg.drop(columns='SalePrice')

rgX['GarageYrBlt'] = rgX['GarageYrBlt'].interpolate(method= 'pad')
rgX['LotFrontage'] = rgX['LotFrontage'].fillna(0)

rgX = rgX.fillna('NaN')

le = preprocessing.LabelEncoder()

for i in rgX.columns:
	if rgX[i].dtypes == 'object':
		rgX[i] = le.fit_transform(rgX[i].astype(str))
	
rgX = Normalizer().fit_transform(rgX)


print(rgY.max())

trainX, testX, trainY, testY = train_test_split(rgX,rgY, test_size=0.2)


#_____________________________________________________________________________

Bayes = linear_model.BayesianRidge()

Bayes.fit(trainX, trainY)

BayesPredict = Bayes.predict(testX)

print("Bayes Score:" , r2_score(testY, BayesPredict), mean_absolute_error(testY, BayesPredict))

#_____________________________________________________________________________

Linear = linear_model.LinearRegression()

Linear.fit(trainX, trainY)

LinearPredict = Linear.predict(testX)

print("Linear Score:", r2_score(testY, LinearPredict), mean_absolute_error(testY, LinearPredict))

#______________________________________________________________________________

Neighbors = KNeighborsRegressor(n_neighbors = 5, weights = 'distance')

Neighbors.fit(trainX, trainY)

NeighborsPredict = Neighbors.predict(testX)

print("KNeighbors Score:", r2_score(testY, NeighborsPredict) , mean_absolute_error(testY, NeighborsPredict))

#______________________________________________________________________________

Tree = DecisionTreeRegressor(max_depth = 7)

Tree.fit(trainX, trainY)

TreePredict = Tree.predict(testX)

print('Tree Score:', r2_score(testY, TreePredict), mean_absolute_error(testY, TreePredict))

#_______________________________________________________________________________

Rede = MLPRegressor(learning_rate_init = 0.02,max_iter = 1000, batch_size = 100)

Rede.fit(trainX, trainY)

RedePredict = Rede.predict(testX)

print('Rede Neural Score:', r2_score(testY, RedePredict), mean_absolute_error(testY, RedePredict))