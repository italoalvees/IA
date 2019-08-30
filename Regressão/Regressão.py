import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split, cross_val_score

rg = pd.read_csv('train.csv')

rg = rg.drop(columns='Id')


for i in range(10):
	print(i)

for i in rg.columns:
	if(type(rg[i]) == "<class 'str'>"):
		rg[i] = rg[i].fillna("nan")
	else:
		rg[i] = rg[i].fillna(0)
	print(rg[i])



for i in rg.columns:
	le = preprocessing.LabelEncoder()
	rg[i] = le.fit_transform(rg[i])

print(rg)

rg = Normalizer().fit_transform(rg)

print(rg)