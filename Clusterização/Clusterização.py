import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.cluster import AgglomerativeClustering

le = preprocessing.LabelEncoder()


data = pd.read_csv('finaldata.csv')
#print(data)

data['category'] = le.fit_transform(data['category'])

# d = Normalizer().fit_transform(data)

# print(data.columns)

# data = pd.DataFrame(d, columns = data.columns)

times = ['Arsenal:', "Bournemouth:", 'Brighton' ,'Burnley:', 'Cardiff:', 'Chelsea:', 'Crystal Palace:', 'Everton:', "Fulham:", 'Huddersfield:', 'Leicester:', 'Liverpool:', 'Manchester City:', 'Manchester United:', 'NewCastle United:', 'Southampton:', 'Tottenham:', 'Watford:', 'West Ham:', 'Wolverhampton:' ]

#_______________________________________________________________________________

km = KMeans(n_clusters = 3 , init = 'random')

km.fit(data)

clusters = km.predict(data)

#_______________________________________________________________________________

plt.scatter(data['general_league_position'], data['Total'], c=clusters, cmap='viridis')


centers = km.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=10, alpha=0.6);


#_______________________________________________________________________________


hie = AgglomerativeClustering(n_clusters = 3)

clHie = hie.fit_predict(data)

#_______________________________________________________________________________

plt.scatter(data['general_league_position'], data['Total'], c=clHie, cmap='viridis')

plt.show()
#_______________________________________________________________________________

for i in range(len(times)):
	print(times[i],'K:',clusters[i],'H:',clHie[i])






