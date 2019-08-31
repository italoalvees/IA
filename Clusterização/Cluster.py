import pandas as pd

transfer = pd.read_csv('transfer.csv', index_col = 'team')
epl = pd.read_csv('epl_1819.csv', index_col = 'team')


transfer = transfer.drop(columns = ['end_2018','end_2017','end_2016','end_2015','end_2014','end_2013','end_2012','end_2011','end_2010'])
epl = epl.drop(columns= ['general_matches_played','general_goal_difference','general_squad_size','attack_goals_counter','attack_goals_box','finance _market_average','general_points'])

epl = epl.sort_index()
transfer = transfer.sort_index()

# for i in range(epl.index):
print(epl.index, transfer.index)

finaldata = pd.concat([epl,transfer["end_2019"],transfer['Total']], axis = 1)


finaldata.to_csv('finaldata.csv', encoding = 'utf-8', sep = ',', index = False)

print(finaldata)


