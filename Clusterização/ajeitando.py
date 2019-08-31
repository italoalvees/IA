arq1 = open('finaldata.csv', 'r')
texto = arq1.read().replace('"','')
arq1.close()

arq2 = open('finaldata.csv', 'w')
arq2.write(texto)
arq2.close()

print(texto)