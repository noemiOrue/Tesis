#import statsmodels.api as sm
import os
import pandas
import numpy
import scipy
import copy
import _pickle as pickle
from sklearn.preprocessing import MinMaxScaler

def dcg_at_k(r, k):
        
    if r is None or len(r) < 1:
        return 0.0

    r = numpy.asarray(r[:k])
    p = len(r)
    log2i = numpy.log2(range(2, p + 1))
    return r[0] + (r[1:] / log2i).sum()

def ndcg_at_k(r,rBest, k):
    
    dcg_max = dcg_at_k(sorted(rBest, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

#leemos excels

training = {}
path = './output/'
for root, dirs, files in os.walk(path):
    for filename in files:
        if "_" not in filename and "Excels" not in filename and ".png" not in filename and ".txt" not in filename:
            proyectos = pandas.read_excel(path+filename, sheet_name='Sheet1',index_col = "Pais-Año")
            training[filename[:-5]]=proyectos.replace("..",0)

pickle.dump(training, open( "./trainingONG.p", "wb" ) )

training_years = {} #proporcion publico/privado
dinerosONG = {}
espanya_subvenciones = {} #listado de paises subvencionados por españa cada año
espanya_subvenciones_dinero = {} #cantidad de dinero en las subvenciones
anualMAE = {} #dinero anual  MAE
anualPrivados = {} #dinero anual privado 

for ong in training:
    training_years[ong] = {}
    dinerosONG[ong] = {}
    anualMAE[ong] = {}
    anualPrivados[ong] = {}
    for rowPos in range(len(training[ong])):
        #subvenciones
        if training[ong].iloc[[rowPos]].index[0][0:4] not in espanya_subvenciones:
            espanya_subvenciones[training[ong].iloc[[rowPos]].index[0][0:4]] = []
            espanya_subvenciones_dinero[training[ong].iloc[[rowPos]].index[0][0:4]] = {}
        if training[ong].iloc[[rowPos]]["Subvencion_publica"][0] != 0:
            if training[ong].iloc[[rowPos]].index[0][5:] not in espanya_subvenciones_dinero[training[ong].iloc[[rowPos]].index[0][0:4]]:
                espanya_subvenciones_dinero[training[ong].iloc[[rowPos]].index[0][0:4]][training[ong].iloc[[rowPos]].index[0][5:]] = 0 
            espanya_subvenciones[training[ong].iloc[[rowPos]].index[0][0:4]].append(training[ong].iloc[[rowPos]].index[0][5:])
            espanya_subvenciones_dinero[training[ong].iloc[[rowPos]].index[0][0:4]][training[ong].iloc[[rowPos]].index[0][5:]] +=training[ong].iloc[[rowPos]]["Subvencion_publica"][0]
        #dinero en el proyecto
        if training[ong].iloc[[rowPos]].index[0][0:4] not in dinerosONG[ong]:
            dinerosONG[ong][training[ong].iloc[[rowPos]].index[0][0:4]]={}
            anualMAE[ong][training[ong].iloc[[rowPos]].index[0][0:4]] = training[ong].iloc[[rowPos]]["Fondos_Publicos_MAE"][0]
            anualPrivados[ong][training[ong].iloc[[rowPos]].index[0][0:4]] = training[ong].iloc[[rowPos]]["Fondos_Privados_Total"][0]
        if training[ong].iloc[[rowPos]].index[0][5:] not in dinerosONG[ong][training[ong].iloc[[rowPos]].index[0][0:4]]:
            dinerosONG[ong][training[ong].iloc[[rowPos]].index[0][0:4]][training[ong].iloc[[rowPos]].index[0][5:]] = 0
        dinerosONG[ong][training[ong].iloc[[rowPos]].index[0][0:4]][training[ong].iloc[[rowPos]].index[0][5:]]+=training[ong].iloc[[rowPos]]["Dinero_en_el_proyecto"][0]
        
        #fondos
        if training[ong].iloc[[rowPos]].index[0][0:4] not in training_years[ong]:
            training_years[ong][training[ong].iloc[[rowPos]].index[0][0:4]] = {}
            training_years[ong][training[ong].iloc[[rowPos]].index[0][0:4]]["publico"]= training[ong].iloc[[rowPos]]["Fondos_Publicos_Total"][0]
            training_years[ong][training[ong].iloc[[rowPos]].index[0][0:4]]["privado"]= training[ong].iloc[[rowPos]]["Fondos_Privados_Total"][0]

listP = []
for year in espanya_subvenciones:
    for pais in espanya_subvenciones[year]:
        listP.append(pais)
len(set(listP))

listP = set()
for ong in dinerosONG:
    for year in dinerosONG[ong]:
        for pais in  dinerosONG[ong][year]:
            listP.add(pais)
        

pickle.dump(dinerosONG, open( "./dinerosONG.p", "wb" ) )
        
training_years_proporcio = {}
for ong in training_years:
    public = 0
    privat = 0
    training_years_proporcio[ong] = 0
    for year in training_years[ong]:
        public += training_years[ong][year]["publico"]
        privat += training_years[ong][year]["privado"]
    training_years_proporcio[ong] = privat/(public+privat)


# primer: si les ong amb més diners privats van menys a països subvencionats
# Aixo es simplement mirar la proporcio dels paisos on van les ongs segons el diner privat:
# Mirem si (per exemple), les ong de mes de 70% de diner privat van menys (en %) als paisos 
# subvencionats per espanya per a aquell any
training70oMes = {}
trainingMenys70 = {}

for ong in training:
    if training_years_proporcio[ong] >=0.7:
        training70oMes[ong] = {}
        for rowPos in range(len(training[ong])): 
            if training[ong].iloc[[rowPos]].index[0][0:4] not in training70oMes[ong]:
                training70oMes[ong][training[ong].iloc[[rowPos]].index[0][0:4]] = []
            training70oMes[ong][training[ong].iloc[[rowPos]].index[0][0:4]].append(training[ong].iloc[[rowPos]].index[0][5:])
    else:
        trainingMenys70[ong] = {}
        for rowPos in range(len(training[ong])): 
            if training[ong].iloc[[rowPos]].index[0][0:4] not in trainingMenys70[ong]:
                trainingMenys70[ong][training[ong].iloc[[rowPos]].index[0][0:4]] = []
            trainingMenys70[ong][training[ong].iloc[[rowPos]].index[0][0:4]].append(training[ong].iloc[[rowPos]].index[0][5:])


subvencio70oMes = []
subvencioMenys70 = []
for ong in training70oMes:
    for year in training70oMes[ong]:
        for pais in training70oMes[ong][year]:
            if pais in espanya_subvenciones[year]:
                subvencio70oMes.append(1)
            else:
                subvencio70oMes.append(0)
for ong in trainingMenys70:
    for year in trainingMenys70[ong]:
        for pais in trainingMenys70[ong][year]:
            if pais in espanya_subvenciones[year]:
                subvencioMenys70.append(1)
            else:
                subvencioMenys70.append(0)         

subvencio70oMes.count(1)/len(subvencio70oMes)
subvencioMenys70.count(1)/len(subvencioMenys70)

training50oMes = {}
trainingMenys50 = {}

for ong in training:
    if training_years_proporcio[ong] >=0.5:
        training50oMes[ong] = {}
        for rowPos in range(len(training[ong])): 
            if training[ong].iloc[[rowPos]].index[0][0:4] not in training50oMes[ong]:
                training50oMes[ong][training[ong].iloc[[rowPos]].index[0][0:4]] = []
            training50oMes[ong][training[ong].iloc[[rowPos]].index[0][0:4]].append(training[ong].iloc[[rowPos]].index[0][5:])
    else:
        trainingMenys50[ong] = {}
        for rowPos in range(len(training[ong])): 
            if training[ong].iloc[[rowPos]].index[0][0:4] not in trainingMenys50[ong]:
                trainingMenys50[ong][training[ong].iloc[[rowPos]].index[0][0:4]] = []
            trainingMenys50[ong][training[ong].iloc[[rowPos]].index[0][0:4]].append(training[ong].iloc[[rowPos]].index[0][5:])

subvencio50oMes = []
subvencioMenys50 = []
for ong in training50oMes:
    for year in training50oMes[ong]:
        for pais in training50oMes[ong][year]:
            if pais in espanya_subvenciones[year]:
                subvencio50oMes.append(1)
            else:
                subvencio50oMes.append(0)
for ong in trainingMenys50:
    for year in trainingMenys50[ong]:
        for pais in trainingMenys50[ong][year]:
            if pais in espanya_subvenciones[year]:
                subvencioMenys50.append(1)
            else:
                subvencioMenys50.append(0)         

subvencio50oMes.count(1)/len(subvencio50oMes)
subvencioMenys50.count(1)/len(subvencioMenys50)

training25oMes = {}
trainingMenys25 = {}

for ong in training:
    if training_years_proporcio[ong] >=0.25:
        training25oMes[ong] = {}
        for rowPos in range(len(training[ong])): 
            if training[ong].iloc[[rowPos]].index[0][0:4] not in training25oMes[ong]:
                training25oMes[ong][training[ong].iloc[[rowPos]].index[0][0:4]] = []
            training25oMes[ong][training[ong].iloc[[rowPos]].index[0][0:4]].append(training[ong].iloc[[rowPos]].index[0][5:])
    else:
        trainingMenys25[ong] = {}
        for rowPos in range(len(training[ong])): 
            if training[ong].iloc[[rowPos]].index[0][0:4] not in trainingMenys25[ong]:
                trainingMenys25[ong][training[ong].iloc[[rowPos]].index[0][0:4]] = []
            trainingMenys25[ong][training[ong].iloc[[rowPos]].index[0][0:4]].append(training[ong].iloc[[rowPos]].index[0][5:])

subvencio25oMes = []
subvencioMenys25 = []
for ong in training25oMes:
    for year in training25oMes[ong]:
        for pais in training25oMes[ong][year]:
            if pais in espanya_subvenciones[year]:
                subvencio25oMes.append(1)
            else:
                subvencio25oMes.append(0)
for ong in trainingMenys25:
    for year in trainingMenys25[ong]:
        for pais in trainingMenys25[ong][year]:
            if pais in espanya_subvenciones[year]:
                subvencioMenys25.append(1)
            else:
                subvencioMenys25.append(0)         

subvencio25oMes.count(1)/len(subvencio25oMes)
subvencioMenys25.count(1)/len(subvencioMenys25)





# segon: mirar si les ongs es gasten els diners als mateixos paisos que les subvencions de espanya.
# son 46 rankings. El final donaria 1 llista per espanya, 45 llistes. 
# La idea a corroborar es: Les ong petites s'assemblen mes a la llista d'espanya

dinerosONGTotal = {}
for ong in dinerosONG:
    dinerosONGTotal[ong] = {}
    for year in dinerosONG[ong]:
        for pais in dinerosONG[ong][year]:
            if pais not in dinerosONGTotal[ong]:
                dinerosONGTotal[ong][pais] = 0
            dinerosONGTotal[ong][pais]+=dinerosONG[ong][year][pais]
                
espanyaONGTotal = {}
for year in espanya_subvenciones_dinero:
    for pais in espanya_subvenciones_dinero[year]:
        if pais not in espanyaONGTotal:
            espanyaONGTotal[pais] = 0
        espanyaONGTotal[pais]+=espanya_subvenciones_dinero[year][pais]

espanyaList = []
for entry in espanyaONGTotal:
    temp = [entry,espanyaONGTotal[entry]]
    espanyaList.append(temp)

espanyaList.sort(key = lambda x: x[1],reverse=True) 



####dos.1

espanyaListPaisesYear = {}
for year in espanya_subvenciones_dinero:
    espanyaListPaisesYear[year] = []
    for entry in espanya_subvenciones_dinero[year]:
        espanyaListPaisesYear[year].append([entry,espanya_subvenciones_dinero[year][entry]])
    espanyaListPaisesYear[year].sort(key = lambda x: x[1],reverse=True)
    for i in range(len(espanyaListPaisesYear[year])):
        espanyaListPaisesYear[year][i].append(len(espanyaListPaisesYear[year])-i)
        

dineroPaisesOrderYear = {}
for ong in training_years:
    dineroPaisesOrderYear[ong] = {}
    for year in dinerosONG[ong]:
        dineroPaisesOrderYear[ong][year] = []
        for entry in dinerosONG[ong][year]:
            dineroPaisesOrderYear[ong][year].append((entry,dinerosONG[ong][year][entry]))
        dineroPaisesOrderYear[ong][year].sort(key = lambda x: x[1],reverse=True)
    
NDCGM70 = []
NDCGm70 = []
NDCGM70P = []
NDCGm70P = []

for ong in training70oMes:
    priorizacionONGYear = []
    mitjanesNDCG = []
    mitjanesNDCGP = []
    for year in dineroPaisesOrderYear[ong]:
        priorizacionONGYear = []
        for entry in dineroPaisesOrderYear[ong][year]:
            pais = entry[0]
            info = (0,0)
            for el in espanyaListPaisesYear[year]:
                if el[0] == pais:
                    info = [el[1],el[2]]
                    break
            priorizacionONGYear.append(info)
            
        valueDinero = ndcg_at_k([entry[0] for entry in priorizacionONGYear],[entry[1] for entry in espanyaListPaisesYear[year]]+[0]*(194-len(espanyaListPaisesYear[year])),len(priorizacionONGYear))
        valuePosicion = ndcg_at_k([entry[1] for entry in priorizacionONGYear],[entry[2] for entry in espanyaListPaisesYear[year]]+[0]*(194-len(espanyaListPaisesYear[year])),len(priorizacionONGYear))
        
        mitjanesNDCG.append(valueDinero)
        mitjanesNDCGP.append(valuePosicion)
    
    NDCGM70.append(numpy.mean(mitjanesNDCG))
    NDCGM70P.append(numpy.mean(mitjanesNDCGP))

numpy.mean(NDCGM70)
numpy.mean(NDCGM70P)

for ong in trainingMenys70:
    priorizacionONGYear = []
    mitjanesNDCG = []
    mitjanesNDCGP = []
    for year in dineroPaisesOrderYear[ong]:
        priorizacionONGYear = []
        for entry in dineroPaisesOrderYear[ong][year]:
            pais = entry[0]
            info = (0,0)
            for el in espanyaListPaisesYear[year]:
                if el[0] == pais:
                    info = [el[1],el[2]]
                    break
            priorizacionONGYear.append(info)
            
        valueDinero = ndcg_at_k([entry[0] for entry in priorizacionONGYear],[entry[1] for entry in espanyaListPaisesYear[year]]+[0]*(194-len(espanyaListPaisesYear[year])),len(priorizacionONGYear))
        valuePosicion = ndcg_at_k([entry[1] for entry in priorizacionONGYear],[entry[2] for entry in espanyaListPaisesYear[year]]+[0]*(194-len(espanyaListPaisesYear[year])),len(priorizacionONGYear))
        
        mitjanesNDCG.append(valueDinero)
        mitjanesNDCGP.append(valuePosicion)
    
    NDCGm70.append(numpy.mean(mitjanesNDCG))
    NDCGm70P.append(numpy.mean(mitjanesNDCGP))

numpy.mean(NDCGm70)
numpy.mean(NDCGm70P)


NDCGM50 = []
NDCGm50 = []
NDCGM50P = []
NDCGm50P = []

for ong in training50oMes:
    priorizacionONGYear = []
    mitjanesNDCG = []
    mitjanesNDCGP = []
    for year in dineroPaisesOrderYear[ong]:
        priorizacionONGYear = []
        for entry in dineroPaisesOrderYear[ong][year]:
            pais = entry[0]
            info = (0,0)
            for el in espanyaListPaisesYear[year]:
                if el[0] == pais:
                    info = [el[1],el[2]]
                    break
            priorizacionONGYear.append(info)
            
        valueDinero = ndcg_at_k([entry[0] for entry in priorizacionONGYear],[entry[1] for entry in espanyaListPaisesYear[year]]+[0]*(194-len(espanyaListPaisesYear[year])),len(priorizacionONGYear))
        valuePosicion = ndcg_at_k([entry[1] for entry in priorizacionONGYear],[entry[2] for entry in espanyaListPaisesYear[year]]+[0]*(194-len(espanyaListPaisesYear[year])),len(priorizacionONGYear))
        
        mitjanesNDCG.append(valueDinero)
        mitjanesNDCGP.append(valuePosicion)
    
    NDCGM50.append(numpy.mean(mitjanesNDCG))
    NDCGM50P.append(numpy.mean(mitjanesNDCGP))

numpy.mean(NDCGM50)
numpy.mean(NDCGM50P)

for ong in trainingMenys50:
    priorizacionONGYear = []
    mitjanesNDCG = []
    mitjanesNDCGP = []
    for year in dineroPaisesOrderYear[ong]:
        priorizacionONGYear = []
        for entry in dineroPaisesOrderYear[ong][year]:
            pais = entry[0]
            info = (0,0)
            for el in espanyaListPaisesYear[year]:
                if el[0] == pais:
                    info = [el[1],el[2]]
                    break
            priorizacionONGYear.append(info)
            
        valueDinero = ndcg_at_k([entry[0] for entry in priorizacionONGYear],[entry[1] for entry in espanyaListPaisesYear[year]]+[0]*(194-len(espanyaListPaisesYear[year])),len(priorizacionONGYear))
        valuePosicion = ndcg_at_k([entry[1] for entry in priorizacionONGYear],[entry[2] for entry in espanyaListPaisesYear[year]]+[0]*(194-len(espanyaListPaisesYear[year])),len(priorizacionONGYear))
        
        mitjanesNDCG.append(valueDinero)
        mitjanesNDCGP.append(valuePosicion)
    
    NDCGm50.append(numpy.mean(mitjanesNDCG))
    NDCGm50P.append(numpy.mean(mitjanesNDCGP))

numpy.mean(NDCGm50)
numpy.mean(NDCGm50P)

NDCGM25 = []
NDCGm25 = []
NDCGM25P = []
NDCGm25P = []

for ong in training25oMes:
    priorizacionONGYear = []
    mitjanesNDCG = []
    mitjanesNDCGP = []
    for year in dineroPaisesOrderYear[ong]:
        priorizacionONGYear = []
        for entry in dineroPaisesOrderYear[ong][year]:
            pais = entry[0]
            info = (0,0)
            for el in espanyaListPaisesYear[year]:
                if el[0] == pais:
                    info = [el[1],el[2]]
                    break
            priorizacionONGYear.append(info)
            
        valueDinero = ndcg_at_k([entry[0] for entry in priorizacionONGYear],[entry[1] for entry in espanyaListPaisesYear[year]]+[0]*(194-len(espanyaListPaisesYear[year])),len(priorizacionONGYear))
        valuePosicion = ndcg_at_k([entry[1] for entry in priorizacionONGYear],[entry[2] for entry in espanyaListPaisesYear[year]]+[0]*(194-len(espanyaListPaisesYear[year])),len(priorizacionONGYear))
        
        mitjanesNDCG.append(valueDinero)
        mitjanesNDCGP.append(valuePosicion)
    
    NDCGM25.append(numpy.mean(mitjanesNDCG))
    NDCGM25P.append(numpy.mean(mitjanesNDCGP))

numpy.mean(NDCGM25)
numpy.mean(NDCGM25P)

for ong in trainingMenys25:
    priorizacionONGYear = []
    mitjanesNDCG = []
    mitjanesNDCGP = []
    for year in dineroPaisesOrderYear[ong]:
        priorizacionONGYear = []
        for entry in dineroPaisesOrderYear[ong][year]:
            pais = entry[0]
            info = (0,0)
            for el in espanyaListPaisesYear[year]:
                if el[0] == pais:
                    info = [el[1],el[2]]
                    break
            priorizacionONGYear.append(info)
            
        valueDinero = ndcg_at_k([entry[0] for entry in priorizacionONGYear],[entry[1] for entry in espanyaListPaisesYear[year]]+[0]*(194-len(espanyaListPaisesYear[year])),len(priorizacionONGYear))
        valuePosicion = ndcg_at_k([entry[1] for entry in priorizacionONGYear],[entry[2] for entry in espanyaListPaisesYear[year]]+[0]*(194-len(espanyaListPaisesYear[year])),len(priorizacionONGYear))
        
        mitjanesNDCG.append(valueDinero)
        mitjanesNDCGP.append(valuePosicion)
    
    NDCGm25.append(numpy.mean(mitjanesNDCG))
    NDCGm25P.append(numpy.mean(mitjanesNDCGP))

numpy.mean(NDCGm25)
numpy.mean(NDCGm25P)

