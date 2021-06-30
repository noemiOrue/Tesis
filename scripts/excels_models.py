#import statsmodels.api as sm
import os
import pandas
import numpy
import copy
import _pickle as pickle
from sklearn.preprocessing import MinMaxScaler


#leemos excels

listColumns = ["ONU","GDP","Public_Grant","Budget_Previous_Year","Donor_Aid_Budget"]
listColumns = listColumns + ["LatinAmerica",'Africa','Colony','Delegation','Visitado'] 

training = {}
path = '../output/'
for root, dirs, files in os.walk(path):
    for filename in files:
        print(filename)
        if "_" not in filename and ".png" not in filename and ".txt" not in filename and "allExcels" not in filename and '~' not in filename:
            proyectos = pandas.read_excel(path+filename, sheet_name='Sheet1',index_col = "Pais-Año")
            proyectos = proyectos[listColumns]
            training[filename[:-5]]=proyectos.replace("..",0)

#training tenim tota la informacio de totes les ong

countriesTotal = set()
for ong in training:
    ongInfo = training[ong]
    for index,row in ongInfo.iterrows():
        country = index[index.index("_")+1:]
        countriesTotal.add(country)
        
ongRaw = pickle.load( open( "./ong.p", "rb" ) )
delegacionesONG = pickle.load( open( "./delegaciones.p", "rb" ) )
paises_ONU = pickle.load( open( "./paises_ONU.p", "rb" ))
spainAid = pickle.load(open( "./dinerosEspanya.p", "rb" ))

oldColonies = ["mexico","guatemala","el salvador","honduras","nicaragua","costa rica","panama"]
oldColonies = oldColonies + ["colombia","venezuela","ecuador","peru","bolivia","chile","argentina"]
oldColonies = oldColonies + ["paraguay","uruguay","cuba","puerto rico"]
oldColonies = oldColonies + ["filipinas","guam", "marruecos","sahara occidental","guinea ecuatorial"] 


PaisosVisitats = {}
llistatPaisos = {}
paisosTotal = set()

for ong in training:
    
    PaisosVisitats[ong]= []
    entries = set()
    listOngs = list(training.keys())
    examples = {}
    for rowPos in range(len(training[ong])):
        pais = training[ong].iloc[[rowPos]].index[0][5:] #pais
        paisosTotal.add(pais) #tots els paisos
        PaisosVisitats[ong].append(pais) #paisos visitats
        if pais not in llistatPaisos: 
            llistatPaisos[pais] = 0 #contem els paisos que es visiten
        llistatPaisos[pais]+=1
        entries.add(training[ong].iloc[[rowPos]].index[0]) # visites fetes!
        indexInfo =training[ong].iloc[[rowPos]].index[0]
        if indexInfo[0:4] not in examples: #exemples per a treure la info del pais en aquell any
            examples[indexInfo[0:4]] = training[ong].iloc[[rowPos]]
        #examples[indexInfo[0:4]][indexInfo[5:]]=training[ong].iloc[[rowPos]]
    
    entriesTotal = set(list(training[ong].index))
    for pais in countriesTotal:
        for year in ("2009","2010","2011","2012","2013","2014","2015","2016"):
            if year+"_"+pais not in entriesTotal:
                if year in examples:
                    newEntry = copy.deepcopy(examples[year]) #copiem les dades amb la referencia de l'any
                else:
                    for yearExample in examples:
                        newEntry = copy.deepcopy(examples[yearExample]) #copiem les dades anteriors... total, canviem tot
                        print(ong,year, pais,yearExample)
                        break
                newEntry.rename(index={newEntry.index[0]:year+"_"+pais},inplace=True)
                #if pais in paises_ONU:
                if "ONU" in paises_ONU[pais]:
                    newEntry["ONU"] = paises_ONU[pais]["ONU"][int(year)]
                else:
                    newEntry["ONU"] = 0
                newEntry["GDP"] = paises_ONU[pais]["money"][int(year)]
                #else:
                #    newEntry["ONU"] = 0
                #    newEntry["GDP"] = 0 #posar el minim
                if pais in spainAid[int(year)]:
                    newEntry["Donor_Aid_Budget"] = spainAid[int(year)][pais] 
                else:
                    newEntry["Donor_Aid_Budget"] =0
                if "SUBVENCIONES" in ongRaw[ong][int(year)]:
                    if pais in ongRaw[ong][int(year)]["SUBVENCIONES"]:
                        newEntry["Public_Grant"]= ongRaw[ong][int(year)]["SUBVENCIONES"][pais]
                    else: newEntry["Public_Grant"]= 0
                else:
                    newEntry["Public_Grant"]= 0
                if pais in oldColonies:
                    newEntry["Colony"] = 1
                else:
                    newEntry["Colony"] = 0
                if int(year) in [2013,2015]:
                    pYear = int(year) -2
                else:
                    pYear = int(year)-1
                if pYear in ongRaw[ong] and "PROYECTOS" in ongRaw[ong][pYear] and pais in ongRaw[ong][pYear]["PROYECTOS"]:
                    newEntry["Budget_Previous_Year"] =ongRaw[ong][pYear]["PROYECTOS"][pais]
                else:
                    newEntry["Budget_Previous_Year"] = 0
                if pais in delegacionesONG[ong][int(year)]:
                    newEntry["Delegation"] = 1
                else:
                    newEntry["Delegation"] = 0
                
                newEntry["Visitado"] = 0
                training[ong] = training[ong].append(newEntry)
               
    training[ong].to_excel("../output/"+ong+"_positivos_negativos.xlsx")
    

PaisosVisitatsNumero = {}
PaisosVisitatsRatio = {}
for ong in PaisosVisitats:
    ong = "economistas sin fronteras"
    PaisosVisitatsNumero[ong]=len(set(PaisosVisitats[ong]))
    PaisosVisitatsRatio[ong] = len(set(PaisosVisitats[ong]))/len(PaisosVisitats[ong])


#PaisosVisitatsRatioONG = {}
#for ong in paisosVisitatsONG:
#    PaisosVisitatsRatioONG[ong] = paisosVisitatsONG[ong].count(1)/len(paisosVisitatsONG[ong]) 


trainingGlobal = pandas.DataFrame()
trainingGlobal_negatiu = pandas.DataFrame()
for root, dirs, files in os.walk(path):
    for filename in files:
        if "_negativos" in filename:
            proyectos = pandas.read_excel(path+filename, sheet_name='Sheet1',index_col = "Pais-Año")
            proyectos = proyectos[listColumns]
            trainingGlobal_negatiu = trainingGlobal_negatiu.append(proyectos)
            
trainingGlobal_negatiu.to_excel("../output/allExcels_negatiu.xlsx",columns=listColumns)
    


len(trainingGlobal_negatiu)




            