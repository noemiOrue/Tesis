#import statsmodels.api as sm
import os
import pandas
import numpy
import copy
import _pickle as pickle
from sklearn.preprocessing import MinMaxScaler

#leemos excels

listColumns = ["ONU","Gross_National_Income","Public_Grant","Total_Fondos","Proporcion_Fondos_Privados","NGO_Country_Budget_Previous_Year","Total_subvencion_en_el_Pais_y_Anyo"]
listColumns = listColumns + ["Vision_ONGD_LatinAmerica",'Vision_ONGD_Africa','Vision_ONGD_Universal',"Anyo_ONG","Internacional",'Colony','Visitado','Dinero_en_el_proyecto'] 


training = {}

path = '../output/'
for root, dirs, files in os.walk(path):
    for filename in files:
        print(filename)
        if "_" not in filename and ".png" not in filename and ".txt" not in filename and "allExcels" not in filename and '~' not in filename:
            proyectos = pandas.read_excel(path+filename, sheet_name='Sheet1',index_col = "Pais-Año")
            proyectos = proyectos[listColumns]
            training[filename[:-5]]=proyectos.replace("..",0)

ongRaw = pickle.load( open( "./ong.p", "rb" ) )


PaisosVisitats = {}
#paisosVisitatsONG = {}
llistatPaisos = {}
paisosTotal = set()
len(list(paisosTotal))
for ong in training:
    
    PaisosVisitats[ong]= []
    entries = set()
    listOngs = list(training.keys())
    #paisosVisitatsONG[ong] = []
    examples = {}
    for rowPos in range(len(training[ong])):
        pais = training[ong].iloc[[rowPos]].index[0][5:] #pais
        paisosTotal.add(pais) #tots els paisos
        PaisosVisitats[ong].append(pais) #paisos visitats
        #paisosVisitatsONG[ong].append(int(training[ong].iloc[[rowPos]]["ONU"][0]))
        if pais not in llistatPaisos: 
            llistatPaisos[pais] = 0 #contem els paisos que es visiten
        llistatPaisos[pais]+=1
        entries.add(training[ong].iloc[[rowPos]].index[0]) # visites fetes!
        if training[ong].iloc[[rowPos]].index[0][0:4] not in examples: #exemples per a treure la info del pais en aquell any
            examples[training[ong].iloc[[rowPos]].index[0][0:4]] = training[ong].iloc[[rowPos]].index[0]
    
    for ong_other in training:
        if ong_other != ong: #per a cada ong diferent a la que tractem
            for rowPos in range(len(training[ong_other])): #per a cada entrada d'aquesta ong
                newValue = training[ong_other].iloc[[rowPos]] #nova entrada
                
                if newValue.index[0] not in entries and newValue.index[0][0:4] in examples: #si no tenim entrada per aquell pais pero si ha actuat en aquell any
                    entries.add(newValue.index[0]) #nova entrada!
                    newEntry = copy.deepcopy(training[ong].loc[[examples[newValue.index[0][0:4]]]]) #copiem les dades amb la referencia de l'any
                    newEntry = newEntry.set_index(newValue.index) #posem l'index
                    newEntry["ONU"] = newValue["ONU"]
                    newEntry["Gross_National_Income"] = newValue["Gross_National_Income"]
                    newEntry["NGO_Country_Budget_Previous_Year"] = 0
                    
                    year = int(newValue.index[0][0:4])-1
                    country = newValue.index[0][5:]
                    if year in ongRaw[ong]:
                        if "PROYECTOS" in ongRaw[ong][year]:
                            if country in ongRaw[ong][year]["PROYECTOS"]:
                                newEntry["NGO_Country_Budget_Previous_Year"] = ongRaw[ong][year]["PROYECTOS"][country]
                            else:
                                newEntry["NGO_Country_Budget_Previous_Year"] = 0
                        else:
                            newEntry["NGO_Country_Budget_Previous_Year"] = 0
                    else:
                        newEntry["NGO_Country_Budget_Previous_Year"] = 0
                    newEntry["Visitado"] = 0
                    newEntry["Dinero_en_el_proyecto"] = 0
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




            