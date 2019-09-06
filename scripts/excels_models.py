#import statsmodels.api as sm
import os
import pandas
import numpy
import copy
import _pickle as pickle
from sklearn.preprocessing import MinMaxScaler

#leemos excels
training = {}

llistatPaisos = {}
path = './output/'
for root, dirs, files in os.walk(path):
    for filename in files:
        if "_" not in filename and ".png" not in filename and ".txt" not in filename and "allExcels" not in filename:
            print(filename)
            proyectos = pandas.read_excel(path+filename, sheet_name='Sheet1',index_col = "Pais-Año")
            training[filename[:-5]]=proyectos.replace("..",0)

ongRaw = pickle.load( open( "./ong.p", "rb" ) )

listColumns = ["ONU","Gross_National_Income","Subvencion_publica","Fondos_Publicos_MAE","Total_Fondos","Proporcion_Fondos_Privados","dinero_anyo_anterior_en_proyectos","Total_subvencion_en_el_Pais_y_Anyo"]
listColumns = listColumns + ["Vision_ONGD_Latinoamerica",'Vision_ONGD_Africa','Vision_ONGD_Universal','Visitado','Dinero_en_el_proyecto'] 


PaisosVisitats = {}
paisosVisitatsONG = {}

for ong in training:
    
    PaisosVisitats[ong]= []
    entries = set()
    listOngs = list(training.keys())
    paisosVisitatsONG[ong] = []
    examples = {}
    for rowPos in range(len(training[ong])):
        pais = training[ong].iloc[[rowPos]].index[0][5:]
        PaisosVisitats[ong].append(pais)
        paisosVisitatsONG[ong].append(int(training[ong].iloc[[rowPos]]["ONU"][0]))
        if pais not in llistatPaisos:
            llistatPaisos[pais] = 0
        llistatPaisos[pais]+=1
        entries.add(training[ong].iloc[[rowPos]].index[0])
        if training[ong].iloc[[rowPos]].index[0][0:4] not in examples:
            examples[training[ong].iloc[[rowPos]].index[0][0:4]] = training[ong].iloc[[rowPos]].index[0]
    
    for ong_other in training:
        if ong_other != ong:
            for rowPos in range(len(training[ong_other])):
                newValue = training[ong_other].iloc[[rowPos]]
                if newValue.index[0] not in entries and newValue.index[0][0:4] in examples:
                    entries.add(newValue.index[0])
                    newEntry = copy.deepcopy(training[ong].loc[[examples[newValue.index[0][0:4]]]])
                    newEntry = newEntry.set_index(newValue.index)
                    newEntry["ONU"] = newValue["ONU"]
                    newEntry["Gross_National_Income"] = newValue["Gross_National_Income"]
                    newEntry["dinero_anyo_anterior_en_proyectos"] = 0
                    
                    year = int(newValue.index[0][0:4])-1
                    country = newValue.index[0][5:]
                    if year in ongRaw[ong]:
                        if "PROYECTOS" in ongRaw[ong][year]:
                            if country in ongRaw[ong][year]["PROYECTOS"]:
                                newEntry["dinero_anyo_anterior_en_proyectos"] = ongRaw[ong][year]["PROYECTOS"][country]
                            else:
                                newEntry["dinero_anyo_anterior_en_proyectos"] = 0
                        else:
                            newEntry["dinero_anyo_anterior_en_proyectos"] = 0
                    else:
                        newEntry["dinero_anyo_anterior_en_proyectos"] = 0
                    newEntry["Visitado"] = 0
                    newEntry["Dinero_en_el_proyecto"] = 0
                    training[ong] = training[ong].append(newEntry)

    training[ong].to_excel("./output/"+ong+"_positivos_negativos.xlsx",columns=listColumns)



PaisosVisitatsNumero = {}
PaisosVisitatsRatio = {}
for ong in PaisosVisitats:
    ong = "economistas sin fronteras"
    PaisosVisitatsNumero[ong]=len(set(PaisosVisitats[ong]))
    PaisosVisitatsRatio[ong] = len(set(PaisosVisitats[ong]))/len(PaisosVisitats[ong])


PaisosVisitatsRatioONG = {}
for ong in paisosVisitatsONG:
    PaisosVisitatsRatioONG[ong] = paisosVisitatsONG[ong].count(1)/len(paisosVisitatsONG[ong]) 




training_negatiu = {}
trainingGlobal = pandas.DataFrame()
trainingGlobal_negatiu = pandas.DataFrame()
path = './output/'
for root, dirs, files in os.walk(path):
    for filename in files:
                   
        if "_negativos" in filename:
            proyectos = pandas.read_excel(path+filename, sheet_name='Sheet1',index_col = "Pais-Año")
            training_negatiu[filename[:filename.index("_")]]=proyectos
            trainingGlobal_negatiu = trainingGlobal_negatiu.append(proyectos)

trainingGlobal_negatiu.to_excel("./output/allExcels_negatiu.xlsx")
    







            