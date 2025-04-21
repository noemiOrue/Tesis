#import statsmodels.api as sm
import os
import pandas
import numpy
import copy
import _pickle as pickle
from sklearn.preprocessing import MinMaxScaler
import unicodedata



#leemos excels

listColumns = ["ONU","GDP","Public_Grant","Budget_Previous_Year","Donor_Aid_Budget"]
listColumns = listColumns + ["LatinAmerica",'Africa','Colony','Delegation','Visitado'] 
listColumns = listColumns + ["ControlofCorruption","RuleofLaw","RegulatoryQuality","GovernmentEffectiveness","Political StabilityNoViolence","VoiceandAccountability","generic"]

training = {}
path = '../output/'
for root, dirs, files in os.walk(path):
    for filename in files:
        print(filename)
        if "_" not in filename and ".png" not in filename and ".txt" not in filename and "allExcels" not in filename and '~' not in filename:
            proyectos = pandas.read_excel(path+filename, sheet_name='Sheet1',index_col = "Pais-Año")
            for col in listColumns:
                if col not in proyectos.columns:
                    proyectos[col] = None  # or 0 if numeric
            proyectos = proyectos[listColumns]
            training[filename[:-5]]=proyectos.replace("..",0)
            
            for index, row in proyectos.iterrows():
                if row["Budget_Previous_Year"] < 100 and row["Budget_Previous_Year"] > 0:
                    print("FOUND")

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
                
                training[ong] = pandas.concat([training[ong], newEntry])
                
               
    training[ong].to_excel("../output/"+ong+"_positivos_negativos.xlsx")
    
import unicodedata
import numpy as np

import pandas

def correctCountries(c):
    if c == "afghanistan":
        return "afganistan"
    elif c == "libya":
        return "libia"
    elif c == "kenya":
        return "kenia"
    elif c == "croatia":
        return "croacia"
    elif c == "greece":
        return "grecia"
    elif c == "italy":
        return "italia"
    elif c == "malaysia":
        return "malasia"
    elif c == "jordan":
        return "jordania"
    elif c == "thailand":
        return "tailandia"
    elif c == "ukraine":
        return "ucrania"
    elif c == "cameroon":
        return "camerun"
    elif c == "egypt, arab rep.":
        return "egipto"
    elif c == "congo, dem. rep.":
        return "republica democratica del congo"
    elif c == "congo, rep.":
        return "republica del congo"
    elif c == "venezuela, rb":
        return "venezuela"
    elif c == "brazil":
        return "brasil"
    elif c == "ethiopia":
        return "etiopia"
    elif c == "morocco":
        return "marruecos"
    elif c == "turkiye":
        return "turquia"
    elif c == "japan":
        return "japon"
    elif c == "russian federation":
        return "rusia"
    elif c == "cote d'ivoire":
        return "costa de marfil"
    elif c == "dominican republic":
        return "republica dominicana"
    elif c == "iran, islamic rep.":
        return "iran"
    elif c == "iraq":
        return "irak"
    elif c == "belize":
        return "belice"
    elif c == "lesotho":
        return "lesoto"
    elif c == "zimbabwe":
        return "zimbabue"
    elif c == "tajikistan":
        return "tayikistan"
    elif c == "libano":
        return "lebanon"
    elif c == "filipinas":
        return "philippines"
    elif c == "moldavia":
        return "moldova"
    elif c == "mauricio":
        return "mauritius"
    elif c == "ruanda":
        return "rwanda"
    elif c == "papua nueva guinea":
        return "papua new guinea"
    elif c == "lituania":
        return "lithuania"
    elif c == "tunez":
        return "tunisia"
    elif c == "sudafrica":
        return "south africa"
    elif c == "corea del norte":
        return "korea, dem. rep."
    elif c == "sierra leona":
        return "sierra leone"
    elif c == "bielorrusia":
        return "belarus"
    elif c == "republica centroafricana":
        return "central african republic"
    elif c == "santo tome y principe":
        return "são tomé and principe"
    return c

df = pandas.read_excel('../dades/wgidataset_processed.xlsx', header=15,sheet_name="VoiceandAccountability")    
dictCountries = {}
dictCountries["VoiceandAccountability"] = {}
for country in df["Country/Territory"].unique():
    countryP = unicodedata.normalize('NFD', country).encode('ascii', 'ignore').decode("utf-8").lower()
    countryP = correctCountries(countryP)
    dictCountries["VoiceandAccountability"][countryP] = {}
    entry = df.loc[df["Country/Territory"] == country]
    for year in [2009,2010,2011,2012,2013,2014,2015,2016]:
        dictCountries["VoiceandAccountability"][countryP][year] = float(entry["Estimate "+str(year)])

df = pandas.read_excel('../dades/wgidataset_processed.xlsx', header=15,sheet_name="Political StabilityNoViolence")    
dictCountries["Political StabilityNoViolence"] = {}
for country in df["Country/Territory"].unique():
    countryP = unicodedata.normalize('NFD', country).encode('ascii', 'ignore').decode("utf-8").lower()
    countryP = correctCountries(countryP)

    dictCountries["Political StabilityNoViolence"][countryP] = {}
    entry = df.loc[df["Country/Territory"] == country]
    for year in [2009,2010,2011,2012,2013,2014,2015,2016]:
        dictCountries["Political StabilityNoViolence"][countryP][year] = float(entry["Estimate "+str(year)])
        
df = pandas.read_excel('../dades/wgidataset_processed.xlsx', header=15,sheet_name="GovernmentEffectiveness")    
dictCountries["GovernmentEffectiveness"] = {}
for country in df["Country/Territory"].unique():
    countryP = unicodedata.normalize('NFD', country).encode('ascii', 'ignore').decode("utf-8").lower()
    countryP = correctCountries(countryP)

    dictCountries["GovernmentEffectiveness"][countryP] = {}
    entry = df.loc[df["Country/Territory"] == country]
    for year in [2009,2010,2011,2012,2013,2014,2015,2016]:
        dictCountries["GovernmentEffectiveness"][countryP][year] = float(entry["Estimate "+str(year)])
df = pandas.read_excel('../dades/wgidataset_processed.xlsx', header=15,sheet_name="RegulatoryQuality")    
dictCountries["RegulatoryQuality"] = {}
for country in df["Country/Territory"].unique():
    countryP = unicodedata.normalize('NFD', country).encode('ascii', 'ignore').decode("utf-8").lower()
    countryP = correctCountries(countryP)

    dictCountries["RegulatoryQuality"][countryP] = {}
    entry = df.loc[df["Country/Territory"] == country]
    for year in [2009,2010,2011,2012,2013,2014,2015,2016]:
        dictCountries["RegulatoryQuality"][countryP][year] = float(entry["Estimate "+str(year)])

df = pandas.read_excel('../dades/wgidataset_processed.xlsx', header=15,sheet_name="RuleofLaw")    
dictCountries["RuleofLaw"] = {}
for country in df["Country/Territory"].unique():
    countryP = unicodedata.normalize('NFD', country).encode('ascii', 'ignore').decode("utf-8").lower()
    countryP = correctCountries(countryP)

    dictCountries["RuleofLaw"][countryP] = {}
    entry = df.loc[df["Country/Territory"] == country]
    for year in [2009,2010,2011,2012,2013,2014,2015,2016]:
        dictCountries["RuleofLaw"][countryP][year] = float(entry["Estimate "+str(year)])

df = pandas.read_excel('../dades/wgidataset_processed.xlsx', header=15,sheet_name="ControlofCorruption")    
dictCountries["ControlofCorruption"] = {}
for country in df["Country/Territory"].unique():
    countryP = unicodedata.normalize('NFD', country).encode('ascii', 'ignore').decode("utf-8").lower()
    countryP = correctCountries(countryP)

    dictCountries["ControlofCorruption"][countryP] = {}
    entry = df.loc[df["Country/Territory"] == country]
    for year in [2009,2010,2011,2012,2013,2014,2015,2016]:
        dictCountries["ControlofCorruption"][countryP][year] = float(entry["Estimate "+str(year)])

for root, dirs, files in os.walk(path):
    for filename in files:
        if "_positivos_negativos" in filename and "còpia" not in filename:
            df = pandas.read_excel('../output/'+filename, header=0,index_col = 0)    
            for newEntry in ["ControlofCorruption","RuleofLaw","RegulatoryQuality","GovernmentEffectiveness","Political StabilityNoViolence","VoiceandAccountability"]:
                df[newEntry]=0
                for index, row in df.iterrows():
                    country =index[index.index("_")+1:]
                    country = unicodedata.normalize('NFD', country).encode('ascii', 'ignore').decode("utf-8").lower()
                    country = correctCountries(country) 
                    year = index[:index.index("_")]
                    
                    if country in dictCountries[newEntry]:
                        value =  dictCountries[newEntry][country]
                        df.loc[index,newEntry] = value[int(year)]
                    else:
                        print("no")
                
            df["generic"] = df[["ControlofCorruption","RuleofLaw","RegulatoryQuality","GovernmentEffectiveness","Political StabilityNoViolence","VoiceandAccountability"]].mean(axis=1)
            df.to_excel("../output/"+filename[:-5]+"_2.xlsx")
                        


for root, dirs, files in os.walk(path):
    for filename in files:
        if filename.endswith(".xlsx") and "_2" in filename:
            ngo_name = filename.split("_")[0]  # Get part before the first "_"
            df = pandas.read_excel("../output/"+filename)
            df["NGO"] = ngo_name  # Add NGO column
            df.to_excel("../output/"+filename, index=False)




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
        if "_negativos_2" in filename:
            proyectos = pandas.read_excel(path+filename, sheet_name='Sheet1',index_col = "Pais-Año")
            proyectos = proyectos[listColumns+["NGO"]]
            proyectos["Pais-Año"] = proyectos.index  # bring index back into a column

            proyectos.set_index(["Pais-Año", "NGO"], inplace=True)
            trainingGlobal_negatiu = pandas.concat([trainingGlobal_negatiu, proyectos])

            
trainingGlobal_negatiu.to_excel("../output/allExcels_negatiu.xlsx",columns=listColumns)








            