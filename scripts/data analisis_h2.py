#import statsmodels.api as sm
import os
import pandas
import numpy
import copy
import _pickle as pickle
from sklearn.preprocessing import MinMaxScaler



#primer: quants cops han repetit de pais en anys consecutius

ongPaisosRepetits = {}
ongPaisosNoRepetits = {}
rep = 0
noRep = 0

dinerosONG = pickle.load( open( "./dinerosONG.p", "rb" ) )
for ong in dinerosONG:
    ongPaisosRepetits[ong] = {}
    ongPaisosNoRepetits[ong] = {}
    for year in dinerosONG[ong]:
        if year != '2009':
            if year not in ongPaisosRepetits[ong]:
                ongPaisosRepetits[ong][year] = []
                ongPaisosNoRepetits[ong][year] = []
            for pais in dinerosONG[ong][year]:
                
                if (str(int(year)-1) in dinerosONG[ong]) and pais in dinerosONG[ong][str(int(year)-1)]:
                    ongPaisosRepetits[ong][year].append(pais)
                    rep+=1
                else:
                    ongPaisosNoRepetits[ong][year].append(pais)
                    noRep+=1

                    


#segon: Una ONG té subvencio per anar a un pais a l'any X. Mirar si a l'any X+1 no tenen subvencio. si hi tornen a anar-hi o no.

trainingONG = pickle.load( open( "./ong.p", "rb" ) )
year = 2010
ong = "acción contra el hambre"
ongRepetitsSubvencio = {}
ongNoRepetitsSubvencio = {}
repSub = 0
noRepSub = 0
for ong in dinerosONG:
    ongRepetitsSubvencio[ong] = {}
    ongNoRepetitsSubvencio[ong] = {}
    for year in trainingONG[ong]:
        if year in [2008,2009,2010,2011,2012,2013,2014,2015]:
            if year not in ongRepetitsSubvencio[ong]:
                ongRepetitsSubvencio[ong][year] = []
                ongNoRepetitsSubvencio[ong][year] = []
            if "SUBVENCIONES" in trainingONG[ong][year]:
                for pais in trainingONG[ong][year]["SUBVENCIONES"]:
                    if (year+1) in trainingONG[ong] and (("SUBVENCIONES" not in trainingONG[ong][year+1]) or (pais not in trainingONG[ong][year+1]["SUBVENCIONES"])):
                        if "PROYECTOS" in trainingONG[ong][year+1] and pais in trainingONG[ong][year+1]["PROYECTOS"]:
                            ongRepetitsSubvencio[ong][year].append(pais)
                            repSub +=1
                        else:
                            ongNoRepetitsSubvencio[ong][year].append(pais)
                            noRepSub+=1




