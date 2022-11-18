# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 23:18:25 2019

@author: bcoma
"""
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy
import unicodedata
import _pickle as pickle
import glob
import copy


import os
import csv

paises = []
file = csv.reader(open("../dades/Datos_vf/paises.csv","r",encoding="utf8"), delimiter=',',quoting=csv.QUOTE_NONE)
for el in file:
    text = el[0].replace('"', '').lower().strip()
    paises.append(unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8"))
paises.append("sudan del sur")
paises.append("myanmar")
paises.append("serbia y montenegro")


def revisionNombres(inputName):
    if inputName in ["accion-contra-el-hambre","accio contra el hambre","accion contra el hambre","fundacion accion contra el hambre"]:
        return "acción contra el hambre"
    elif inputName in ["cruz roja","cruz roja espaola","cruz roja espanola"]:
        return "cruz roja"
    elif inputName in ["caritas","caritas espanola","critas"]:
        return "cáritas"
    elif inputName in ["intermon oxfam","oxfam intermn","fundacion oxfam intermon","fundacio oxfam  intermon, fundacio privada"]:
        return "oxfam intermon"
    elif inputName in ["ayuda en accion","ayuda en accin","fundación ayuda en accion","fundacion ayuda en accion"]:
        return "ayuda en acción"
    elif inputName in ["accion verapaz"]:
        return "acción verapaz"
    elif inputName in ["fundacion save the children"]:
        return "save the children"
    elif inputName in ["manos unidas - comite catolico de la campana contra el hambre en el mundo"]:
        return "manos unidas"
    
    elif inputName in ["amigos de la tierra espa§a","amigos de la tierra espaa","amigos de la tierra españa - adte","amigos de la tierra espana - adte","amigos de la tierra españa","amigos de la tierra espana"]:
        return "amigos de la tierra"

    elif inputName in ["edificando comunidad de nazareth","edificando comunidad nazaret"]:
        return "edificando comunidad de nazaret"

    elif inputName in ["fundación adra,fundacion adra, agencia adventista para el desarrollo y recursos asistenciales"]:
        return "adra"
    elif inputName in ["fundacion codespa"]:
        return "codespa"
    elif inputName in ["fundación alboan","fundacion alboan"]:
        return "alboan"
    elif inputName in ["asociación de investigación y especialización sobre temas iberoamericanos (aieti)","asociacion de investigacion y especializacion sobre temas iberoamericanos (aieti)"]:
        return "aieti"
    elif inputName in ["fundación adsis", "fundacion adsis"]:
        return "adsis"
    elif inputName in ["economistas sin frontera","econimistas sin frontera", "economistas sin fronteras de españa","economistas sin fronteras de espana","fundación economistas sin fronteras","fundacion economistas sin fronteras"]:
        return "economistas sin fronteras"
    elif inputName in ["fundacion amref salud africa", "amref salud africa"]:
        return "amref"
    elif inputName in ["farmaceuticos sin fronteras de espana", "farmaceuticos sin fronteras de españa","farmaceuticos sin frontera"]:
        return "farmaceuticos sin fronteras"
    elif inputName in ["fundacion adra, agencia adventista para el desarrollo y recursos asistenciales"]:
        return "adra"
    elif inputName in ["asociacion de investigacion y especializacion sobre temas iberoamericanos (aieti)","asociacion de investigacion y especializacion sobre temas iberoamericanos"]:
        return "aieti"
    elif inputName in ["fundacion alboan"]:
        return "alboan"
    elif inputName in ["amigos de la tierra espana","amigos de la tierra espana - adte","amigos de la tierra espaa"]:
        return "amigos de la tierra"
    elif inputName in ["asociacion entrepueblos"]:
        return "entrepueblos"
    elif inputName in ["asociacion fontilles"]:
        return "fontilles"
    elif inputName in ["fere ceca","federacion espanola de religiosos de ensenanza - titulares de centros catolicos"]:
        return "fere-ceca"
    elif inputName in ["fisc-compania de maria","fisc compania maria","fisc compania de maria","fisc"]:
        return "fisc-compañia de maria"
    elif inputName in ["fundacion intered"]:
        return "intered"
    elif inputName in ["instituto sindical de cooperacion al desarrollo (iscod)"]:
        return "iscod"
    elif inputName in ["juan ciudad ongd", ""]:
        return "juan ciudad"
    elif inputName in ["fundacion mundubat - mundubat fundazioa"]:
        return "mundubat"
    elif inputName in ["asociacion paz con dignidad"]:
        return "paz con dignidad"
    elif inputName in ["ocasha-cristianos con el sur"]:
        return "ocasha"
    elif inputName in ["asociacion paz con dignidad"]:
        return "paz con dignidad"
    elif inputName in ["fundacion jovenes y desarrollo"]:
        return "jovenes y desarrollo"
    elif inputName in ["proyde promocion y desarrollo","asociacion proyde"]:
        return "proyde"
    elif inputName in ["movimiento por la paz, el desarme y la libertad","movimiento por la paz -mpdl-","movimiento por la paz"]:
        return "mpdl"
    elif inputName in ["fundacion cideal de cooperacion e investigacion","fundacion cideal","centro de comunicacion, investigacion y documentacion entre europa, espana y america latina"]:
        return "cideal"
    elif inputName in ["fundacion de religiosos para la salud","fundacion de religiosos para la salud (frs)","fundacin de religiosos para la salud","fundacion religiosa salud","fundacion religiosos para la salud"]:
        return "frs"
    elif inputName in ["fundacion benefica del valle","fundacion del valle","fundacion valle"]:
        return "fundación valle"
    elif inputName in ["fundacion entreculturas - fe y alegria","fundacion entreculturas"]:
        return "entreculturas"
    elif inputName in ["fundacion iberoamerica-europa","fundacion iberoamerica europa","iberoamerica","iberomerica"]:
        return "iberoamerica europa"
    elif inputName in ["asociacion juvenil madreselva","fundacion madreselva"]:
        return "madreselva"
    elif inputName in ["fundacion para el desarrollo de la enfermeria - fuden","fundacion para el desarrollo de la enfermeria"]:
        return "fuden"
    elif inputName in ["fundacion promocion social de la cultura","fundacion promocion social","fundacin promocin social","funacion promocion social","fundacio promocion social","promocion social"]:
        return "promoción social"
    elif inputName in ["fundacion pueblos hermanos"]:
        return "pueblos hermanos"
    elif inputName in ["federacion de asociaciones medicus mundi en espana"]:
        return "medicus mundi"
    elif inputName in ["centro de estudios y solidaridad con america latina"]:
        return "cesal"
    elif inputName in ["fundacion de ayuda contra la drogadiccion"]:
        return "fad"
    elif inputName in ["educacion sin fronteras","educacion sin fronteras - espana","educacion sin fronteras-espana"]:
        return "educo"
    return inputName

def revisionPaises(inputName):
    if inputName in ["territorios palestinos","ribera occidental y gaza"]:
        return "palestina"
    if inputName in ["guinea bissau"]:
        return "guinea-bissau"
    if inputName in ["guinea conakri","guinea conakry"]:
        return "guinea"
    if inputName in ["poblacion saharaui"]:
        return "sahara occidental"
    if inputName in ["argelia"]:
        return "algeria"
    if inputName in ["argelia"]:
        return "algeria"
    if inputName in ["congo (republica democratica del congo)","r.d. congo","congo, republica democratica del"]:
        return "republica democratica del congo"
    if inputName in ["bosnia-herzegovina"]:
        return "bosnia y herzegovina"
    if inputName in ["zimbabwe","zimbaue","zimbaue "]:
        return "zimbabue"
    if inputName in ["rep. dominicana"]:
        return "republica dominicana"
    if inputName in ["belarus"]:
        return "bielorrusia"
    if inputName in ["congo, rep.","congo ,rep","congo, republica del"]:
        return "republica del congo"
    if inputName in ["kasajstan","kazajstan"]:
        return "kazajistan"
    if inputName in ["birmania"]:
        return "myanmar"
    if inputName in ["djibouti"]:
        return "yibuti"
    if inputName in ["kirguizstan"]:
        return "kirguistan"
    if inputName in ["rep. centroafricana"]:
        return "republica centroafricana"
    if inputName in ["papua-nueva guinea"]:
        return "papua nueva guinea"
    if inputName in ["azerbaijan"]:
        return "azerbaiyan"
    if inputName in ["rwanda"]:
        return "ruanda"
    if inputName in ["botswana"]:
        return "botsuana"
    if inputName in ["san vicente"]:
        return "san vicente y las granadinas"
    if inputName in ["moldova","republica de moldova"]:
        return "moldavia"
    if inputName in ["pakistán"]:
        return "pakistan"
    if inputName in ["vanuata"]:
        return "vanuatu"
    if inputName in ["viet nam"]:
        return "vietnam"
    if inputName in ["republica arabe siria"]:
        return "siria"
    if inputName in ["cote d'ivoire"]:
        return "costa de marfil"
    if inputName in ["kenya"]:
        return "kenia"
    if inputName in ["eswatini","esuatini"]:
        return "swazilandia"
    if inputName in ["egipto, republica arabe de","republica arabe de egipto"]:
        return "egipto"
    if inputName in ["federacion de rusia"]:
        return "rusia"
    if inputName in ["timor-leste"]:
        return "timor oriental"
    if inputName in ["iraq"]:
        return "irak"
    if inputName in ["iran, republica islamica del"]:
        return "iran"
    if inputName in ["lesotho"]:
        return "lesoto"
    if inputName in ["macedonia del norte"]:
        return "macedonia"
    if inputName in ["republica democratica popular lao","lao"]:
        return "laos"
    if inputName in ["hong kong, region administrativa especial"]:
        return "hong kong"
    if inputName in ["corea, republica de"]:
        return "corea del sur"
    if inputName in ["corea, republica popular democratica de"]:
        return "corea del norte"
    return inputName

oldColonies = ["mexico","guatemala","el salvador","honduras","nicaragua","costa rica","panama"]
oldColonies = oldColonies + ["colombia","venezuela","ecuador","peru","bolivia","chile","argentina"]
oldColonies = oldColonies + ["paraguay","uruguay","cuba","puerto rico"]
oldColonies = oldColonies + ["filipinas","guam", "marruecos","sahara occidental","guinea ecuatorial"] 
oldColonies = oldColonies + ["republica dominicana","bahamas","antigua y barbuda", "trinidad y tobago"] 
oldColonies = oldColonies + ["jamaica","barbados","santa lucia","guyana","haiti"]
oldColonies = oldColonies + ["bermudas"]

print(oldColonies.sort())

delegacionesONG = {}
path = 'C:/Users/bcoma/Documents/GitHub/NGO paper/dades/delegaciones'
for root, dirs,files in os.walk(path):
    for dire in dirs:
        ong = dire[dire.index("_")+1:]
        ong =  unicodedata.normalize('NFD', ong.lower().strip()).encode('ascii', 'ignore').decode("utf-8") 
        ong = revisionNombres(ong)
        delegacionesONG[ong] = {}
        for file in glob.glob(path+"/"+dire+"/*20*"):
            print("DINTRE")
            fileName = os.path.basename(file)
            if "20" in file:
                f = open(file,encoding='utf-8', errors='ignore')
                delegacionesONG[ong][int(fileName[0:4])] = []
                dades = f.readlines()
                for line in dades[1:]:
                    if line != '\n':
                        pais = line[:line.index("(")].strip()
                        pais = unicodedata.normalize('NFD', pais.lower().strip()).encode('ascii', 'ignore').decode("utf-8") 
                        pais = revisionPaises(pais)
                        delegacionesONG[ong][int(fileName[:4])].append(pais)
        if 2012 in delegacionesONG[ong]:
            delegacionesONG[ong][2013] = copy.deepcopy(delegacionesONG[ong][2012])
        if 2015 in delegacionesONG[ong]:
            delegacionesONG[ong][2014] = copy.deepcopy(delegacionesONG[ong][2015])


dinerosEspanya = {}
ong = {}
#Historico subvenciones 2009 a 2016 ["SUBVENCIONES"]
PC_subvenciones = pd.read_excel('../dades/Datos_vf/Histórico subvenciones convocatorias AECID 1992-2018 (exc. CAP).xls', sheet_name='Proyectos y Convenos')
PC_Acciones = pd.read_excel('../dades/Datos_vf/Histórico subvenciones convocatorias AECID 1992-2018 (exc. CAP).xls', sheet_name='Acciones')

new_header = PC_subvenciones.iloc[0] 
PC_subvenciones = PC_subvenciones[1:] 
PC_subvenciones.columns = new_header 
q2013 =0
for i in range(0,len(PC_subvenciones)):
    info = PC_subvenciones.iloc[i]
    
    if info['Año de Adjudicación'] >= 2008 and info['Año de Adjudicación'] <= 2016:
        if info['País'].lower() not in ["sin determinar","españa","varios"]:
            
            ong_nom = unicodedata.normalize('NFD', info["ONGD"].lower().strip()).encode('ascii', 'ignore').decode("utf-8") 
            ong_nom = revisionNombres(ong_nom)
            
            if ong_nom not in ong:
                ong[ong_nom] = {}
            if info['Año de Adjudicación'] not in ong[ong_nom]:
                ong[ong_nom][info['Año de Adjudicación']] = {}
            if info['Año de Adjudicación'] not in dinerosEspanya:
                dinerosEspanya[info['Año de Adjudicación']] = {}
            if "SUBVENCIONES" not in ong[ong_nom][info['Año de Adjudicación']] :
                ong[ong_nom][info['Año de Adjudicación']]["SUBVENCIONES"] = {}
            if info["Importe total subvención (€)"]!="RENUNCIA":
                    paisos = info['País'].replace(' y ', ' , ').replace(".","").split(',')
                    
                    for j in range(len(paisos)):
                        paisos[j]= revisionPaises(unicodedata.normalize('NFD', paisos[j].lower().strip()).encode('ascii', 'ignore').decode("utf-8") )
                    for pais in paisos:
                        if pais in paises:
                            
                            if pais not in ong[ong_nom][info['Año de Adjudicación']]["SUBVENCIONES"]:
                                ong[ong_nom][info['Año de Adjudicación']]["SUBVENCIONES"][pais] = 0
                            if pais not in dinerosEspanya[info['Año de Adjudicación']]:
                                dinerosEspanya[info['Año de Adjudicación']][pais]=0
                            ong[ong_nom][info['Año de Adjudicación']]["SUBVENCIONES"][pais]+= float(info["Importe total subvención (€)"])/len(paisos)
                            dinerosEspanya[info['Año de Adjudicación']][pais]+=float(info["Importe total subvención (€)"])/len(paisos)
                        else:
                            pass
                    

new_header = PC_Acciones.iloc[0] 
PC_Acciones = PC_Acciones[1:] 
PC_Acciones.columns = new_header 

for i in range(0,len(PC_Acciones)):
    info = PC_Acciones.iloc[i]
    if info['Año de Adjudicación'] >= 2014 and info['Año de Adjudicación'] <= 2016:
        if info['País'].lower() not in ["sin determinar","españa","varios"]:
            ong_nom = unicodedata.normalize('NFD', info["ENTE RECEPTOR DE LA SUBVENCIÓN"].lower().strip()).encode('ascii', 'ignore').decode("utf-8") 
            ong_nom = revisionNombres(ong_nom)
            if ong_nom not in ong:
                ong[ong_nom] = {}
            
            if info['Año de Adjudicación'] not in ong[ong_nom]:
                ong[ong_nom][info['Año de Adjudicación']] = {}
            if info['Año de Adjudicación'] not in dinerosEspanya:
                dinerosEspanya[info['Año de Adjudicación']] = {}
            if "SUBVENCIONES" not in ong[ong_nom][info['Año de Adjudicación']] :
                ong[ong_nom][info['Año de Adjudicación']]["SUBVENCIONES"] = {}
            if info["Importe total subvención (€)"]!="RENUNCIA":
                paisos = info['País'].replace(' y ', ' , ').replace(".","").split(',')
                for j in range(len(paisos)):
                    paisos[j]= revisionPaises(unicodedata.normalize('NFD', paisos[j].lower().strip()).encode('ascii', 'ignore').decode("utf-8") )
                for pais in paisos:
                    if pais in paises:
                        if pais not in ong[ong_nom][info['Año de Adjudicación']]["SUBVENCIONES"]:
                            ong[ong_nom][info['Año de Adjudicación']]["SUBVENCIONES"][pais] = 0
                        if pais not in dinerosEspanya[info['Año de Adjudicación']]:
                            dinerosEspanya[info['Año de Adjudicación']][pais]=0
                        ong[ong_nom][info['Año de Adjudicación']]["SUBVENCIONES"][pais]+= float(info["Importe total subvención (€)"])/len(paisos)
                        dinerosEspanya[info['Año de Adjudicación']][pais]+=float(info["Importe total subvención (€)"])/len(paisos)

                    else:
                        pass

#ingresos ong 2014 a 2015 ["INGRESOS"]

ingresos2014 = pd.read_excel('../dades/Datos_vf/ongd9_informe2014-2015_Ingresos.xlsx', sheet_name='Fondos2014')
new_header = ingresos2014.iloc[4]
ingresos2014 = ingresos2014[5:] 
ingresos2014.columns = new_header 
for i in range(0,len(ingresos2014)):
    #i = 10
    info = ingresos2014.iloc[i]
    info = info.replace(" ",0)
    posTotalPrivados = list(info.index).index("TOTAL servicios prestados")+1
    posTotalPublicos = list(info.index).index("Otros fondos públicos")+1
    if isinstance(info["Nombre"], str):
        ong_nom = unicodedata.normalize('NFD', info["Nombre"].lower().strip()).encode('ascii', 'ignore').decode("utf-8") 
        ong_nom = revisionNombres(ong_nom)
        if ong_nom not in ong:
            ong[ong_nom] = {}
        if 2014 not in ong[ong_nom]:
            ong[ong_nom][2014] = {}
        if "INGRESOS" not in ong[ong_nom][2014]:
            ong[ong_nom][2014]["INGRESOS"] = {}
        ong[ong_nom][2014]["INGRESOS"]["TOTAL cuotas periódicas"] = float(info["TOTAL  cuotas periódicas"])
        ong[ong_nom][2014]["INGRESOS"]["TOTAL donaciones"] = float(info["TOTAL donaciones"])
        ong[ong_nom][2014]["INGRESOS"]["TOTAL EMPRESAS"] = float(info["TOTAL EMPRESAS"])
        ong[ong_nom][2014]["INGRESOS"]["TOTAL venta productos"] = float(info["TOTAL venta productos"])
        ong[ong_nom][2014]["INGRESOS"]["TOTAL servicios prestados"] = float(info["TOTAL servicios prestados"])
        ong[ong_nom][2014]["INGRESOS"]["TOTAL fondos privados"] = float(list(info)[posTotalPrivados])
        ong[ong_nom][2014]["INGRESOS"]["TOTAL ESTATAL"] = float(info["TOTAL ESTATAL"])
        ong[ong_nom][2014]["INGRESOS"]["TOTAL AUTONÓMICO"] = float(info["TOTAL AUTONÓMICO"])
        ong[ong_nom][2014]["INGRESOS"]["TOTAL INTERNACIONAL"] = float(info["TOTAL INTERNACIONAL"])
        ong[ong_nom][2014]["INGRESOS"]["Otros fondos públicos"] = float(info["Otros fondos públicos"])
        ong[ong_nom][2014]["INGRESOS"]["TOTAL fondos públicos"] = float(list(info)[posTotalPublicos])
        
ingresos2015 = pd.read_excel('../dades/Datos_vf/ongd9_informe2014-2015_Ingresos.xlsx', sheet_name='Fondos2015')
new_header = ingresos2015.iloc[4] 
ingresos2015 = ingresos2015[5:]
ingresos2015.columns = new_header 
for i in range(0,len(ingresos2015)):
    info = ingresos2015.iloc[i]
    info = info.replace(" ",0)
    posTotalPrivados = list(info.index).index("TOTAL otros servicios")+1
    posTotalPublicos = list(info.index).index("Otros fondos públicos")+1
    if isinstance(info["Nombre"], str):

        ong_nom = unicodedata.normalize('NFD', info["Nombre"].lower().strip()).encode('ascii', 'ignore').decode("utf-8") 
        ong_nom = revisionNombres(ong_nom)
        if ong_nom not in ong:
            ong[ong_nom] = {}
        if 2015 not in ong[ong_nom]:
            ong[ong_nom][2015] = {}
        if "INGRESOS" not in ong[ong_nom][2015]:
            ong[ong_nom][2015]["INGRESOS"] = {}
        ong[ong_nom][2015]["INGRESOS"]["TOTAL cuotas periódicas"] = float(info["TOTAL  cuotas periódicas"])
        ong[ong_nom][2015]["INGRESOS"]["TOTAL donaciones"] = float(info["TOTAL donaciones"])
        ong[ong_nom][2015]["INGRESOS"]["TOTAL EMPRESAS"] = float(info["TOTAL EMPRESAS"])
        ong[ong_nom][2015]["INGRESOS"]["TOTAL venta productos"] = float(info["TOTAL ventas"])
        ong[ong_nom][2015]["INGRESOS"]["TOTAL otros servicios"] = float(info["TOTAL otros servicios"])
        ong[ong_nom][2015]["INGRESOS"]["TOTAL fondos privados"] = float(list(info)[posTotalPrivados])
        ong[ong_nom][2015]["INGRESOS"]["TOTAL ESTATAL"] = float(info["TOTAL ESTATAL"])
        ong[ong_nom][2015]["INGRESOS"]["TOTAL AUTONÓMICO"] = float(info["TOTAL AUTONÓMICO"])
        ong[ong_nom][2015]["INGRESOS"]["TOTAL INTERNACIONAL"] = float(info["TOTAL INTERNACIONAL"])
        ong[ong_nom][2015]["INGRESOS"]["Otros fondos públicos"] = float(info["Otros fondos públicos"])
        ong[ong_nom][2015]["INGRESOS"]["TOTAL fondos públicos"] = float(list(info)[posTotalPublicos])


ingresos2016 = pd.read_excel('../dades/Datos_vf/ongd9_informe2016_Ingresos.xlsx', sheet_name='Fondos')
new_header = ingresos2016.iloc[4] 
ingresos2016 = ingresos2016[5:] 
ingresos2016.columns = new_header 
for i in range(0,len(ingresos2016)):
    info = ingresos2016.iloc[i]
    info = info.replace(" ",0)
    info = info.replace("s.d.",0)
    posTotalPrivados = list(info.index).index("TOTAL servicios prestados")+1
    posTotalPublicos = list(info.index).index("Otros   ")+1
    if isinstance(info["Nombre"], str):
        ong_nom = unicodedata.normalize('NFD', info["Nombre"].lower().strip()).encode('ascii', 'ignore').decode("utf-8") 
        ong_nom = revisionNombres(ong_nom)
        if ong_nom not in ong:
            ong[ong_nom] = {}
        
        if 2016 not in ong[ong_nom]:
            ong[ong_nom][2016] = {}
        if "INGRESOS" not in ong[ong_nom][2016]:
            ong[ong_nom][2016]["INGRESOS"] = {}
        ong[ong_nom][2016]["INGRESOS"]["TOTAL cuotas periódicas"] = float(info["TOTAL cuotas periódicas"])
        ong[ong_nom][2016]["INGRESOS"]["TOTAL donaciones"] = float(info["TOTAL doncaciones"])
        ong[ong_nom][2016]["INGRESOS"]["TOTAL EMPRESAS"] = float(info["TOTAL  Fondo de empresas fundaciones"])
        ong[ong_nom][2016]["INGRESOS"]["TOTAL venta productos"] = float(info["TOTAL venta de productos"])
        ong[ong_nom][2016]["INGRESOS"]["TOTAL otros servicios"] = float(info["TOTAL servicios prestados"])
        ong[ong_nom][2016]["INGRESOS"]["TOTAL fondos privados"] = float(list(info)[posTotalPrivados])
        ong[ong_nom][2016]["INGRESOS"]["TOTAL ESTATAL"] = float(info["TOTAL fondos MAE y otros ministerios"])
        ong[ong_nom][2016]["INGRESOS"]["TOTAL AUTONÓMICO"] = float(info["TOTAL fondos descentralizados"])
        ong[ong_nom][2016]["INGRESOS"]["TOTAL INTERNACIONAL"] = float(info["TOTAL fondos internacionales"])
        ong[ong_nom][2016]["INGRESOS"]["Otros fondos públicos"] = float(info["Otros   "])
        ong[ong_nom][2016]["INGRESOS"]["TOTAL fondos públicos"] = float(list(info)[posTotalPublicos])


ingresos = pd.read_excel('../dades/Datos_vf/Coordinadora ONG Desarrollo España - Datos año 2013.xls', sheet_name='Anex 2')
header_ingresos = ingresos.iloc[5] 
ingresos = ingresos[7:]
ingresos.columns = header_ingresos

i = 0
for i in range(0,len(ingresos)):
    info = ingresos.iloc[i]
    info = info.replace(" ",0)
    info = info.replace("n.d.",0)
    if isinstance(info[1], str):
        ong_nom = unicodedata.normalize('NFD', info[1].lower().strip()).encode('ascii', 'ignore').decode("utf-8") 
        ong_nom = revisionNombres(ong_nom)
        if ong_nom not in ong:
            ong[ong_nom] = {}
        if 2013 not in ong[ong_nom]:
            ong[ong_nom][2013] = {}
        ong[ong_nom][2013]["INGRESOS"] = {}
        
        ong[ong_nom][2013]["INGRESOS"]["TOTAL cuotas periódicas"] = float(info[3])+float(info[4])
        
        ong[ong_nom][2013]["INGRESOS"]["TOTAL donaciones"] = float(info[5])
        ong[ong_nom][2013]["INGRESOS"]["TOTAL EMPRESAS"] = float(info[6])
        ong[ong_nom][2013]["INGRESOS"]["TOTAL venta productos"] = float(info[7])
        ong[ong_nom][2013]["INGRESOS"]["TOTAL otros servicios"] = float(info[8])
        ong[ong_nom][2013]["INGRESOS"]["TOTAL fondos privados"] = float(info[9])

        ong[ong_nom][2013]["INGRESOS"]["TOTAL ESTATAL"] = float(info[12])
        ong[ong_nom][2013]["INGRESOS"]["TOTAL AUTONÓMICO"] = float(info[13])
        ong[ong_nom][2013]["INGRESOS"]["TOTAL INTERNACIONAL"] = float(info[14])
        ong[ong_nom][2013]["INGRESOS"]["Otros fondos públicos"] = float(info[15])
        ong[ong_nom][2013]["INGRESOS"]["TOTAL fondos públicos"] = float(info[16])

ingresos = pd.read_excel('../dades/Datos_vf/Coordinadora ONG Desarrollo España - Datos año 2012.xls', sheet_name='Anex 2')
header_ingresos = ingresos.iloc[5] #grab the first row for the header
ingresos = ingresos[7:]
ingresos.columns = header_ingresos

i = 0
for i in range(0,len(ingresos)):
    info = ingresos.iloc[i]
    info = info.replace(" ",0)
    info = info.replace("n.d.",0)
    if isinstance(info[1], str):

        ong_nom = unicodedata.normalize('NFD', info[1].lower().strip()).encode('ascii', 'ignore').decode("utf-8") 
        ong_nom = revisionNombres(ong_nom)
        if ong_nom not in ong:
            ong[ong_nom] = {}
        if 2012 not in ong[ong_nom]:
            ong[ong_nom][2012] = {}
        ong[ong_nom][2012]["INGRESOS"] = {}
        
        ong[ong_nom][2012]["INGRESOS"]["TOTAL cuotas periódicas"] = float(info[3])+float(info[4])
        
        ong[ong_nom][2012]["INGRESOS"]["TOTAL donaciones"] = float(info[5])
        ong[ong_nom][2012]["INGRESOS"]["TOTAL EMPRESAS"] = float(info[6])
        ong[ong_nom][2012]["INGRESOS"]["TOTAL venta productos"] = float(info[7])
        ong[ong_nom][2012]["INGRESOS"]["TOTAL otros servicios"] = float(info[8])
        ong[ong_nom][2012]["INGRESOS"]["TOTAL fondos privados"] = float(info[9])

        
        ong[ong_nom][2012]["INGRESOS"]["TOTAL ESTATAL"] = float(info[12])
        ong[ong_nom][2012]["INGRESOS"]["TOTAL AUTONÓMICO"] = float(info[13])
        ong[ong_nom][2012]["INGRESOS"]["TOTAL INTERNACIONAL"] = float(info[14])
        ong[ong_nom][2012]["INGRESOS"]["Otros fondos públicos"] = float(info[15])
        ong[ong_nom][2012]["INGRESOS"]["TOTAL fondos públicos"] = float(info[16])



voluntarios = pd.read_excel('../dades/Datos_vf/Coordinadora ONG Desarrollo España - Datos año 2012.xls', sheet_name='Anex 9-bsocial')
voluntarios = voluntarios[6:]
i = 0
for i in range(0,len(voluntarios)):
    info = voluntarios.iloc[i]
    info = info.replace("n.d.",0)
    if isinstance(info[1], str):
        ong_nom = unicodedata.normalize('NFD', info[1].lower().strip()).encode('ascii', 'ignore').decode("utf-8") 
        ong_nom = revisionNombres(ong_nom)
        if ong_nom not in ong:
            ong[ong_nom] = {}
        if 2012 not in ong[ong_nom]:
            ong[ong_nom][2012] = {}
        ong[ong_nom][2012]["VOLUNTARIOS"] = {}
        ong[ong_nom][2012]["VOLUNTARIOS"]["España"] = float(info[6])
        ong[ong_nom][2012]["VOLUNTARIOS"]["Extranjero"] = float(info[7])
    
    

voluntarios = pd.read_excel('../dades/Datos_vf/Coordinadora ONG Desarrollo España - Datos año 2013.xls', sheet_name='Anex 9-bsocial')
voluntarios = voluntarios[6:]
i = 0
for i in range(0,len(voluntarios)):
    info = voluntarios.iloc[i]
    info = info.replace("n.d.",0)
    if isinstance(info[1], str):
        ong_nom = unicodedata.normalize('NFD', info[1].lower().strip()).encode('ascii', 'ignore').decode("utf-8") 
        ong_nom = revisionNombres(ong_nom)
        if ong_nom not in ong:
            ong[ong_nom] = {}
        if 2013 not in ong[ong_nom]:
            ong[ong_nom][2013] = {}
        ong[ong_nom][2013]["VOLUNTARIOS"] = {}
        ong[ong_nom][2013]["VOLUNTARIOS"]["España"] = float(info[6])
        ong[ong_nom][2013]["VOLUNTARIOS"]["Extranjero"] = float(info[7])


voluntarios = pd.read_excel('../dades/Datos_vf/ongd11_informe2014-2015_Base Social.xls', sheet_name="base social 2014")
header_voluntarios = voluntarios.iloc[1] 
voluntarios = voluntarios[3:]
voluntarios.columns = header_voluntarios
i = 0

for i in range(0,len(voluntarios)):
    info = voluntarios.iloc[i]
    info = info.replace("n.d.",0)
    info = info.replace(" ",0)
    if isinstance(info[0], str):
        ong_nom = unicodedata.normalize('NFD', info[0].lower().strip()).encode('ascii', 'ignore').decode("utf-8") 
        ong_nom = revisionNombres(ong_nom)
        if ong_nom not in ong:
            ong[ong_nom] = {}
        if 2014 not in ong[ong_nom]:
            ong[ong_nom][2014] = {}
        ong[ong_nom][2014]["VOLUNTARIOS"] = {}
        ong[ong_nom][2014]["VOLUNTARIOS"]["España"] = float(info["Total España"])
        ong[ong_nom][2014]["VOLUNTARIOS"]["Extranjero"] = float(info["Total extranjero"])
    

voluntarios = pd.read_excel('../dades/Datos_vf/ongd11_informe2014-2015_Base Social.xls', sheet_name="base social 2015")
header_voluntarios = voluntarios.iloc[1] #grab the first row for the header
voluntarios = voluntarios[3:]
voluntarios.columns = header_voluntarios
i = 0

for i in range(0,len(voluntarios)):
    info = voluntarios.iloc[i]
    info = info.replace("n.d.",0)
    info = info.replace(" ",0)
    if isinstance(info[0], str):
        ong_nom = unicodedata.normalize('NFD', info[0].lower().strip()).encode('ascii', 'ignore').decode("utf-8") 
        ong_nom = revisionNombres(ong_nom)
        if ong_nom not in ong:
            ong[ong_nom] = {}
        if 2015 not in ong[ong_nom]:
            ong[ong_nom][2015] = {}
        ong[ong_nom][2015]["VOLUNTARIOS"] = {}
        ong[ong_nom][2015]["VOLUNTARIOS"]["España"] = float(info["Total España"])
        ong[ong_nom][2015]["VOLUNTARIOS"]["Extranjero"] = float(info["Total extranjero"])




voluntarios = pd.read_excel('../dades/Datos_vf/ongd11_informe2016_Base Social.xls', sheet_name="base social")
header_voluntarios = voluntarios.iloc[1]
voluntarios = voluntarios[3:]
voluntarios.columns = header_voluntarios
i = 0

for i in range(0,len(voluntarios)):
    info = voluntarios.iloc[i]
    info = info.replace("n.d.",0)
    info = info.replace(" ",0)
    if isinstance(info[0], str):
        ong_nom = unicodedata.normalize('NFD', info[0].lower().strip()).encode('ascii', 'ignore').decode("utf-8") 
        ong_nom = revisionNombres(ong_nom)
        if ong_nom not in ong:
            ong[ong_nom] = {}
        if 2016 not in ong[ong_nom]:
            ong[ong_nom][2016] = {}
        ong[ong_nom][2016]["VOLUNTARIOS"] = {}
        ong[ong_nom][2016]["VOLUNTARIOS"]["España"] = float(info["Total España"])
        ong[ong_nom][2016]["VOLUNTARIOS"]["Extranjero"] = float(info["Total extranjero"])
    

#trabajadores ong 2014 a 2015 ["TRABAJADORES"]
treballadors = pd.read_excel('../dades/Datos_vf/ongd8_informe2014-2015_Recursos Humanos.xls', sheet_name='Remunerado 2014')
header_treballadors = treballadors.iloc[2] #grab the first row for the header
treballadors = treballadors[3:] #take the data less the header row
header_treballadors[0]="ONG"
header_treballadors[5]="Extranjero"
header_treballadors[7]="Local"

treballadors.columns = header_treballadors 

i = 0
for i in range(0,len(treballadors)):
    info = treballadors.iloc[i]
    if isinstance(info["ONG"], str):
        ong_nom = unicodedata.normalize('NFD', info["ONG"].lower().strip()).encode('ascii', 'ignore').decode("utf-8") 
        ong_nom = revisionNombres(ong_nom)
        if ong_nom not in ong:
            ong[ong_nom] = {}
        if 2014 not in ong[ong_nom]:
            ong[ong_nom][2014] = {}
        ong[ong_nom][2014]["TRABAJADORES"] = {}
        
        ong[ong_nom][2014]["TRABAJADORES"]["España"] = info["Oficina Central "] + info["Delegaciones "]
        ong[ong_nom][2014]["TRABAJADORES"]["Extranjero"] = info["Extranjero"]
        ong[ong_nom][2014]["TRABAJADORES"]["Local"] = info["Local"]
                

treballadors = pd.read_excel('../dades/Datos_vf/ongd8_informe2014-2015_Recursos Humanos.xls', sheet_name='Remunerado 2015')
header_treballadors = treballadors.iloc[2]
treballadors = treballadors[3:] 
header_treballadors[0]="ONG"
header_treballadors[5]="Extranjero"
header_treballadors[7]="Local"

treballadors.columns = header_treballadors 

i = 0
for i in range(0,len(treballadors)):
    info = treballadors.iloc[i]
    if isinstance(info["ONG"], str):
        ong_nom = unicodedata.normalize('NFD', info["ONG"].lower().strip()).encode('ascii', 'ignore').decode("utf-8") 
        ong_nom = revisionNombres(ong_nom)
        if ong_nom not in ong:
            ong[ong_nom] = {}
        if 2015 not in ong[ong_nom]:
            ong[ong_nom][2015] = {}
        ong[ong_nom][2015]["TRABAJADORES"] = {}
        
        ong[ong_nom][2015]["TRABAJADORES"]["España"] = info["Oficina Central "] + info["Delegaciones "]
        ong[ong_nom][2015]["TRABAJADORES"]["Extranjero"] = info["Extranjero"]
        ong[ong_nom][2015]["TRABAJADORES"]["Local"] = info["Local"]

treballadors = pd.read_excel('../dades/Datos_vf/ongd8_informe2016_Recursos Humanos.xls', sheet_name='Remunerado')
treballadors = treballadors[1:] 

i = 0
for i in range(0,len(treballadors)):
    info = treballadors.iloc[i]
    if isinstance(info["Nombre"], str):
        ong_nom = unicodedata.normalize('NFD', info["Nombre"].lower().strip()).encode('ascii', 'ignore').decode("utf-8") 
        ong_nom = revisionNombres(ong_nom)
        if ong_nom not in ong:
            ong[ong_nom] = {}
        if 2016 not in ong[ong_nom]:
            ong[ong_nom][2016] = {}
        ong[ong_nom][2016]["TRABAJADORES"] = {}
        
        ong[ong_nom][2016]["TRABAJADORES"]["España"] = info["Total España"]
        ong[ong_nom][2016]["TRABAJADORES"]["Extranjero"] = info["En el \nextranjero/ Cooperantes"]
        ong[ong_nom][2016]["TRABAJADORES"]["Local"] = info["Personal Local"]



treballadors = pd.read_excel('../dades/Datos_vf/Coordinadora ONG Desarrollo España - Datos año 2012.xls', sheet_name='Anex 4')
header_treballadors = treballadors.iloc[5]
treballadors = treballadors[8:]
header_treballadors[1]="ONG"
header_treballadors[3]="En España"
treballadors.columns = header_treballadors

i = 0
for i in range(0,len(treballadors)):
    info = treballadors.iloc[i]
    if isinstance(info["ONG"], str):
        ong_nom = unicodedata.normalize('NFD', info["ONG"].lower().strip()).encode('ascii', 'ignore').decode("utf-8") 
        ong_nom = revisionNombres(ong_nom)
        if ong_nom not in ong:
            ong[ong_nom] = {}
        if 2012 not in ong[ong_nom]:
            ong[ong_nom][2012] = {}
        ong[ong_nom][2012]["TRABAJADORES"] = {}
        ong[ong_nom][2012]["TRABAJADORES"]["España"] = info["En España"] + info[4]
        ong[ong_nom][2012]["TRABAJADORES"]["Extranjero"] = info[5]
        ong[ong_nom][2012]["TRABAJADORES"]["Local"] = info[-5]


treballadors = pd.read_excel('../dades/Datos_vf/Coordinadora ONG Desarrollo España - Datos año 2013.xls', sheet_name='Anex 4')
header_treballadors = treballadors.iloc[5] 
treballadors = treballadors[8:]
header_treballadors[1]="ONG"
header_treballadors[3]="En España"
treballadors.columns = header_treballadors

i = 0
for i in range(0,len(treballadors)):
    info = treballadors.iloc[i]
    if isinstance(info["ONG"], str):
        ong_nom = unicodedata.normalize('NFD', info["ONG"].lower().strip()).encode('ascii', 'ignore').decode("utf-8") 
        ong_nom = revisionNombres(ong_nom)
        if ong_nom not in ong:
            ong[ong_nom] = {}
        if 2013 not in ong[ong_nom]:
            ong[ong_nom][2013] = {}
        ong[ong_nom][2013]["TRABAJADORES"] = {}
        ong[ong_nom][2013]["TRABAJADORES"]["España"] = info["En España"] + info[4]
        ong[ong_nom][2013]["TRABAJADORES"]["Extranjero"] = info[5]
        ong[ong_nom][2013]["TRABAJADORES"]["Local"] = info[-5]


# ["PROYECTOS"] 2013 a 2016    
path = '../dades/Datos_vf/projectes2013/'
for root, dirs, files in os.walk(path):
    for filename in files:
        
        file = csv.reader(open(path+filename,"r",encoding="utf-8"), delimiter='\t',quoting=csv.QUOTE_NONE)
        info = []
        for el in file:
            for el2 in el:
                info.append(el2)
        ong_nom = unicodedata.normalize('NFD', filename[:-5].lower().strip()).encode('ascii', 'ignore').decode("utf-8") 
        ong_nom = revisionNombres(ong_nom)
        if not ong_nom in ong:
            ong[ong_nom] = {}
        if not 2013 in ong[ong_nom]:
            ong[ong_nom][2013] = {}
        if not 2012 in ong[ong_nom]:
            ong[ong_nom][2012] = {}
        ong[ong_nom][2013]["PROYECTOS"] = {}
        ong[ong_nom][2012]["PROYECTOS"] = {}
        for j in range(len(info)):
            pais= revisionPaises(unicodedata.normalize('NFD', info[j].lower().strip()).encode('ascii', 'ignore').decode("utf-8") )
            if pais in paises:
                ong[ong_nom][2013]["PROYECTOS"][pais]= float(info[j+2][:info[j+2].index('€')-1].replace('.', '').replace(',','.'))/2.0
                ong[ong_nom][2012]["PROYECTOS"][pais]= float(info[j+2][:info[j+2].index('€')-1].replace('.', '').replace(',','.'))/2.0
            else:
                pass
                
proyectos = pd.read_excel('../dades/Datos_vf/ongd14_informe2014-2015_proyectos paises.xlsx', sheet_name='ongd paises')
header_paises = proyectos.iloc[0]
header_proyectos = proyectos.iloc[1]
proyectos = proyectos[2:] 
proyectos.columns = header_proyectos

i = 0
for i in range(0,len(proyectos)):
    info = proyectos.iloc[i]
    ong_nom = unicodedata.normalize('NFD', info["Rótulos de fila"].lower().strip()).encode('ascii', 'ignore').decode("utf-8") 
    ong_nom = revisionNombres(ong_nom)
    if ong_nom not in ong:
        ong[ong_nom] = {}
    if 2014 not in ong[ong_nom]:
        ong[ong_nom][2014] = {}
    if 2015 not in ong[ong_nom]:
        ong[ong_nom][2015] = {}
    ong[ong_nom][2014]["PROYECTOS"] = {}
    ong[ong_nom][2015]["PROYECTOS"] = {}
    for j in range(len(info)):
        if j > 3:
            if not numpy.isnan(info[j]) and not isinstance(header_paises[j], str):
                dineritos = float(info[j])/2.0
                if numpy.isnan(dineritos):
                    dineritos = 0
                pais= revisionPaises(unicodedata.normalize('NFD', header_paises[j-1].lower().strip()).encode('ascii', 'ignore').decode("utf-8") )
                if pais in paises:
                    ong[ong_nom][2014]["PROYECTOS"][pais]=dineritos
                    ong[ong_nom][2015]["PROYECTOS"][pais]=dineritos
                else:
                    pass
            elif not numpy.isnan(info[j]) and isinstance(header_paises[j], str) and j < (len(info)-1):
                if numpy.isnan(info[j+1]):
                    pais= revisionPaises(unicodedata.normalize('NFD', header_paises[j].lower().strip()).encode('ascii', 'ignore').decode("utf-8") )
                    if pais in paises:
                        ong[ong_nom][2014]["PROYECTOS"][pais]=0
                        ong[ong_nom][2015]["PROYECTOS"][pais]=0
                    else:
                        pass


proyectos = pd.read_excel('../dades/Datos_vf/informe2016_ongd por paises.xlsx', sheet_name='ongd paises')
header_paises = proyectos.iloc[0]
header_proyectos = proyectos.iloc[1] 
proyectos = proyectos[2:] 
proyectos.columns = header_proyectos

i = 0
for i in range(0,len(proyectos)):
    info = proyectos.iloc[i]
    ong_nom = unicodedata.normalize('NFD', info["Rótulos de fila"].lower().strip()).encode('ascii', 'ignore').decode("utf-8") 
    ong_nom = revisionNombres(ong_nom)
    if ong_nom not in ong:
        ong[ong_nom] = {}
    if 2016 not in ong[ong_nom]:
        ong[ong_nom][2016] = {}
    ong[ong_nom][2016]["PROYECTOS"] = {}
    for j in range(len(info)):
        if j > 3:
            if not numpy.isnan(info[j]) and not isinstance(header_paises[j], str):
                dineritos = info[j]
                if numpy.isnan(dineritos):
                    dineritos = 0
                pais= revisionPaises(unicodedata.normalize('NFD', header_paises[j-1].lower().strip()).encode('ascii', 'ignore').decode("utf-8") )
                if pais in paises:
                    ong[ong_nom][2016]["PROYECTOS"][pais]=dineritos
                else:
                    pass
            elif not numpy.isnan(info[j]) and isinstance(header_paises[j], str) and j < (len(info)-1):
                if numpy.isnan(info[j+1]):
                    pais= revisionPaises(unicodedata.normalize('NFD', header_paises[j].lower().strip()).encode('ascii', 'ignore').decode("utf-8") )
                    if pais in paises:
                        ong[ong_nom][2016]["PROYECTOS"][pais]=0            
                    else:
                        pass
                
# ['INGRESOS'] y ['TRABAJADORES'] 2009

proyectos = pd.read_excel('../dades/Datos_vf/Informe2008-2009_ongd por paises + dineritos + personal.xlsx', sheet_name='2009')
header = proyectos.iloc[0]
proyectos = proyectos[1:] 
header[0] = "ong"
header[1] = "pais"
header[2]
proyectos.columns = header

i = 0
for i in range(0,len(proyectos)):
    info = proyectos.iloc[i]
    if info["pais"] != " " and info["pais"] != 0:
        info = info.replace(" ",0)
        info = info.replace("n/i",0)
        ong_nom = unicodedata.normalize('NFD', info["ong"].lower().strip()).encode('ascii', 'ignore').decode("utf-8") 
        ong_nom = revisionNombres(ong_nom)
        if ong_nom not in ong:
            ong[ong_nom] = {}
        if 2009 not in ong[ong_nom]:
            ong[ong_nom][2009] = {}
        ong[ong_nom][2009]["PROYECTOS"] = {}
        
        for j in range(len(info)):
            pais= revisionPaises(unicodedata.normalize('NFD', info["pais"].lower().strip()).encode('ascii', 'ignore').decode("utf-8") )
            if pais in paises:
                dineritos = info["3. Importe de euros por País"]
                ong[ong_nom][2009]["PROYECTOS"][pais]=dineritos
            else:
                pass
        
        ong[ong_nom][2009]["INGRESOS"] = {}
        ong[ong_nom][2009]["INGRESOS"]["TOTAL cuotas periódicas"] = float(info["13.1 Cuotas periódicas y apadrinamientos"])
        ong[ong_nom][2009]["INGRESOS"]["TOTAL donaciones"] = float(info["13.2 donaciones"])
        ong[ong_nom][2009]["INGRESOS"]["TOTAL EMPRESAS"] = float(info["13.3 Fondo de empresas fundaciones"])
        ong[ong_nom][2009]["INGRESOS"]["TOTAL venta productos"] = float(info["13.4 Venta de productos"])
        ong[ong_nom][2009]["INGRESOS"]["TOTAL servicios prestados"] = float(info["13.5 Servicios prestados y  Otros fondos"])
        ong[ong_nom][2009]["INGRESOS"]["TOTAL fondos privados"] = float(info["13.6 Total Fondos privados"])
        ong[ong_nom][2009]["INGRESOS"]["TOTAL ESTATAL"] = float(info["12.1 MAE y otros ministerios"])
        ong[ong_nom][2009]["INGRESOS"]["TOTAL AUTONÓMICO"] = float(info["12.2 Cooperación descentralizada"])
        ong[ong_nom][2009]["INGRESOS"]["TOTAL INTERNACIONAL"] = float(info["12.3 Ámbito internacional"])
        ong[ong_nom][2009]["INGRESOS"]["Otros fondos públicos"] = float(info["12.4 Otros fondos"])
        ong[ong_nom][2009]["INGRESOS"]["TOTAL fondos públicos"] = float(info["12.5 Total Fondos Públicos"])
        
        ong[ong_nom][2009]["TRABAJADORES"] = {}
        ong[ong_nom][2009]["TRABAJADORES"]["España"] = float(info["15.1 Oficina Central"]) +float(info["15.2 Delegaciones"])
        ong[ong_nom][2009]["TRABAJADORES"]["Extranjero"] = float(info["15.3 En el extranjero"])
        ong[ong_nom][2009]["TRABAJADORES"]["Local"] = float(info["15. 6 Personal Local"])
        
        
# ['INGRESOS'], ['TRABAJADORES'] y ['PROYECTOS'] 2011

path = 'C:/Users/bcoma/Documents/Noemi/Tesis/dades/Datos_vf/2011/'
for root, dirs, files in os.walk(path):
    for filename in files:
        file = csv.reader(open(path+filename,"r",encoding="latin-1"), delimiter='\t',quoting=csv.QUOTE_NONE)
        info = []
        for el in file:
            for el2 in el:
                info.append(el2)
        ong_nom = unicodedata.normalize('NFD', filename[:filename.index('_')].lower().strip()).encode('ascii', 'ignore').decode("utf-8") 
        ong_nom = revisionNombres(ong_nom)
        
        if ong_nom not in ong:
            ong[ong_nom] = {}
        if 2011 not in ong[ong_nom]:
            ong[ong_nom][2011] = {}
        if ("Fondos privados" in info):
            ong[ong_nom][2011]["INGRESOS"] = {}
        
            posMoney = -1
            for j in range(len(info)):
                if '\x80' in info[j]: 
                    posMoney = j
                    break
            if ong_nom in ["aieti","frs"]:
                ong[ong_nom][2011]["INGRESOS"]["TOTAL cuotas periódicas"] = 0
                ong[ong_nom][2011]["INGRESOS"]["TOTAL donaciones"] = 0
                ong[ong_nom][2011]["INGRESOS"]["TOTAL EMPRESAS"] = 0
                ong[ong_nom][2011]["INGRESOS"]["TOTAL venta productos"] = 0
                ong[ong_nom][2011]["INGRESOS"]["TOTAL servicios prestados"] = 0
                
                ong[ong_nom][2011]["INGRESOS"]["TOTAL ESTATAL"] = 0
                ong[ong_nom][2011]["INGRESOS"]["TOTAL AUTONÓMICO"] = 0
                ong[ong_nom][2011]["INGRESOS"]["TOTAL INTERNACIONAL"] = 0
                ong[ong_nom][2011]["INGRESOS"]["Otros fondos públicos"] = 0
                
                ong[ong_nom][2011]["INGRESOS"]["TOTAL fondos privados"] = 0
                ong[ong_nom][2011]["INGRESOS"]["TOTAL fondos públicos"] = 0
                
            else:
                ong[ong_nom][2011]["INGRESOS"]["TOTAL cuotas periódicas"] = float(info[posMoney+1][:-2].replace('.', '').replace(',','.'))+float(info[posMoney+2][:-2].replace('.', '').replace(',','.'))+float(info[posMoney+3][:-2].replace('.', '').replace(',','.'))
                ong[ong_nom][2011]["INGRESOS"]["TOTAL donaciones"] = float(info[posMoney+4][:-2].replace('.', '').replace(',','.'))
                ong[ong_nom][2011]["INGRESOS"]["TOTAL EMPRESAS"] = float(info[posMoney+5][:-2].replace('.', '').replace(',','.'))#+float(info[posMoney+5][:-2])
                ong[ong_nom][2011]["INGRESOS"]["TOTAL venta productos"] = float(info[posMoney+6][:-2].replace('.', '').replace(',','.'))
                ong[ong_nom][2011]["INGRESOS"]["TOTAL servicios prestados"] = float(info[posMoney+7][:-2].replace('.', '').replace(',','.'))+float(info[posMoney+8][:-2].replace('.', '').replace(',','.'))
                ong[ong_nom][2011]["INGRESOS"]["TOTAL fondos privados"] = float(info[posMoney][:-2].replace('.', '').replace(',','.'))
                
                ong[ong_nom][2011]["INGRESOS"]["TOTAL ESTATAL"] = float(info[posMoney+10][:-2].replace('.', '').replace(',','.'))
                ong[ong_nom][2011]["INGRESOS"]["TOTAL AUTONÓMICO"] = float(info[posMoney+11][:-2].replace('.', '').replace(',','.'))
                ong[ong_nom][2011]["INGRESOS"]["TOTAL INTERNACIONAL"] = float(info[posMoney+12][:-2].replace('.', '').replace(',','.'))
                ong[ong_nom][2011]["INGRESOS"]["Otros fondos públicos"] = float(info[posMoney+13][:-2].replace('.', '').replace(',','.'))
                ong[ong_nom][2011]["INGRESOS"]["TOTAL fondos públicos"] = float(info[posMoney+9][:-2].replace('.', '').replace(',','.'))

            ong[ong_nom][2011]["TRABAJADORES"] = {}
            ong[ong_nom][2011]["VOLUNTARIOS"] = {}

            if not "En España" in info:
                ong[ong_nom][2011]["TRABAJADORES"]["España"] = "No info"
                ong[ong_nom][2011]["TRABAJADORES"]["Extranjero"] = "No info"
            else:    
                ong[ong_nom][2011]["TRABAJADORES"]["España"] = float(info[info.index('En España')+3][:-9])
                ong[ong_nom][2011]["TRABAJADORES"]["Extranjero"] = float(info[info.index('En España')+4][:-9])
            if 'PERSONAL LOCAL' in info:
                ong[ong_nom][2011]["TRABAJADORES"]["Local"] = float(info[info.index('PERSONAL LOCAL')+4][:-9])
            else:
                ong[ong_nom][2011]["TRABAJADORES"]["Local"] = 'No info'
            
            if "Personal voluntario en Cooperación" in info:
                ong[ong_nom][2011]["VOLUNTARIOS"]["España"] = float(info[info.index('Personal voluntario en Cooperación')+4])+float(info[info.index('Personal voluntario en Cooperación')+5])
                ong[ong_nom][2011]["VOLUNTARIOS"]["Extranjero"] = float(info[info.index('Personal voluntario en Cooperación')+6])
            else:
                ong[ong_nom][2011]["VOLUNTARIOS"]["España"] = 0
                ong[ong_nom][2011]["VOLUNTARIOS"]["Extranjero"] = 0
        else:
            ong[ong_nom][2011]["PROYECTOS"] = {}

            for j in range(len(info)):
                paises_proyecto = []
                dinerito = 0
                if "proyecto" in info[j]:
                    dinerito = info[j+1]
                    j-=1
                    pais= revisionPaises(unicodedata.normalize('NFD', info[j].lower().strip()).encode('ascii', 'ignore').decode("utf-8") )
                    paises_proyecto.append(pais)
                    while True:
                        j-=1
                        pais= revisionPaises(unicodedata.normalize('NFD', info[j].lower().strip()).encode('ascii', 'ignore').decode("utf-8") )
                        if pais in paises:
                            paises_proyecto.append(pais)
                        else:
                            break
                        
                for pais in paises_proyecto:
                    ong[ong_nom][2011]["PROYECTOS"][pais]= float(dinerito[:-1].replace('.', '').replace(',','.'))/len(paises_proyecto)
                paises_proyecto = [] 


# ['INGRESOS'], ['TRABAJADORES'] y ['PROYECTOS'] 2010
path = 'C:/Users/bcoma/Documents/Noemi/Tesis/dades/Datos_vf/2010/'
for root, dirs, files in os.walk(path):
    for filename in files:
        #filename = "acción contra el hambre_paises_ 2010"
        
        typeLectura = 0
        try:
            file= csv.reader(open(path+filename,"r",encoding="utf8"), delimiter='\t',quoting=csv.QUOTE_NONE)
            info = []
            for el in file:
                for el2 in el:
                    info.append(el2)
        except:
            typeLectura = 1
            file= csv.reader(open(path+filename,"r",encoding="latin-1"), delimiter='\t',quoting=csv.QUOTE_NONE)
            info = []
            for el in file:
                for el2 in el:
                    info.append(el2)
        ong_nom = unicodedata.normalize('NFD', filename[:filename.index('_')].lower().strip()).encode('ascii', 'ignore').decode("utf-8") 
        ong_nom = revisionNombres(ong_nom)
        if ong_nom not in ong:
            ong[ong_nom] = {}
        if 2010 not in ong[ong_nom]:
            ong[ong_nom][2010] = {}
        if "Fondos privados" in info:
            ong[ong_nom][2010]["INGRESOS"] = {}
        
            posMoney = -1
            for j in range(len(info)):
                if ('€' in info[j] and typeLectura==0) or ('\x80' in info[j] and typeLectura==1): 
                    posMoney = j
                    break
            ong[ong_nom][2010]["INGRESOS"]["TOTAL cuotas periódicas"] = float(info[posMoney+1][:-2].replace('.', '').replace(',','.'))+float(info[posMoney+2][:-2].replace('.', '').replace(',','.'))+float(info[posMoney+3][:-2].replace('.', '').replace(',','.'))
            ong[ong_nom][2010]["INGRESOS"]["TOTAL donaciones"] = float(info[posMoney+4][:-2].replace('.', '').replace(',','.'))
            ong[ong_nom][2010]["INGRESOS"]["TOTAL EMPRESAS"] = float(info[posMoney+5][:-2].replace('.', '').replace(',','.'))#+float(info[posMoney+5][:-2])
            ong[ong_nom][2010]["INGRESOS"]["TOTAL venta productos"] = float(info[posMoney+6][:-2].replace('.', '').replace(',','.'))
            ong[ong_nom][2010]["INGRESOS"]["TOTAL servicios prestados"] = float(info[posMoney+7][:-2].replace('.', '').replace(',','.'))+float(info[posMoney+8][:-2].replace('.', '').replace(',','.'))
            ong[ong_nom][2010]["INGRESOS"]["TOTAL fondos privados"] = float(info[posMoney][:-2].replace('.', '').replace(',','.'))

            
            
            ong[ong_nom][2010]["INGRESOS"]["TOTAL ESTATAL"] = float(info[posMoney+10][:-2].replace('.', '').replace(',','.'))
            ong[ong_nom][2010]["INGRESOS"]["TOTAL AUTONÓMICO"] = float(info[posMoney+11][:-2].replace('.', '').replace(',','.'))
            ong[ong_nom][2010]["INGRESOS"]["TOTAL INTERNACIONAL"] = float(info[posMoney+12][:-2].replace('.', '').replace(',','.'))
            ong[ong_nom][2010]["INGRESOS"]["Otros fondos públicos"] = float(info[posMoney+13][:-2].replace('.', '').replace(',','.'))
            ong[ong_nom][2010]["INGRESOS"]["TOTAL fondos públicos"] = float(info[posMoney+9][:-2].replace('.', '').replace(',','.'))

            
            
            ong[ong_nom][2010]["TRABAJADORES"] = {}
            ong[ong_nom][2010]["VOLUNTARIOS"] = {}
            if not "En España" in info:
                ong[ong_nom][2010]["TRABAJADORES"]["España"] = "No info"
                ong[ong_nom][2010]["TRABAJADORES"]["Extranjero"] = "No info"
            else:
                ong[ong_nom][2010]["TRABAJADORES"]["España"] = float(info[info.index('En España')+3][:-9])
                ong[ong_nom][2010]["TRABAJADORES"]["Extranjero"] = float(info[info.index('En España')+4][:-9])
            if 'PERSONAL LOCAL' in info:
                ong[ong_nom][2010]["TRABAJADORES"]["Local"] = float(info[info.index('PERSONAL LOCAL')+4][:-9])
            else:
                ong[ong_nom][2010]["TRABAJADORES"]["Local"] = 'No info'
            
            if "Personal voluntario en Cooperación" in info:
                ong[ong_nom][2010]["VOLUNTARIOS"]["España"] = float(info[info.index('Personal voluntario en Cooperación')+4])+float(info[info.index('Personal voluntario en Cooperación')+5])
                ong[ong_nom][2010]["VOLUNTARIOS"]["Extranjero"] = float(info[info.index('Personal voluntario en Cooperación')+6])
            else:
                ong[ong_nom][2010]["VOLUNTARIOS"]["España"] = 0
                ong[ong_nom][2010]["VOLUNTARIOS"]["Extranjero"] = 0
        else:
            ong[ong_nom][2010]["PROYECTOS"] = {}
            for j in range(len(info)):
                paises_proyecto = []
                dinerito = 0
                if "proyecto" in info[j]:
                    dinerito = info[j+1]
                    j-=1
                    pais= revisionPaises(unicodedata.normalize('NFD', info[j].lower().strip()).encode('ascii', 'ignore').decode("utf-8") )
                    paises_proyecto.append(pais)
                    while True:
                        j-=1
                        pais= revisionPaises(unicodedata.normalize('NFD', info[j].lower().strip()).encode('ascii', 'ignore').decode("utf-8") )
                        if pais in paises:
                            paises_proyecto.append(pais)
                        else:
                            break
                for pais in paises_proyecto:
                    ong[ong_nom][2010]["PROYECTOS"][pais]= float(dinerito[:-1].replace('.', '').replace(',','.'))/len(paises_proyecto)
                paises_proyecto = [] 



# ['INGRESOS'], ['TRABAJADORES'] y ['PROYECTOS'] 2009

path = 'C:/Users/bcoma/Documents/Noemi/Tesis/dades/Datos_vf/2009/'
for root, dirs, files in os.walk(path):
    for filename in files:
        typeLectura = 0
        try:
            file= csv.reader(open(path+filename,"r",encoding="utf8"), delimiter='\t',quoting=csv.QUOTE_NONE)
            info = []
            for el in file:
                for el2 in el:
                    info.append(el2)
        except:
            typeLectura = 1
            file= csv.reader(open(path+filename,"r",encoding="latin-1"), delimiter='\t',quoting=csv.QUOTE_NONE)
            info = []
            for el in file:
                for el2 in el:
                    info.append(el2)
        ong_nom = unicodedata.normalize('NFD', filename[:filename.index('_')].lower().strip()).encode('ascii', 'ignore').decode("utf-8") 
        ong_nom = revisionNombres(ong_nom)
        if ong_nom not in ong:
            ong[ong_nom] = {}
        if 2009 not in ong[ong_nom]:
            ong[ong_nom][2009] = {}
        if "Fondos privados" in info:
            ong[ong_nom][2009]["INGRESOS"] = {}
        
            posMoney = -1
            for j in range(len(info)):
                if ('€' in info[j] and typeLectura==0) or ('\x80' in info[j] and typeLectura==1): 
                    posMoney = j
                    break
            ong[ong_nom][2009]["INGRESOS"]["TOTAL cuotas periódicas"] = float(info[posMoney+1][:-2].replace('.', '').replace(',','.'))+float(info[posMoney+2][:-2].replace('.', '').replace(',','.'))
            ong[ong_nom][2009]["INGRESOS"]["TOTAL donaciones"] = float(info[posMoney+3][:-2].replace('.', '').replace(',','.'))
            ong[ong_nom][2009]["INGRESOS"]["TOTAL EMPRESAS"] = float(info[posMoney+4][:-2].replace('.', '').replace(',','.'))#+float(info[posMoney+5][:-2])
            ong[ong_nom][2009]["INGRESOS"]["TOTAL venta productos"] = float(info[posMoney+5][:-2].replace('.', '').replace(',','.'))
            ong[ong_nom][2009]["INGRESOS"]["TOTAL servicios prestados"] = float(info[posMoney+6][:-2].replace('.', '').replace(',','.'))+float(info[posMoney+8][:-2].replace('.', '').replace(',','.'))
            ong[ong_nom][2009]["INGRESOS"]["TOTAL fondos privados"] = float(info[posMoney][:-2].replace('.', '').replace(',','.'))

            
            
            ong[ong_nom][2009]["INGRESOS"]["TOTAL ESTATAL"] = float(info[posMoney+9][:-2].replace('.', '').replace(',','.'))
            ong[ong_nom][2009]["INGRESOS"]["TOTAL AUTONÓMICO"] = float(info[posMoney+10][:-2].replace('.', '').replace(',','.'))
            ong[ong_nom][2009]["INGRESOS"]["TOTAL INTERNACIONAL"] = float(info[posMoney+11][:-2].replace('.', '').replace(',','.'))
            ong[ong_nom][2009]["INGRESOS"]["Otros fondos públicos"] = float(info[posMoney+12][:-2].replace('.', '').replace(',','.'))
            ong[ong_nom][2009]["INGRESOS"]["TOTAL fondos públicos"] = float(info[posMoney+8][:-2].replace('.', '').replace(',','.'))

        
            ong[ong_nom][2009]["TRABAJADORES"] = {}
            ong[ong_nom][2009]["VOLUNTARIOS"] = {}
        
            if not "Oficina central" in info:
                ong[ong_nom][2009]["TRABAJADORES"]["España"] = "No info"
                ong[ong_nom][2009]["TRABAJADORES"]["Extranjero"] = "No info"
            else:
                ong[ong_nom][2009]["TRABAJADORES"]["España"] = float(info[info.index('En el extranjero')+2][:-9])+float(info[info.index('En el extranjero')+3][:-9])
                ong[ong_nom][2009]["TRABAJADORES"]["Extranjero"] = float(info[info.index('En el extranjero')+4][:-9])
            if 'PERSONAL LOCAL' in info:
                ong[ong_nom][2009]["TRABAJADORES"]["Local"] = float(info[info.index('desglose recursos...')-1][:-9])
            else:
                ong[ong_nom][2009]["TRABAJADORES"]["Local"] = 'No info'
            
            if "Personal voluntario en Cooperación" in info:
                ong[ong_nom][2009]["VOLUNTARIOS"]["España"] = float(info[info.index('Personal voluntario en Cooperación')+4])+float(info[info.index('Personal voluntario en Cooperación')+5])
                ong[ong_nom][2009]["VOLUNTARIOS"]["Extranjero"] = float(info[info.index('Personal voluntario en Cooperación')+6])
            else:
                ong[ong_nom][2009]["VOLUNTARIOS"]["España"] = 0
                ong[ong_nom][2009]["VOLUNTARIOS"]["Extranjero"] = 0
        
        else:
            ong[ong_nom][2009]["PROYECTOS"] = {}
            for j in range(len(info)):
                paises_proyecto = []
                dinerito = 0
                if "proyecto" in info[j]:
                    dinerito = info[j+1]
                    j-=1
                    pais= revisionPaises(unicodedata.normalize('NFD', info[j].lower().strip()).encode('ascii', 'ignore').decode("utf-8") )
                    paises_proyecto.append(pais)
                    while True:
                        j-=1
                        pais= revisionPaises(unicodedata.normalize('NFD', info[j].lower().strip()).encode('ascii', 'ignore').decode("utf-8") )
                        if pais in paises:
                            paises_proyecto.append(pais)
                        else:
                            
                            break
                        
                for pais in paises_proyecto:
                    ong[ong_nom][2009]["PROYECTOS"][pais]= float(dinerito[:-1].replace('.', '').replace(',','.'))/len(paises_proyecto)
                paises_proyecto = [] 

path = 'C:/Users/bcoma/Documents/Noemi/Tesis/dades/Datos_vf/2008/'
for root, dirs, files in os.walk(path):
    for filename in files:
        typeLectura = 0
        try:
            file= csv.reader(open(path+filename,"r",encoding="utf8"), delimiter='\t',quoting=csv.QUOTE_NONE)
            info = []
            for el in file:
                for el2 in el:
                    info.append(el2)
        except:
            
            typeLectura = 1
            file= csv.reader(open(path+filename,"r",encoding="latin-1"), delimiter='\t',quoting=csv.QUOTE_NONE)
            info = []
            for el in file:
                for el2 in el:
                    info.append(el2)
        ong_nom = unicodedata.normalize('NFD', filename[:filename.index('_')].lower().strip()).encode('ascii', 'ignore').decode("utf-8") 
        ong_nom = revisionNombres(ong_nom)
        if ong_nom not in ong:
            ong[ong_nom] = {}
        if 2008 not in ong[ong_nom]:
            ong[ong_nom][2008] = {}
        if not "recursos" in filename:
            
            ong[ong_nom][2008]["PROYECTOS"] = {}
            j=0
            while j < len(info):
                pais= revisionPaises(unicodedata.normalize('NFD', info[j].lower().strip()).encode('ascii', 'ignore').decode("utf-8") )
                
                if pais in paises or "sin localizar" in pais:
                    llistatPaisos = []
                    llistatPaisos.append(pais)
                
                if "€" in info[j] and "Total Fondos" not in info[j] and "Realizado" not in info[j]:
                    dinerito = float(info[j].split("/")[-1][:-1].replace('.', '').replace(',','.'))/len(llistatPaisos)
                    for pais in llistatPaisos:
                        ong[ong_nom][2008]["PROYECTOS"][pais]= float(dinerito)
                    llistatPaisos = []
                j+=1
                    

# ['ONU'] TODOS LOS AÑOS
paises_ONU = {}
onu = pd.read_excel('../dades/Datos_vf/Países Priorizados ONU.xlsx', sheet_name='PMA')
for i in range(len(onu)):
    info = onu.iloc[i]
    pais_onu = revisionPaises(unicodedata.normalize('NFD', info["PAÍS"].lower().strip()).encode('ascii', 'ignore').decode("utf-8") ) 
    for year in onu.columns[1:]:    
        
        if pais_onu not in paises_ONU:
            paises_ONU[pais_onu] = {}
            paises_ONU[pais_onu]["ONU"] = {}
        paises_ONU[pais_onu]["ONU"][year]= info[year]
        if year == 2016:
            paises_ONU[pais_onu]["ONU"][2015]= info[year]    

###ingressos anuals països
file = pd.read_excel('../dades/Data_Extract_From_Indicadores_del_desarrollo_mundial/GDP per capal constant 2010 USD.xls', sheet_name='Data')
years = [2009,2010,2011,2012,2013,2014,2015,2016]
for i in range(len(file)):    
    info = file.iloc[i]
    if isinstance(info["Country Name"], str):
        pais_onu = revisionPaises(unicodedata.normalize('NFD', info["Country Name"].lower().strip()).encode('ascii', 'ignore').decode("utf-8") ) 
        
        if pais_onu not in paises_ONU:
            paises_ONU[pais_onu] = {}
        paises_ONU[pais_onu]["money"] = {}
        for j in range(len(years)):
            paises_ONU[pais_onu]["money"][years[j]]= info[str(years[j])]
paises_ONU["venezuela"]["money"][2015] = 13137
paises_ONU["venezuela"]["money"][2016] = 10984
paises_ONU["eritrea"]["money"][2012] = 417.09
paises_ONU["eritrea"]["money"][2013] = 343.26
paises_ONU["eritrea"]["money"][2014] = 447.27
paises_ONU["eritrea"]["money"][2015] = 436.09
paises_ONU["eritrea"]["money"][2016] = 469.23
paises_ONU["sudan del sur"]["money"][2016] = 171.03
paises_ONU["yibuti"]["money"][2009] = 1225
paises_ONU["yibuti"]["money"][2011] = 1402.3
paises_ONU["yibuti"]["money"][2012] = 1615.45
paises_ONU["yibuti"]["money"][2013] = 1634.83
paises_ONU["yibuti"]["money"][2014] = 1726.3
paises_ONU["yibuti"]["money"][2015] = 2216.57
paises_ONU["yibuti"]["money"][2016] = 2316.31
paises_ONU["somalia"]["money"][2009] = 112.1
paises_ONU["somalia"]["money"][2010] = 100.56
paises_ONU["somalia"]["money"][2011] = 298.44
paises_ONU["somalia"]["money"][2012] = 324.71
paises_ONU["somalia"]["money"][2013] = 329.58
paises_ONU["somalia"]["money"][2014] = 324.64
paises_ONU["somalia"]["money"][2015] = 386.32
paises_ONU["somalia"]["money"][2016] = 390
paises_ONU["siria"]["money"][2009] = 2696.92
paises_ONU["siria"]["money"][2010] = 3114.95
paises_ONU["siria"]["money"][2011] = 3373.49
paises_ONU["siria"]["money"][2012] = 4125.78
paises_ONU["siria"]["money"][2013] = 1526.54
paises_ONU["siria"]["money"][2014] = 1366.25
paises_ONU["siria"]["money"][2015] = 1404.51
paises_ONU["siria"]["money"][2016] = 941.02
paises_ONU["corea del norte"]["money"][2009] = 519.25
paises_ONU["corea del norte"]["money"][2010] = 629.84
paises_ONU["corea del norte"]["money"][2011] = 671.42
paises_ONU["corea del norte"]["money"][2012] = 733.86
paises_ONU["corea del norte"]["money"][2013] = 735.33
paises_ONU["corea del norte"]["money"][2014] = 768.41
paises_ONU["corea del norte"]["money"][2015] = 856.79
paises_ONU["corea del norte"]["money"][2016] = 880.78

paises_ONU["sahara occidental"] = {}
paises_ONU["sahara occidental"]["money"] = {}
paises_ONU["sahara occidental"]["money"][2009] = paises_ONU["marruecos"]["money"][2009]*0.33
paises_ONU["sahara occidental"]["money"][2010] = paises_ONU["marruecos"]["money"][2010]*0.33
paises_ONU["sahara occidental"]["money"][2011] = paises_ONU["marruecos"]["money"][2011]*0.33
paises_ONU["sahara occidental"]["money"][2012] = paises_ONU["marruecos"]["money"][2012]*0.33
paises_ONU["sahara occidental"]["money"][2013] = paises_ONU["marruecos"]["money"][2013]*0.33
paises_ONU["sahara occidental"]["money"][2014] = paises_ONU["marruecos"]["money"][2014]*0.33
paises_ONU["sahara occidental"]["money"][2015] = paises_ONU["marruecos"]["money"][2015]*0.33
paises_ONU["sahara occidental"]["money"][2016] = paises_ONU["marruecos"]["money"][2016]*0.33
paises_ONU["islas cook"] = {}
paises_ONU["islas cook"]["money"] = {}
paises_ONU["islas cook"]["money"][2009] = 10886
paises_ONU["islas cook"]["money"][2010] = 12300
paises_ONU["islas cook"]["money"][2011] = 14022
paises_ONU["islas cook"]["money"][2012] = 15564
paises_ONU["islas cook"]["money"][2013] = 15550
paises_ONU["islas cook"]["money"][2014] = 17314
paises_ONU["islas cook"]["money"][2015] = 16448
paises_ONU["islas cook"]["money"][2016] = 16925



paises_ONU["serbia y montenegro"] = {}
paises_ONU["serbia y montenegro"]["money"]= {}
for year in paises_ONU["serbia"]["money"]:
    paises_ONU["serbia y montenegro"]["money"][year] = (paises_ONU["serbia"]["money"][year]*11+paises_ONU["montenegro"]["money"][year])/12

#paises_onu tinc diners i prioritat    
    
mision = pd.read_excel('../dades/Datos_vf/misioìn todas ong.xlsx', sheet_name='misiónONGD')
header = mision.iloc[0]
mision = mision[1:] 
mision.columns = header

for i in range(len(mision)):    
    info = mision.iloc[i]
    ongd = revisionNombres(unicodedata.normalize('NFD', info["ONGD"].lower().strip()).encode('ascii', 'ignore').decode("utf-8") )
    
    if ongd not in ong:
        ong[ongd] = {}
    ong[ongd]["priorizacion"] = [info[1],info[2],info[3],info[4]]
    

mision = pd.read_excel('../dades/Datos_vf/misioìn todas ong.xlsx', sheet_name='forma jurídica')

for i in range(len(mision)):    
    info = mision.iloc[i]
    ongd = revisionNombres(unicodedata.normalize('NFD', info["Nombre"].lower().strip()).encode('ascii', 'ignore').decode("utf-8") )    
    ongd = revisionNombres(ongd)
    
    if ongd not in ong:
        ong[ongd] = {}
    ong[ongd]["info"] = [info[4]]
    
    
universal = pd.read_excel('../Listado ONGD socias.xlsx', sheet_name='Hoja1')
new_header = universal.iloc[0] 
universal = universal[1:] 
universal.columns = new_header 


for i in range(len(universal)):  
    
    info = universal.iloc[i]
    try:
        ongd = revisionNombres(unicodedata.normalize('NFD', info["Nombre habitual"].lower().strip()).encode('ascii', 'ignore').decode("utf-8") )    
        ongd = revisionNombres(ongd)
        if ongd in ong:
            ong[ongd]["anyo"] = info["Año constitución"]
            ong[ongd]["internacional"] = info["international =1 Nacional=0"]
    except:
        pass
    
import xlsxwriter

ongsFer = ["acción contra el hambre","cruz roja","cáritas","oxfam intermon","ayuda en acción","acción verapaz","medicus mundi","manos unidas"]
ongsFer = ongsFer + ["edificando comunidad de nazaret","farmamundi","cesal","codespa","alboan","aieti","fad", "adsis","educo","economistas sin fronteras"]
ongsFer = ongsFer + ["amref","farmaceuticos sin fronteras","adra","amigos de la tierra","fontilles","fere-ceca","fisc-compañia de maria","intered","juan ciudad"]
ongsFer = ongsFer + ["mundubat","paz con dignidad","prosalus","sed","iscod","medicos del mundo","jovenes y desarrollo","mpdl","entrepueblos","cideal","fundación valle"]
ongsFer = ongsFer + ["frs","entreculturas","iberoamerica europa","fuden","promoción social","proyde","pueblos hermanos"]

for ong_nom in ongsFer:
    print(ong_nom)
    posExcel = 1
    workbook = xlsxwriter.Workbook("../output/"+str(ong_nom)+".xlsx")
    worksheet = workbook.add_worksheet()
    worksheet.set_column('A:A', 20)
    bold = workbook.add_format({'bold': True})
    worksheet.write('A1', 'Pais-Año',bold)
    worksheet.write('B1', 'ONU',bold)
    worksheet.write('C1', 'GDP',bold)
    worksheet.write('D1', 'Public_Grant',bold)
    worksheet.write('E1', 'Budget_Previous_Year',bold)
    worksheet.write('F1', 'LatinAmerica',bold)
    worksheet.write('G1', 'Africa',bold)
    worksheet.write('H1', 'Confessional',bold)
    worksheet.write('I1', 'Universal',bold)
    worksheet.write('J1', 'Public_Funds_MAE',bold)
    #fer mae i "no mare"
    worksheet.write('K1', 'Public_Funds_Decentralized',bold)
    worksheet.write('L1', 'Public_Funds_Internacional',bold)
    worksheet.write('M1', 'Public_Funds_Other',bold)
    worksheet.write('N1', 'Public_Funds_Total',bold)
    worksheet.write('O1', 'Private_Funds_Cuotas',bold)
    worksheet.write('P1', 'Private_Funds_Donations',bold)
    worksheet.write('Q1', 'Private_Funds_Companies',bold)
    worksheet.write('R1', 'Fondos_Privados_Venta',bold)
    worksheet.write('S1', 'Fondos_Privados_Servicios',bold)
    worksheet.write('T1', 'Fondos_Privados_Total',bold)
    worksheet.write('U1', 'Personal_Remunerado_España',bold)
    worksheet.write('V1', 'Personal_Remunerado_En_el_Extranjero',bold)
    worksheet.write('W1', 'Personal_Remunerado_Local',bold)
    worksheet.write('X1', 'Personal_Remunerado_Total',bold)
    worksheet.write('Y1', 'Tamaño',bold)
    worksheet.write('Z1', 'Forma_Juridica_Fundacion',bold)
    worksheet.write('AA1', 'Forma_Juridica_Asociacion',bold)
    worksheet.write('AB1', 'Forma_Juridica_Federacion',bold)
    worksheet.write('AC1', 'Forma_Juridica_Otra',bold)
    worksheet.write('AD1', 'Voluntarios_Espanya',bold)
    worksheet.write('AE1', 'Voluntarios_Extranjero',bold)
    worksheet.write('AF1', 'Donor_Aid_Budget',bold)
    worksheet.write('AG1', 'Total_Funds',bold)
    worksheet.write('AH1', '%_Private_Funds',bold)
    worksheet.write('AI1', '%_MAE_Funds',bold)
    worksheet.write('AJ1', 'Anyo_ONG',bold)
    worksheet.write('AK1', 'Internacional',bold)
    worksheet.write("AL1", "Colony", bold)
    worksheet.write("AM1", "Delegation", bold)
    worksheet.write('AN1', 'Visitado',bold)
    worksheet.write('AO1', 'Dinero_en_el_proyecto',bold)
    
    for year in [2009,2010,2011,2012,2013,2014,2015,2016]:
        print(year)
        if year in ong[ong_nom]:
            paisos = []
            if "PROYECTOS" in ong[ong_nom][year]:
                paisos = ong[ong_nom][year]["PROYECTOS"].keys()
                for pais in paisos:
                    totalFondos = 0
                    totalFondosPrivados = 0
                    if pais in paises:
                        
                        worksheet.write(posExcel, 0, str(year)+"_"+str(pais))
                        if pais in paises_ONU and "ONU" in paises_ONU[pais]:
                            if year in paises_ONU[pais]["ONU"]:
                                worksheet.write(posExcel, 1, paises_ONU[pais]["ONU"][year])
                            else:
                                worksheet.write(posExcel, 1, 0)
                        else:
                            worksheet.write(posExcel, 1, 0)
                        if pais in paises_ONU:
                            if "money" in paises_ONU[pais]:
                                if year in paises_ONU[pais]["money"] and not (numpy.isnan(paises_ONU[pais]["money"][year])):
                                    worksheet.write(posExcel, 2, paises_ONU[pais]["money"][year])
                                else:
                                    worksheet.write(posExcel, 2, 0)
                            else:
                                worksheet.write(posExcel, 2, 0)
                        else:
                            worksheet.write(posExcel, 2, 0)
                        if "SUBVENCIONES" in ong[ong_nom][year]:
                            if pais in ong[ong_nom][year]["SUBVENCIONES"]:
                                worksheet.write(posExcel, 3, ong[ong_nom][year]["SUBVENCIONES"][pais])
                            #else:
                            #    if ((year-1) in ong[ong_nom]) and ("SUBVENCIONES" in ong[ong_nom][year-1]) and (pais in ong[ong_nom][year-1]["SUBVENCIONES"]):
                            #        worksheet.write(posExcel, 3, ong[ong_nom][year-1]["SUBVENCIONES"][pais])
                            else:
                                worksheet.write(posExcel, 3, 0)
                        else:
                            worksheet.write(posExcel, 3, 0)
                        
                        if year in [2013,2015]:
                            pYear = year -2
                        else:
                            pYear = year-1
                        if pYear in ong[ong_nom]:
                            if "PROYECTOS" in ong[ong_nom][pYear]: 
                                if pais in ong[ong_nom][pYear]["PROYECTOS"]:
                                    worksheet.write(posExcel, 4, ong[ong_nom][pYear]["PROYECTOS"][pais])
                                else:
                                    worksheet.write(posExcel, 4, 0)
                            else:
                                worksheet.write(posExcel, 4, 0)
                        else:
                            worksheet.write(posExcel, 4, 0)
                        
                        if "priorizacion" in ong[ong_nom]:
                            worksheet.write(posExcel, 5, ong[ong_nom]["priorizacion"][0])
                            worksheet.write(posExcel, 6, ong[ong_nom]["priorizacion"][1])
                            worksheet.write(posExcel, 7, ong[ong_nom]["priorizacion"][2])
                            worksheet.write(posExcel, 8, ong[ong_nom]["priorizacion"][3])
                        else:
                            worksheet.write(posExcel, 5, 0)
                            worksheet.write(posExcel, 6, 0)
                            worksheet.write(posExcel, 7, 0)
                            worksheet.write(posExcel, 8, 0)
                            
                        fondosMAE = 0
                        if "INGRESOS" in ong[ong_nom][year]:
                            if "TOTAL ESTATAL" in ong[ong_nom][year]["INGRESOS"]:
                                worksheet.write(posExcel, 9, ong[ong_nom][year]["INGRESOS"]["TOTAL ESTATAL"])
                                fondosMAE = ong[ong_nom][year]["INGRESOS"]["TOTAL ESTATAL"]
                            else:
                                worksheet.write(posExcel, 9,0)
                            if "TOTAL AUTONÓMICO" in ong[ong_nom][year]["INGRESOS"]:
                                worksheet.write(posExcel, 10, ong[ong_nom][year]["INGRESOS"]["TOTAL AUTONÓMICO"])
                            else:
                                worksheet.write(posExcel, 10,0)
                            if "TOTAL INTERNACIONAL" in ong[ong_nom][year]["INGRESOS"]:
                                worksheet.write(posExcel, 11, ong[ong_nom][year]["INGRESOS"]["TOTAL INTERNACIONAL"])
                            else:
                                worksheet.write(posExcel, 11,0)
                            if "Otros fondos públicos" in ong[ong_nom][year]["INGRESOS"]:
                                worksheet.write(posExcel, 12, ong[ong_nom][year]["INGRESOS"]["Otros fondos públicos"])
                            else:
                                worksheet.write(posExcel, 12,0)
                            worksheet.write(posExcel, 13, ong[ong_nom][year]["INGRESOS"]["TOTAL fondos públicos"])
                            totalFondos = float(ong[ong_nom][year]["INGRESOS"]["TOTAL fondos públicos"])
                            if "TOTAL cuotas periódicas" in ong[ong_nom][year]["INGRESOS"]:
                                worksheet.write(posExcel, 14, ong[ong_nom][year]["INGRESOS"]["TOTAL cuotas periódicas"])
                            else:
                                worksheet.write(posExcel, 14,0)
                            if "TOTAL donaciones" in ong[ong_nom][year]["INGRESOS"]:
                                worksheet.write(posExcel, 15, ong[ong_nom][year]["INGRESOS"]["TOTAL donaciones"])
                            else:
                                worksheet.write(posExcel, 15,0)
                            if "TOTAL EMPRESAS" in ong[ong_nom][year]["INGRESOS"]:
                                worksheet.write(posExcel, 16, ong[ong_nom][year]["INGRESOS"]["TOTAL EMPRESAS"])
                            else:
                                worksheet.write(posExcel, 16,0)
                            if "TOTAL venta productos" in ong[ong_nom][year]["INGRESOS"]:
                                worksheet.write(posExcel, 17, ong[ong_nom][year]["INGRESOS"]["TOTAL venta productos"])
                            else:
                                worksheet.write(posExcel, 17,0)
                            if "TOTAL otros servicios" in ong[ong_nom][year]["INGRESOS"]:
                                worksheet.write(posExcel, 18, ong[ong_nom][year]["INGRESOS"]["TOTAL otros servicios"])
                            else:
                                worksheet.write(posExcel, 18,0)
                            worksheet.write(posExcel, 19, ong[ong_nom][year]["INGRESOS"]["TOTAL fondos privados"] )
                            totalFondos+=float(ong[ong_nom][year]["INGRESOS"]["TOTAL fondos privados"])
                        
                        
                        if "TRABAJADORES" in ong[ong_nom][year]:
                            totalT = 0
                            if "España" in ong[ong_nom][year]["TRABAJADORES"] and ong[ong_nom][year]["TRABAJADORES"]["España"] not in ["No info","n.d.n.d.",'  ']:
                                worksheet.write(posExcel, 20, ong[ong_nom][year]["TRABAJADORES"]["España"])
                                totalT+=ong[ong_nom][year]["TRABAJADORES"]["España"]
                            else:
                                worksheet.write(posExcel, 20,0)
                            
            
                            if "Extranjero" in ong[ong_nom][year]["TRABAJADORES"] and ong[ong_nom][year]["TRABAJADORES"]["Extranjero"] not in ["No info","n.d.","n.d",' ']:
                                worksheet.write(posExcel, 21, ong[ong_nom][year]["TRABAJADORES"]["Extranjero"])
                                totalT+=ong[ong_nom][year]["TRABAJADORES"]["Extranjero"]
                            else:
                                worksheet.write(posExcel, 21,0)
                            if "Local" in ong[ong_nom][year]["TRABAJADORES"] and ong[ong_nom][year]["TRABAJADORES"]["Local"] not in ["No info","n.d.","n.d",' ']:
                                worksheet.write(posExcel, 22, ong[ong_nom][year]["TRABAJADORES"]["Local"])
                                totalT+=ong[ong_nom][year]["TRABAJADORES"]["Local"]
                            else:
                                worksheet.write(posExcel, 22,0)
                            worksheet.write(posExcel, 23, totalT)
                        
                        worksheet.write(posExcel, 25, 0)
                        worksheet.write(posExcel, 26, 0)
                        worksheet.write(posExcel, 27, 0)
                        worksheet.write(posExcel, 28, 0)
                        if "info" in ong[ong_nom]:
                            if ong[ong_nom]["info"][0]=="Fundacion":
                                worksheet.write(posExcel, 25, 1)
                            elif ong[ong_nom]["info"][0]=="Asociación":
                                worksheet.write(posExcel, 26, 1)
                            elif ong[ong_nom]["info"][0]=="Federación":
                                worksheet.write(posExcel, 27, 1)
                            else:
                                worksheet.write(posExcel, 28, 1)
                            
                        if "VOLUNTARIOS" in ong[ong_nom][year]:
                            if "España" in ong[ong_nom][year]["VOLUNTARIOS"] and ong[ong_nom][year]["VOLUNTARIOS"]["España"] not in ["No info","n.d.n.d.",'  ']:
                                worksheet.write(posExcel, 29, ong[ong_nom][year]["VOLUNTARIOS"]["España"])
                            else:
                                worksheet.write(posExcel, 29,0)
                            
            
                            if "Extranjero" in ong[ong_nom][year]["VOLUNTARIOS"] and ong[ong_nom][year]["VOLUNTARIOS"]["Extranjero"] not in ["No info","n.d.","n.d",' ']:
                                worksheet.write(posExcel, 30, ong[ong_nom][year]["VOLUNTARIOS"]["Extranjero"])
                            else:
                                worksheet.write(posExcel, 30,0)
                        
                        if year in dinerosEspanya and pais in dinerosEspanya[year]:
                            
                            worksheet.write(posExcel, 31,dinerosEspanya[year][pais])
                        else:
                            worksheet.write(posExcel, 31,0)
                        
                        print("TOTAL FONDOS",totalFondos)
                        worksheet.write(posExcel, 32,totalFondos)
                        if totalFondos == 0:
                            worksheet.write(posExcel, 33,0)
                        else:
                            worksheet.write(posExcel, 33,float(ong[ong_nom][year]["INGRESOS"]["TOTAL fondos privados"])/float(totalFondos))
                        if totalFondos == 0:
                            worksheet.write(posExcel, 34,0)
                        else:
                            worksheet.write(posExcel, 34,float(fondosMAE)/float(totalFondos))
                        if "anyo" in ong[ong_nom]:
                            worksheet.write(posExcel, 35,ong[ong_nom]["anyo"])
                            try:
                                worksheet.write(posExcel, 36,ong[ong_nom]["internacional"])
                            except:
                                worksheet.write(posExcel, 36,0)
                        else:
                            worksheet.write(posExcel, 35,0)
                            worksheet.write(posExcel, 36,0)
                        if pais in oldColonies:
                            worksheet.write(posExcel, 37,1)
                        else:
                            worksheet.write(posExcel, 37,0)
                        if year in delegacionesONG[ong_nom]:
                            print("DINTRE1")
                            if pais in delegacionesONG[ong_nom][year]:
                                print("DINTRE2")
                                worksheet.write(posExcel, 38,1)
                            else:
                                worksheet.write(posExcel, 38,0)
                        else: 
                            worksheet.write(posExcel, 38,0)
                        worksheet.write(posExcel, 39,1)
                        worksheet.write(posExcel, 40,ong[ong_nom][year]["PROYECTOS"][pais])
                        
                        posExcel+=1
                    else:
                        pass
    workbook.close()       
                
pickle.dump(ong, open( "./ong.p", "wb" ) )
pickle.dump(delegacionesONG, open( "./delegaciones.p", "wb" ) )
pickle.dump(paises_ONU, open( "./paises_ONU.p", "wb" ) )
pickle.dump(dinerosEspanya,open( "./dinerosEspanya.p", "wb" ) )

"""
diputados = pd.read_excel('./data_congreso_diputados.xlsx', sheet_name='taulamare')
diputados["proporcion fondos privados"] = ""
diputados["total fondos"] = ""
diputados["año constitución"] = ""
for i in range(len(diputados)):  
    info = mision.iloc[i]
    info.index
    if info["Congede"]==1:
        nomONG = revisionNombres(unicodedata.normalize('NFD', info["Organization NAME"].lower().strip()).encode('ascii', 'ignore').decode("utf-8") )    
        nomONG = revisionNombres(nomONG)
        year = info["Year"]
        if nomONG in ong and year in ong[nomONG]:
             
             diputados.iloc[i, diputados.columns.get_loc('año constitución')] = ong[nomONG]["anyo"]
       
            
             if "INGRESOS" in ong[nomONG][year]:
            
                fondosPrivados = float(ong[ong_nom][year]["INGRESOS"]["TOTAL fondos privados"])/(float(ong[ong_nom][year]["INGRESOS"]["TOTAL fondos públicos"])+float(ong[ong_nom][year]["INGRESOS"]["TOTAL fondos privados"]))
                #info["proporcion fondos privados"] = fondosPrivados                     
                diputados.iloc[i, diputados.columns.get_loc('proporcion fondos privados')] = fondosPrivados
                diputados.iloc[i, diputados.columns.get_loc('total fondos')] = float(ong[ong_nom][year]["INGRESOS"]["TOTAL fondos públicos"])+float(ong[ong_nom][year]["INGRESOS"]["TOTAL fondos privados"])
       
            #diputados.iloc[i, diputados.columns.get_loc('proporcion fondos privados')] = fondosPrivados
            

diputados.to_excel('./data_congreso_diputados_modified.xlsx', sheet_name='taulamare')
"""


