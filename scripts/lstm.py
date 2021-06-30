"""
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
"""

import numpy as np
import random
import sys, os, json
from math import sqrt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation,Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import optimizers
import shap
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn import preprocessing
import _pickle as pickle
import tensorflow.keras
import copy
import seaborn as sns
import matplotlib.pyplot as plt


tensorflow.test.is_gpu_available()
np.random.seed(7)
from tensorflow import set_random_seed

data = pd.read_excel("../output/allExcels_negatiu.xlsx",index_col = 0, header=0)
data.columns
corrPearson = data.corr(method='pearson')
corrSpearman = data.corr(method='spearman')
corrKendall = data.corr(method='kendall')

#sns.heatmap(corrSpearman)
corrSpearman_plot = copy.deepcopy(corrSpearman)

nameVariables = list(corrSpearman.columns)
nameVariables[0] = "UN LDCs"
nameVariables[1] = "GDP per capita"
nameVariables[2] = "Public Grant"
nameVariables[3] = "Budget Previous Year"
nameVariables[4] = "Donor Aid Budget"
nameVariables[5] = "Latin America Mission"
nameVariables[6] = "Africa Mission"
nameVariables[9] = "Project Developed"
corrSpearman_plot.drop(["Visitado"],axis=1,inplace=True)
corrSpearman_plot.drop(["Visitado"],axis=0,inplace=True)


mask = np.triu(np.ones_like(corrSpearman_plot, dtype=np.bool))
plt.figure(figsize=(16, 6))
# Store heatmap object in a variable to easily access it when you want to include more features (such as title).
# Set the range of values to be displayed on the colormap from -1 to 1, and set the annotation to True to display the correlation values on the heatmap.
heatmap = sns.heatmap(corrSpearman_plot,mask=mask, annot=True,xticklabels=nameVariables[:9], yticklabels=nameVariables[:9],vmin=-1, vmax=1,cmap="RdYlBu_r")
plt.xticks(rotation=45)
# Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
heatmap.set_title("Spearman's 'Correlation Heatmap", fontdict={'fontsize':12}, pad=12);

corrSpearman_plot_2 = copy.deepcopy(corrSpearman)
nameVariables[0] = "UN LDCs"
nameVariables[1] = "GDP per capita"
nameVariables[2] = "Public Grant"
nameVariables[3] = "Budget Previous Year"
nameVariables[4] = "Donor Aid Budget"
nameVariables[5] = "Latin America Mission"
nameVariables[6] = "Africa Mission"
nameVariables[9] = "Project Developed"
corrSpearman_plot_2.columns=nameVariables
corrSpearman_plot_2.index=nameVariables
corrSpearman_plot_2.drop(["Project Developed"],axis=0,inplace=True)

plt.figure(figsize=(4, 8))
heatmap = sns.heatmap(corrSpearman_plot_2[['Project Developed']].sort_values(by='Project Developed', ascending=False), vmin=-1, vmax=1, annot=True,cmap="RdYlBu_r")
heatmap.set_title('Features Correlating with Project Developed', fontdict={'fontsize':12});

#data = data.drop('%_MAE_Funds',1)
#data = data.drop('Total_Funds',1)
#data = data.drop('%_Private_Funds', 1)

data['GDP'] = np.log(data['GDP'])
data['Public_Grant'] = np.log(data['Public_Grant'])
data['Budget_Previous_Year'] = np.log(data['Budget_Previous_Year']) #skew data
data['Donor_Aid_Budget'] = np.log(data['Donor_Aid_Budget'])
data[data < 0] = 0

scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(data)


training_LSTM = {}
y_LSTM = {}

path = '../output/'
for root, dirs, files in os.walk(path):
    for filename in files:
        if "_negativos" in filename:
            
            name_ONG = filename[:filename.index("_")]
            training_LSTM[name_ONG] = {}
            y_LSTM[name_ONG] = {}
            proyectos = pd.read_excel(path+filename, sheet_name='Sheet1',index_col = "Pais-Año")

            proyectos['Budget_Previous_Year'] = np.log(proyectos['Budget_Previous_Year'])
            proyectos['Donor_Aid_Budget'] = np.log(proyectos['Donor_Aid_Budget'])
            proyectos['GDP'] = np.log(proyectos['GDP'])
            proyectos['Public_Grant'] = np.log(proyectos['Public_Grant'])
            proyectos[proyectos < 0] = 0
            
            for index, row in proyectos.iterrows():
                age = index[:4]
                country = index[5:]
                if country not in training_LSTM[name_ONG]:
                    training_LSTM[name_ONG][country] = {}
                    y_LSTM[name_ONG][country] = {}
                
                row = scaler.transform([row])
                y_LSTM[name_ONG][country][age]= row[0][-1]
            
                training_LSTM[name_ONG][country][age]= row[0][:-1]

training_LSTM_8 = []
y_LSTM_8 = []           
qND = 0
for ong in training_LSTM:
    for country in training_LSTM[ong]:
        ages = ["2009","2010","2011","2012","2013","2014","2015","2016"]
               
        newdata = []
        
        for posAge in range(len(ages)):
            if ages[posAge] in training_LSTM[ong][country]:
                data = training_LSTM[ong][country][ages[posAge]]
                newdata.append(data)
            else:
                print(ong,country)
                qND+=1
                newdata.append([0,0,0,0,0,0,0,0,0])
            if ages[posAge]=="2016":
                yR = 0
                y = 0
                if ages[posAge] in training_LSTM[ong][country]:
                    y = y_LSTM[ong][country][ages[posAge]]
                training_LSTM_8.append(newdata)
                y_LSTM_8.append(y)
        

training_LSTM_8_pad = sequence.pad_sequences(training_LSTM_8,dtype='float64')              


import time
inici = time.time()

model_bin = Sequential()

model_bin.add(LSTM(100, implementation=2, input_shape=(training_LSTM_8_pad.shape[1], training_LSTM_8_pad.shape[2]),
                           return_sequences=True,recurrent_dropout=0.2))
model_bin.add(BatchNormalization())
model_bin.add(LSTM(100, implementation=2,recurrent_dropout=0.2))
#model_bin.add(Dropout(0.2))
model_bin.add(BatchNormalization())
model_bin.add(Dense(1, activation="sigmoid"))
nadam_opt = optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999)

model_bin.compile(loss='binary_crossentropy', optimizer=nadam_opt)
early_stopping = EarlyStopping(patience=0,mode="min",monitor='val_loss')
model_bin.fit(training_LSTM_8_pad, y_LSTM_8, validation_split=0.1,callbacks=[early_stopping],epochs=1000)
final = time.time()


model_bin = Sequential()

model_bin.add(LSTM(100, implementation=2, input_shape=(training_LSTM_8_pad.shape[1], training_LSTM_8_pad.shape[2]),
                           recurrent_dropout=0.2,return_sequences=True))
model_bin.add(BatchNormalization())
model_bin.add(LSTM(100, implementation=2,recurrent_dropout=0.2))
model_bin.add(BatchNormalization())
model_bin.add(Dense(1, activation="sigmoid"))
nadam_opt = optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999)

model_bin.compile(loss='binary_crossentropy', optimizer=nadam_opt)
model_bin.fit(training_LSTM_8_pad, y_LSTM_8, epochs=4)
#early_stopping = EarlyStopping(patience=0,mode="min",monitor='val_loss')
final = time.time()


model_bin.save("./binaryKeras.model")
model_bin = tensorflow.keras.models.load_model("./binaryKeras.model")

explainer_bin = shap.DeepExplainer(model_bin,training_LSTM_8_pad)

training_LSTM_8_pad_B = explainer_bin.shap_values(training_LSTM_8_pad)

pickle.dump(training_LSTM_8_pad_B,open("./fitxerShapleyLSTM_8_B","wb")) 

training_LSTM_8_pad_B = pickle.load(open("./fitxerShapleyLSTM_8_B","rb"))


nameVariables = list(proyectos.columns)
nameVariables[0] = "UN LDCs"
nameVariables[1] = "GDP per capita"
nameVariables[2] = "Public Grant"
nameVariables[3] = "Budget Previous Year"
nameVariables[4] = "Donor Aid Budget"
nameVariables[5] = "Latin America Mission"
nameVariables[6] = "Africa Mission"

shap_Specific= []
shap_SpecificValues = []
shap_Specific_P= []
shap_SpecificValues_P = []
len(training_LSTM_8_pad_B[0][0])
variables = []
for i in range(len(training_LSTM_8_pad_B[0])):
    valSV = []
    valFeature = []
    for j in range(len(training_LSTM_8_pad_B[0][0])): #0 --> 7 (8 anys)
        for k in range(len(training_LSTM_8_pad_B[0][0][0])): #0-->8 (9 variables) 
            valSV.append(training_LSTM_8_pad_B[0][i][j][k])
            valFeature.append(training_LSTM_8_pad[i][j][k])
            if i ==0:
                variable =nameVariables[k]
                variable = variable+"_"+["2009","2010","2011","2012","2013","2014","2015","2016"][j]
                variables.append(variable)
    shap_Specific.append(copy.deepcopy(valSV))
    shap_SpecificValues.append(copy.deepcopy(valFeature))
    if y_LSTM_8[i]==1:
        shap_Specific_P.append(copy.deepcopy(valSV))
        shap_SpecificValues_P.append(copy.deepcopy(valFeature))


shap.summary_plot(np.array(shap_Specific),features=np.array(shap_SpecificValues),feature_names=variables,plot_type="bar",max_display=10)
shap.summary_plot(np.array(shap_Specific),features=np.array(shap_SpecificValues),feature_names=variables,max_display=10)

shap.summary_plot(np.array(shap_Specific_P),features=np.array(shap_SpecificValues_P),feature_names=variables,plot_type="bar",max_display=10)
shap.summary_plot(np.array(shap_Specific_P),features=np.array(shap_SpecificValues_P),feature_names=variables,max_display=10)

shap_Specific_H1_H2 = []
shap_SpecificValues_H1_H2 = []
pos_H1_H2 = [63,64,65,67,68,69]
for i in range(len(shap_Specific)):
    entry = []
    entryValues = []
    for j in pos_H1_H2:
        entry.append(shap_Specific[i][j])
        entryValues.append(shap_SpecificValues[i][j])
    shap_Specific_H1_H2.append(copy.deepcopy(entry))
    shap_SpecificValues_H1_H2.append(copy.deepcopy(entryValues))



shap_Specific_H1_H2_P = []
shap_SpecificValues_H1_H2_P = []
for i in range(len(shap_Specific_P)):
    entry = []
    entryValues = []
    for j in pos_H1_H2:
        entry.append(shap_Specific_P[i][j])
        entryValues.append(shap_SpecificValues_P[i][j])
    shap_Specific_H1_H2_P.append(copy.deepcopy(entry))
    shap_SpecificValues_H1_H2_P.append(copy.deepcopy(entryValues))


shap.summary_plot(np.array(shap_Specific_H1_H2),features=np.array(shap_SpecificValues_H1_H2),feature_names=["UN LDCs_2016","GDP per capita_2016","Public Grant_2016","Donor Aid Budget_2016","Latin America Mission_2016","Africa Mission_2016"],plot_type="bar",max_display=10)
shap.summary_plot(np.array(shap_Specific_H1_H2),features=np.array(shap_SpecificValues_H1_H2),feature_names=["UN LDCs_2016","GDP per capita_2016","Public Grant_2016","Donor Aid Budget_2016","Latin America Mission_2016","Africa Mission_2016"],max_display=10)

shap.summary_plot(np.array(shap_Specific_H1_H2_P),features=np.array(shap_SpecificValues_H1_H2_P),feature_names=["UN LDCs_2016","GDP per capita_2016","Public Grant_2016","Donor Aid Budget_2016","Latin America Mission_2016","Africa Mission_2016"],plot_type="bar",max_display=10)
shap.summary_plot(np.array(shap_Specific_H1_H2_P),features=np.array(shap_SpecificValues_H1_H2_P),feature_names=["UN LDCs_2016","GDP per capita_2016","Public Grant_2016","Donor Aid Budget_2016","Latin America Mission_2016","Africa Mission_2016"],max_display=10)
       





















absSHAP_B = []
absSHAP_P = []
SHAP_P = []
absSHAP_B_num = []
absSHAP_P_num = []
SHAP_P_num = []
dicSV_P = []


for i in range(len(training_LSTM_8_pad_B[0][0])):
    data = []
    data0 = []
    for j in range(len(training_LSTM_8_pad_B[0][0][0])):
        data.append([])
        data0.append(0)
    absSHAP_B.append(copy.deepcopy(data))
    absSHAP_B_num.append(copy.deepcopy(data0))

    absSHAP_P.append(copy.deepcopy(data))
    absSHAP_P_num.append(copy.deepcopy(data0))
    
    SHAP_P.append(copy.deepcopy(data))
    SHAP_P_num.append(copy.deepcopy(data0))
    
    dicSV_P.append(copy.deepcopy(data0))
        
        

for entry in training_LSTM_8_pad_B[0]:
    for i in range(len(entry)):
        for j in range(len(entry[0])):
            absSHAP_B[i][j].append(abs(entry[i][j]))
    
         
posEntry = 0
for entry in training_LSTM_8_pad_B[0]:
    if y_LSTM_8[posEntry]==1:
        pos = np.argwhere(entry == np.amax(entry))
        dicSV_P[pos[0][0]][pos[0][1]]+=1
        for i in range(len(entry)):
            for j in range(len(entry[0])):
                absSHAP_P[i][j].append(abs(entry[i][j]))
                SHAP_P[i][j].append(entry[i][j])
    posEntry+=1

for i in range(len(entry)):
    for j in range(len(entry[0])):
        absSHAP_B_num[i][j] = sum(absSHAP_B[i][j])/len(absSHAP_B[i][j])  

for i in range(len(absSHAP_B)):
    for j in range(len(absSHAP_B[0])):
        absSHAP_P_num[i][j] = sum(absSHAP_P[i][j])/len(absSHAP_P[i][j]) 
        SHAP_P_num[i][j] = sum(SHAP_P[i][j])/len(SHAP_P[i][j]) 


  
for i in range(len(list(proyectos.columns))):
    print(i,proyectos.columns[i])


model_bin = tensorflow.keras.models.load_model("./binaryKeras.model")
for ong in training_LSTM:
    print(ong)
bigNames = ["acción contra el hambre","cáritas","cruz roja","medicos del mundo","oxfam intermon","fontilles","educo","adsis","entreculturas","manos unidas"]

training_LSTM_8_Big = []
y_LSTM_8_Big = []           

for ong in training_LSTM:
    if ong in bigNames:
        print(ong)
        for country in training_LSTM[ong]:
            ages = ["2009","2010","2011","2012","2013","2014","2015","2016"]
                   
            newdata = []
            for posAge in range(len(ages)):
                if ages[posAge] in training_LSTM[ong][country]:
                    data = training_LSTM[ong][country][ages[posAge]]
                    newdata.append(data)
                else:
                    newdata.append([0,0,0,0,0,0,0,0,0,0,0])
                if ages[posAge]=="2016":
                    yR = 0
                    y = 0
                    if ages[posAge] in training_LSTM[ong][country]:
                        y = y_LSTM[ong][country][ages[posAge]]
                    training_LSTM_8_Big.append(newdata)
                    y_LSTM_8_Big.append(y)

training_LSTM_8_pad_Big = sequence.pad_sequences(training_LSTM_8_Big,dtype='float64')              
explainer_bin = shap.DeepExplainer(model_bin,training_LSTM_8_pad)
training_LSTM_8_pad_B_Big = explainer_bin.shap_values(training_LSTM_8_pad_Big)


shap_Specific_Big= []
shap_SpecificValues_Big = []
shap_Specific_P_Big= []
shap_SpecificValues_P_Big = []
variables = []
for i in range(len(training_LSTM_8_pad_B_Big[0])):
    valSV = []
    valFeature = []
    for j in range(len(training_LSTM_8_pad_B_Big[0][0])):
        for k in range(len(training_LSTM_8_pad_B_Big[0][0][0])): 
            valSV.append(training_LSTM_8_pad_B_Big[0][i][j][k])
            valFeature.append(training_LSTM_8_pad_Big[i][j][k])
            if i ==0:
                variable = proyectos.columns[k]
                variable = variable+"_"+["2009","2010","2011","2012","2013","2014","2015","2016"][j]
                variables.append(variable)
    shap_Specific_Big.append(copy.deepcopy(valSV))
    shap_SpecificValues_Big.append(copy.deepcopy(valFeature))
    if y_LSTM_8_Big[i]==1:
        shap_Specific_P_Big.append(copy.deepcopy(valSV))
        shap_SpecificValues_P_Big.append(copy.deepcopy(valFeature))


shap.summary_plot(np.array(shap_Specific_Big),features=np.array(shap_SpecificValues_Big),feature_names=variables,plot_type="bar",max_display=10)
shap.summary_plot(np.array(shap_Specific_Big),features=np.array(shap_SpecificValues_Big),feature_names=variables,max_display=10)


smallNames = ["amref","acción verapaz","edificando comunidad de nazaret","farmaceuticos sin fronteras","fisc-compañia de maria","pueblos hermanos"]

training_LSTM_8_Small = []
y_LSTM_8_Small = []           

for ong in training_LSTM:
    if ong in smallNames:
        print(ong)
        for country in training_LSTM[ong]:
            ages = ["2009","2010","2011","2012","2013","2014","2015","2016"]
                   
            newdata = []
            for posAge in range(len(ages)):
                if ages[posAge] in training_LSTM[ong][country]:
                    data = training_LSTM[ong][country][ages[posAge]]
                    newdata.append(data)
                else:
                    newdata.append([0,0,0,0,0,0,0,0,0,0,0,0,0])
                if ages[posAge]=="2016":
                    yR = 0
                    y = 0
                    if ages[posAge] in training_LSTM[ong][country]:
                        y = y_LSTM[ong][country][ages[posAge]]
                    training_LSTM_8_Small.append(newdata)
                    y_LSTM_8_Small.append(y)

training_LSTM_8_pad_Small = sequence.pad_sequences(training_LSTM_8_Small,dtype='float64')              
explainer_bin = shap.DeepExplainer(model_bin,training_LSTM_8_pad)
training_LSTM_8_pad_B_Small = explainer_bin.shap_values(training_LSTM_8_pad_Small)


shap_Specific_Small= []
shap_SpecificValues_Small = []
shap_Specific_P_Small= []
shap_SpecificValues_P_Small = []
variables = []
for i in range(len(training_LSTM_8_pad_B_Small[0])):
    valSV = []
    valFeature = []
    for j in range(len(training_LSTM_8_pad_B_Small[0][0])):
        for k in range(len(training_LSTM_8_pad_B_Small[0][0][0])): 
            valSV.append(training_LSTM_8_pad_B_Small[0][i][j][k])
            valFeature.append(training_LSTM_8_pad_Small[i][j][k])
            if i ==0:
                variable = proyectos.columns[k]
                variable = variable+"_"+["2009","2010","2011","2012","2013","2014","2015","2016"][j]
                variables.append(variable)
    shap_Specific_Small.append(copy.deepcopy(valSV))
    shap_SpecificValues_Small.append(copy.deepcopy(valFeature))
    if y_LSTM_8_Small[i]==1:
        shap_Specific_P_Small.append(copy.deepcopy(valSV))
        shap_SpecificValues_P_Small.append(copy.deepcopy(valFeature))


shap.summary_plot(np.array(shap_Specific_Small),features=np.array(shap_SpecificValues_Small),feature_names=variables,plot_type="bar",max_display=10)
shap.summary_plot(np.array(shap_Specific_Small),features=np.array(shap_SpecificValues_Small),feature_names=variables,max_display=10)



training_LSTM_8_Medium = []
y_LSTM_8_Medium = []           

for ong in training_LSTM:
    if ong not in smallNames and ong not in bigNames:
        print(ong)
        for country in training_LSTM[ong]:
            ages = ["2009","2010","2011","2012","2013","2014","2015","2016"]
                   
            newdata = []
            for posAge in range(len(ages)):
                if ages[posAge] in training_LSTM[ong][country]:
                    data = training_LSTM[ong][country][ages[posAge]]
                    newdata.append(data)
                else:
                    newdata.append([0,0,0,0,0,0,0,0,0,0,0,0,0])
                if ages[posAge]=="2016":
                    yR = 0
                    y = 0
                    if ages[posAge] in training_LSTM[ong][country]:
                        y = y_LSTM[ong][country][ages[posAge]]
                    training_LSTM_8_Medium.append(newdata)
                    y_LSTM_8_Medium.append(y)

training_LSTM_8_pad_Medium = sequence.pad_sequences(training_LSTM_8_Medium,dtype='float64')              
explainer_bin = shap.DeepExplainer(model_bin,training_LSTM_8_pad)
training_LSTM_8_pad_B_Medium = explainer_bin.shap_values(training_LSTM_8_pad_Medium)


shap_Specific_Medium= []
shap_SpecificValues_Medium = []
shap_Specific_P_Medium= []
shap_SpecificValues_P_Medium = []
variables = []
for i in range(len(training_LSTM_8_pad_B_Medium[0])):
    valSV = []
    valFeature = []
    for j in range(len(training_LSTM_8_pad_B_Medium[0][0])):
        for k in range(len(training_LSTM_8_pad_B_Medium[0][0][0])): 
            valSV.append(training_LSTM_8_pad_B_Medium[0][i][j][k])
            valFeature.append(training_LSTM_8_pad_Medium[i][j][k])
            if i ==0:
                variable = proyectos.columns[k]
                variable = variable+"_"+["2009","2010","2011","2012","2013","2014","2015","2016"][j]
                variables.append(variable)
    shap_Specific_Medium.append(copy.deepcopy(valSV))
    shap_SpecificValues_Medium.append(copy.deepcopy(valFeature))
    if y_LSTM_8_Medium[i]==1:
        shap_Specific_P_Medium.append(copy.deepcopy(valSV))
        shap_SpecificValues_P_Medium.append(copy.deepcopy(valFeature))


shap.summary_plot(np.array(shap_Specific_Medium),features=np.array(shap_SpecificValues_Medium),feature_names=variables,plot_type="bar",max_display=10)
shap.summary_plot(np.array(shap_Specific_Medium),features=np.array(shap_SpecificValues_Medium),feature_names=variables,max_display=10)








































    
shap_Specific.append([training_LSTM_8_pad_B[0][i][7][6],training_LSTM_8_pad_B[0][i][7][12],training_LSTM_8_pad_B[0][i][5][6],training_LSTM_8_pad_B[0][i][3][6],training_LSTM_8_pad_B[0][i][6][6],training_LSTM_8_pad_B[0][i][7][12]])
shap_SpecificValues.append([training_LSTM_8_pad[i][7][6],training_LSTM_8_pad[i][7][12],training_LSTM_8_pad[i][5][6],training_LSTM_8_pad[i][3][6],training_LSTM_8_pad[i][6][6],training_LSTM_8_pad[i][7][12]])
if y_LSTM_8[i]==1:
    shap_Specific.append([training_LSTM_8_pad_B[0][i][7][6],training_LSTM_8_pad_B[0][i][7][12],training_LSTM_8_pad_B[0][i][5][6],training_LSTM_8_pad_B[0][i][3][6],training_LSTM_8_pad_B[0][i][6][6],training_LSTM_8_pad_B[0][i][7][12]])
    shap_SpecificValues.append([training_LSTM_8_pad[i][7][6],training_LSTM_8_pad[i][7][12],training_LSTM_8_pad[i][5][6],training_LSTM_8_pad[i][3][6],training_LSTM_8_pad[i][6][6],training_LSTM_8_pad[i][7][12]])
    






absSHAP_B_num[7][7]
absSHAP_B_num[0][7]
absSHAP_B_num[7][0]
absSHAP_B_num[7][12]


d = training_LSTM_8_pad_B[0][0]



 

test_0 = training_LSTM_8_pad_B[0][0]
test_4 = training_LSTM_8_pad_B[0][4]

len(test_0)
num = 0
for el in test_4:
    for el2 in el:
        num+=el2

predictions = model_bin.predict(training_LSTM_8_pad)
predictions[4]
num+np.mean(y_LSTM_8)




training_LSTM_8_pad_B = pickle.load(open("./fitxerShapleyLSTM_8_B","rb"))
training_LSTM_8_pad_R = pickle.load(open("./fitxerShapleyLSTM_8_R","rb"))













######################################regressio 8





model = Sequential()

model.add(LSTM(100, implementation=2, input_shape=(training_LSTM_8_pad.shape[1], training_LSTM_8_pad.shape[2]),
                           recurrent_dropout=0.2,return_sequences=True))
model.add(BatchNormalization())

model.add(LSTM(100, implementation=2, input_shape=(training_LSTM_8_pad.shape[1], training_LSTM_8_pad.shape[2]),
                   recurrent_dropout=0.2))
model.add(BatchNormalization())

model.add(Dense(1,activation="sigmoid"))
nadam_opt = optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

model.compile(loss='mean_squared_error', optimizer=nadam_opt,metrics = ["mae"])
es = EarlyStopping(monitor='val_loss',patience=100)
model.fit(training_LSTM_8_pad, np.asarray(yR_LSTM_8), epochs=50)
final = time.time()

#model.save("./regKeras.model")


#training_LSTM_8_pad_copy = training_LSTM_8_pad[:]
#random.shuffle(training_LSTM_8_pad_copy)
#background_8 = training_LSTM_8_pad_copy[:5000]
pickle.dump(training_LSTM_8_pad,open("./background_8","wb"))  

explainer = shap.DeepExplainer(model,training_LSTM_8_pad)

training_LSTM_8_pad_R = explainer.shap_values(training_LSTM_8_pad)

pickle.dump(training_LSTM_8_pad_R,open("./fitxerShapleyLSTM_8_R","wb"))  




data = [[2,4],[3,5]]
data2 = data[:]

data2[0]=1


absSHAP_R = []
absPosSHAP_R = []
absNegSHAP_R = []
sSHAP_R = []
sPosSHAP_R = []
sNegSHAP_R = []
for i in range(len(training_LSTM_8_pad_R[0][0])):
    data = []
    for j in range(len(training_LSTM_8_pad_R[0][0][0])):
        data.append([])
    absSHAP_R.append(copy.deepcopy(data))
    absPosSHAP_R.append(copy.deepcopy(data))
    absNegSHAP_R.append(copy.deepcopy(data))
    sSHAP_R.append(copy.deepcopy(data))
    sPosSHAP_R.append(copy.deepcopy(data))
    sNegSHAP_R.append(copy.deepcopy(data))

i = 0
j = 0
k = 0

for i in range(len(training_LSTM_8_pad_R[0])):
    for j in range(len(training_LSTM_8_pad_R[0][0])):
        for k in range(len(training_LSTM_8_pad_R[0][0][0])):
            #print(training_LSTM_8_pad_R[0][i][j][k],abs(training_LSTM_8_pad_R[0][i][j][k]))
            absSHAP_R[j][k].append(abs(training_LSTM_8_pad_R[0][i][j][k]))
            sSHAP_R[j][k].append(training_LSTM_8_pad_R[0][i][j][k])

absSHAP_R[0][0][1]
sSHAP_R[0][0][1]

for i in range(len(yR_LSTM_8)):
    if yR_LSTM_8[i]>0:
        for j in range(len(training_LSTM_8_pad_R[0][0])):
            for k in range(len(training_LSTM_8_pad_R[0][0][0])):
                absPosSHAP_R[j][k].append(abs(training_LSTM_8_pad_R[0][i][j][k]))
                sPosSHAP_R[j][k].append(training_LSTM_8_pad_R[0][i][j][k])
    else:
        for j in range(len(training_LSTM_8_pad_R[0][0])):
            for k in range(len(training_LSTM_8_pad_R[0][0][0])):
                absNegSHAP_R[j][k].append(abs(training_LSTM_8_pad_R[0][i][j][k]))
                sNegSHAP_R[j][k].append(training_LSTM_8_pad_R[0][i][j][k])
   

for i in range(len(absSHAP_R)):
    for j in range(len(absSHAP_R[i])):
        absSHAP_R[i][j]=sum(absSHAP_R[i][j])/len(absSHAP_R[i][j])
        absPosSHAP_R[i][j]=sum(absPosSHAP_R[i][j])/len(absPosSHAP_R[i][j])
        absNegSHAP_R[i][j]=sum(absNegSHAP_R[i][j])/len(absNegSHAP_R[i][j])
        sSHAP_R[i][j]=sum(sSHAP_R[i][j])/len(sSHAP_R[i][j])
        sPosSHAP_R[i][j]=sum(sPosSHAP_R[i][j])/len(sPosSHAP_R[i][j])
        sNegSHAP_R[i][j]=sum(sNegSHAP_R[i][j])/len(sNegSHAP_R[i][j])


absSHAP_R_pad = sequence.pad_sequences(absSHAP_R,dtype='float32')              
absPosSHAP_R_pad = sequence.pad_sequences(absPosSHAP_R,dtype='float32')              
absNegSHAP_R_pad = sequence.pad_sequences(absNegSHAP_R,dtype='float32')  
sSHAP_R_pad = sequence.pad_sequences(sSHAP_R,dtype='float32')              
sPosSHAP_R_pad = sequence.pad_sequences(sPosSHAP_R,dtype='float32')              
sNegSHAP_R_pad = sequence.pad_sequences(sNegSHAP_R,dtype='float32')              




















################################################3
absSHAP_B = []
absPosSHAP_B = []
absNegSHAP_B = []
sSHAP_B = []
sPosSHAP_B = []
sNegSHAP_B = []
for i in range(len(training_LSTM_8_pad_B[0][0])):
    data = []
    for j in range(len(training_LSTM_8_pad_B[0][0][0])):
        data.append([])
    absSHAP_B.append(copy.deepcopy(data))
    absPosSHAP_B.append(copy.deepcopy(data))
    absNegSHAP_B.append(copy.deepcopy(data))
    sSHAP_B.append(copy.deepcopy(data))
    sPosSHAP_B.append(copy.deepcopy(data))
    sNegSHAP_B.append(copy.deepcopy(data))

i = 0
j = 0
k = 0

for i in range(len(training_LSTM_8_pad_B[0])):
    for j in range(len(training_LSTM_8_pad_B[0][0])):
        for k in range(len(training_LSTM_8_pad_B[0][0][0])):
            #print(training_LSTM_8_pad_B[0][i][j][k],abs(training_LSTM_8_pad_B[0][i][j][k]))
            absSHAP_B[j][k].append(abs(training_LSTM_8_pad_B[0][i][j][k]))
            sSHAP_B[j][k].append(training_LSTM_8_pad_B[0][i][j][k])

absSHAP_B[0][0][1]
sSHAP_B[0][0][1]

for i in range(len(yR_LSTM_8)):
    if yR_LSTM_8[i]>0:
        for j in range(len(training_LSTM_8_pad_B[0][0])):
            for k in range(len(training_LSTM_8_pad_B[0][0][0])):
                absPosSHAP_B[j][k].append(abs(training_LSTM_8_pad_B[0][i][j][k]))
                sPosSHAP_B[j][k].append(training_LSTM_8_pad_B[0][i][j][k])
    else:
        for j in range(len(training_LSTM_8_pad_B[0][0])):
            for k in range(len(training_LSTM_8_pad_B[0][0][0])):
                absNegSHAP_B[j][k].append(abs(training_LSTM_8_pad_B[0][i][j][k]))
                sNegSHAP_B[j][k].append(training_LSTM_8_pad_B[0][i][j][k])
   

for i in range(len(absSHAP_B)):
    for j in range(len(absSHAP_B[i])):
        absSHAP_B[i][j]=sum(absSHAP_B[i][j])/len(absSHAP_B[i][j])
        absPosSHAP_B[i][j]=sum(absPosSHAP_B[i][j])/len(absPosSHAP_B[i][j])
        absNegSHAP_B[i][j]=sum(absNegSHAP_B[i][j])/len(absNegSHAP_B[i][j])
        sSHAP_B[i][j]=sum(sSHAP_B[i][j])/len(sSHAP_B[i][j])
        sPosSHAP_B[i][j]=sum(sPosSHAP_B[i][j])/len(sPosSHAP_B[i][j])
        sNegSHAP_B[i][j]=sum(sNegSHAP_B[i][j])/len(sNegSHAP_B[i][j])


absSHAP_B_pad = sequence.pad_sequences(absSHAP_B,dtype='float32')              
absPosSHAP_B_pad = sequence.pad_sequences(absPosSHAP_B,dtype='float32')              
absNegSHAP_B_pad = sequence.pad_sequences(absNegSHAP_B,dtype='float32')  
sSHAP_B_pad = sequence.pad_sequences(sSHAP_B,dtype='float32')              
sPosSHAP_B_pad = sequence.pad_sequences(sPosSHAP_B,dtype='float32')              
sNegSHAP_B_pad = sequence.pad_sequences(sNegSHAP_B,dtype='float32')    





































