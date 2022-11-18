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
np.__version__
from scipy import stats
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
from tensorflow.keras import regularizers

import shap
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn import preprocessing, metrics
import _pickle as pickle
import tensorflow.keras
import copy
#import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow
tensorflow.__version__

from tensorflow.compat.v1.keras.backend import get_session
tensorflow.compat.v1.disable_v2_behavior()
tensorflow.test.is_gpu_available()
np.random.seed(7)
tensorflow.random.set_seed(7)

#data = data.drop('%_MAE_Funds',1)
#data = data.drop('Total_Funds',1)
#data = data.drop('%_Private_Funds', 1)

data = pd.read_excel("../output/allExcels_negatiu.xlsx",index_col = 0, header=0)
data.columns
data_original = copy.deepcopy(data)
data['GDP'] = np.log(data['GDP'])
data['Public_Grant'] = np.log(data['Public_Grant'])
data['Budget_Previous_Year'] = np.log(data['Budget_Previous_Year']) #skew data
data['Donor_Aid_Budget'] = np.log(data['Donor_Aid_Budget'])
data[data < 0] = 0

data2 = data.loc[data["ControlofCorruption"]!=0]
resCorr = data.corr(method="pearson")
resCorr2 = data2.corr(method="pearson")


scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(data.values)


training_LSTM = {}
y_LSTM = {}


pos = 0
path = '../output/'
for root, dirs, files in os.walk(path):
    for filename in files:
        if "_negativos_2" in filename:
            
            
            name_ONG = filename[:filename.index("_")]
            training_LSTM[name_ONG] = {}
            y_LSTM[name_ONG] = {}
            proyectos = pd.read_excel(path+filename,index_col =0)
            posVisitado = proyectos.columns.get_loc("Visitado")

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
                y_LSTM[name_ONG][country][age]= row["Visitado"]
                
                row = scaler.transform([row])
                
                row2 = np.delete(row,posVisitado)
                training_LSTM[name_ONG][country][age]= row2

q = []
q1 = []
q2 = []
for ong in training_LSTM:
    for country in training_LSTM[ong]:
        for year in training_LSTM[ong][country]:
            if training_LSTM[ong][country][year][2]>0:
                q.append(training_LSTM[ong][country][year][-1])
                if training_LSTM[ong][country][year][-1] == 0 and str(int(year)+1) in training_LSTM[ong][country]:
                    q1.append(training_LSTM[ong][country][str(int(year)+1)][-1])
                
(q.count(1)+q1.count(1))/len(q)

training_LSTM_8 = []
training_LSTM_8_Big = []
training_LSTM_8_Big_Y = []
training_LSTM_8_Small = []
training_LSTM_8_Medium = []
training_LSTM_8_Small_Y = []
training_LSTM_8_Medium_Y = []

y_LSTM_8 = []  
y_LSTM_8_Small = []  
training_LSTM_4 = []
training_LSTM_2_2 = []    
training_LSTM_8_noPath = []
training_LSTM_8_noPath4Years = []
qND = 0
bigNGOs = []
smallNGOs = []
mediumNGOs = []

for ong in training_LSTM:
    big = False
    small = False
    medium = False
    if "cruz roja" in ong or "cáritas" in ong or "acción contra el hambre" in ong or "verapaz" in ong or "medicos del mundo" in ong or "oxfam" in ong:# or "fontilles" in ong or "educo" in ong or "adsis" in ong or "entreculturas" in ong or "manos unidas" in ong or "mpdl" in ong:
        big = True
    elif "verapaz" in ong or "amref" in ong or "nazaret" in ong or "farmaceuticos" in ong or "fisc" in ong or "pueblos hermanos" in ong:
        small = True
    else:
        medium = True
    for country in training_LSTM[ong]:
        ages = ["2009","2010","2011","2012","2013","2014","2015","2016"]
               
        newdata = []
        newdata4 = []
        newdata2_2 = []
        newdata_noPath = []
        newdata_noPath4Years = []
        
        for posAge in range(len(ages)):
            if ages[posAge] in training_LSTM[ong][country]:
                data = training_LSTM[ong][country][ages[posAge]]
                #if ages[posAge] != "2009":
                #    data[5] = 0
                #    data[6] = 0
                #    data[7] = 0
                    
                newdata.append(data)
                data2 = copy.deepcopy(data)
                data2[3] = 0
                data2[8] = 0
                newdata_noPath.append(data2)
                if posAge <4:
                    newdata4.append(data)
                    newdata_noPath4Years.append(data2)

                else:
                    newdata4.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
                    newdata_noPath4Years.append(data)
                    
                if posAge%2==0:
                    newdata2_2.append(data)
                else:
                    newdata2_2.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
                    
            else:
                print(ong,country)
                qND+=1
                newdata.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
                if posAge <4:
                    newdata4.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
                if posAge%2==0:
                    newdata2_2.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            if ages[posAge]=="2016":
                yR = 0
                y = 0
                if ages[posAge] in training_LSTM[ong][country]:
                    y = y_LSTM[ong][country][ages[posAge]]
                training_LSTM_8.append(newdata)
                
                training_LSTM_4.append(newdata4)
                training_LSTM_2_2.append(newdata2_2)
                
                training_LSTM_8_noPath.append(newdata_noPath)
                training_LSTM_8_noPath4Years.append(newdata_noPath4Years)
                
                y_LSTM_8.append(y)
                bigNGOs.append(big)
                smallNGOs.append(small)
                mediumNGOs.append(medium)
                if big:
                    training_LSTM_8_Big.append(newdata)
                    training_LSTM_8_Big_Y.append(y)
                elif small:
                    training_LSTM_8_Small.append(newdata)
                    training_LSTM_8_Small_Y.append(y)
                    y_LSTM_8_Small.append(y)
                else:
                    training_LSTM_8_Medium.append(newdata)
                    training_LSTM_8_Medium_Y.append(y)
                    
    

                
training_LSTM_8_pad = sequence.pad_sequences(np.array(training_LSTM_8),dtype='float32')              
training_LSTM_8_pad_noPath = sequence.pad_sequences(np.array(training_LSTM_8_noPath),dtype='float32')              
training_LSTM_8_pad_noPath4Years = sequence.pad_sequences(np.array(training_LSTM_8_noPath4Years),dtype='float32')              

training_LSTM_4_pad = sequence.pad_sequences(np.array(training_LSTM_4),dtype='float32')              
training_LSTM_2_2_pad = sequence.pad_sequences(np.array(training_LSTM_2_2),dtype='float32') 
training_LSTM_8_pad_big = sequence.pad_sequences(np.array(training_LSTM_8_Big),dtype='float32')              
training_LSTM_8_pad_small = sequence.pad_sequences(np.array(training_LSTM_8_Small),dtype='float32') 
training_LSTM_8_pad_medium = sequence.pad_sequences(np.array(training_LSTM_8_Medium),dtype='float32')                      

################################# 8
import time
inici = time.time()
"""
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
early_stopping = EarlyStopping(patience=5,mode="min",monitor='val_loss')
model_bin.fit(training_LSTM_8_pad, np.array(y_LSTM_8), validation_split=0.1,callbacks=[early_stopping],epochs=1000)
final = time.time()

"""

model_bin = Sequential()

model_bin.add(LSTM(100, implementation=2, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),input_shape=(training_LSTM_8_pad.shape[1], training_LSTM_8_pad.shape[2]),
                           return_sequences=True,recurrent_dropout=0.2))
model_bin.add(BatchNormalization())
model_bin.add(LSTM(100, implementation=2, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),recurrent_dropout=0.2))
#model_bin.add(Dropout(0.2))
model_bin.add(BatchNormalization())
model_bin.add(Dense(1,kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), activation="sigmoid"))
nadam_opt = optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999)

model_bin.compile(loss='binary_crossentropy', optimizer=nadam_opt)
early_stopping = EarlyStopping(patience=5,mode="min",monitor='val_loss')
model_bin.fit(training_LSTM_8_pad, np.array(y_LSTM_8), validation_split=0.1,callbacks=[early_stopping],epochs=1000)
final = time.time()



model_bin = Sequential()

model_bin.add(LSTM(100, implementation=2,kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), input_shape=(training_LSTM_8_pad.shape[1], training_LSTM_8_pad.shape[2]),
                           recurrent_dropout=0.2,return_sequences=True))
model_bin.add(BatchNormalization())
model_bin.add(LSTM(100, implementation=2,kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),recurrent_dropout=0.2))
model_bin.add(BatchNormalization())
model_bin.add(Dense(1, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),activation="sigmoid"))
nadam_opt = optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999)

model_bin.compile(loss='binary_crossentropy', optimizer=nadam_opt)
model_bin.fit(training_LSTM_8_pad, np.array(y_LSTM_8), epochs=26)
#early_stopping = EarlyStopping(patience=0,mode="min",monitor='val_loss')
final = time.time()
"""
model_bin = Sequential()

model_bin.add(LSTM(100, implementation=2, input_shape=(training_LSTM_8_pad.shape[1], training_LSTM_8_pad.shape[2]),
                           return_sequences=True,recurrent_dropout=0.2))
model_bin.add(BatchNormalization())
model_bin.add(Dense(1, activation="relu"))

#model_bin.add(Dropout(0.2))
model_bin.add(BatchNormalization())
model_bin.add(Dense(1, activation="sigmoid"))
nadam_opt = optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999)

model_bin.compile(loss='binary_crossentropy', optimizer=nadam_opt)
early_stopping = EarlyStopping(patience=5,mode="min",monitor='val_loss')
model_bin.fit(training_LSTM_8_pad, np.array(y_LSTM_8), validation_split=0.1,callbacks=[early_stopping],epochs=1000)
final = time.time()


"""
model_bin.save("./binaryKeras_1.model")
model_bin = tensorflow.keras.models.load_model("./binaryKeras_1.model")



explainer_bin = shap.DeepExplainer(model_bin,training_LSTM_8_pad)

training_LSTM_8_pad_B = explainer_bin.shap_values(training_LSTM_8_pad)

pickle.dump(training_LSTM_8_pad_B,open("./fitxerShapleyLSTM_8_B_1","wb")) 

training_LSTM_8_pad_B = pickle.load(open("./fitxerShapleyLSTM_8_B_1","rb"))


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


shap.summary_plot(np.array(shap_Specific),features=np.array(shap_SpecificValues),feature_names=variables,plot_type="bar",max_display=15)
shap.summary_plot(np.array(shap_Specific),features=np.array(shap_SpecificValues),feature_names=variables,max_display=15)





predictions = model_bin.predict(training_LSTM_8_pad)[:,0]
predictions01 = model_bin.predict_classes(training_LSTM_8_pad)


display = metrics.PrecisionRecallDisplay.from_predictions(y_LSTM_8, predictions, name="Predictive model using all data")
_ = display.ax_.set_title("")

metrics.average_precision_score(y_LSTM_8,predictions)
metrics.f1_score(y_LSTM_8,predictions01)
metrics.precision_score(y_LSTM_8,predictions01)
metrics.recall_score(y_LSTM_8,predictions01)
#shap.summary_plot(np.array(shap_Specific_P),features=np.array(shap_SpecificValues_P),feature_names=variables,plot_type="bar",max_display=10)
#shap.summary_plot(np.array(shap_Specific_P),features=np.array(shap_SpecificValues_P),feature_names=variables,max_display=10)


for i in range(len(variables)):
    print(i, variables[i])
shap_Specific_H1_H2 = []
shap_SpecificValues_H1_H2 = []
pos_H1_H2 = [63,64,65,67,68,69,70]
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


shap.summary_plot(np.array(shap_Specific_H1_H2),features=np.array(shap_SpecificValues_H1_H2),feature_names=["UN LDCs_2016","GDP per capita_2016","Public Grant_2016","Latin America Mission_2016","Africa Mission_2016"],plot_type="bar",max_display=10)
shap.summary_plot(np.array(shap_Specific_H1_H2),features=np.array(shap_SpecificValues_H1_H2),feature_names=["UN LDCs_2016","GDP per capita_2016","Public Grant_2016","Donor Aid Budget_2016","Latin America Mission_2016","Africa Mission_2016","Colony_2016"],max_display=10)

shap.summary_plot(np.array(shap_Specific_H1_H2_P),features=np.array(shap_SpecificValues_H1_H2_P),feature_names=["UN LDCs_2016","GDP per capita_2016","Public Grant_2016","Donor Aid Budget_2016","Latin America Mission_2016","Africa Mission_2016"],plot_type="bar",max_display=10)
shap.summary_plot(np.array(shap_Specific_H1_H2_P),features=np.array(shap_SpecificValues_H1_H2_P),feature_names=["UN LDCs_2016","GDP per capita_2016","Public Grant_2016","Donor Aid Budget_2016","Latin America Mission_2016","Africa Mission_2016"],max_display=10)




#############

model_bin4 = Sequential()

model_bin4.add(LSTM(100, implementation=2, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),input_shape=(training_LSTM_4_pad.shape[1], training_LSTM_4_pad.shape[2]),
                           return_sequences=True,recurrent_dropout=0.2))
model_bin4.add(BatchNormalization())
model_bin4.add(LSTM(100, implementation=2, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),recurrent_dropout=0.2))
#model_bin.add(Dropout(0.2))
model_bin4.add(BatchNormalization())
model_bin4.add(Dense(1,kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), activation="sigmoid"))
nadam_opt = optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999)
model_bin4.compile(loss='binary_crossentropy', optimizer=nadam_opt)
early_stopping = EarlyStopping(patience=5,mode="min",monitor='val_loss')
model_bin4.fit(training_LSTM_4_pad, np.array(y_LSTM_8), validation_split=0.1,callbacks=[early_stopping],epochs=1000)
final = time.time()



model_bin4 = Sequential()

model_bin4.add(LSTM(100, implementation=2,kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), input_shape=(training_LSTM_4_pad.shape[1], training_LSTM_4_pad.shape[2]),
                           recurrent_dropout=0.2,return_sequences=True))
model_bin4.add(BatchNormalization())
model_bin4.add(LSTM(100, implementation=2,kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),recurrent_dropout=0.2))
model_bin4.add(BatchNormalization())
model_bin4.add(Dense(1, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),activation="sigmoid"))
nadam_opt = optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999)

model_bin4.compile(loss='binary_crossentropy', optimizer=nadam_opt)
model_bin4.fit(training_LSTM_4_pad, y_LSTM_8, epochs=17)
#early_stopping = EarlyStopping(patience=0,mode="min",monitor='val_loss')


model_bin4.save("./binaryKeras_4.model")
model_bin4 = tensorflow.keras.models.load_model("./binaryKeras_4.model")

explainer_bin4 = shap.DeepExplainer(model_bin4,training_LSTM_4_pad)
training_LSTM_4_pad_B = explainer_bin4.shap_values(training_LSTM_4_pad)

#training_LSTM_8_pad_B = explainer_bin.shap_values(training_LSTM_8_pad)

pickle.dump(training_LSTM_4_pad_B,open("./training_LSTM_4_pad_B","wb")) 

training_LSTM_4_pad_B = pickle.load(open("./training_LSTM_4_pad_B","rb"))



predictions_4 = model_bin4.predict(training_LSTM_8_pad)[:,0]
predictions01_4 = model_bin4.predict_classes(training_LSTM_8_pad)

metrics.average_precision_score(y_LSTM_8,predictions_4)

display = metrics.PrecisionRecallDisplay.from_predictions(y_LSTM_8, predictions_4, name="Predictive model using first 4 years")
_ = display.ax_.set_title("")

metrics.average_precision_score(y_LSTM_8,predictions_4)
metrics.f1_score(y_LSTM_8,predictions01_4)
metrics.precision_score(y_LSTM_8,predictions01_4)
metrics.recall_score(y_LSTM_8,predictions01_4)



shap_Specific_4= []
shap_SpecificValues_4 = []
shap_Specific_P_4= []
shap_SpecificValues_P_4 = []
variables = []
for i in range(len(training_LSTM_4_pad_B[0])):
    valSV = []
    valFeature = []
    for j in range(len(training_LSTM_4_pad_B[0][0])): #0 --> 7 (8 anys)
        for k in range(len(training_LSTM_4_pad_B[0][0][0])): #0-->8 (9 variables) 
            valSV.append(training_LSTM_4_pad_B[0][i][j][k])
            valFeature.append(training_LSTM_4_pad[i][j][k])
            if i ==0:
                variable =nameVariables[k]
                variable = variable+"_"+["2009","2010","2011","2012","2013","2014","2015","2016"][j]
                variables.append(variable)
    shap_Specific_4.append(copy.deepcopy(valSV))
    shap_SpecificValues_4.append(copy.deepcopy(valFeature))
    if y_LSTM_8[i]==1:
        shap_Specific_P_4.append(copy.deepcopy(valSV))
        shap_SpecificValues_P_4.append(copy.deepcopy(valFeature))


shap.summary_plot(np.array(shap_Specific_4),features=np.array(shap_SpecificValues_4),feature_names=variables,max_display=15)
shap.summary_plot(np.array(shap_Specific_4),features=np.array(shap_SpecificValues_4),feature_names=variables,plot_type="bar",max_display=15)


#######################################
model_bin2 = Sequential()

model_bin2.add(LSTM(100, implementation=2, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),input_shape=(training_LSTM_2_2_pad.shape[1], training_LSTM_2_2_pad.shape[2]),
                           return_sequences=True,recurrent_dropout=0.2))
model_bin2.add(BatchNormalization())
model_bin2.add(LSTM(100, implementation=2, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),recurrent_dropout=0.2))
#model_bin.add(Dropout(0.2))
model_bin2.add(BatchNormalization())
model_bin2.add(Dense(1,kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), activation="sigmoid"))
nadam_opt = optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999)
model_bin2.compile(loss='binary_crossentropy', optimizer=nadam_opt)
early_stopping = EarlyStopping(patience=5,mode="min",monitor='val_loss')
model_bin2.fit(training_LSTM_2_2_pad, np.array(y_LSTM_8), validation_split=0.1,callbacks=[early_stopping],epochs=1000)


model_bin2 = Sequential()

model_bin2.add(LSTM(100, implementation=2,kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), input_shape=(training_LSTM_2_2_pad.shape[1], training_LSTM_2_2_pad.shape[2]),
                           recurrent_dropout=0.2,return_sequences=True))
model_bin2.add(BatchNormalization())
model_bin2.add(LSTM(100, implementation=2,kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),recurrent_dropout=0.2))
model_bin2.add(BatchNormalization())
model_bin2.add(Dense(1, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),activation="sigmoid"))
nadam_opt = optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999)

model_bin2.compile(loss='binary_crossentropy', optimizer=nadam_opt)
model_bin2.fit(training_LSTM_2_2_pad, y_LSTM_8, epochs=29)

model_bin2.save("./binaryKeras_2.model")
model_bin2 = tensorflow.keras.models.load_model("./binaryKeras_2.model")


explainer_bin2 = shap.DeepExplainer(model_bin2,training_LSTM_2_2_pad)
training_LSTM_2_pad_B = explainer_bin2.shap_values(training_LSTM_2_2_pad)



pickle.dump(training_LSTM_2_pad_B,open("./training_LSTM_2_pad_B","wb")) 

training_LSTM_2_pad_B = pickle.load(open("./training_LSTM_2_pad_B","rb"))


predictions_2 = model_bin2.predict(training_LSTM_2_2_pad)[:,0]
predictions01_2 = model_bin2.predict_classes(training_LSTM_2_2_pad)

metrics.average_precision_score(y_LSTM_8,predictions_2)



#display = metrics.PrecisionRecallDisplay.from_predictions(y_LSTM_8, predictions_2, name="Predictive model using data every two years")
#_ = display.ax_.set_title("")

metrics.average_precision_score(y_LSTM_8,predictions_2)
metrics.f1_score(y_LSTM_8,predictions01_2)
metrics.precision_score(y_LSTM_8,predictions01_2)
metrics.recall_score(y_LSTM_8,predictions01_2)







shap_Specific_2= []
shap_SpecificValues_2 = []
shap_Specific_P_2= []
shap_SpecificValues_P_2 = []
variables = []
for i in range(len(training_LSTM_2_pad_B[0])):
    valSV = []
    valFeature = []
    for j in range(len(training_LSTM_2_pad_B[0][0])): #0 --> 7 (8 anys)
        for k in range(len(training_LSTM_2_pad_B[0][0][0])): #0-->8 (9 variables) 
            valSV.append(training_LSTM_2_pad_B[0][i][j][k])
            valFeature.append(training_LSTM_2_2_pad[i][j][k])
            if i ==0:
                variable =nameVariables[k]
                variable = variable+"_"+["2009","2010","2011","2012","2013","2014","2015","2016"][j]
                variables.append(variable)
    shap_Specific_2.append(copy.deepcopy(valSV))
    shap_SpecificValues_2.append(copy.deepcopy(valFeature))
    if y_LSTM_8[i]==1:
        shap_Specific_P_2.append(copy.deepcopy(valSV))
        shap_SpecificValues_P_2.append(copy.deepcopy(valFeature))


shap.summary_plot(np.array(shap_Specific_2),features=np.array(shap_SpecificValues_2),feature_names=variables,max_display=15)
shap.summary_plot(np.array(shap_Specific_2),features=np.array(shap_SpecificValues_2),feature_names=variables,plot_type="bar",max_display=15)


##################

model_bin_noPath= Sequential()

model_bin_noPath.add(LSTM(100, implementation=2, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),input_shape=(training_LSTM_8_pad.shape[1], training_LSTM_8_pad.shape[2]),
                           return_sequences=True,recurrent_dropout=0.2))
model_bin_noPath.add(BatchNormalization())
model_bin_noPath.add(LSTM(100, implementation=2, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),recurrent_dropout=0.2))
#model_bin.add(Dropout(0.2))
model_bin_noPath.add(BatchNormalization())
model_bin_noPath.add(Dense(1,kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), activation="sigmoid"))
nadam_opt = optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999)

model_bin_noPath.compile(loss='binary_crossentropy', optimizer=nadam_opt)
early_stopping = EarlyStopping(patience=5,mode="min",monitor='val_loss')
model_bin_noPath.fit(training_LSTM_8_pad_noPath, np.array(y_LSTM_8), validation_split=0.1,callbacks=[early_stopping],epochs=1000)


model_bin_noPath = Sequential()

model_bin_noPath.add(LSTM(100, implementation=2,kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), input_shape=(training_LSTM_8_pad.shape[1], training_LSTM_8_pad.shape[2]),
                           recurrent_dropout=0.2,return_sequences=True))
model_bin_noPath.add(BatchNormalization())
model_bin_noPath.add(LSTM(100, implementation=2,kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),recurrent_dropout=0.2))
model_bin_noPath.add(BatchNormalization())
model_bin_noPath.add(Dense(1, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),activation="sigmoid"))
nadam_opt = optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999)

model_bin_noPath.compile(loss='binary_crossentropy', optimizer=nadam_opt)
model_bin_noPath.fit(training_LSTM_8_pad_noPath, np.array(y_LSTM_8), epochs=22)
#early_stopping = EarlyStopping(patience=0,mode="min",monitor='val_loss')

model_bin_noPath.save("./binaryKeras_noPath.model")

model_bin_noPath = tensorflow.keras.models.load_model("./binaryKeras_noPath.model")

explainer_bin_noPath = shap.DeepExplainer(model_bin_noPath,training_LSTM_8_pad_noPath)

training_LSTM_8_pad_B_noPath = explainer_bin_noPath.shap_values(training_LSTM_8_pad_noPath)


pickle.dump(training_LSTM_8_pad_B_noPath,open("./training_LSTM_8_pad_B_noPath","wb")) 

training_LSTM_8_pad_B_noPath = pickle.load(open("./training_LSTM_8_pad_B_noPath","rb"))




shap_Specific_noPath= []
shap_SpecificValues_noPath = []
variables = []
for i in range(len(training_LSTM_8_pad_B_noPath[0])):
    valSV = []
    valFeature = []
    for j in range(len(training_LSTM_8_pad_B_noPath[0][0])): #0 --> 7 (8 anys)
        for k in range(len(training_LSTM_8_pad_B_noPath[0][0][0])): #0-->8 (9 variables) 
            valSV.append(training_LSTM_8_pad_B_noPath[0][i][j][k])
            valFeature.append(training_LSTM_8_pad_noPath[i][j][k])
            if i ==0:
                variable =nameVariables[k]
                variable = variable+"_"+["2009","2010","2011","2012","2013","2014","2015","2016"][j]
                variables.append(variable)
    shap_Specific_noPath.append(copy.deepcopy(valSV))
    shap_SpecificValues_noPath.append(copy.deepcopy(valFeature))
    


shap.summary_plot(np.array(shap_Specific_noPath),features=np.array(shap_SpecificValues_noPath),feature_names=variables,max_display=15)
shap.summary_plot(np.array(shap_Specific_noPath),features=np.array(shap_SpecificValues_noPath),feature_names=variables,plot_type="bar",max_display=15)



predictions_noPath = model_bin_noPath.predict(training_LSTM_8_pad_noPath)[:,0]
predictions01_noPath = model_bin_noPath.predict_classes(training_LSTM_8_pad)

metrics.average_precision_score(y_LSTM_8,predictions_noPath)

display = metrics.PrecisionRecallDisplay.from_predictions(y_LSTM_8, predictions_noPath, name="Predictive model using H1 and H2 data")
_ = display.ax_.set_title("")

metrics.average_precision_score(y_LSTM_8,predictions_noPath)
metrics.f1_score(y_LSTM_8,predictions01_noPath)
metrics.precision_score(y_LSTM_8,predictions01_noPath)
metrics.recall_score(y_LSTM_8,predictions01_noPath)




########


model_binSmall = Sequential()

model_binSmall.add(LSTM(100, implementation=2, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),input_shape=(training_LSTM_8_pad_small.shape[1], training_LSTM_8_pad_small.shape[2]),
                           return_sequences=True,recurrent_dropout=0.2))
model_binSmall.add(BatchNormalization())
model_binSmall.add(LSTM(100, implementation=2, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),recurrent_dropout=0.2))
#model_bin.add(Dropout(0.2))
model_binSmall.add(BatchNormalization())
model_binSmall.add(Dense(1,kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), activation="sigmoid"))
nadam_opt = optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999)
model_binSmall.compile(loss='binary_crossentropy', optimizer=nadam_opt)
early_stopping = EarlyStopping(patience=5,mode="min",monitor='val_loss')
model_binSmall.fit(training_LSTM_8_pad_small, np.array(y_LSTM_8_Small), validation_split=0.1,callbacks=[early_stopping],epochs=1000)


model_binSmall = Sequential()

model_binSmall.add(LSTM(100, implementation=2,kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), input_shape=(training_LSTM_8_pad_small.shape[1], training_LSTM_8_pad_small.shape[2]),
                           recurrent_dropout=0.2,return_sequences=True))
model_binSmall.add(BatchNormalization())
model_binSmall.add(LSTM(100, implementation=2,kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),recurrent_dropout=0.2))
model_binSmall.add(BatchNormalization())
model_binSmall.add(Dense(1, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),activation="sigmoid"))
nadam_opt = optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999)

model_binSmall.compile(loss='binary_crossentropy', optimizer=nadam_opt)
model_binSmall.fit(training_LSTM_8_pad_small, y_LSTM_8_Small, epochs=50)


model_binSmall.save("./binaryKeras_Small.model")


explainer_bin_Small = shap.DeepExplainer(model_binSmall,training_LSTM_8_pad_small)
training_LSTM_Small_pad_B = explainer_bin_Small.shap_values(training_LSTM_8_pad_small)



shap_Specific_Small= []
shap_SpecificValues_Small= []

variables = []
for i in range(len(training_LSTM_Small_pad_B[0])):
    valSV = []
    valFeature = []
    for j in range(len(training_LSTM_Small_pad_B[0][0])): #0 --> 7 (8 anys)
        for k in range(len(training_LSTM_Small_pad_B[0][0][0])): #0-->8 (9 variables) 
            valSV.append(training_LSTM_Small_pad_B[0][i][j][k])
            valFeature.append(training_LSTM_8_pad_small[i][j][k])
            if i ==0:
                variable =nameVariables[k]
                variable = variable+"_"+["2009","2010","2011","2012","2013","2014","2015","2016"][j]
                variables.append(variable)
    shap_Specific_Small.append(copy.deepcopy(valSV))
    shap_SpecificValues_Small.append(copy.deepcopy(valFeature))
    


shap.summary_plot(np.array(shap_Specific_Small),features=np.array(shap_SpecificValues_Small),feature_names=variables,max_display=15)
shap.summary_plot(np.array(shap_Specific_2),features=np.array(shap_SpecificValues_2),feature_names=variables,plot_type="bar",max_display=15)




###########


shapleyValuesBig = []
for i in range(len(training_LSTM_8_pad_B[0])):
    if bigNGOs[i]:
        shapleyValuesBig.append(training_LSTM_8_pad_B[0][i])
        
predictionsBig = model_bin.predict(training_LSTM_8_pad_big)[:,0]

metrics.average_precision_score(training_LSTM_8_Big_Y,predictionsBig)


metrics.f1_score(training_LSTM_8_Big_Y,predictionsBig)

shap_SpecificBig= []
shap_SpecificBigValues= []

variables = []
for i in range(len(shapleyValuesBig)):
    valSV = []
    valFeature = []
    for j in range(len(shapleyValuesBig[0])): #0 --> 7 (8 anys)
        for k in range(len(shapleyValuesBig[0][0])): #0-->8 (9 variables) 
            valSV.append(shapleyValuesBig[i][j][k])
            valFeature.append(training_LSTM_8_pad_big[i][j][k])
            if i ==0:
                variable =nameVariables[k]
                variable = variable+"_"+["2009","2010","2011","2012","2013","2014","2015","2016"][j]
                variables.append(variable)
    shap_SpecificBig.append(copy.deepcopy(valSV))
    shap_SpecificBigValues.append(copy.deepcopy(valFeature))

shap.summary_plot(np.array(shap_SpecificBig),features=np.array(shap_SpecificBigValues),feature_names=variables,plot_type="bar",max_display=15)
shap.summary_plot(np.array(shap_SpecificBig),features=np.array(shap_SpecificBigValues),feature_names=variables,max_display=15)



shapleyValuesSmall = []
for i in range(len(training_LSTM_8_pad_B[0])):
    if smallNGOs[i]:
        shapleyValuesSmall.append(training_LSTM_8_pad_B[0][i])
        


shap_SpecificSmall= []
shap_SpecificSmallValues= []
predictionsSmall = model_bin.predict(training_LSTM_8_pad_small)[:,0]


predictionsSmall = model_bin.predict(training_LSTM_8_pad_small)[:,0]

metrics.average_precision_score(training_LSTM_8_Small_Y,predictionsSmall)

variables = []
for i in range(len(shapleyValuesSmall)):
    valSV = []
    valFeature = []
    for j in range(len(shapleyValuesSmall[0])): #0 --> 7 (8 anys)
        for k in range(len(shapleyValuesSmall[0][0])): #0-->8 (9 variables) 
            valSV.append(shapleyValuesSmall[i][j][k])
            valFeature.append(training_LSTM_8_pad_small[i][j][k])
            if i ==0:
                variable =nameVariables[k]
                variable = variable+"_"+["2009","2010","2011","2012","2013","2014","2015","2016"][j]
                variables.append(variable)
    shap_SpecificSmall.append(copy.deepcopy(valSV))
    shap_SpecificSmallValues.append(copy.deepcopy(valFeature))

shap.summary_plot(np.array(shap_SpecificSmall),features=np.array(shap_SpecificSmallValues),feature_names=variables,plot_type="bar",max_display=15)
shap.summary_plot(np.array(shap_SpecificSmall),features=np.array(shap_SpecificSmallValues),feature_names=variables,max_display=15)






shapleyValuesMedium = []
for i in range(len(training_LSTM_8_pad_B[0])):
    if mediumNGOs[i]:
        shapleyValuesMedium.append(training_LSTM_8_pad_B[0][i])
        


shap_SpecificMedium= []
shap_SpecificMediumValues= []

predictionsMedium = model_bin.predict(training_LSTM_8_pad_medium)[:,0]


variables = []
for i in range(len(shapleyValuesMedium)):
    valSV = []
    valFeature = []
    for j in range(len(shapleyValuesMedium[0])): #0 --> 7 (8 anys)
        for k in range(len(shapleyValuesMedium[0][0])): #0-->8 (9 variables) 
            valSV.append(shapleyValuesMedium[i][j][k])
            valFeature.append(training_LSTM_8_pad_medium[i][j][k])
            if i ==0:
                variable =nameVariables[k]
                variable = variable+"_"+["2009","2010","2011","2012","2013","2014","2015","2016"][j]
                variables.append(variable)
    shap_SpecificMedium.append(copy.deepcopy(valSV))
    shap_SpecificMediumValues.append(copy.deepcopy(valFeature))

shap.summary_plot(np.array(shap_SpecificMedium),features=np.array(shap_SpecificMediumValues),feature_names=variables,plot_type="bar",max_display=15)
shap.summary_plot(np.array(shap_SpecificMedium),features=np.array(shap_SpecificMediumValues),feature_names=variables,max_display=15)


predictionsMedium = model_bin.predict(training_LSTM_8_pad_medium)[:,0]

metrics.average_precision_score(training_LSTM_8_Medium_Y,predictionsMedium)



shap.summary_plot(np.array(shap_SpecificMedium+shap_SpecificSmall),features=np.array(shap_SpecificMediumValues+shap_SpecificSmallValues),feature_names=variables,plot_type="bar",max_display=15)
shap.summary_plot(np.array(shap_SpecificMedium+shap_SpecificSmall),features=np.array(shap_SpecificMediumValues+shap_SpecificSmallValues),feature_names=variables,max_display=15)






































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





































