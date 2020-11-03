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

from sklearn import preprocessing
import _pickle as pickle
import tensorflow.keras
import copy

#dataO = pd.read_excel("../output/allExcels_negatiu.xlsx",index_col = 0, header=0)

data = pd.read_excel("../output/allExcels_negatiu.xlsx",index_col = 0, header=0)

data['Gross_National_Income'] = np.log(data['Gross_National_Income'])
data['Public_Grant'] = np.log(data['Public_Grant'])
data['Total_Fondos'] = np.log(data['Total_Fondos'])
data['NGO_Country_Budget_Previous_Year'] = np.log(data['NGO_Country_Budget_Previous_Year']) #skew data
data['Total_subvencion_en_el_Pais_y_Anyo'] = np.log(data['Total_subvencion_en_el_Pais_y_Anyo'])
data['Dinero_en_el_proyecto'] = np.log(data['Dinero_en_el_proyecto']) #skew data

data[data < 0] = 0

scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(data)

np.random.seed(7)

#data_scaled = pd.DataFrame(x_scaled, columns = data.columns,index=data.index)

training_LSTM = {}
y_LSTM = {}
yR_LSTM = {}

path = '../output/'
for root, dirs, files in os.walk(path):
    for filename in files:
        if "_negativos" in filename:
            name_ONG = filename[:filename.index("_")]
            training_LSTM[name_ONG] = {}
            y_LSTM[name_ONG] = {}
            yR_LSTM[name_ONG] = {}
            print(name_ONG)
            proyectos = pd.read_excel(path+filename, sheet_name='Sheet1',index_col = "Pais-Año")
            proyectos['NGO_Country_Budget_Previous_Year'] = np.log(proyectos['NGO_Country_Budget_Previous_Year'])
            proyectos['Total_subvencion_en_el_Pais_y_Anyo'] = np.log(proyectos['Total_subvencion_en_el_Pais_y_Anyo'])
            proyectos['Total_Fondos'] = np.log(proyectos['Total_Fondos'])

            proyectos['Gross_National_Income'] = np.log(proyectos['Gross_National_Income'])
            proyectos['Public_Grant'] = np.log(proyectos['Public_Grant'])
            proyectos['Dinero_en_el_proyecto'] = np.log(proyectos['Dinero_en_el_proyecto'])
            proyectos[proyectos < 0] = 0
            
            for index, row in proyectos.iterrows():
                age = index[:4]
                country = index[5:]
                if country not in training_LSTM[name_ONG]:
                    training_LSTM[name_ONG][country] = {}
                    y_LSTM[name_ONG][country] = {}
                    yR_LSTM[name_ONG][country] = {}
                
                row = scaler.transform([row])
                y_LSTM[name_ONG][country][age]= row[0][-2]
                yR_LSTM[name_ONG][country][age]= row[0][-1]
                
                training_LSTM[name_ONG][country][age]= row[0][:-2]
                #training_LSTM[name_ONG][country][age]= row

training_LSTM_2 = []
y_LSTM_2 = []
yR_LSTM_2 = []
training_LSTM_4 = []
training_LSTM_8 = []
y_LSTM_8 = []
yR_LSTM_8 = []
            

for ong in training_LSTM:
    for country in training_LSTM[ong]:
        ages = ["2009","2010","2011","2012","2013","2014","2015","2016"]
        for posAge in range(len(ages)-1):
            newdata = []
            if ages[posAge] in training_LSTM[ong][country]:
                data = training_LSTM[ong][country][ages[posAge]]
                newdata.append(data)
            else:
                newdata.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            yR = 0
            y = 0
            if ages[posAge+1] in training_LSTM[ong][country]:
                data = training_LSTM[ong][country][ages[posAge+1]]
                newdata.append(data)
                yR = yR_LSTM[ong][country][ages[posAge+1]]
                y = y_LSTM[ong][country][ages[posAge+1]]
            else:
                newdata.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            training_LSTM_2.append(newdata)
            y_LSTM_2.append(y)
            yR_LSTM_2.append(yR)
        
        newdata = []
        for posAge in range(len(ages)):
            if ages[posAge] in training_LSTM[ong][country]:
                data = training_LSTM[ong][country][ages[posAge]]
                newdata.append(data)
            else:
                newdata.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            
            if ages[posAge]=="2016":
                yR = 0
                y = 0
                if ages[posAge] in training_LSTM[ong][country]:
                    yR = yR_LSTM[ong][country][ages[posAge]]
                    y = y_LSTM[ong][country][ages[posAge]]
                training_LSTM_8.append(newdata)
                y_LSTM_8.append(y)
                yR_LSTM_8.append(yR)

training_LSTM_8_pad = sequence.pad_sequences(training_LSTM_8,dtype='float32')              


training_LSTM_2_pad = sequence.pad_sequences(training_LSTM_2,dtype='float32')              

######################################regressio 8
import time
inici = time.time()
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



model_bin = Sequential()

model_bin.add(LSTM(100, implementation=2, input_shape=(training_LSTM_8_pad.shape[1], training_LSTM_8_pad.shape[2]),
                           recurrent_dropout=0.2,return_sequences=True))
model_bin.add(BatchNormalization())
model_bin.add(LSTM(100, implementation=2,recurrent_dropout=0.2))
model_bin.add(BatchNormalization())
model_bin.add(Dense(1, activation="sigmoid"))
nadam_opt = optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

model_bin.compile(loss='binary_crossentropy', optimizer=nadam_opt)
model_bin.fit(training_LSTM_8_pad, y_LSTM_8, epochs=50)
final = time.time()

model_bin.save("./binaryKeras.model")

explainer_bin = shap.DeepExplainer(model_bin,training_LSTM_8_pad)

training_LSTM_8_pad_B = explainer_bin.shap_values(training_LSTM_8_pad)

pickle.dump(training_LSTM_8_pad_B,open("./fitxerShapleyLSTM_8_B","wb"))  

training_LSTM_8_pad_B = pickle.load(open("./fitxerShapleyLSTM_8_B","rb"))
training_LSTM_8_pad_R = pickle.load(open("./fitxerShapleyLSTM_8_R","rb"))


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





































