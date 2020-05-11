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
import sys, os, json
from math import sqrt
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import model_from_json
from keras.preprocessing import sequence
from load_dataset import load_dataset


# launch command
# python main.py 100 4 data/BPI12_train.csv data/BPI12_test.csv data/BPI12_train.csv
# python main.py 100 6 data/vendita_polizze_assicurative_train.csv data/vendita_polizze_assicurative_test.csv

if len(sys.argv) < 4:
    sys.exit("python main.py n_neurons n_layers fileTrain fileTest")
n_neurons = int(sys.argv[1])
n_layers = int(sys.argv[2])
fileTrain = sys.argv[3]
fileTest = sys.argv[4]

# fix random seed for reproducibility
np.random.seed(7)

X_train, X_test, y_train, y_test = load_dataset(fileTrain, fileTest)

X_train = sequence.pad_sequences(X_train)
print("DEBUG: training shape", X_train.shape)
X_test = sequence.pad_sequences(X_test, maxlen=X_train.shape[1])
maxlen = X_train.shape[1]
print("DEBUG: test shape", X_test.shape)

filename = fileTrain.replace('_train.csv', '')
filename = filename.replace('data/', '')
file_trained = "model/model_" + filename + "_" + str(n_neurons) + "_" + str(n_layers) + ".json"

if not os.path.isfile(file_trained):
    model = Sequential()
    if n_layers == 1:
        model.add(LSTM(n_neurons, implementation=2, input_shape=(X_train.shape[1], X_train.shape[2]),
                       recurrent_dropout=0.2))
        model.add(BatchNormalization())
    else:
        for i in range(n_layers - 1):
            model.add(LSTM(n_neurons, implementation=2, input_shape=(X_train.shape[1], X_train.shape[2]),
                           recurrent_dropout=0.2, return_sequences=True))
            model.add(BatchNormalization())
        model.add(LSTM(n_neurons, implementation=2, recurrent_dropout=0.2))
        model.add(BatchNormalization())

    # add output layer (regression)
    model.add(Dense(1))

    # compiling the model, creating the callbacks
    model.compile(loss='mae', optimizer='Nadam', metrics=['mean_squared_error', 'mae', 'mape'])
    print(model.summary())
    early_stopping = EarlyStopping(patience=42)
    model_checkpoint = ModelCheckpoint(
        "model/model_" + filename + "_" + str(n_neurons) + "_" + str(n_layers) + "_weights_best.h5",
        monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, mode='auto',
                                   epsilon=0.0001, cooldown=0, min_lr=0)

    # train the model (maxlen)
    model.fit(X_train, y_train, validation_split=0.2, callbacks=[early_stopping, model_checkpoint, lr_reducer],
              epochs=1, batch_size=1000)
    # saving model shape to file
    model_json = model.to_json()
    with open(file_trained, "w") as json_file:
        json_file.write(model_json)
    with open("model/" + filename + "_train_shape.json", "w") as json_file:
        shape = {'shape': X_train.shape}
        json.dump(shape, json_file)
    print("Created model and saved weights")
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(scores)
    print("Root Mean Squared Error: %.4f MAE: %.4f MAPE: %.4f%%" % (sqrt(scores[1] / ((24.0 * 3600) ** 2)), scores[2] / (24.0 * 3600), scores[3]))

else:
    model = model_from_json(
        open("model/model_" + filename + "_" + str(n_neurons) + "_" + str(n_layers) + ".json").read())
    # load saved weigths to the test model
    model.load_weights("model/model_" + filename + "_" + str(n_neurons) + "_" + str(n_layers) + "_weights_best.h5")
    # Compile model (required to make predictions)
    model.compile(loss='mae', optimizer='Nadam', metrics=['mean_squared_error', 'mae', 'mape'])
    print("Loaded model and weights from file")

    scores = model.evaluate(X_test, y_test, verbose=0)
    print(scores)
    print("Root Mean Squared Error: %.4f MAE: %.4f MAPE: %.4f%%" % (
    sqrt(scores[1] / ((24.0 * 3600) ** 2)), scores[2] / (24.0 * 3600), scores[3]))
