# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 00:15:55 2021

@author: bcoma
"""

import numpy as np
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

dataSpearman = copy.deepcopy(data)
dataSpearman.drop(["Visitado"],axis=1,inplace=True)
dataSpearman.drop(["ONU"],axis=1,inplace=True)
dataSpearman.drop(["Colony"],axis=1,inplace=True)
dataSpearman.drop(["LatinAmerica"],axis=1,inplace=True)
dataSpearman.drop(["Africa"],axis=1,inplace=True)
dataSpearman.drop(["Delegation"],axis=1,inplace=True)
nameVariables = list(dataSpearman.columns)
corrSpearman = dataSpearman.corr(method='spearman')


nameVariables[0] = "GDP per capita"
nameVariables[1] = "Public Grant"
nameVariables[2] = "Budget Previous Year"
nameVariables[3] = "Donor Aid Budget"

mask = np.triu(np.ones_like(corrSpearman, dtype=np.bool))
plt.figure(figsize=(16, 6))
# Store heatmap object in a variable to easily access it when you want to include more features (such as title).
# Set the range of values to be displayed on the colormap from -1 to 1, and set the annotation to True to display the correlation values on the heatmap.
heatmap = sns.heatmap(corrSpearman,mask=mask, annot=True,xticklabels=nameVariables, yticklabels=nameVariables,vmin=-1, vmax=1,cmap="RdYlBu_r")
plt.xticks(rotation=45)
# Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
heatmap.set_title("Spearman Correlation Heatmap", fontdict={'fontsize':12}, pad=12);

from sklearn.metrics import matthews_corrcoef

dataChi = copy.deepcopy(data)
dataChi.drop(["Visitado"],axis=1,inplace=True)
dataChi.drop(["GDP"],axis=1,inplace=True)
dataChi.drop(["Public_Grant"],axis=1,inplace=True)
dataChi.drop(["Donor_Aid_Budget"],axis=1,inplace=True)
dataChi.drop(["Budget_Previous_Year"],axis=1,inplace=True)

nameVariables[0] = "GDP per capita"
nameVariables[1] = "Public Grant"
nameVariables[2] = "Budget Previous Year"
nameVariables[3] = "Donor Aid Budget"

corrPearson = dataChi.corr(method='pearson')

mask = np.triu(np.ones_like(corrPearson, dtype=np.bool))
plt.figure(figsize=(16, 6))
# Store heatmap object in a variable to easily access it when you want to include more features (such as title).
# Set the range of values to be displayed on the colormap from -1 to 1, and set the annotation to True to display the correlation values on the heatmap.
heatmap = sns.heatmap(corrPearson,mask=mask, annot=True,xticklabels=nameVariables, yticklabels=nameVariables,vmin=-1, vmax=1,cmap="RdYlBu_r")
plt.xticks(rotation=45)
# Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
heatmap.set_title("Spearman Correlation Heatmap", fontdict={'fontsize':12}, pad=12);









matthews_corrcoef(dataChi["ONU"], dataChi["Colony"])


dataSpearman = copy.deepcopy(data)
dataSpearman.drop(["Visitado"],axis=1,inplace=True)
dataSpearman.drop(["ONU"],axis=1,inplace=True)
dataSpearman.drop(["Colony"],axis=1,inplace=True)
dataSpearman.drop(["LatinAmerica"],axis=1,inplace=True)
dataSpearman.drop(["Africa"],axis=1,inplace=True)
dataSpearman.drop(["Delegation"],axis=1,inplace=True)
nameVariables = list(dataSpearman.columns)
corrSpearman = dataSpearman.corr(method='spearman')


nameVariables[0] = "GDP per capita"
nameVariables[1] = "Public Grant"
nameVariables[2] = "Budget Previous Year"
nameVariables[3] = "Donor Aid Budget"

mask = np.triu(np.ones_like(corrSpearman, dtype=np.bool))
plt.figure(figsize=(16, 6))
# Store heatmap object in a variable to easily access it when you want to include more features (such as title).
# Set the range of values to be displayed on the colormap from -1 to 1, and set the annotation to True to display the correlation values on the heatmap.
heatmap = sns.heatmap(corrSpearman,mask=mask, annot=True,xticklabels=nameVariables, yticklabels=nameVariables,vmin=-1, vmax=1,cmap="RdYlBu_r")
plt.xticks(rotation=45)
# Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
heatmap.set_title("Spearman Correlation Heatmap", fontdict={'fontsize':12}, pad=12);










corrPearson = data.corr(method='pearson')
corrSpearman = data.corr(method='spearman')
corrKendall = data.corr(method='kendall')

stats.pointbiserialr(data["ONU"],data["GDP"])
stats.pointbiserialr(data["Delegation"],data["Budget_Previous_Year"])
stats.pointbiserialr(data["Colony"],data["Donor_Aid_Budget"])

pd.crosstab(data["Visitado"],data["Delegation"])

table()
stats.pointbiserialr(,)


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
