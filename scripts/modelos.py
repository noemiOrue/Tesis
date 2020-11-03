# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 20:28:03 2019

@author: bcoma
"""

import pandas
import shap
import numpy as np
import copy

trainingGlobal_negatiu = pandas.read_excel("../output/allExcels_negatiu.xlsx", sheet_name='Sheet1',index_col = "Pais-Año")


#trainingGlobal =trainingGlobal.replace("..",0)
trainingGlobal_negatiu =trainingGlobal_negatiu.replace("..",0)




provaRF_negatiu = copy.deepcopy(trainingGlobal_negatiu)
del provaRF_negatiu['Visitado']
del provaRF_negatiu['Dinero_en_el_proyecto']

import catboost as cbt
import shap

cbt.__version__

cbt_model_df_R = cbt.CatBoostRegressor(train_dir="./entrenament/",bootstrap_type = 'Bernoulli',subsample = 0.66,rsm = 0.66,iterations=1000, random_seed=99,loss_function='RMSE',eval_metric="MAE")
cbt_model_df_R.fit(provaRF_negatiu,trainingGlobal_negatiu['Dinero_en_el_proyecto'])
explainer_regressor = shap.TreeExplainer(cbt_model_df_R)
shap_values_regressor = explainer_regressor.shap_values(provaRF_negatiu)
shap.summary_plot(shap_values_regressor, provaRF_negatiu, plot_type="bar")            
shap.summary_plot(shap_values_regressor, provaRF_negatiu)  


cbt_model_df_C = cbt.CatBoostClassifier(train_dir="./entrenament/",bootstrap_type = 'Bernoulli',subsample = 0.66,rsm = 0.66,iterations=1000, random_seed=99)
cbt_model_df_C.fit(provaRF_negatiu,trainingGlobal_negatiu['Visitado'])
explainer_classifier = shap.TreeExplainer(cbt_model_df_C)
shap_values_classifier = explainer_classifier.shap_values(provaRF_negatiu)
shap.summary_plot(shap_values_classifier, provaRF_negatiu, plot_type="bar")            
shap.summary_plot(shap_values_classifier, provaRF_negatiu) 



from sklearn.ensemble import RandomForestClassifier

# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt',
                               max_depth = 2)
# Fit on training data
model.fit(provaRF_negatiu,trainingGlobal_negatiu['Visitado'])

explainer_classifier = shap.TreeExplainer(model)
shap_values_classifier = explainer_classifier.shap_values(provaRF_negatiu)
shap.summary_plot(shap_values_classifier, provaRF_negatiu, plot_type="bar")            


bst = train(params, dmatrix, num_boost_round=1)

import xgboost

params = {
  'colsample_bynode': 0.8,
  'learning_rate': 1,
  'max_depth': 5,
  'num_parallel_tree': 100,
  'objective': 'binary:logistic',
  'subsample': 0.8,
}

