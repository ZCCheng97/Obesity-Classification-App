import pandas as pd
import pickle
import numpy as np

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import streamlit as st

import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings(action='ignore', category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# replace with input from user
Gender = "Male"
data_dict = {
"Gender":Gender,
"Age":                               26,
"Height":                            1.74,
"Weight":                            56,
"family_history_with_overweight":"yes",
"FAVC":                             "yes",
"FCVC":                              2,
"NCP":                               3,
"CAEC":                               "Sometimes",
"SMOKE":                              "no",
"CH2O":                              2,
"SCC":                                "no",
"FAF":                               1,
"TUE":                               1,
"CALC":                               "Sometimes",
"MTRANS":                             "Public_Transportation"
}
test = pd.DataFrame(data_dict, index = [0])

# test = pd.read_csv("data/train.csv")

d = {0: 'Insufficient_Weight',
 1: 'Normal_Weight',
 4: 'Obesity_Type_I',
 5: 'Obesity_Type_II',
 6: 'Obesity_Type_III',
 2: 'Overweight_Level_I',
 3: 'Overweight_Level_II'}
d_to = {'Insufficient_Weight':0,
 'Normal_Weight':1,
 'Obesity_Type_I':4,
 'Obesity_Type_II':5,
 'Obesity_Type_III':6,
 'Overweight_Level_I':2,
 'Overweight_Level_II':3}
cats = pd.Index(['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE',
       'SCC', 'CALC', 'MTRANS'],
      dtype='object')
conts = pd.Index(['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'BMI',
       'CAEC', 'CALC'],
      dtype='object')
CAEC = {'no':0,'Sometimes':1,'Frequently':2,'Always':3}
CALC = {'no':0,'Sometimes':1,'Frequently':2,'Always':2}
preprocessor = pickle.load(open("model/preprocessor.pkl", 'rb'))

test[cats] = test[cats].apply(lambda x: x.astype('category'))
test["BMI"] = test.Weight/(test.Height)**2
test.CAEC = test.CAEC.map(CAEC)
test.CALC = test.CALC.map(CALC)
test.CAEC = test.CAEC.astype("int8")
test.CALC = test.CALC.astype("int8")
test.FAVC = test.FAVC.cat.codes
test.SCC = test.SCC.cat.codes
test.SMOKE = test.SMOKE.cat.codes
test.family_history_with_overweight = test.family_history_with_overweight.cat.codes
test.NCP = test.NCP.round()
test = preprocessor.transform(test)

voting_clf = pickle.load(open("model/finalized_obesity_model.sav", 'rb'))
output = voting_clf.predict(test)

prediction = np.vectorize(d.__getitem__)(output[:5])
print(prediction)