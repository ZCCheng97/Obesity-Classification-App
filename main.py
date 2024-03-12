import pandas as pd
import numpy as np
from utils import *

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings(action='ignore', category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=UserWarning)

st.write("""
# Obesity Risk Prediction

This app predicts your obesity risk. Fill in your details in the sidebar. Predictions are generated in real time!

Data obtained from [Obesity or CVD risk](https://www.kaggle.com/datasets/aravindpcoder/obesity-or-cvd-risk-classifyregressorcluster) on Kaggle. 
Original Authors: Fabio Mendoza Palechor and Alexis de la Hoz Manotas, from the Universidad de la Costa, CUC, Colombia.
""")

st.sidebar.header('Input your personal datails:')

test = user_input_features()

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

test = process_data(test, cats)
voting_clf = pickle.load(open("model/finalized_obesity_model.sav", 'rb'))
output = voting_clf.predict(test)

prediction = np.vectorize(d.__getitem__)(output)

st.subheader('Prediction')
weight_cats = np.array([v for _, v in sorted(d.items())])
st.write(weight_cats[output])