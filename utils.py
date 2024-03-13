import pandas as pd
import pickle

import streamlit as st

def process_data(df,categories):
      """
      Processes user input. Carries out ordinal label encoding, one-hot encoding and scaling of continuous values. 
      """
      CAEC = {'no':0,'Sometimes':1,'Frequently':2,'Always':3}
      CALC = {'no':0,'Sometimes':1,'Frequently':2,'Always':2}
      
      output_df = df.copy()
      preprocessor = pickle.load(open("model/preprocessor.pkl", 'rb'))

      output_df[categories] = output_df[categories].apply(lambda x: x.astype('category'))
      output_df["BMI"] = output_df.Weight/(output_df.Height)**2
      output_df.CAEC = output_df.CAEC.map(CAEC)
      output_df.CALC = output_df.CALC.map(CALC)
      output_df.CAEC = output_df.CAEC.astype("int8")
      output_df.CALC = output_df.CALC.astype("int8")
      output_df.FAVC = output_df.FAVC.cat.codes
      output_df.SCC = output_df.SCC.cat.codes
      output_df.SMOKE = output_df.SMOKE.cat.codes
      output_df.family_history_with_overweight = output_df.family_history_with_overweight.cat.codes
      output_df.NCP = output_df.NCP.round()
      output_df = preprocessor.transform(output_df)

      return output_df

def height_valid(string):
      try:
            float(string)
      except ValueError:
            return False
      if float(string) < 1.45 or float(string) > 1.98:
            return False
      else:
            return True
      
def weight_valid(string):
      try:
            float(string)
      except ValueError:
            return False
      if float(string) < 39.0 or float(string) > 173.0:
            return False
      else:
            return True

def user_input_features():
      """
      Handles user input on streamlit and generates a dataframe of the inputs.
      """
      Gender = st.sidebar.selectbox('Gender',('Male','Female'))
      Age = st.sidebar.slider('Age (in years)', 14.0,61.0,25.0, 0.5)
      Height= st.sidebar.text_input('Height (in metres)', 1.6)
      Weight= st.sidebar.text_input('Weight (in kilograms)', 60.0)
      family_history_with_overweight= st.sidebar.selectbox('Anyone in your family with a history of being overweight?',('yes','no'))
      FAVC= st.sidebar.selectbox('Do you frequently consume high caloric foods?',('yes','no'))
      FCVC= st.sidebar.selectbox('How frequently do you consume vegetables?',('Frequently','Sometimes','Almost never'))
      NCP=  st.sidebar.selectbox('How many main meals do you have in a day?',(1,2,3,4))
      CAEC= st.sidebar.selectbox('How often do you eat in between meals?',('Frequently','Sometimes','no'))
      SMOKE=st.sidebar.selectbox('Are you a smoker?',('yes','no'))
      CH2O= st.sidebar.selectbox('How frequently do you consume (plain) water?',('Frequently','Sometimes','Almost never'))
      SCC=  'no'
      FAF=  st.sidebar.selectbox('How frequently do you engage in physical activity?',('Frequently','Sometimes','Almost never',"Never"))
      TUE=  st.sidebar.selectbox('How often do you engage with technology devices?',('Sometimes','Almost never',"Never"))
      CALC= st.sidebar.selectbox('How often do you eat consume alcohol?',('Frequently','Sometimes','no'))
      MTRANS= st.sidebar.selectbox('What is your primary mode of transportation?',('Public_Transportation',"Automobile","Walking", "Bike","Motorbike"))
      
      input_valid = height_valid(Height) and weight_valid(Weight)
      if not height_valid(Height):
            st.error("Please input a valid height value between 1.45 to 1.98 metres.")
            st.stop()
      if not weight_valid(Weight):
            st.error("Please input a valid weight value between 39 to 173 kilograms.")
            st.stop()

      freq_to_ord = {"Frequently": 3, 
                  "Sometimes":2, 
                  "Almost never":1, 
                  "Never":0}
      
      data = {
      "Gender":Gender,
      "Age":                               Age,
      "Height":                            float(Height),
      "Weight":                            float(Weight),
      "family_history_with_overweight":family_history_with_overweight,
      "FAVC":                             FAVC,
      "FCVC":                              freq_to_ord[FCVC],
      "NCP":                               NCP,
      "CAEC":                               CAEC,
      "SMOKE":                              SMOKE,
      "CH2O":                              freq_to_ord[CH2O],
      "SCC":                                SCC,
      "FAF":                               freq_to_ord[FAF],
      "TUE":                              freq_to_ord[TUE],
      "CALC":                               CALC,
      "MTRANS":                             MTRANS
}
      features = pd.DataFrame(data, index=[0])
      return features, input_valid