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

def user_input_features():
        """
        Handles user input on streamlit and generates a dataframe of the inputs.
        """
        Gender = st.sidebar.selectbox('Gender',('Male','Female'))
        Age = st.sidebar.slider('Age (in years)', 14.0,61.0,25.0)
        Height= st.sidebar.slider('Height (in metres)', 1.45, 1.98, 1.6)
        Weight= st.sidebar.slider('Weight (in kilograms)', 39.0,173.0,60.0)
        family_history_with_overweight= st.sidebar.selectbox('Anyone in your family with a history of being overweight?',('yes','no'))
        FAVC= st.sidebar.selectbox('Do you frequently consume high caloric foods?',('yes','no'))
        FCVC= st.sidebar.selectbox('On a scale of 1 to 3, how frequently do you consume vegetables?',(1,2,3))
        NCP=  st.sidebar.selectbox('How many main meals do you have in a day?',(1,2,3,4))
        CAEC= st.sidebar.selectbox('How often do you eat in between meals?',('Frequently','Sometimes','no'))
        SMOKE=st.sidebar.selectbox('Are you a smoker?',('yes','no'))
        CH2O= st.sidebar.selectbox('On a scale of 1 to 3, how frequently do you consume (plain) water?',(1,2,3))
        SCC=  'no'
        FAF=  st.sidebar.selectbox('On a scale of 0 to 3, how frequently do you engage in physical activity?',(0,1,2,3))
        TUE=  st.sidebar.selectbox('On a scale of 0 to 2, how often do you engage with technology devices?',(0,1,2))
        CALC= st.sidebar.selectbox('How often do you eat consume alcohol?',('Frequently','Sometimes','no'))
        MTRANS= st.sidebar.selectbox('What is your primary mode of transportation?',('Public_Transportation',"Automobile","Walking", "Bike","Motorbike"))
        data = {
            "Gender":Gender,
            "Age":                               Age,
            "Height":                            Height,
            "Weight":                            Weight,
            "family_history_with_overweight":family_history_with_overweight,
            "FAVC":                             FAVC,
            "FCVC":                              FCVC,
            "NCP":                               NCP,
            "CAEC":                               CAEC,
            "SMOKE":                              SMOKE,
            "CH2O":                              CH2O,
            "SCC":                                SCC,
            "FAF":                               FAF,
            "TUE":                              TUE,
            "CALC":                               CALC,
            "MTRANS":                             MTRANS
}
        features = pd.DataFrame(data, index=[0])
        return features