from tensorflow.keras.models import load_model
import dill
import pandas as pd
import numpy as np
import streamlit as st

model=load_model('model.h5')

with open('preprocessor.pkl','rb') as file:
    preprocessor=dill.load(file)

st.title('diabetes_prediction')


Pregnancies=st.number_input('Pregnancies')
Glucose=st.number_input('Gluscose')
BloodPressure=st.number_input('BloodPressure') 
SkinThickness=st.number_input('SkinThickness') 
Insulin=st.number_input('Insulin') 
BMI=st.number_input('BMI') 
DiabetesPedigreeFunction=st.number_input('DiabetesPedigreeFunction') 
Age=st.number_input('Age')

user_input={
    "Pregnancies": [Pregnancies],
    "Glucose": [Glucose],
    "BloodPressure": [BloodPressure],
    "SkinThickness": [SkinThickness],
    "Insulin": [Insulin],
    "BMI": [BMI],
    "DiabetesPedigreeFunction": [DiabetesPedigreeFunction],
    "Age": [Age]
}

df=pd.DataFrame(user_input)

df=preprocessor.transform(df)

encoded_df=pd.DataFrame(df)

pred=model.predict(encoded_df)

prob=pred[0][0]

if prob>0.5:
     st.write("Diabetes found")
else:
     st.write("Diabetes not found")    






