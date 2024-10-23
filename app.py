<<<<<<< HEAD
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.models import load_model 
import pickle

##load the trained model

model = load_model('model.h5')
## load the model, scalar pickle, onehot

model = load_model('model.h5')

with open ('onehor_encoded_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)
    
with open ('scalar.pkl','rb') as file:
    scaler = pickle.load(file)
    
with open ('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

## steamlit app
st.title("customer churn prediction")

## user input
CreditScore = st.number_input('Credit score')
Geography = st.selectbox('Geography',onehot_encoder_geo.categories_[0])
Gender = st.selectbox('Gender',label_encoder_gender.classes_)
Age = st.slider('Age',18,48)
Tenure = st.slider('Tenure',0,10)
Balance = st.number_input("Balance")
NumOfProducts = st.slider('Num Of Products',1,4)
HasCrCard = st.selectbox('Has Cr Card',[0,1])
IsActiveMember = st.selectbox('Is an Active Member',[0,1])
EstimatedSalary = st.number_input("Estimated Salary")


inputdata = {
    'CreditScore' : [CreditScore],
    'Geography':[Geography],
    'Gender':[Gender],
    'Age':[Age],
    'Tenure':[Tenure],
    'Balance':[Tenure],
    'NumOfProducts':[NumOfProducts],
    'HasCrCard':[HasCrCard],
    'IsActiveMember':[IsActiveMember],
    'EstimatedSalary':[EstimatedSalary]
}

geo_encoded = onehot_encoder_geo.transform([[Geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']) )

input_df = pd.DataFrame(inputdata)

input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])
inputdata = pd.concat([input_df.drop('Geography',axis=1) ,geo_encoded_df], axis=1)

inputdata_scaled = scaler.transform(inputdata)

## prediction churn
prediction = model.predict(inputdata_scaled)
pred_prob = prediction[0][0]

st.write(f"the churn probability {pred_prob:.4f}")

if pred_prob >.5:
    st.write("Customer is likely to churn")
else:
    st.write("Customer is not likely to churn")
=======
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.models import load_model 
import pickle

##load the trained model

model = load_model('model.h5')
## load the model, scalar pickle, onehot

model = load_model('model.h5')

with open ('onehor_encoded_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)
    
with open ('scalar.pkl','rb') as file:
    scaler = pickle.load(file)
    
with open ('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

## steamlit app
st.title("customer churn prediction")

## user input
CreditScore = st.number_input('Credit score')
Geography = st.selectbox('Geography',onehot_encoder_geo.categories_[0])
Gender = st.selectbox('Gender',label_encoder_gender.classes_)
Age = st.slider('Age',18,48)
Tenure = st.slider('Tenure',0,10)
Balance = st.number_input("Balance")
NumOfProducts = st.slider('Num Of Products',1,4)
HasCrCard = st.selectbox('Has Cr Card',[0,1])
IsActiveMember = st.selectbox('Is an Active Member',[0,1])
EstimatedSalary = st.number_input("Estimated Salary")


inputdata = {
    'CreditScore' : [CreditScore],
    'Geography':[Geography],
    'Gender':[Gender],
    'Age':[Age],
    'Tenure':[Tenure],
    'Balance':[Tenure],
    'NumOfProducts':[NumOfProducts],
    'HasCrCard':[HasCrCard],
    'IsActiveMember':[IsActiveMember],
    'EstimatedSalary':[EstimatedSalary]
}

geo_encoded = onehot_encoder_geo.transform([[Geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']) )

input_df = pd.DataFrame(inputdata)

input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])
inputdata = pd.concat([input_df.drop('Geography',axis=1) ,geo_encoded_df], axis=1)

inputdata_scaled = scaler.transform(inputdata)

## prediction churn
prediction = model.predict(inputdata_scaled)
pred_prob = prediction[0][0]

st.write(f"the churn probability {pred_prob:.4f}")

if pred_prob >.5:
    st.write("Customer is likely to churn")
else:
    st.write("Customer is not likely to churn")
>>>>>>> 23f9b04cd905f9fc14907236bafa793d445c0087
