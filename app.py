# Importing all the libraries
import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load all the encoders and scaler
with open('encode_geography.pkl','rb') as file:
    encode_geo=pickle.load(file)

with open('label_encode_gender.pkl','rb') as file:
    label_gender=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

#### Streamlit App:
st.title("Customer Churn Prediction")

# User Imput:
CreditScore = st.number_input('Credit Score')
Geography = st.selectbox('Geography', encode_geo.categories_[0])
Gender = st.selectbox('Gender', label_gender.classes_)
Age = st.slider('Age',18,92)
Tenure = st.slider('Tenure',0,10)
Balance = st.number_input('Balance')
NumOfProducts = st.slider('Number of Products',1,4)
HasCrCard = st.selectbox('Has Credit Card',[0,1])
IsActiveMember = st.selectbox('Is Active Member',[0,1])
EstimatedSalary = st.number_input('Estimated Salary')

# Prepare the input data
input_data= pd.DataFrame({
    'CreditScore': [CreditScore],
    'Gender': [label_gender.transform([Gender])[0]],
    'Age': [Age],
    'Tenure': [Tenure],
    'Balance': [Balance],
    'NumOfProducts': [NumOfProducts],
    'HasCrCard': [HasCrCard],
    'IsActiveMember': [IsActiveMember],
    'EstimatedSalary': [EstimatedSalary]
})

# One-hot encode Geography
geo_encoded= encode_geo.transform([[Geography]]).toarray()
geo_encoded_df= pd.DataFrame(geo_encoded, columns= encode_geo.get_feature_names_out(['Geography']))

# Combine the input data with one hot encoded column
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict Churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("the customer is not likely to churn.")