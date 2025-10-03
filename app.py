import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st
import pickle


#LOADING THE MODEL
from tensorflow.keras.models import load_model
model=load_model('model.h5')

#LOADING THE SCALED PICKLE, ENCODED PICKLE

with open('sc.pkl','rb') as file:
  sc=pickle.load(file)

with open('le.pkl','rb') as file:
  le=pickle.load(file)

with open('oh.pkl','rb') as file:
  oh=pickle.load(file)
  
#STREAMLIT APP

st.title('Customer Churn Prediction')

#USER INPUT
credit_score=st.number_input('Credit Score')
geography=st.selectbox('Geography',oh.categories_[0]) # Choosing between France, Germany, Spain
gender=st.selectbox('Gender',le.classes_)
age=st.slider('Age',0,100)
tenure=st.slider('Tenure',0,10)
balance=st.number_input('Balance')
products=st.slider('Number of Products',1,4)
card=st.selectbox('Has Credit Card',[0,1])
active=st.selectbox('Is Active Member',[0,1])
salary=st.number_input('Estimated Salary')

#Converting user input into data frame
input=pd.DataFrame({'CreditScore':[credit_score],'Geography':[geography],'Gender':[gender],'Age':[age],'Tenure':[tenure],'Balance':[balance],'NumOfProducts':[products],'HasCrCard':[card],'IsActiveMember':[active],'EstimatedSalary':[salary]})


#CONVERTING CATEGORICAL VALUE TO NUMERICAL
input['Gender']=le.transform(input['Gender'])
geo=oh.transform(input[['Geography']])
input.drop(['Geography'], axis=1,inplace=True)
input[['France','Germany','Spain']]=geo.toarray()

#SCALING THE DATA
input=sc.transform(input)

#MAKING PREDICTIONS
pred=model.predict(input)
st.write(f"Prediction Probability: {pred[0][0]}")


if pred[0][0] > 0.5:
      st.write('Customer is likely to churn')
else:
  st.write('Customer is not likely to churn')


  