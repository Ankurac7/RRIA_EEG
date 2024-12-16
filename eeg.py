# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 18:15:01 2021

@author: ankur
"""

import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('C:\\Users\\ANKUR\\Downloads\\trained_model.sav', 'rb'))


# creating a function for Prediction
def eeg(input_data):
    #changing the input data as numpy array
    input_data_as_numpy_array=np.asarray(input_data)

    #reshape the array as we arre predicting for one instance
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)
    if (prediction[0]==0):
        return "The student has learned"
    else:
        return "The student hasn't learned"

  
    
  
def main():
    # giving a title
    st.title('EEG Project Web App')
    
    # getting the input data from the user
    SubjectID = st.text_input('Subject ID')
    VideoID = st.text_input('VideoID')
    Attention = st.text_input('Attention value')
    Mediation = st.text_input('Mediation value')
    Raw = st.text_input('Raw level')
    Delta = st.text_input('Delta value')
    Theta = st.text_input('Theta value')
    Alpha1 = st.text_input('Alpha1 value')
    Alpha2 = st.text_input('Alpha2 value')
    Beta1 = st.text_input('Beta1 value')
    Beta2 = st.text_input('Beta2 value')
    Gamma1 = st.text_input('Gamma1 value')
    Gamma2 = st.text_input('Gamma2 value')
    userdefinedlabeln = st.text_input('User Defined Label')
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    if st.button('EEG Test Result'):
        diagnosis = eeg([SubjectID,VideoID,Attention,Mediation,Raw,Delta,Theta,Alpha1,Alpha2,Beta1,Beta2,Gamma1,Gamma2,userdefinedlabeln])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()