# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open('C:\\Users\\ANKUR\\Downloads\\trained_model.sav', 'rb'))
input_data=(0.00E+00,0.00E+00,5.60E+01,4.30E+01,2.78E+02,3.02E+05,9.06E+04,3.37E+04,2.40E+04,2.79E+04,4.51E+04,3.32E+04,8.29E+03,0.00E+00)
#changing the input data as numpy array
input_data_as_numpy_array=np.asarray(input_data)

#reshape the array as we arre predicting for one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=loaded_model.predict(input_data_reshaped)
print(prediction)
if (prediction[0]==0):
  print("The student has learned")
else:
  print("The student hasn't learned")
