import numpy as np
import pickle

# Loading the saved Model
loaded_model = pickle.load(open('./trained_model.sav', 'rb'))

input_data = (0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0)

# Changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('You are not diagnosed with Lassa Fever')
else: 
  print('You are diagnosed with Lassa Fever')