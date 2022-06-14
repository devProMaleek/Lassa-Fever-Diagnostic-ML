import numpy as np
import streamlit as st
import pickle

# Loading the saved Model
loaded_model = pickle.load(
    open('trained_model.sav', 'rb'))

# Creating a function for Prediction


def lassa_fever_prediction(input_data):
    # Changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'You are not diagnosed with Lassa Fever'
    else:
        return 'You are diagnosed with Lassa Fever'

# Convert User report 
def convert_userInput_into_number(fieldName):
    if fieldName == "Yes":
        fieldName = 1
    else:
        fieldName = 0
    return fieldName

def main():
    # Page Title
    st.header   ('Welcome to Lassa Fever Diagnostic System')

    # Getting the input from the user
    
    col1, col2 = st.columns(2)
    
    with col1:
        breathing_problem = st.radio(
            "Are you experiencing Breathing Problem?",
            ('Yes', 'No'), horizontal=True)
        breathing_problem = convert_userInput_into_number(breathing_problem)
        
        fever = st.radio(
            "Are you experiencing Serious Fever?",
            ('Yes', 'No'), horizontal=True)
        fever = convert_userInput_into_number(fever)

        dry_cough = st.radio(
            "Do you have Dry Cough?",
            ('Yes', 'No'), horizontal=True)
        dry_cough = convert_userInput_into_number(dry_cough)

        sore_throat = st.radio(
            "Do you have Sore Throat?",
            ('Yes', 'No'), horizontal=True)
        sore_throat = convert_userInput_into_number(sore_throat)

        running_nose = st.radio(
            "Do you have Runny Nose?",
            ('Yes', 'No'), horizontal=True)
        running_nose = convert_userInput_into_number(running_nose)

        chest_pain = st.radio(
            "Do you have Chest Pain?",
            ('Yes', 'No'), horizontal=True)
        chest_pain = convert_userInput_into_number(chest_pain)
        
        headache = st.radio(
            "Are you experiencing Headache?",
            ('Yes', 'No'), horizontal=True)
        headache = convert_userInput_into_number(headache)
        
    with col2:

        Diabetes = st.radio(
            "Are you Diabetic?",
            ('Yes', 'No'), horizontal=True)
        Diabetes = convert_userInput_into_number(Diabetes)

        hypertension = st.radio(
            "Are you hypertensive?",
            ('Yes', 'No'), horizontal=True)
        hypertension = convert_userInput_into_number(hypertension)

        fatigue = st.radio(
            "Are you experiencing Fatigue(Weakness)?",
            ('Yes', 'No'), horizontal=True)
        fatigue = convert_userInput_into_number(fatigue)

        diarrhoea = st.radio(
            "Do you have Diarrhoea?",
            ('Yes', 'No'), horizontal=True)
        diarrhoea = convert_userInput_into_number(diarrhoea)

        vomiting = st.radio(
            "Are you Vomiting?",
            ('Yes', 'No'), horizontal=True)
        vomiting = convert_userInput_into_number(vomiting)

        hearing_loss = st.radio(
            "Do you have Hearing Problem?",
            ('Yes', 'No'), horizontal=True  )
        hearing_loss = convert_userInput_into_number(hearing_loss)

    # All User Input
    user_input = [breathing_problem, fever, dry_cough, sore_throat, running_nose, chest_pain,
                  headache, Diabetes, hypertension, fatigue, diarrhoea, vomiting, hearing_loss]
   
    # Code for prediction
    diagnosis = ''

    # Prediction Button
    btn = st.button("Lassa Fever Test Result")

    if btn:
        diagnosis = lassa_fever_prediction(user_input)
    
    # Printing the Results
    st.success(diagnosis)


        
if __name__ == '__main__':
    main() 
