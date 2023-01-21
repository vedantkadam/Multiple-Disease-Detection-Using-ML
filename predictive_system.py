import streamlit as st
import numpy as np
import pickle


## load a saved model
final_model = pickle.load(open('C:/Users/vedantkadam/Desktop/Disease Detection Using ML/final_model.sav','rb'))


## creating function for prediction

def diabetes_prediction(input_data):
    input_data = (5,166,72,19,175,25.8,0.587,51)

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = final_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person is not diabetic'
        
    else:
        return 'The person is  diabetic'


def main():

    st.title("Diabetes detection 🩸")

    Pregnancies =st.text_input("Number of Pregnancies")
    Glucose =st.text_input("Enter Glucose Vlaue")
    BloodPressure =st.text_input("Enter BloodPressure Result")
    SkinThickness =st.text_input("Mention your SkinThickness")
    Insulin =st.text_input("Insulin level")
    BMI =st.text_input("BMI of your Body")
    DiabetesPedigreeFunction =st.text_input("Number of DiabetesPedigreeFunction")
    Age =st.text_input("What is your Age")

    # code for prediction

    diagnosis = ""

    # craeting button for prediction
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    

    st.success(diagnosis)



if __name__ == '__main__':
    main()
