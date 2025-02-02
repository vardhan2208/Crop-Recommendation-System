import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('Crop_recommendation.csv')

features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']

Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, target ,test_size = 0.2,random_state =42)

RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain,Ytrain)  

print(RF.predict([[90, 42, 43, 20.87, 82, 6.5, 202.9]]) )

crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 
    7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 
    12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 
    17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans", 
    21: "Chickpea", 22: "Coffee"
}

st.title("Crop Prediction System 🌾")
st.write("Enter the following details to predict the best crop for cultivation:")

# Input fields
N = st.number_input("Nitrogen Content", min_value=0.0, value=0.0, step=1.0)
P = st.number_input("Phosphorus Content", min_value=0.0, value=0.0, step=1.0)
K = st.number_input("Potassium Content", min_value=0.0, value=0.0, step=1.0)
temp = st.number_input("Temperature (°C)", min_value=-10.0, value=25.0, step=0.1)
humidity = st.number_input("Humidity (%)", min_value=0.0, value=50.0, step=0.1)
ph = st.number_input("Soil pH", min_value=0.0, value=7.0, step=0.1)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=100.0, step=1.0)

# Predict button
if st.button("Predict"):
    # Prepare the input
    feature_list = [[N, P, K, temp, humidity, ph, rainfall]]

    try:
        prediction = RF.predict(feature_list)
        # predicted_crop = crop_dict.get(prediction[0], "Unknown")
        # Display result
        if prediction:
            st.success(f"🌟 The best crop to be cultivated is: **{prediction}**!")
        else:
            st.error("Sorry, we could not determine the best crop with the provided data.")
    except Exception as e:
        st.error(f"Error occurred: {e}")