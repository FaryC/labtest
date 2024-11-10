import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved model and encoders
with open('model_penguin_65130701937.pkl', 'rb') as file:
    model, species_encoder, island_encoder, sex_encoder = pickle.load(file)

# Streamlit app
st.title("Penguin Species Prediction")

# Input features
island = st.selectbox("Island", island_encoder.classes_)
culmen_length_mm = st.number_input("Culmen Length (mm)", min_value=0.0)
culmen_depth_mm = st.number_input("Culmen Depth (mm)", min_value=0.0)
flipper_length_mm = st.number_input("Flipper Length (mm)", min_value=0.0)
body_mass_g = st.number_input("Body Mass (g)", min_value=0.0)
sex = st.selectbox("Sex", sex_encoder.classes_)

# Create a dataframe for prediction
new_data = pd.DataFrame({
    'island': [island],
    'culmen_length_mm': [culmen_length_mm],
    'culmen_depth_mm': [culmen_depth_mm],
    'flipper_length_mm': [flipper_length_mm],
    'body_mass_g': [body_mass_g],
    'sex': [sex]
})

# Encode categorical features
new_data['island'] = island_encoder.transform(new_data['island'])
new_data['sex'] = sex_encoder.transform(new_data['sex'])

# Make prediction
prediction = model.predict(new_data)
predicted_species = species_encoder.inverse_transform(prediction)[0]

# Display the prediction
st.write(f"Predicted Penguin Species: {predicted_species}")
