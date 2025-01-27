import streamlit as st
import numpy as np
import joblib

# Load your trained model
model = joblib.load('xgboost_model_for_house_price.joblib')

# List of all address features (same as during training)
address_features =['Shahran', 'Pardis', 'Shahrake Qods', 'Shahrake Gharb',
       'North Program Organization', 'Andisheh', 'West Ferdows Boulevard',
       'Narmak', 'Saadat Abad', 'Zafar', 'Islamshahr', 'Pirouzi',
       'Shahrake Shahid Bagheri', 'Moniriyeh', 'Velenjak', 'Amirieh',
       'Southern Janatabad', 'Salsabil', 'Zargandeh', 'Feiz Garden',
       'Water Organization', 'ShahrAra', 'Gisha', 'Ray', 'Abbasabad',
       'Ostad Moein', 'Farmanieh', 'Parand', 'Punak', 'Qasr-od-Dasht',
       'Aqdasieh', 'Pakdasht', 'Railway', 'Central Janatabad',
       'East Ferdows Boulevard', 'Pakdasht KhatunAbad', 'Sattarkhan',
       'Baghestan', 'Shahryar', 'Northern Janatabad', 'Daryan No',
       'Southern Program Organization', 'Rudhen', 'West Pars', 'Afsarieh',
       'Marzdaran', 'Dorous', 'Sadeghieh', 'Chahardangeh', 'Baqershahr',
       'Jeyhoon', 'Lavizan', 'Shams Abad', 'Fatemi',
       'Keshavarz Boulevard', 'Kahrizak', 'Qarchak',
       'Northren Jamalzadeh', 'Azarbaijan', 'Bahar',
       'Persian Gulf Martyrs Lake', 'Beryanak', 'Heshmatieh',
       'Elm-o-Sanat', 'Golestan', 'Shahr-e-Ziba', 'Pasdaran',
       'Chardivari', 'Gheitarieh', 'Kamranieh', 'Gholhak', 'Heravi',
       'Hashemi', 'Dehkade Olampic', 'Damavand', 'Republic', 'Zaferanieh',
       'Qazvin Imamzadeh Hassan', 'Niavaran', 'Valiasr', 'Qalandari',
       'Amir Bahador', 'Ekhtiarieh', 'Ekbatan', 'Absard', 'Haft Tir',
       'Mahallati', 'Ozgol', 'Tajrish', 'Abazar', 'Koohsar', 'Hekmat',
       'Parastar', 'Lavasan', 'Majidieh', 'Southern Chitgar', 'Karimkhan',
       'Si Metri Ji', 'Karoon', 'Northern Chitgar', 'East Pars', 'Kook',
       'Air force', 'Sohanak', 'Komeil', 'Azadshahr', 'Zibadasht',
       'Amirabad', 'Dezashib', 'Elahieh', 'Mirdamad', 'Razi', 'Jordan',
       'Mahmoudieh', 'Shahedshahr', 'Yaftabad', 'Mehran', 'Nasim Shahr',
       'Tenant', 'Chardangeh', 'Fallah', 'Eskandari', 'Shahrakeh Naft',
       'Ajudaniye', 'Tehransar', 'Nawab', 'Yousef Abad',
       'Northern Suhrawardi', 'Villa', 'Hakimiyeh', 'Nezamabad',
       'Garden of Saba', 'Tarasht', 'Azari', 'Shahrake Apadana', 'Araj',
       'Vahidieh', 'Malard', 'Shahrake Azadi', 'Darband', 'Vanak',
       'Tehran Now', 'Darabad', 'Eram', 'Atabak', 'Sabalan', 'SabaShahr',
       'Shahrake Madaen', 'Waterfall', 'Ahang', 'Salehabad', 'Pishva',
       'Enghelab', 'Islamshahr Elahieh', 'Ray - Montazeri',
       'Firoozkooh Kuhsar', 'Ghoba', 'Mehrabad', 'Southern Suhrawardi',
       'Abuzar', 'Dolatabad', 'Hor Square', 'Taslihat', 'Kazemabad',
       'Robat Karim', 'Ray - Pilgosh', 'Ghiyamdasht', 'Telecommunication',
       'Mirza Shirazi', 'Gandhi', 'Argentina', 'Seyed Khandan',
       'Shahrake Quds', 'Safadasht', 'Khademabad Garden', 'Hassan Abad',
       'Chidz', 'Khavaran', 'Boloorsazi', 'Mehrabad River River',
       'Varamin - Beheshti', 'Shoosh', 'Thirteen November', 'Darakeh',
       'Aliabad South', 'Alborz Complex', 'Firoozkooh', 'Vahidiyeh',
       'Shadabad', 'Naziabad', 'Javadiyeh', 'Yakhchiabad']

# Page title
st.title("House Price Prediction in Tehran")
st.write("Enter the details of the house below to get the predicted price.")

# Input fields
area = st.number_input("Area (sqm):", min_value=1, value=50)
room = st.number_input("Number of Rooms:", min_value=1, value=1)
parking = st.selectbox("Parking:", ["Yes", "No"])
warehouse = st.selectbox("Warehouse:", ["Yes", "No"])
elevator = st.selectbox("Elevator:", ["Yes", "No"])
address = st.selectbox("Address:", address_features)

# Convert categorical inputs to numerical
parking_num = 1 if parking == "Yes" else 0
warehouse_num = 1 if warehouse == "Yes" else 0
elevator_num = 1 if elevator == "Yes" else 0

# Create address feature vector
address_vector = [1 if addr == address else 0 for addr in address_features]

# Predict button
if st.button("Predict Price"):
    # Prepare input for the model
    input_features = np.array([[area, room, parking_num, warehouse_num, elevator_num] + address_vector])

    # Predict
    predicted_price = model.predict(input_features)[0]

    # Display result
    st.success(f"The predicted price of the house in {address} is approximately {predicted_price:,.0f} Toman.")