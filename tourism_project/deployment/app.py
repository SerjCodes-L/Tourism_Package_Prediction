import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "Tourism_Package_Prediction_v1.joblib")
model = joblib.load(model_path)

st.title("Tourism Package Prediction App - Serjith Paramba")
st.write("""
This application predicts the likelihood of a customer purchasing a tourism package based on the provided information.
""")


Age                             = st.number_input("Customer Age", 0, 120)
Gender                          = st.selectbox("Customer Sex", ["Male", "Female"])
MaritalStatus                   = st.selectbox("Customer Marital Status",["Simgle", "Family"])
Occupation                      = st.selectbox("Customer Occupation", ["Free Lancer","Large Business","Occupation", "Salaried", "Small Business"])
Designation                     = st.selectbox("Customer Designation", ["AVP","Executive","Manager","Senior Manager","VP"])
MonthlyIncome                   = st.number_input("Customer Monthly Income", 0, 999999)
Passport                        = st.selectbox("Customer has Passport", ["0","1"])
OwnCar                          = st.selectbox("Customer Owns Car",["0","1"])
CityTier                        = st.selectbox("Customer City Tier", ["1","2","3"])
NumberOfTrips                   = st.number_input("Number Of Trips", 0, 100)
prodTaken                       = st.selectbox("Product Purchased in past?", ["0","1"])
NumberOfPersonVisiting          = st.number_input("Number Of Person Visiting", 0, 30)
NumberOfChildrenVisiting        = st.number_input("Number Of Children Visiting", 0, 15 )
TypeofContact                   = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
PreferredPropertyStar           = st.selectbox("Preferred Property Star",["2","3","4","5"])
ProductPitched                  = st.number_input("Product Pitched", 0, 1)
DurationOfPitch                 = st.number_input("Duration Of Pitch", 0, 24)
PitchSatisfactionScore          = st.selectbox("Pitch Satisfaction Score",["1","2","3","4","5"])
NumberOfFollowups               = st.number_input("Number Of Followups", 0, 50)

input_data = pd.DataFrame([{
"Age": Age,
"Gender": Gender,
"MaritalStatus": MaritalStatus,
"Occupation": Occupation,
"Designation":Designation,
"MonthlyIncome": MonthlyIncome,
"Passport" : Passport,
"OwnCar": OwnCar,
"CityTier": CityTier,
"NumberOfTrips": NumberOfTrips,
"prodTaken":  prodTaken,
"NumberOfPersonVisiting": NumberOfPersonVisiting,
"NumberOfChildrenVisiting": NumberOfChildrenVisiting,
"TypeofContact": TypeofContact,
"PreferredPropertyStar" : PreferredPropertyStar,
"ProductPitched" : ProductPitched,
"DurationOfPitch": DurationOfPitch,
"PitchSatisfactionScore": PitchSatisfactionScore,
"NumberOfFollowups" : NumberOfFollowups,
 }])

if st.button("Predict Tourism Package Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Customer is likely to Purchase" if prediction == 1 else "Customer is not likely to Purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
