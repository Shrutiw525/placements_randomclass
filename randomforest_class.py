# build streamlit app to predict the placement status of a student based on the input features using the trained Random Forest Classifier model. The app should have input fields for each feature and a button to make the prediction. Display the predicted placement status
# add some ui elements to make the app more user-friendly and visually appealing. You can use Streamlit's built-in components to create a clean and interactive interface for users to input their data and see the results.
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("placement.csv")

# no need to print dataset
# no need to print dataset description
# no need to plot boxplot
# no need to print value counts of target variable
# CGPA,AptitudeScore,Projects,Internships,CodingScore

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["Placed"] = le.fit_transform(df["Placed"])
x = df.drop("Placed", axis=1)
y = df["Placed"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)
# build a streamlit app to predict the placement status of a student based on the input features using the trained Random Forest Classifier model. The app should have input fields for each feature and a button to make the prediction. Display the predicted placement status
# add some ui elements to make the app more user-friendly and visually appealing. You can use

import streamlit as st
st.set_page_config(page_title="Student Placement Predictor", page_icon="🎓", layout="centered")
st.title("🎓 Student Placement Prediction using Random Forest Classifier")
# Sidebar for info and model details
st.sidebar.header("About")
st.sidebar.info("Enter the details of the student and click Predict to see if they are likely to be placed.")
with st.sidebar.expander("Model Details"):
    st.write("""
    **Model:** Random Forest Classifier  
    **Features:** CGPA, Aptitude Score, Projects, Internships, Coding Score  
    **Trained on:** placement.csv
    """)
# Use columns for a cleaner layout
col1, col2 = st.columns(2)
with col1:
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5, help="Cumulative Grade Point Average")
    aptitude_score = st.number_input("Aptitude Score", min_value=0, max_value=100, value=80, help="Score in aptitude test")
with col2:
    projects = st.number_input("Number of Projects", min_value=0, value=2, help="Total number of projects completed")
    internships = st.number_input("Number of Internships", min_value=0, value=1, help="Total number of internships completed")
    coding_score = st.number_input("Coding Score", min_value=0, max_value=100, value=85, help="Score in coding test")
input_data = [[cgpa, aptitude_score, projects, internships, coding_score]]
colA, colB = st.columns([1,1])
predict_clicked = colA.button("Predict", use_container_width=True)
reset_clicked = colB.button("Reset", use_container_width=True)
if reset_clicked:
    st.experimental_rerun()
if predict_clicked:
    # Input validation
    if cgpa < 0 or cgpa > 10 or aptitude_score < 0 or aptitude_score > 100 or coding_score < 0 or coding_score > 100:
        st.warning("Please enter valid values for CGPA, Aptitude Score, and Coding Score.")
    else:
        prediction = model.predict(input_data)
        placement_status = "Placed" if prediction[0] == 1 else "Not Placed"
        if placement_status == "Placed":
            st.success(f"Predicted Placement Status: {placement_status}")
        else:
            st.error(f"Predicted Placement Status: {placement_status}")