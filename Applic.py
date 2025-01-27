import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page setup
st.set_page_config(page_title="Cardiovascular Disease Prediction", layout="wide")

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", ["Home", "Data Information", "Prediction"])

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

# Define the exact features used during training
TRAINING_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "thalach", 
    "exang", "oldpeak", "slope", "ca"
]

# Page content
if page == "Home":
    image_url = "https://raw.githubusercontent.com/Papaxristos/Cardio-disease-prediction/refs/heads/main/heart-attack.webp"
    st.image(image_url, use_container_width=True, caption="Cardiovascular Health")
    st.write(""" Cardiovascular diseases (CVDs) are the leading cause of death globally, taking an estimated 17.9 million lives each year. CVDs are a group of disorders of the heart and blood vessels and include coronary heart disease, cerebrovascular disease, rheumatic heart disease and other conditions. More than four out of five CVD deaths are due to heart attacks and strokes, and one third of these deaths occur prematurely in people under 70 years of age.

The most important behavioural risk factors of heart disease and stroke are unhealthy diet, physical inactivity, tobacco use and harmful use of alcohol. Amongst environmental risk factors, air pollution is an important factor. The effects of behavioural risk factors may show up in individuals as raised blood pressure, raised blood glucose, raised blood lipids, and overweight and obesity. These “intermediate risks factors” can be measured in primary care facilities and indicate an increased risk of heart attack, stroke, heart failure and other complications.

Cessation of tobacco use, reduction of salt in the diet, eating more fruit and vegetables, regular physical activity and avoiding harmful use of alcohol have been shown to reduce the risk of cardiovascular disease. Health policies that create conducive environments for making healthy choices affordable and available, as well as improving air quality and reducing pollution, are essential for motivating people to adopt and sustain healthy behaviours.

Identifying those at highest risk of CVDs and ensuring they receive appropriate treatment can prevent premature deaths. Access to noncommunicable disease medicines and basic health technologies in all primary health care facilities is essential to ensure that those in need receive treatment and counselling.""")

elif page == "Data Information":
    st.subheader("Data Information")
    st.write("""This data set dates from 1988 and consists of four databases: Cleveland, Hungary, Switzerland, and Long Beach V. It contains 76 attributes, including the predicted attribute, but all published experiments refer to using a subset of 14 of them. The "target" field refers to the presence of heart disease in the patient. It is integer valued 0 = no disease and 1 = disease.""")
    
    # Load a sample dataset
    df = pd.DataFrame({
        "age": [52, 53, 70, 61, 62, 58, 58, 55, 46, 54, 71, 43, 51, 52],
        "sex": ["male", "male", "female", "female", "male", "male", "female", "female", "male", "male", "female", "female", "male", "male"],
        "trestbps": [125, 140, 145, 148, 138, 100, 114, 160, 120, 122, 112, 132, 140, 128],
        "chol": [212, 203, 174, 203, 294, 248, 318, 289, 249, 286, 149, 341, 298, 204],
        "thalach": [168, 155, 125, 161, 106, 122, 140, 145, 144, 116, 125, 136, 122, 156],
        "target": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
    })
    
    # Visualizations
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.histplot(df['age'], kde=False, bins=10, ax=ax[0], color="blue", edgecolor="black")
    ax[0].set_title("Age Distribution")
    sns.histplot(df['chol'], kde=False, bins=10, ax=ax[1], color="red", edgecolor="black")
    ax[1].set_title("Cholesterol Distribution")
    st.pyplot(fig)

    st.write("Sample data used for visualization:")
    st.table(df)

elif page == "Prediction":
    st.subheader("Make a Prediction")
    st.write("Adjust the settings below to enter the data:")

    # Collect user input with detailed descriptions
    input_data = {
        "age": st.slider("Age (years): The age of the individual.", 20, 100, step=1),
        "sex": st.radio("Gender: Select 'male' or 'female'.", options=["female", "male"], index=0),
        "cp": st.slider("Chest Pain Type (1-4): A rating for chest pain intensity (1: minimal, 4: severe).", 1, 4, step=1),
        "trestbps": st.slider("Resting Blood Pressure (mm Hg): Blood pressure when the individual is at rest.", 80, 200, step=1),
        "chol": st.slider("Cholesterol (mg/dl): Total cholesterol level in the blood.", 100, 400, step=1),
        "thalach": st.slider("Max Heart Rate (bpm): Maximum heart rate achieved during exercise.", 60, 220, step=1),
        "exang": st.radio("Exercise Induced Angina: Did exercise cause chest pain? (0: No, 1: Yes)", options=[0, 1], index=0),
        "oldpeak": st.slider("ST Depression (oldpeak): Depression in ST segment caused by exercise.", 0.0, 6.0, step=0.1),
        "slope": st.radio("Slope of ST: Slope of the peak exercise ST segment (0: Downsloping, 1: Flat, 2: Upsloping)", options=[0, 1, 2], index=0),
        "ca": st.slider("Number of Major Vessels (ca): Number of vessels colored by fluoroscopy (0-4).", 0, 4, step=1)
    }

    # Convert categorical input to numeric
    input_data["sex"] = 1 if input_data["sex"] == "male" else 0

    # Create a DataFrame from user input
    input_df = pd.DataFrame([input_data])

    # Load sample data from the "Data Information" section
    df = pd.DataFrame({
        "age": [52, 53, 70, 61, 62, 58, 58, 55, 46, 54, 71, 43, 51, 52],
        "sex": [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        "trestbps": [125, 140, 145, 148, 138, 100, 114, 160, 120, 122, 112, 132, 140, 128],
        "chol": [212, 203, 174, 203, 294, 248, 318, 289, 249, 286, 149, 341, 298, 204],
        "thalach": [168, 155, 125, 161, 106, 122, 140, 145, 144, 116, 125, 136, 122, 156],
        "exang": [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        "oldpeak": [0.0, 1.4, 1.3, 1.6, 3.6, 2.2, 1.0, 0.6, 0.0, 1.2, 1.9, 0.1, 0.4, 0.3],
        "slope": [2, 2, 1, 2, 1, 2, 1, 0, 2, 2, 0, 1, 2, 2],
        "ca": [0, 0, 1, 2, 0, 1, 1, 2, 0, 1, 1, 0, 1, 1]
    })

    # Visualization: User Input vs Sample Data
    st.write("### Comparison of User Input with Sample Data")
    features_to_plot = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
    
    for feature in features_to_plot:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(df[feature], kde=True, color="blue", label="Sample Data", ax=ax)
        ax.axvline(input_data[feature], color="red", linestyle="--", label=f"User Input ({input_data[feature]})")
        ax.set_title(f"Distribution of {feature} with User Input")
        ax.set_xlabel(f"{feature} values")
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)

    # Check if features match the training data
    if set(input_df.columns) == set(TRAINING_FEATURES):
        try:
            # Make prediction
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]
            st.write(f"The predicted outcome is: **{'Disease' if prediction == 1 else 'No Disease'}**")
            st.write(f"Probability of No Disease: **{probabilities[0]:.2f}**")
            st.write(f"Probability of Disease: **{probabilities[1]:.2f}**")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.error("The input features do not match the features used during training.")


