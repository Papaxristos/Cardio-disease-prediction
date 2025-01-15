import streamlit as st
import pandas as pd
import joblib

# Page setup
st.set_page_config(page_title="Cardiovascular Disease Prediction", layout="wide")

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", ["Home", "Data Information", "Prediction"])

# Page content
if page == "Home":
    image_url = "https://raw.githubusercontent.com/Papaxristos/Cardio-disease-prediction/refs/heads/main/heart-attack.webp"

    st.image(image_url, use_container_width=True, caption="Cardiovascular Health")
    st.write("""
    ### Cardiovascular Diseases (CVDs)
    Cardiovascular diseases (CVDs) are the leading cause of death globally, taking an estimated 17.9 million lives each year. CVDs are a group of disorders of the heart and blood vessels and include coronary heart disease, cerebrovascular disease, rheumatic heart disease and other conditions. More than four out of five CVD deaths are due to heart attacks and strokes, and one third of these deaths occur prematurely in people under 70 years of age.

    The most important behavioural risk factors of heart disease and stroke are unhealthy diet, physical inactivity, tobacco use and harmful use of alcohol. Amongst environmental risk factors, air pollution is an important factor. The effects of behavioural risk factors may show up in individuals as raised blood pressure, raised blood glucose, raised blood lipids, and overweight and obesity. These “intermediate risks factors” can be measured in primary care facilities and indicate an increased risk of heart attack, stroke, heart failure and other complications.

    Cessation of tobacco use, reduction of salt in the diet, eating more fruit and vegetables, regular physical activity and avoiding harmful use of alcohol have been shown to reduce the risk of cardiovascular disease. Health policies that create conducive environments for making healthy choices affordable and available, as well as improving air quality and reducing pollution, are essential for motivating people to adopt and sustain healthy behaviours.

    Identifying those at highest risk of CVDs and ensuring they receive appropriate treatment can prevent premature deaths. Access to noncommunicable disease medicines and basic health technologies in all primary health care facilities is essential to ensure that those in need receive treatment and counselling.
    """)

elif page == "Data Information":
    st.subheader("Data Information")
    st.write("""
    This data set dates from 1988 and consists of four databases: Cleveland, Hungary, Switzerland, and Long Beach V. It contains 76 attributes, including the predicted attribute, but all published experiments refer to using a subset of 14 of them. The "target" field refers to the presence of heart disease in the patient. It is integer valued 0 = no disease and 1 = disease.
    """)
    
    data = {
        "age": [52.0, 53.0, 70.0, 61.0, 62.0, 58.0, 58.0, 55.0, 46.0, 54.0, 71.0, 43.0, 51.0, 52.0],
        "trestbps": [125.0, 140.0, 145.0, 148.0, 138.0, 100.0, 114.0, 160.0, 120.0, 122.0, 112.0, 132.0, 140.0, 128.0],
        "chol": [212.0, 203.0, 174.0, 203.0, 294.0, 248.0, 318.0, 289.0, 249.0, 286.0, 149.0, 341.0, 298.0, 204.0],
        "thalach": [168.0, 155.0, 125.0, 161.0, 106.0, 122.0, 140.0, 145.0, 144.0, 116.0, 125.0, 136.0, 122.0, 156.0],
        "target": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "cp_1": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "cp_2": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "cp_3": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "exang_1": [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        "slope_1": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "slope_2": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    df = pd.DataFrame(data)
    st.table(df)

elif page == "Prediction":
    st.subheader("Make a Prediction")
    st.write("Adjust the settings below to enter the data:")

    # Collect user inputs
    input_data = {
        "age": st.slider("Age (years)", 20, 100, step=1),
        "sex": st.radio("Gender", options=["female", "male"], index=0),
        "cp": st.slider("Chest Pain Type (1-4)", 1, 4, step=1),
        "trestbps": st.slider("Resting Blood Pressure (mm Hg)", 80, 200, step=1),
        "chol": st.slider("Cholesterol (mg/dl)", 100, 400, step=1),
        "fbs": st.radio("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], index=0),
        "restecg": st.radio("Resting ECG", options=[0, 1, 2], index=0),
        "thalach": st.slider("Max Heart Rate (bpm)", 60, 220, step=1),
        "exang": st.radio("Exercise Induced Angina", options=[0, 1], index=0),
        "oldpeak": st.slider("Depression Induced by Exercise (oldpeak)", 0.0, 6.0, step=0.1),
        "slope": st.radio("Slope of ST", options=[0, 1, 2], index=0),
        "ca": st.slider("Number of Major Vessels (ca)", 0, 4, step=1),
        "thal": st.slider("Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)", 1, 3, step=1)
    }

    # Load model
    @st.cache_resource
    def load_model():
        return joblib.load('model.pkl')

    model = load_model()

    # Prediction
    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        
        # Select only expected features
        expected_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                            'thalach', 'exang', 'oldpeak']
        
        input_df = input_df[expected_features]
        
        # Convert gender to numeric
        input_df['sex'] = input_df['sex'].apply(lambda x: 1 if x.lower() == "male" else 0)

        try:
            # Compute probability
            prediction_prob = model.predict_proba(input_df)[:, 1][0] * 100
            st.write(f"Probability of cardiovascular disease: {prediction_prob:.2f}%")
            if prediction_prob > 50:
                st.error("High risk of cardiovascular disease!")
            else:
                st.success("Low risk of cardiovascular disease!")
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
