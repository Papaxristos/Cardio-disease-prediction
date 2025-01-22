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
    This dataset, originating from 1988, consolidates information from four sources: Cleveland, Hungary, Switzerland, and Long Beach V. Although it includes 76 attributes in total, research often focuses on 14 key features. The target variable indicates the presence of heart disease, where "0" means no disease and "1" means disease. By analyzing these features, we aim to create a simple and intuitive tool to help predict cardiovascular disease risk and support better health decisions!
    """)
    data = {
        "age": [52, 53, 70, 61, 62, 58, 58, 55, 46, 54, 71, 43, 51, 52],
        "trestbps": [125, 140, 145, 148, 138, 100, 114, 160, 120, 122, 112, 132, 140, 128],
        "chol": [212, 203, 174, 203, 294, 248, 318, 289, 249, 286, 149, 341, 298, 204],
        "thalach": [168, 155, 125, 161, 106, 122, 140, 145, 144, 116, 125, 136, 122, 156],
        "target": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        "cp_1": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "cp_2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "cp_3": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "exang_1": [0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1],
        "slope_1": [0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1],
        "slope_2": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }

    df = pd.DataFrame(data)
    st.table(df)

elif page == "Prediction":
    st.subheader("Make a Prediction")
    st.write("Adjust the settings below to enter the data:")

    # Συλλογή δεδομένων από τον χρήστη
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

    # Φόρτωση μοντέλου
    @st.cache_resource
    def load_model():
        return joblib.load('model.pkl')

    model = load_model()

    # Πρόβλεψη
    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])

        # Μόνο τα απαραίτητα χαρακτηριστικά
        expected_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                             'thalach', 'exang', 'oldpeak']

        input_df = input_df[expected_features]
        input_df['sex'] = input_df['sex'].apply(lambda x: 1 if x.lower() == "male" else 0)

        try:
            prediction_prob = model.predict_proba(input_df)[:, 1][0] * 100
            st.write(f"Probability of cardiovascular disease: {prediction_prob:.2f}%")
            if prediction_prob > 50:
                st.error("High risk of cardiovascular disease!")
            else:
                st.success("Low risk of cardiovascular disease!")

            # Visualization
            st.subheader("Visualization")
            st.write("The following chart shows the prediction probability:")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.bar(["Low Risk", "High Risk"], [100 - prediction_prob, prediction_prob], color=['green', 'red'])
            ax.set_ylabel("Probability (%)")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

