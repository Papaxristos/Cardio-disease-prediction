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

# Page content
if page == "Home":
    image_url = "https://raw.githubusercontent.com/Papaxristos/Cardio-disease-prediction/refs/heads/main/heart-attack.webp"
    st.image(image_url, use_container_width=True, caption="Cardiovascular Health")
    st.write(""" Cardiovascular diseases (CVDs) are the leading cause of death globally, taking an estimated 17.9 million lives each year. CVDs are a group of disorders of the heart and blood vessels and include coronary heart disease, cerebrovascular disease, rheumatic heart disease and other conditions. More than four out of five CVD deaths are due to heart attacks and strokes, and one third of these deaths occur prematurely in people under 70 years of age.

The most important behavioural risk factors of heart disease and stroke are unhealthy diet, physical inactivity, tobacco use and harmful use of alcohol. Amongst environmental risk factors, air pollution is an important factor. The effects of behavioural risk factors may show up in individuals as raised blood pressure, raised blood glucose, raised blood lipids, and overweight and obesity. These “intermediate risks factors” can be measured in primary care facilities and indicate an increased risk of heart attack, stroke, heart failure and other complications.

Cessation of tobacco use, reduction of salt in the diet, eating more fruit and vegetables, regular physical activity and avoiding harmful use of alcohol have been shown to reduce the risk of cardiovascular disease. Health policies that create conducive environments for making healthy choices affordable and available, as well as improving air quality and reducing pollution, are essential for motivating people to adopt and sustain healthy behaviours.

Identifying those at highest risk of CVDs and ensuring they receive appropriate treatment can prevent premature deaths. Access to noncommunicable disease medicines and basic health technologies in all primary health care facilities is essential to ensure that those in need receive treatment and counselling. """)  # Brief description of cardiovascular disease

elif page == "Data Information":
    st.subheader("Data Information")
    st.write("""
    This data set dates from 1988 and consists of four databases: Cleveland, Hungary, Switzerland, and Long Beach V. It contains 76 attributes, including the predicted attribute, but all published experiments refer to using a subset of 14 of them. The "target" field refers to the presence of heart disease in the patient. It is integer valued 0 = no disease and 1 = disease.
    """)
    # Data visualization (without the prediction button)
    df = pd.DataFrame({
        "age": [52, 53, 70, 61, 62, 58, 58, 55, 46, 54, 71, 43, 51, 52],
        "trestbps": [125, 140, 145, 148, 138, 100, 114, 160, 120, 122, 112, 132, 140, 128],
        "chol": [212, 203, 174, 203, 294, 248, 318, 289, 249, 286, 149, 341, 298, 204],
        "thalach": [168, 155, 125, 161, 106, 122, 140, 145, 144, 116, 125, 136, 122, 156],
        "target": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
    })

    # Visualizing the data (no prediction button)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].hist(df['age'], bins=10, color='skyblue', edgecolor='black')
    ax[0].set_title("Age Distribution")
    ax[1].hist(df['chol'], bins=10, color='salmon', edgecolor='black')
    ax[1].set_title("Cholesterol Distribution")
    st.pyplot(fig)

    st.table(df)

elif page == "Prediction":
    st.subheader("Make a Prediction")
    st.write("Adjust the settings below to enter the data:")

    # Collect user input
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

    # Load the model
    @st.cache_resource
    def load_model():
        return joblib.load('model.pkl')

    model = load_model()

    # Create a DataFrame from user input
    input_df = pd.DataFrame([input_data])

    # Convert categorical columns to numeric (e.g., sex: female -> 0, male -> 1)
    input_df['sex'] = input_df['sex'].apply(lambda x: 1 if x == "male" else 0)

    # Prediction button
    if st.button("Predict"):
        # Preprocessing input
        expected_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak']
        input_df = input_df[expected_features]

        try:
            prediction_prob = model.predict_proba(input_df)[:, 1][0] * 100
            st.write(f"Probability of cardiovascular disease: {prediction_prob:.2f}%")
            if prediction_prob > 50:
                st.error("High risk of cardiovascular disease!")
            else:
                st.success("Low risk of cardiovascular disease!")

            # Visualization of the probability
            fig, ax = plt.subplots()
            ax.bar(["Low Risk", "High Risk"], [100 - prediction_prob, prediction_prob], color=['green', 'red'])
            ax.set_ylabel("Probability (%)")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

        # EDA Section: Display after prediction
        st.subheader("Exploratory Data Analysis (EDA)")

        # Bar chart of selected features
        features = ['age', 'chol', 'thalach']
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(features, input_df[features].values.flatten(), color=['skyblue', 'salmon', 'lightgreen'])
        ax.set_title("Selected Features Bar Chart")
        ax.set_ylabel("Value")
        st.pyplot(fig)

        # Spider/Radar chart for all input features
        def plot_radar(features, values):
            labels = features
            num_vars = len(labels)

            # Compute angle for each feature
            angles = [n / float(num_vars) * 2 * 3.1416 for n in range(num_vars)]
            values = list(values)  # Ensure values is a list if it's a pandas series
            values += values[:1]  # Close the circle by repeating the first value
            angles += angles[:1]  # Close the circle by repeating the first angle

            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            ax.fill(angles, values, color='cyan', alpha=0.25)
            ax.plot(angles, values, color='blue', linewidth=2)
            ax.set_yticklabels([])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels)
            ax.set_title("Input Feature Radar Chart")
            st.pyplot(fig)

        plot_radar(['age', 'chol', 'thalach', 'oldpeak'], input_df[['age', 'chol', 'thalach', 'oldpeak']].values.flatten())

