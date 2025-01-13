import streamlit as st
import pandas as pd
import joblib

# Πρέπει να είναι η πρώτη εντολή Streamlit
st.set_page_config(page_title="Πρόβλεψη Καρδιοαγγειακών Νοσημάτων", layout="wide")

# Επιλογή γλώσσας
language = st.sidebar.selectbox("Επιλέξτε γλώσσα / Select Language", ["Ελληνικά", "English"])

# Εμφάνιση εικόνας
image_path = "C:\\Users\\user\\Desktop\\heart.jpg"  # Ενημέρωσε το path
st.image(image_path, use_container_width=True, caption="Cardiovascular Health")

# Τίτλος εφαρμογής
if language == "Ελληνικά":
    st.title("Εφαρμογή Πρόβλεψης για Καρδιοαγγειακά Νοσήματα")
else:
    st.title("Cardiovascular Disease Prediction App")

# Ορισμός χαρακτηριστικών και περιγραφών ανά γλώσσα
if language == "Ελληνικά":
    training_features = [
        'Ηλικία (χρόνια)', 'Αρτηριακή πίεση σε ηρεμία (mm Hg)', 'Ολική χοληστερόλη (mg/dl)',
        'Μέγιστος καρδιακός ρυθμός (bpm)', 'Τυπικός πόνος στο στήθος', 'Μη καρδιακός πόνος στο στήθος',
        'Άτυπος πόνος στο στήθος', 'Πόνος στο στήθος μετά από άσκηση',
        'Επίπεδη κλίση ST στο καρδιογράφημα', 'Ανοδική κλίση ST στο καρδιογράφημα'
    ]
    feature_descriptions = {
        'Ηλικία (χρόνια)': "Η ηλικία του ατόμου σε χρόνια.",
        'Αρτηριακή πίεση σε ηρεμία (mm Hg)': "Η πίεση του αίματος σε ηρεμία (σε mm Hg).",
        'Ολική χοληστερόλη (mg/dl)': "Η χοληστερόλη (σε mg/dl).",
        'Μέγιστος καρδιακός ρυθμός (bpm)': "Η μέγιστη καρδιακή συχνότητα κατά τη διάρκεια άσκησης (σε bpm).",
        'Τυπικός πόνος στο στήθος': "Πόνος στο στήθος που συνήθως συνδέεται με καρδιακά προβλήματα.",
        'Μη καρδιακός πόνος στο στήθος': "Πόνος στο στήθος που δεν σχετίζεται με την καρδιά.",
        'Άτυπος πόνος στο στήθος': "Πόνος που δεν έχει τυπικά χαρακτηριστικά καρδιακού πόνου.",
        'Πόνος στο στήθος μετά από άσκηση': "Πόνος στο στήθος που εμφανίζεται μετά από άσκηση.",
        'Επίπεδη κλίση ST στο καρδιογράφημα': "Επίπεδη κλίση του τμήματος ST στο καρδιογράφημα.",
        'Ανοδική κλίση ST στο καρδιογράφημα': "Ανοδική κλίση του τμήματος ST στο καρδιογράφημα."
    }
else:
    training_features = [
        'Age (years)', 'Resting Blood Pressure (mm Hg)', 'Total Cholesterol (mg/dl)',
        'Maximum Heart Rate (bpm)', 'Typical Chest Pain', 'Non-Cardiac Chest Pain',
        'Atypical Chest Pain', 'Chest Pain After Exercise',
        'Flat ST Slope on ECG', 'Upsloping ST Slope on ECG'
    ]
    feature_descriptions = {
        'Age (years)': "The person's age in years.",
        'Resting Blood Pressure (mm Hg)': "Blood pressure at rest (in mm Hg).",
        'Total Cholesterol (mg/dl)': "Cholesterol level (in mg/dl).",
        'Maximum Heart Rate (bpm)': "Maximum heart rate during exercise (in bpm).",
        'Typical Chest Pain': "Chest pain usually associated with heart problems.",
        'Non-Cardiac Chest Pain': "Chest pain unrelated to heart issues.",
        'Atypical Chest Pain': "Pain that does not have typical cardiac characteristics.",
        'Chest Pain After Exercise': "Chest pain occurring after physical exercise.",
        'Flat ST Slope on ECG': "Flat slope of the ST segment on the ECG.",
        'Upsloping ST Slope on ECG': "Upsloping slope of the ST segment on the ECG."
    }

# Εισαγωγή δεδομένων
input_data = {}
for feature in training_features:
    st.markdown(f"### {feature}")
    st.markdown(f"- {feature_descriptions[feature]}")
    value = st.number_input(f"Enter value for {feature}", min_value=0, max_value=300, step=1)
    input_data[feature] = value

# Φόρτωση μοντέλου
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

# Πρόβλεψη
def make_prediction(model, input_data):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("High risk!")
    else:
        st.success("Low risk!")

if st.button("Predict"):
    make_prediction(model, input_data)
