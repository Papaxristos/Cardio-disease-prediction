import streamlit as st
import pandas as pd
import joblib  # Για φόρτωση του μοντέλου

# Εμφάνιση τίτλου
st.title("Εφαρμογή Πρόβλεψης για Καρδιοαγγειακά Νοσήματα")

# Χαρακτηριστικά και ορισμοί
training_features = [
    'Ηλικία (χρόνια)', 'Αρτηριακή πίεση σε ηρεμία (mm Hg)', 'Ολική χοληστερόλη (mg/dl)', 'Μέγιστος καρδιακός ρυθμός (bpm)',
    'Τυπικός πόνος στο στήθος', 'Μη καρδιακός πόνος στο στήθος', 'Άτυπος πόνος στο στήθος',
    'Πόνος στο στήθος μετά από άσκηση',
    'Επίπεδη κλίση ST στο καρδιογράφημα', 'Ανοδική κλίση ST στο καρδιογράφημα'
]

feature_ranges = {
    'Ηλικία (χρόνια)': (18, 100),
    'Αρτηριακή πίεση σε ηρεμία (mm Hg)': (80, 200),
    'Ολική χοληστερόλη (mg/dl)': (100, 600),
    'Μέγιστος καρδιακός ρυθμός (bpm)': (60, 250),
    'Τυπικός πόνος στο στήθος': (0.0, 1.0),
    'Μη καρδιακός πόνος στο στήθος': (0.0, 1.0),
    'Άτυπος πόνος στο στήθος': (0.0, 1.0),
    'Πόνος στο στήθος μετά από άσκηση': (0.0, 1.0),
    'Επίπεδη κλίση ST στο καρδιογράφημα': (0.0, 1.0),
    'Ανοδική κλίση ST στο καρδιογράφημα': (0.0, 1.0)
}

# Περιγραφές χαρακτηριστικών
feature_descriptions = {
    'Ηλικία (χρόνια)': "Η ηλικία του ατόμου σε χρόνια.",
    'Αρτηριακή πίεση σε ηρεμία (mm Hg)': "Η πίεση του αίματος σε ηρεμία (σε mm Hg).",
    'Ολική χοληστερόλη (mg/dl)': "Η χοληστερόλη (σε mg/dl).",
    'Μέγιστος καρδιακός ρυθμός (bpm)': "Η μέγιστη καρδιακή συχνότητα κατά τη διάρκεια άσκησης (σε bpm).",
    'Τυπικός πόνος στο στήθος': "Αν το άτομο έχει τυπικό πόνο θώρακα.",
    'Μη καρδιακός πόνος στο στήθος': "Αν το άτομο έχει μη ανγειακό πόνο θώρακα.",
    'Άτυπος πόνος στο στήθος': "Αν το άτομο έχει μη τυπικό πόνο θώρακα.",
    'Πόνος στο στήθος μετά από άσκηση': "Αν το άτομο έχει πόνο στο στήθος μετά από άσκηση.",
    'Επίπεδη κλίση ST στο καρδιογράφημα': "Αν η κλίση του ST segment στο καρδιογράφημα είναι επίπεδη.",
    'Ανοδική κλίση ST στο καρδιογράφημα': "Αν η κλίση του ST segment στο καρδιογράφημα είναι ανοδική."
}

# Εισαγωγή δεδομένων
input_data = {}
for feature in training_features:
    st.markdown(f"### {feature}:")
    st.markdown(f"- **Προτεινόμενη κλίμακα**: {feature_ranges[feature]}")
    st.markdown(f"- **Επεξήγηση**: {feature_descriptions[feature]}")
    min_val, max_val = feature_ranges[feature]
    value = st.number_input(f"Δώστε τιμή για {feature}:",
                            min_value=min_val, max_value=max_val, step=0.1 if isinstance(min_val, float) else 1)
    input_data[feature] = value

# Φόρτωση μοντέλου
@st.cache_resource
def load_model():
    try:
        return joblib.load('model.pkl')
    except Exception as e:
        st.error(f"Σφάλμα στη φόρτωση του μοντέλου: {e}")
        return None

model = load_model()

# Πρόβλεψη
def make_prediction(model, input_data):
    if model:
        try:
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[:, 1][0]
            if prediction == 1:
                st.error(f"Υψηλός κίνδυνος! Πιθανότητα: {proba*100:.2f}%")
            else:
                st.success(f"Χαμηλός κίνδυνος. Πιθανότητα: {proba*100:.2f}%")
        except Exception as e:
            st.error(f"Σφάλμα στην πρόβλεψη: {e}")

if st.button('Πρόβλεψη'):
    make_prediction(model, input_data)
