import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ===============================
# LOAD MODEL DAN FEATURE NAMES
# ===============================
@st.cache_resource
def load_model():
    return joblib.load("model_catboost.pkl")

@st.cache_resource
def load_features():
    return joblib.load("feature_names.pkl")

model = load_model()
feature_names = load_features()

# ===============================
# LOAD DATASET UNTUK ENCODING
# ===============================
df = pd.read_csv("stroke.csv")

# DROP ID jika ada
if "id" in df.columns:
    df = df.drop(columns=["id"])

encoders = {}
cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    le = LabelEncoder()
    le.fit(df[col])
    encoders[col] = le

# ===============================
# STREAMLIT UI
# ===============================
st.title("Prediksi Risiko Stroke")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", encoders["gender"].classes_)
    age = st.number_input("Age", 0, 120, 30)
    hypertension = st.selectbox("Hypertension", [0,1])
    heart_disease = st.selectbox("Heart Disease", [0,1])
    ever_married = st.selectbox("Ever Married", encoders["ever_married"].classes_)

with col2:
    work_type = st.selectbox("Work Type", encoders["work_type"].classes_)
    Residence_type = st.selectbox("Residence Type", encoders["Residence_type"].classes_)
    avg_glucose_level = st.number_input("Glucose Level", value=100.0)
    bmi = st.number_input("BMI", value=25.0)
    smoking_status = st.selectbox("Smoking Status", encoders["smoking_status"].classes_)

# ===============================
# PREDICT
# ===============================
if st.button("Prediksi"):

    input_dict = {
        "gender": encoders["gender"].transform([gender])[0],
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": encoders["ever_married"].transform([ever_married])[0],
        "work_type": encoders["work_type"].transform([work_type])[0],
        "Residence_type": encoders["Residence_type"].transform([Residence_type])[0],
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": encoders["smoking_status"].transform([smoking_status])[0],
    }

    features = pd.DataFrame([input_dict])

    # FIX UTAMA: sesuaikan hanya fitur yang benar-benar ada
    valid_features = [col for col in feature_names if col != "id"]

    features = features[valid_features]

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("Pasien Berisiko Stroke")
    else:
        st.success("Pasien Tidak Berisiko Stroke")

st.write("Feature names dari model:")
st.write(feature_names)

st.write("Kolom dari input Streamlit:")
st.write(features.columns.tolist())



