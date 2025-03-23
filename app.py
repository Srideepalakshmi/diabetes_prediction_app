import streamlit as st
import numpy as np
import joblib

# 🎯 Streamlit App Title
st.title("🩺 AI-Powered Diabetes Prediction System")
st.write("Enter your health parameters and predict the risk of diabetes.")

# 🔹 Load Model and Scaler
MODEL_PATH = "C:/Users/Raj/Documents/Diabetes_Prediction/rf_model.pkl"
SCALER_PATH = "C:/Users/Raj/Documents/Diabetes_Prediction/scaler.pkl"

try:
    model = joblib.load(MODEL_PATH)  # Load Trained Model
    scaler = joblib.load(SCALER_PATH)  # Load Scaler
    st.success("✅ Model and Scaler Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Error loading model or dataset: {e}")
    st.stop()

# 📌 Feature Names (Update based on dataset)
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
            "BMI", "Diabetes Pedigree Function", "Age"]

# 🔹 User Input for Symptoms
input_features = []
for feature in FEATURES:
    value = st.number_input(f"{feature.replace('_', ' ').title()}:", min_value=0.0, step=0.1, format="%.2f")
    input_features.append(value)

# 🔹 Convert Input to NumPy Array & Scale It
if len(input_features) == len(FEATURES):
    input_array = np.array(input_features).reshape(1, -1)  # Reshape Input
    try:
        input_scaled = scaler.transform(input_array)  # Apply Scaling
    except Exception as e:
        st.error(f"❌ Scaling Error: {e}")
        st.stop()

    # 🔹 Predict Diabetes on Button Click
    if st.button("🔍 Predict Diabetes Risk"):
        try:
            prediction = model.predict(input_scaled)[0]  # Get Prediction
            probability = model.predict_proba(input_scaled)[0][1] * 100  # Probability of Diabetes

            # 🎯 Display Result
            if prediction == 1:
                st.error(f"⚠️ High Risk of Diabetes! ({probability:.2f}%)")
            else:
                st.success(f"✅ Low Risk of Diabetes ({100 - probability:.2f}%)")

            # 🔹 Debugging Output (Remove in Production)
            st.write(f"📝 Model Raw Prediction: {prediction}")
            st.write(f"📊 Diabetes Probability: {probability:.2f}%")

        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")
else:
    st.warning("⚠️ Please enter values for all health parameters.")
