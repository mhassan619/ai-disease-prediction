import streamlit as st
import pandas as pd
import joblib
from Disease_Prediction import load_disease_data, preprocess_data, train_models

st.set_page_config("AI Disease Prediction", "🩺", layout="wide")

st.markdown("""
<style>
body { background-color:#F4F6F8; }
.card {
background:white; padding:20px; border-radius:15px;
box-shadow:0 6px 20px rgba(0,0,0,0.1)
}
.title { font-size:36px; color:#1E88E5; font-weight:700 }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧠 AI Disease Prediction System</div>', unsafe_allow_html=True)
st.markdown("Modern Machine Learning Medical Dashboard")
st.markdown("---")

disease_type = st.sidebar.selectbox(
    "Select Disease",
    ["breast_cancer", "diabetes", "heart_disease"]
)

df, disease_name, target_col = load_disease_data(disease_type)
X, y, scaler, features = preprocess_data(df, target_col)

c1, c2, c3 = st.columns(3)
c1.metric("Patients", df.shape[0])
c2.metric("Features", len(features))
c3.metric("Disease %", f"{y.mean()*100:.1f}%")

st.dataframe(df.head(), use_container_width=True)

if "trained" not in st.session_state:
    st.session_state.trained = False

if st.button("🚀 Train Models"):
    with st.spinner("Training..."):
        results = train_models(X, y)

    best = max(results, key=lambda x: results[x]["f1"])
    joblib.dump(results[best]["model"], f"{disease_type}_model.pkl")
    joblib.dump(scaler, f"{disease_type}_scaler.pkl")

    st.session_state.trained = True
    st.success(f"Best Model: {best}")

    for m, r in results.items():
        st.markdown(f"""
        <div class="card">
        <b>{m}</b><br>
        Accuracy: {r['accuracy']:.2f}<br>
        Precision: {r['precision']:.2f}<br>
        Recall: {r['recall']:.2f}<br>
        F1: {r['f1']:.2f}
        </div><br>
        """, unsafe_allow_html=True)

st.markdown("## 🧪 Patient Diagnosis")

inputs = {}
cols = st.columns(3)
for i, f in enumerate(features):
    inputs[f] = cols[i % 3].number_input(f, 0.0, 500.0, 0.0)

if st.button("🔍 Predict"):
    if not st.session_state.trained:
        st.warning("Train model first!")
        st.stop()

    model = joblib.load(f"{disease_type}_model.pkl")
    scaler = joblib.load(f"{disease_type}_scaler.pkl")

    df_input = pd.DataFrame([inputs])[features]
    scaled = scaler.transform(df_input)

    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    if pred == 1:
        st.error(f"⚠️ Disease Detected ({prob:.2%})")
    else:

        st.success(f"✅ No Disease ({1-prob:.2%})")
