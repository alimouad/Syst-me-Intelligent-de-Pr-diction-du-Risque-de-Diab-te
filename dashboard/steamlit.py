import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
	page_title="GlucoRiskAI",
	page_icon="🩺",
	layout="centered",
)

st.markdown(
	"""
	<style>
	.risk-card { padding: 1.2rem; border-radius: 12px; color: white; }
	.risk-high { background: #b42318; }
	.risk-low { background: #067647; }
	</style>
	""",
	unsafe_allow_html=True,
)

st.title("GlucoRiskAI")
st.caption("Evaluation indicative du risque diabetique")

with st.form("patient_form"):
	st.subheader("Constantes cliniques")
	col1, col2 = st.columns(2)

	with col1:
		pregnancies = st.number_input("Grossesses", min_value=0, max_value=20, value=1)
		glucose = st.number_input("Glucose", min_value=1.0, max_value=400.0, value=120.0)
		blood_pressure = st.number_input("Pression arterielle", min_value=1.0, max_value=250.0, value=70.0)
		skin_thickness = st.number_input("Epaisseur cutanee", min_value=0.0, max_value=150.0, value=20.0)

	with col2:
		insulin = st.number_input("Insuline", min_value=0.0, max_value=1000.0, value=80.0)
		bmi = st.number_input("IMC", min_value=1.0, max_value=100.0, value=25.0)
		pedigree = st.number_input("Fonction pedigree", min_value=0.0, max_value=3.0, value=0.4)
		age = st.number_input("Age", min_value=1, max_value=120, value=30)

	submitted = st.form_submit_button("Evaluer le risque", type="primary")

if submitted:
	payload = {
		"pregnancies": pregnancies,
		"glucose": glucose,
		"blood_pressure": blood_pressure,
		"skin_thickness": skin_thickness,
		"insulin": insulin,
		"bmi": bmi,
		"diabetes_pedigree_function": pedigree,
		"age": age,
	}

	try:
		response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
		response.raise_for_status()
		result = response.json()
		css_class = "risk-high" if result["risk_level"] == "high" else "risk-low"
		st.markdown(
			f"<div class='risk-card {css_class}'><h2>{result['risk_category']}</h2>"
			f"<p>{result['advice']}</p></div>",
			unsafe_allow_html=True,
		)
	except requests.RequestException as error:
		st.error(f"API indisponible: {error}")
