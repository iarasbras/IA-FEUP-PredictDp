import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = "models/dropout_model.pkl"


DISTRICTS = [
    "Aveiro",
    "Beja",
    "Braga",
    "Bragança",
    "Castelo Branco",
    "Coimbra",
    "Évora",
    "Faro",
    "Guarda",
    "Leiria",
    "Lisboa",
    "Portalegre",
    "Porto",
    "Santarém",
    "Setúbal",
    "Viana do Castelo",
    "Vila Real",
    "Viseu",
]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def risk_color(risk: str) -> str:
    if risk == "Low":
        return "green"
    if risk == "Medium":
        return "orange"
    return "red"


st.set_page_config(
    page_title="Student Dropout Predictor",
    page_icon="🎓",
    layout="centered",
)

st.title("Student Dropout Predictor")

st.write(
    "This proof-of-concept uses artificial FEUP-inspired academic data to estimate "
    "student dropout risk based on school-based indicators such as attendance, grades, "
    "course progress, assignment completion, district of origin, and Moodle activity."
)

model = load_model()

st.subheader("Student Information")

attendance = st.slider("Attendance (%)", 0, 100, 75)
average_grade = st.slider("Average grade (0–20)", 0.0, 20.0, 12.0, 0.1)
failed_courses = st.slider("Failed courses", 0, 10, 2)
completed_courses = st.slider("Completed courses", 0, 12, 6)
assignment_completion = st.slider("Assignment completion (%)", 0, 100, 70)

district_of_origin = st.selectbox(
    "District of origin",
    DISTRICTS,
    index=DISTRICTS.index("Porto"),
)

high_school_average = st.slider("High school average (0–20)", 0.0, 20.0, 14.0, 0.1)
platform_logins = st.slider("Moodle / platform logins per week", 0, 20, 5)

input_data = pd.DataFrame([{
    "attendance_percentage": attendance,
    "average_grade": average_grade,
    "failed_courses": failed_courses,
    "completed_courses": completed_courses,
    "assignment_completion_percentage": assignment_completion,
    "district_of_origin": district_of_origin,
    "high_school_average": high_school_average,
    "platform_logins_per_week": platform_logins,
}])

if st.button("Predict Dropout Risk"):
    prediction = model.predict(input_data)[0]

    confidence = None
    probability_map = {}

    if hasattr(model.named_steps["model"], "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        classes = model.named_steps["model"].classes_
        probability_map = dict(zip(classes, probabilities))
        confidence = probability_map[prediction]

    color = risk_color(prediction)

    st.markdown(
        f"""
        <div style="padding: 20px; border-radius: 12px; background-color: #f5f5f5;">
            <h2 style="color: {color};">Predicted Risk: {prediction}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if confidence is not None:
        st.write(f"Model confidence: **{confidence:.2%}**")

    if probability_map:
        st.subheader("Risk Probabilities")
        probability_df = pd.DataFrame({
            "Risk Level": list(probability_map.keys()),
            "Probability": [round(value, 4) for value in probability_map.values()],
        })
        st.dataframe(probability_df)

    st.subheader("Suggested Interpretation")

    if prediction == "Low":
        st.success("The student appears to be progressing normally.")
    elif prediction == "Medium":
        st.warning("The student may benefit from academic monitoring or mentoring.")
    else:
        st.error("The student should be prioritized for early academic intervention.")

    st.subheader("Input Summary")
    st.dataframe(input_data)