import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = "models/dropout_model.pkl"

RISK_COLORS = {
    "Low": "#58854e",
    "Medium": "#e6c260",
    "High": "#912431",
}

FEATURE_COLUMNS = [
    "high_school_average",
    "district_of_origin",
    "average_grade",
    "attendance_percentage",
    "courses_per_semester",
    "failed_courses",
    "completed_courses",
    "assignment_completion_percentage",
    "platform_logins_per_week",
]

DISTRICTS = [
    "Açores",
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
    "Madeira",
    "Portalegre",
    "Porto",
    "Santarém",
    "Setúbal",
    "Viana do Castelo",
    "Vila Real",
    "Viseu",
]

DEFAULT_COURSES = 6
DEFAULT_FAILED = 2
DEFAULT_COMPLETED = 4


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def style_risk_column(value: str) -> str:
    color = RISK_COLORS.get(value, "#a04329")
    return f"color: {color}; font-weight: 700"


def initialize_course_state():
    if "courses_per_semester" not in st.session_state:
        st.session_state["courses_per_semester"] = DEFAULT_COURSES

    if "failed_courses" not in st.session_state:
        st.session_state["failed_courses"] = DEFAULT_FAILED

    if "completed_courses" not in st.session_state:
        st.session_state["completed_courses"] = DEFAULT_COMPLETED

    if "course_warning_msg" not in st.session_state:
        st.session_state["course_warning_msg"] = ""

    validate_course_values(show_warning=False)


def validate_course_values(show_warning: bool = True):
    courses = st.session_state.get("courses_per_semester", DEFAULT_COURSES)
    failed = st.session_state.get("failed_courses", DEFAULT_FAILED)
    completed = st.session_state.get("completed_courses", DEFAULT_COMPLETED)

    adjusted = False

    if failed > courses:
        failed = courses
        st.session_state["failed_courses"] = failed
        adjusted = True

    if completed > courses:
        completed = courses
        st.session_state["completed_courses"] = completed
        adjusted = True

    if failed + completed > courses:
        completed = max(0, courses - failed)
        st.session_state["completed_courses"] = completed
        adjusted = True

    st.session_state["course_warning_msg"] = (
        "Adjusted course values so failed + completed does not exceed total courses."
        if adjusted and show_warning
        else ""
    )


def get_risk_explanations(input_row: dict) -> list[str]:
    reasons = []

    if input_row["attendance_percentage"] < 70:
        reasons.append("Low attendance may indicate academic disengagement.")

    if input_row["average_grade"] < 12:
        reasons.append("The current average grade is close to the minimum passing threshold.")

    if input_row["failed_courses"] >= 3:
        reasons.append("Several failed courses increase the probability of academic delay.")

    if input_row["assignment_completion_percentage"] < 65:
        reasons.append("Low assignment completion suggests inconsistent academic participation.")

    if input_row["platform_logins_per_week"] < 5:
        reasons.append("Low Moodle activity may indicate reduced engagement with course materials.")

    if input_row["completed_courses"] < max(1, input_row["courses_per_semester"] // 2):
        reasons.append("The number of completed courses is low compared with the expected workload.")

    if input_row["district_of_origin"] not in [
        "Porto",
        "Aveiro",
        "Braga",
        "Viana do Castelo",
        "Vila Real",
    ]:
        reasons.append("The student comes from outside the closest northern districts, which may affect adaptation.")

    if not reasons:
        reasons.append("No critical academic risk indicators were detected.")

    return reasons


def get_recommendation(prediction: str) -> str:
    if prediction == "Low":
        return "The student currently shows stable academic engagement."

    if prediction == "Medium":
        return "The student may benefit from mentoring, tutoring, or periodic academic follow-up."

    return "The student should be prioritized for immediate academic intervention and support."


st.set_page_config(
    page_title="Student Dropout Predictor",
    layout="centered",
)

st.markdown(
    """
    <style>
        input[type="range"] {
            accent-color: #a04329 !important;
        }

        .stButton > button {
            border: 2px solid #a04329 !important;
            box-shadow: 0 0 0 1px rgba(174, 63, 50, 0.18),
                        0 0 14px rgba(174, 63, 50, 0.18) !important;
        }

        .stButton > button:hover {
            border-color: #a04329 !important;
            box-shadow: 0 0 0 1px rgba(174, 63, 50, 0.28),
                        0 0 18px rgba(174, 63, 50, 0.3) !important;
        }

        a[href^="#"] {
            display: none !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

initialize_course_state()

st.title("Student Dropout Predictor", anchor=False)

st.caption("Machine Learning Early-Warning System for Academic Retention")

st.markdown(
    """
    <div style="text-align: justify; max-width: 900px; line-height: 1.6;">
        This machine-learning-based early-warning system uses FEUP-inspired synthetic data
        to simulate academic and behavioural patterns associated with student dropout risk.
        The model estimates whether a student presents a Low, Medium, or High dropout risk
        level based on academic performance, engagement indicators, and institutional activity.
    </div>
    """,
    unsafe_allow_html=True,
)

model = load_model()

st.subheader("Student Information", anchor=False)

MIN_GRADE = 9.5

high_school_average = st.slider(
    "High school average (0–20)",
    MIN_GRADE,
    20.0,
    16.0,
    0.1,
)

district_of_origin = st.selectbox(
    "Region of origin",
    DISTRICTS,
    index=DISTRICTS.index("Porto"),
)

average_grade = st.slider(
    "Current average grade (0–20)",
    MIN_GRADE,
    20.0,
    14.0,
    0.1,
)

courses_per_semester = st.slider(
    "Courses per semester",
    min_value=1,
    max_value=10,
    key="courses_per_semester",
    on_change=validate_course_values,
)

failed_courses = st.slider(
    "Failed courses",
    min_value=0,
    max_value=st.session_state["courses_per_semester"],
    key="failed_courses",
    on_change=validate_course_values,
)

completed_courses = st.slider(
    "Completed courses",
    min_value=0,
    max_value=st.session_state["courses_per_semester"],
    key="completed_courses",
    on_change=validate_course_values,
)

if st.session_state.get("course_warning_msg"):
    st.markdown(
        f"""
        <div style="
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
            background-color: rgba(230, 194, 96, 0.12);
            border-left: 4px solid {RISK_COLORS['Medium']};
        ">
            <span style="
                color: {RISK_COLORS['Medium']};
                font-weight: 600;
            ">
                {st.session_state["course_warning_msg"]}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

attendance = st.slider(
    "Attendance (%)",
    0,
    100,
    75,
)

assignment_completion = st.slider(
    "Assignment completion (%)",
    0,
    100,
    70,
)

platform_logins = st.number_input(
    "Moodle / platform logins per week",
    min_value=0,
    value=5,
)

input_row = {
    "high_school_average": high_school_average,
    "district_of_origin": district_of_origin,
    "average_grade": average_grade,
    "attendance_percentage": attendance,
    "courses_per_semester": courses_per_semester,
    "failed_courses": failed_courses,
    "completed_courses": completed_courses,
    "assignment_completion_percentage": assignment_completion,
    "platform_logins_per_week": platform_logins,
}

input_data = pd.DataFrame([input_row])

_, button_right = st.columns([3, 1])

with button_right:
    predict_clicked = st.button(
    "Predict Dropout Risk",
    width="stretch",
)

if predict_clicked:
    prediction = model.predict(input_data)[0]

    confidence = None
    probability_map = {}

    if hasattr(model.named_steps["model"], "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        classes = model.named_steps["model"].classes_
        probability_map = dict(zip(classes, probabilities))
        confidence = probability_map[prediction]

    border_color = RISK_COLORS.get(prediction, "#a04329")

    st.markdown(
        f"""
        <div style="
            padding: 28px 24px;
            margin-top: 14px;
            margin-bottom: 12px;
            border-radius: 12px;
            background-color: transparent;
            border-left: 5px solid {border_color};
        ">
            <div style="
                color: {border_color};
                margin: 0;
                font-size: 32px;
                font-weight: 700;
            ">
                Predicted Risk: {prediction}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if confidence is not None:
        st.write(f"Model confidence: **{confidence:.2%}**")

    recommendation = get_recommendation(prediction)

    st.markdown(
        f"""
        <div style="
            padding: 12px 16px;
            margin: 16px 0;
            border-radius: 4px;
            background-color: rgba(160, 67, 41, 0.12);
            border-left: 4px solid {border_color};
        ">
            <span style="
                color: {border_color};
                font-weight: 500;
            ">
                {recommendation}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if probability_map:
        st.subheader("Risk Probabilities", anchor=False)

        probability_df = pd.DataFrame({
            "Risk Level": list(probability_map.keys()),
            "Probability": [
                round(value, 4)
                for value in probability_map.values()
            ],
        })

        st.table(probability_df)

    st.subheader("Main Risk Factors", anchor=False)

    for reason in get_risk_explanations(input_row):
        st.write(f"- {reason}")

    st.subheader("Input Summary", anchor=False)

    input_numeric_columns = input_data.select_dtypes(
        include=["number"]
    ).columns

    input_summary = input_data.style.format({
        column: "{:.2f}"
        for column in input_numeric_columns
    })

    st.dataframe(
    input_summary,
    width="stretch",
    hide_index=True,
)

st.divider()

st.subheader("Batch Prediction From CSV", anchor=False)

st.write(
    "Upload a CSV file containing student records to generate dropout risk predictions for multiple students simultaneously."
)

uploaded_csv = st.file_uploader(
    "Upload CSV",
    type=["csv"],
)

if uploaded_csv is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_csv)

    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")

    else:
        missing_columns = [
            column
            for column in FEATURE_COLUMNS
            if column not in uploaded_df.columns
        ]

        if missing_columns:
            st.error(
                "The uploaded CSV is missing required columns: "
                + ", ".join(missing_columns)
            )

        else:
            batch_predictions = model.predict(
                uploaded_df[FEATURE_COLUMNS]
            )

            results_df = uploaded_df.copy()
            results_df["risk"] = batch_predictions

            high_risk_count = (results_df["risk"] == "High").sum()
            medium_risk_count = (results_df["risk"] == "Medium").sum()
            low_risk_count = (results_df["risk"] == "Low").sum()

            st.subheader("Cohort Overview", anchor=False)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Students analyzed", len(results_df))
            c2.metric("High risk", int(high_risk_count))
            c3.metric("Medium risk", int(medium_risk_count))
            c4.metric("Low risk", int(low_risk_count))

            st.subheader("Risk Distribution", anchor=False)

            risk_distribution = results_df["risk"].value_counts().reset_index()
            risk_distribution.columns = ["Risk Level", "Students"]

            st.bar_chart(
                risk_distribution.set_index("Risk Level")
            )

            st.subheader("Predicted Student Risk Table", anchor=False)

            ordered_columns = []

            if "student_number" in results_df.columns:
                ordered_columns.append("student_number")

            ordered_columns.append("risk")

            for column in results_df.columns:
                if column not in ordered_columns:
                    ordered_columns.append(column)

            results_df = results_df[ordered_columns]

            numeric_columns = results_df.select_dtypes(
                include=["number"]
            ).columns

            formatters = {
                column: "{:.2f}"
                for column in numeric_columns
                if column != "student_number"
            }

            if "student_number" in results_df.columns:
                formatters["student_number"] = "{:.0f}"

            styled_results = (
                results_df.style
                .map(style_risk_column, subset=["risk"])
                .format(formatters)
            )

            st.dataframe(
            styled_results,
            width="stretch",
            hide_index=True,
        )