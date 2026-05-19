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


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def style_risk_column(value: str) -> str:
    color = RISK_COLORS.get(value, "#a04329")
    return f"color: {color}; font-weight: 700"


st.set_page_config(
    page_title="Student Dropout Predictor",
    layout="centered",
)

st.markdown("""
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
</style>
""", unsafe_allow_html=True)

st.title("Student Dropout Predictor")

st.caption(
    "Machine Learning Early-Warning System for Academic Retention"
)

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

st.subheader("Student Information")

MIN_GRADE = 9.5

high_school_average = st.slider(
    "High school average (0–20)",
    MIN_GRADE,
    20.0,
    14.0,
    0.1,
)

district_of_origin = st.selectbox(
    "District of origin",
    DISTRICTS,
    index=DISTRICTS.index("Porto"),
)

average_grade = st.slider(
    "Current average grade (0–20)",
    MIN_GRADE,
    20.0,
    12.0,
    0.1,
)

attendance = st.slider(
    "Attendance (%)",
    0,
    100,
    75,
)

courses_per_semester = st.slider(
    "Courses per semester",
    1,
    10,
    6,
)

failed_courses = st.slider(
    "Failed courses",
    0,
    courses_per_semester,
    2,
)

completed_courses = st.slider(
    "Completed courses",
    0,
    max(0, courses_per_semester - failed_courses),
    4,
)

if failed_courses + completed_courses > courses_per_semester:
    completed_courses = courses_per_semester - failed_courses

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

input_data = pd.DataFrame([{
    "high_school_average": high_school_average,
    "district_of_origin": district_of_origin,
    "average_grade": average_grade,
    "attendance_percentage": attendance,
    "courses_per_semester": courses_per_semester,
    "failed_courses": failed_courses,
    "completed_courses": completed_courses,
    "assignment_completion_percentage": assignment_completion,
    "platform_logins_per_week": platform_logins,
}])

_, button_right = st.columns([3, 1])

with button_right:
    predict_clicked = st.button(
        "Predict Dropout Risk",
        use_container_width=True,
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

    if prediction == "Low":

        st.markdown(
            """
            <div style="
                padding: 12px 16px;
                margin: 16px 0;
                border-radius: 4px;
                background-color: rgba(88, 133, 78, 0.15);
                border-left: 4px solid #58854e;
            ">
                <span style="
                    color: #58854e;
                    font-weight: 500;
                ">
                    The student currently shows stable academic engagement.
                    Regular monitoring is recommended.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif prediction == "Medium":

        st.markdown(
            """
            <div style="
                padding: 12px 16px;
                margin: 16px 0;
                border-radius: 4px;
                background-color: rgba(230, 194, 96, 0.15);
                border-left: 4px solid #e6c260;
            ">
                <span style="
                    color: #e6c260;
                    font-weight: 500;
                ">
                    The student may benefit from mentoring, tutoring,
                    or periodic academic follow-up.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div style="
                padding: 12px 16px;
                margin: 16px 0;
                border-radius: 4px;
                background-color: rgba(145, 36, 49, 0.15);
                border-left: 4px solid #912431;
            ">
                <span style="
                    color: #912431;
                    font-weight: 500;
                ">
                    The student should be prioritized for immediate
                    academic intervention and support.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if probability_map:

        st.subheader("Risk Probabilities")

        probability_df = pd.DataFrame({
            "Risk Level": list(probability_map.keys()),
            "Probability": [
                round(value, 4)
                for value in probability_map.values()
            ],
        })

        st.table(probability_df)

    st.subheader("Input Summary")

    input_numeric_columns = input_data.select_dtypes(
        include=["number"]
    ).columns

    input_summary = input_data.style.format({
        column: "{:.2f}"
        for column in input_numeric_columns
    })

    st.dataframe(
        input_summary,
        use_container_width=True,
        hide_index=True,
    )

st.divider()

st.subheader("Batch Prediction From CSV")

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
                use_container_width=True,
                hide_index=True,
            )