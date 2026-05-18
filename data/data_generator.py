import os
import numpy as np
import pandas as pd


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
    "Açores",
    "Madeira",
]


DISTRICT_PROBABILITIES = [
    0.08,
    0.02,
    0.10,
    0.03,
    0.03,
    0.07,
    0.02,
    0.03,
    0.03,
    0.05,
    0.09,
    0.02,
    0.19,
    0.03,
    0.05,
    0.05,
    0.04,
    0.03,
    0.02,
    0.02,
]


DISTRICT_DISTANCE = {
    "Porto": 0,
    "Aveiro": 1,
    "Braga": 2,
    "Viana do Castelo": 3,
    "Vila Real": 4,
    "Viseu": 5,
    "Coimbra": 5,
    "Leiria": 6,
    "Lisboa": 7,
    "Santarém": 7,
    "Setúbal": 8,
    "Castelo Branco": 8,
    "Guarda": 9,
    "Portalegre": 10,
    "Bragança": 10,
    "Évora": 11,
    "Beja": 12,
    "Faro": 12,
    "Açores": 18,
    "Madeira": 20,
}

MIN_GRADE = 9.5

PROFILE_PROBABILITIES = [0.35, 0.45, 0.20]
PROFILE_LABELS = ["strong", "average", "at_risk"]

PROFILE_SETTINGS = {
    "strong": {
        "high_school_mean": 17.5,
        "high_school_sd": 0.9,
        "attendance_mean": 95.0,
        "attendance_sd": 2.8,
        "logins_mean": 8.6,
        "logins_sd": 1.2,
        "assignment_mean": 84.0,
        "assignment_sd": 6.0,
        "course_base": 6,
        "course_high": 9,
        "base_grade": 15.0,
        "grade_noise": 0.7,
        "risk_base": 8.0,
    },
    "average": {
        "high_school_mean": 15.0,
        "high_school_sd": 1.0,
        "attendance_mean": 85.0,
        "attendance_sd": 5.5,
        "logins_mean": 7.6,
        "logins_sd": 1.4,
        "assignment_mean": 74.0,
        "assignment_sd": 8.0,
        "course_base": 5,
        "course_high": 10,
        "base_grade": 14.0,
        "grade_noise": 0.9,
        "risk_base": 18.0,
    },
    "at_risk": {
        "high_school_mean": 13.5,
        "high_school_sd": 1.2,
        "attendance_mean": 68.0,
        "attendance_sd": 10.5,
        "logins_mean": 4.0,
        "logins_sd": 1.6,
        "assignment_mean": 59.0,
        "assignment_sd": 11.0,
        "course_base": 4,
        "course_high": 10,
        "base_grade": 12.0,
        "grade_noise": 1.2,
        "risk_base": 38.0,
    },
}


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-value))


def generate_student_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    profiles = rng.choice(PROFILE_LABELS, size=n_samples, p=PROFILE_PROBABILITIES)

    districts = rng.choice(
        DISTRICTS,
        size=n_samples,
        p=DISTRICT_PROBABILITIES,
    )
    district_distance = np.array([DISTRICT_DISTANCE[district] for district in districts])

    profile_settings = [PROFILE_SETTINGS[profile] for profile in profiles]

    academic_ability = rng.normal(0.0, 1.0, n_samples)
    engagement = rng.normal(0.0, 1.0, n_samples)
    stress = rng.normal(0.0, 1.0, n_samples)

    high_school_average = np.clip(
        np.array([settings["high_school_mean"] for settings in profile_settings])
        + academic_ability * 0.8
        + rng.normal(0, [settings["high_school_sd"] for settings in profile_settings], n_samples),
        MIN_GRADE,
        20,
    )

    courses_per_semester = np.array(
        [rng.integers(settings["course_base"], settings["course_high"] + 1) for settings in profile_settings]
    )

    attendance = np.clip(
        np.array([settings["attendance_mean"] for settings in profile_settings])
        + engagement * 4.0
        + academic_ability * 2.0
        - district_distance * 0.7
        + rng.normal(0, [settings["attendance_sd"] for settings in profile_settings], n_samples),
        0,
        100,
    )

    assignment_completion = np.clip(
        np.array([settings["assignment_mean"] for settings in profile_settings])
        + engagement * 5.0
        + academic_ability * 2.2
        - district_distance * 0.35
        + rng.normal(0, [settings["assignment_sd"] for settings in profile_settings], n_samples),
        0,
        100,
    )

    platform_logins = np.clip(
        np.round(
            np.array([settings["logins_mean"] for settings in profile_settings])
            + engagement * 1.2
            + (attendance - 80) / 18.0
            - district_distance * 0.08
            + rng.normal(0, [settings["logins_sd"] for settings in profile_settings], n_samples)
        ),
        0,
        30,
    ).astype(int)

    grade_pressure = (
        np.array([settings["base_grade"] for settings in profile_settings])
        + (high_school_average - 14.0) * 0.40
        + (attendance - 85.0) * 0.06
        + (assignment_completion - 75.0) * 0.035
        + (platform_logins - 7.0) * 0.12
        - district_distance * 0.10
        + academic_ability * 0.5
        - stress * 0.35
        + rng.normal(0, [settings["grade_noise"] for settings in profile_settings], n_samples)
    )

    average_grade = np.clip(grade_pressure, MIN_GRADE, 20)

    fail_probability = _sigmoid(
        -2.2
        + (13.0 - average_grade) * 0.95
        + np.maximum(high_school_average - average_grade - 1.0, 0) * 1.0
        + (75 - attendance) * 0.10
        + np.maximum(75 - attendance, 0) * 0.12
        + np.maximum(5 - platform_logins, 0) * 0.55
        + (70 - assignment_completion) * 0.03
        + district_distance * 0.05
        + stress * 0.25
    )

    failed_courses = rng.binomial(courses_per_semester, np.clip(fail_probability, 0.02, 0.85))
    completed_courses = courses_per_semester - failed_courses

    gap = np.maximum(high_school_average - average_grade, 0)
    severe_gap = np.maximum(gap - 2.5, 0)
    attendance_gap = np.maximum(75 - attendance, 0)
    strong_attendance_bonus = np.maximum(attendance - 90, 0)
    low_login_gap = np.maximum(7 - platform_logins, 0)

    risk_score = (
        np.array([settings["risk_base"] for settings in profile_settings])
        + attendance_gap * 1.8
        + np.maximum(attendance_gap - 10, 0) * 2.2
        + failed_courses * 6.5
        + (courses_per_semester - completed_courses) * 0.8
        + np.maximum(13.0 - average_grade, 0) * 4.8
        + gap * 1.1
        + severe_gap * 3.0
        + np.maximum(high_school_average - 14.5, 0)
        * np.maximum(13.0 - average_grade, 0)
        * 2.2
        + low_login_gap * 2.4
        + np.maximum(6 - platform_logins, 0) * 1.2
        + (100 - assignment_completion) * 0.08
        + district_distance * np.where(attendance < 75, 0.55, 0.18)
        - strong_attendance_bonus * 0.25
        + rng.normal(0, 3.0, n_samples)
    )

    risk_score = np.clip(risk_score, 0, None)

    dropout_risk = pd.cut(
        risk_score,
        bins=[-np.inf, 25, 55, np.inf],
        labels=["Low", "Medium", "High"],
    )

    data = pd.DataFrame({
        "attendance_percentage": attendance.round(2),
        "average_grade": average_grade.round(2),
        "courses_per_semester": courses_per_semester,
        "failed_courses": failed_courses,
        "completed_courses": completed_courses,
        "assignment_completion_percentage": assignment_completion.round(2),
        "district_of_origin": districts,
        "high_school_average": high_school_average.round(2),
        "platform_logins_per_week": platform_logins,
        "dropout_risk": dropout_risk,
    })

    return data


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    df = generate_student_data()
    output_path = "data/student_dropout_testing_data.csv"
    df.to_csv(output_path, index=False)

    print(f"Dataset created successfully: {output_path}")
    print(df.head())
    print("\nRisk distribution:")
    print(df["dropout_risk"].value_counts())