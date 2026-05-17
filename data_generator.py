import os
import numpy as np
import pandas as pd


def generate_student_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)

    attendance = np.random.normal(75, 15, n_samples).clip(0, 100)
    average_grade = np.random.normal(12, 4, n_samples).clip(0, 20)
    courses_per_semester = np.random.randint(1, 11, n_samples)
    failed_courses = np.random.binomial(courses_per_semester, 0.2)
    completed_courses = np.random.binomial(
        courses_per_semester - failed_courses, 0.8
    )
    assignment_completion = np.random.normal(70, 20, n_samples).clip(0, 100)
    high_school_average = np.random.normal(14, 3, n_samples).clip(0, 20)
    platform_logins = np.random.poisson(5, n_samples).clip(0, 20)

    districts = np.random.choice(
        [
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
        ],
        size=n_samples,
        p=[

    0.08,

    0.02,

    0.10,

    0.03,

    0.03,

    0.06,

    0.02,

    0.03,

    0.03,

    0.05,

    0.08,

    0.02,

    0.26,

    0.03,

    0.04,

    0.05,

    0.04,

    0.03,

],
    )

    district_risk = {
        "Porto": 0,
        "Aveiro": 4,
        "Braga": 4,
        "Viana do Castelo": 6,
        "Vila Real": 7,
        "Viseu": 8,
        "Coimbra": 9,
        "Bragança": 10,
        "Guarda": 11,
        "Leiria": 11,
        "Lisboa": 13,
        "Castelo Branco": 13,
        "Santarém": 14,
        "Setúbal": 15,
        "Portalegre": 16,
        "Évora": 17,
        "Beja": 18,
        "Faro": 18,
    }

    risk_score = (
        (100 - attendance) * 0.22
        + (20 - average_grade) * 2.2
        + failed_courses * 5.5
        + (12 - completed_courses) * 3.0
        + (100 - assignment_completion) * 0.20
        + (20 - high_school_average) * 1.3
        + (20 - platform_logins) * 0.9
        + np.array([district_risk[district] for district in districts])
        + np.random.normal(0, 8, n_samples)
    )

    dropout_risk = pd.cut(
        risk_score,
        bins=[-np.inf, 45, 75, np.inf],
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
    output_path = "data/student_dropout_data.csv"
    df.to_csv(output_path, index=False)

    print(f"Dataset created successfully: {output_path}")
    print(df.head())
    print("\nRisk distribution:")
    print(df["dropout_risk"].value_counts())