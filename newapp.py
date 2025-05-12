import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the dataset
df = pd.read_csv("student_habits_performance.csv")

# Define target and features
X = df.drop(columns=["student_id", "exam_score"])
y = df["exam_score"]

# Identify categorical and numeric features
categorical_features = X.select_dtypes(include='object').columns.tolist()
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessing and pipeline
transformer = make_column_transformer(
    (OneHotEncoder(drop='first'), categorical_features),
    (StandardScaler(), numeric_features),
    remainder='passthrough'
)
model = LinearRegression()
pipeline = Pipeline([('transformer', transformer), ('model', model)])
pipeline.fit(X, y)

# Streamlit App UI
st.header("Student Exam Score Predictor")

# Collect input values from user
age = st.slider("Age", 16, 30, 21)
gender = st.selectbox("Gender", df['gender'].unique())
study_hours = st.number_input("Study Hours per Day", 0.0, 12.0, 3.0)
social_media_hours = st.number_input("Social Media Hours", 0.0, 10.0, 2.0)
netflix_hours = st.number_input("Netflix Hours", 0.0, 10.0, 1.0)
part_time_job = st.selectbox("Part-time Job", df['part_time_job'].unique())
attendance_percentage = st.slider("Attendance (%)", 0.0, 100.0, 85.0)
sleep_hours = st.number_input("Sleep Hours", 0.0, 12.0, 7.0)
diet_quality = st.selectbox("Diet Quality", df['diet_quality'].unique())
exercise_frequency = st.slider("Exercise Frequency (per week)", 0, 14, 3)
parent_edu = st.selectbox("Parental Education Level", df['parental_education_level'].unique())
internet_quality = st.selectbox("Internet Quality", df['internet_quality'].unique())
mental_health_rating = st.slider("Mental Health Rating (1–10)", 1, 10, 5)
extracurricular = st.selectbox("Extracurricular Participation", df['extracurricular_participation'].unique())

# Combine into input DataFrame
input_df = pd.DataFrame({
    'age': [age],
    'gender': [gender],
    'study_hours_per_day': [study_hours],
    'social_media_hours': [social_media_hours],
    'netflix_hours': [netflix_hours],
    'part_time_job': [part_time_job],
    'attendance_percentage': [attendance_percentage],
    'sleep_hours': [sleep_hours],
    'diet_quality': [diet_quality],
    'exercise_frequency': [exercise_frequency],
    'parental_education_level': [parent_edu],
    'internet_quality': [internet_quality],
    'mental_health_rating': [mental_health_rating],
    'extracurricular_participation': [extracurricular]
})

# Make prediction
prediction = pipeline.predict(input_df)[0]
st.subheader("Predicted Exam Score")
st.write(f"{prediction:.2f} out of 100")
