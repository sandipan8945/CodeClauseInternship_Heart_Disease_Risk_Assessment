import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load the heart disease dataset from Kaggle
heart_disease_data = pd.read_csv(r"C:\Users\mails\Downloads\heart_disease_dataset.csv")

# Preprocess the data
heart_disease_data['sex'] = heart_disease_data['sex'].map({'Male': 0, 'Female': 1})
heart_disease_data['cp'] = heart_disease_data['cp'].map({'Typical Angina': 0, 'Atypical Angina': 1, 'Non-Anginal Pain': 2, 'Asymptomatic': 3})
heart_disease_data['fbs'] = heart_disease_data['fbs'].map({'Yes': 1, 'No': 0})
heart_disease_data['restecg'] = heart_disease_data['restecg'].map({'Normal': 0, 'ST-T Wave Abnormality': 1, 'Left Ventricular Hypertrophy': 2})
heart_disease_data['exang'] = heart_disease_data['exang'].map({'Yes': 1, 'No': 0})
heart_disease_data['slope'] = heart_disease_data['slope'].map({'Up-sloping': 0, 'Flat': 1, 'Down-sloping': 2})
heart_disease_data['thal'] = heart_disease_data['thal'].map({'Normal': 0, 'Fixed Defect': 1, 'Reversible Defect': 2})

# Split the data into features (X) and target (y)
X = heart_disease_data.drop('target', axis=1)
y = heart_disease_data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Save the trained model to a file
pickle.dump(model, open("heart_disease_model.pkl", "wb"))

# Load the trained model
model = pickle.load(open("heart_disease_model.pkl", "rb"))

# Streamlit UI
st.title("Heart Disease Prediction App")

# Create input fields for health metrics
age = st.number_input("Age", min_value=0, max_value=120)
sex = st.selectbox("Sex", options=["Male", "Female"])
cp = st.selectbox("Chest Pain Type", options=["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)")
chol = st.number_input("Serum Cholesterol (mg/dl)")
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["Yes", "No"])
restecg = st.selectbox("Resting Electrocardiographic Results", options=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
thalach = st.number_input("Maximum Heart Rate Achieved")
exang = st.selectbox("Exercise Induced Angina", options=["Yes", "No"])
oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest")
slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=["Up-sloping", "Flat", "Down-sloping"])
ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=4)
thal = st.selectbox("Thalassemia", options=["Normal", "Fixed Defect", "Reversible Defect"])

if st.button("Make a Prediction"):
    # Map user input to numerical values
    sex_map = {'Male': 0, 'Female': 1}
    cp_map = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-Anginal Pain': 2, 'Asymptomatic': 3}
    fbs_map = {'Yes': 1, 'No': 0}
    restecg_map = {'Normal': 0, 'ST-T Wave Abnormality': 1, 'Left Ventricular Hypertrophy': 2}
    exang_map = {'Yes': 1, 'No': 0}
    slope_map = {'Up-sloping': 0, 'Flat': 1, 'Down-sloping':  2}
    thal_map = {'Normal': 0, 'Fixed Defect': 1, 'Reversible Defect': 2}

    sex = sex_map[sex]
    cp = cp_map[cp]
    fbs = fbs_map[fbs]
    restecg = restecg_map[restecg]
    exang = exang_map[exang]
    slope = slope_map[slope]
    thal = thal_map[thal]

    # Create a list of user input
    user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

    # Make a prediction
    prediction = model.predict([user_input])

    # Show the prediction result
    st.write("You have a {}% chance of having a heart disease.".format(int(prediction[0] * 100)))