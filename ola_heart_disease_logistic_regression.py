import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('heart-disease.csv')
df = data[['age',
           'sex',
           'cp',
           'trestbps',
           'chol',
           'fbs',
           'restecg',
           'thalach',
           'exang',
           'oldpeak',
           'target']].dropna()

x = df[['age',
        'sex',
        'cp',
        'trestbps',
        'chol',
        'fbs',
        'restecg',
        'thalach',
        'exang',
        'oldpeak']]
y = df['target']

feature_train, feature_test, target_train, target_test = train_test_split(x, y, test_size=0.2)

# Model selection
model = LogisticRegression()
model.fit(feature_train, target_train)

st.header('Heart Disease Prediction')
st.sidebar.subheader('Diagnosis')
age = st.sidebar.number_input('Age', min_value=25)
sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
cp = st.sidebar.number_input('Chest Pain (0-4)', min_value=0, max_value=4)
trestbps = st.sidebar.text_input('Resting Blood Pressure')
chol = st.sidebar.text_input('Serum Cholesterol')
fbs = st.sidebar.selectbox('Fast Blood Sugar > 120 mg/dl', ['Yes', 'No'])
restecg = st.sidebar.number_input('Resting Electrocardiograph Results (0-2)', min_value=0, max_value=2)
thalach = st.sidebar.number_input('Max Heart Rate Achieved', min_value=95)
exang = st.sidebar.selectbox('Exercise Induced Angina', ['Yes', 'No'])
oldpeak = st.sidebar.number_input('Depression induced by exercise relative to rest', min_value=0.1)

if sex == 'Male':
    sex = 1
else:
    sex = 0

if exang == 'Yes':
    exang = 1
else:
    exang = 0

if fbs == 'Yes':
    fbs = 1
else:
    fbs = 0

feature = {
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak]}
features = pd.DataFrame(feature)
dt = pd.DataFrame(feature)
st.dataframe(dt, width=900)

st.write('1 - means that the patient has heart disease')
st.write('0 - means that the patient does not have heart disease')

if st.button('Check Prediction'):
    prediction = model.predict(features)
    # st.write('The prediction on having a heart disease is:', prediction)
    st.write(f'The prediction on having a heart disease is: {prediction[0]}')

