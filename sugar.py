import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

st.write("""
# DIABETES DETECTION
Detect if someone has diabetes using Machine Learning and Python !
\n By Debasmita Dutta
""")

image=Image.open('./image.jpg')
st.image(image,caption='CHECK DIABETES WITH MACHINE LEARNING',use_column_width=True)

df=pd.read_csv('./data.csv')

#st.subheader('Data Information :')

#st.dataframe(df)

#st.write(df.describe())

#chart = st.bar_chart(df)

X = df.iloc[:,0:8].values
Y = df.iloc[:,-1].values

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)

def get_user_input():
    pregnancies = st.sidebar.slider('Pregnancy  Month',0,10,1)
    glucose = st.sidebar.slider('Glucose Level(PP)',0,600,117)
    if glucose>=450:
        st.warning("Consult Doctor to take Insulin doses !")
    blood_pressure = st.sidebar.slider('Blood  Pressure',0,122,72)
    skin_thickness = st.sidebar.slider('Skin  Thickness',0,99,23)
    insulin = st.sidebar.slider('Insulin  Level',0.0,846.0,30.0)
    BMI = st.sidebar.slider('BMI',0.0,67.1,32.0)
    DPF = st.sidebar.slider('DPF',0.078,2.42,0.3725)
    age = st.sidebar.slider('Age',21,81,24)


    user_data= {'Pregnancy_Month':pregnancies,
                'Glucose Level(PP)':glucose,
                'Blood_Pressure':blood_pressure,
                'Skin_Thickness':skin_thickness,
                'Insulin_Level':insulin,
                'BMI':BMI,
                'DPF':DPF,
                'Age':age
               }

    features = pd.DataFrame(user_data,index=[0])
    return features


user_data = get_user_input()

st.subheader('User Input :')
st.write(user_data)


RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train,Y_train)

st.subheader('Model Test Accuracy Score : ')
st.write(str(accuracy_score(Y_test,RandomForestClassifier.predict(X_test))*100)+'%')

prediction = RandomForestClassifier.predict(user_data)

st.subheader('Diagnosis : ')
#st.write(prediction)

if prediction==1:
    st.error("Consult a doctor ! The Patient is having Diabetes.")
else:
    st.success("The patient is safe from Diabetes.")
    

