


import streamlit as st


import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

st.header("Are You a LinkedIn User?")
st.subheader("The purpose of this app is to predict if you are a LinkedIn user based off of the questions that follow.")
user = pd.read_csv("social_media_usage.csv")



z = pd.DataFrame ({
    'Column 1': [1,0,0],
    'Column 2': [0,0,1]
})


import numpy as np

def linkedinfunction (x):
    linkedfunction = np.where(x==1,
                             1,
                             0)
    
    return(x)
linkedinfunction(z)

SS = pd.DataFrame({
    "sm_li":np.where(user["web1h"] == 1, 1, 0),
    "income":np.where(user["income"]> 9, np.nan, user["income"]),
    "education":np.where(user["educ2"] >= 8, np.nan, user["educ2"]),
    "parent":np.where(user["par"] == 1, 1, 0),
    "marital":np.where(user["marital"] > 1,1,0),
    "female":np.where(user["gender"] >= 2,1,0),
    "age":np.where(user["age"] > 97, np.nan, user["age"])
}).dropna()





y = SS["sm_li"]
X = SS[["age", "female", "marital", "parent", "education", "income"]]


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       
                                                    test_size=0.2,    
                                                    random_state=987) 

lr = LogisticRegression(random_state=987, class_weight='balanced')

lr_model = lr.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)

confusion_matrix(y_test, y_pred)











user_age=st.slider("What is your age?", min_value=10, max_value=97, value=18, step=1)
user_gender= st.selectbox("What is your gender? if female select 1, if male select 0", options=[1,0])
user_married= st.selectbox("Are you married? if so select 1, if not select 0", options=[1,0])
user_parent= st.selectbox("Are you a parent? If so select 1, if not select 0", options=[1,0])
st.markdown("What is your education? 1 - Less than High School, 2 = High School Incomplete, 3 = High School Diploma, 4 = Some College, 5 = Two-year associate degree from a college or university, 6 = Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB), 7 = Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school), 8 = Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)")
user_education= st.slider("What is your highest level of education", min_value=1, max_value=8, value=1, step=1)
st.markdown("What is your level of income?1 = Less than $10,000, 2 = $10,000 to under $20,000, 3 = $20,000 to under $30,000, 4 = $30,000 to under $40,000, 5 = $40,000 to under $50,000, 6 = $50,000 to under $75,000, 7 = $75,000 to under $100,000, 8 = $100,000 to under $150,000, 9 = $150,000 or more?")
user_income= st.slider("What is your income", min_value=1, max_value=9, value=1, step=1)

#User inputs used in the Logistic Regression Model
person = [user_income, user_education, user_parent, user_married, user_gender, user_age]

# Predict class, given input feature
predicted_class = lr.predict([person])

# Generate probability of positive class (=1)
probs = lr.predict_proba([person])


st.subheader('Is this person a LinkedIn user?')
if predicted_class > 0:
    label= "You are a LinkedIn user!"
else:
    label= "You are not a LinkedIn user."

result = st.button("CLICK HERE to see your results!")
if result:
    st.write(label)

st.subheader('What is the probability that this person is a LinkedIn user')
st.write(probs)