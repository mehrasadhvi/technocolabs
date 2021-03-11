#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pickle
import numpy as np
model=pickle.load(open('model.pkl','rb'))


def predict_blood(recency,frequency,time,monetary):
    input=np.array([[recency,frequency,time,monetary]]).astype(np.float64)
    prediction=model.predict_proba(input)
    pred='{0:.{1}f}'.format(prediction[0][0], 2)
    return float(pred)

def main():
    st.title("Mini Project")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Prediction Blood Donation ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    recency = st.text_input("Recency","Type Here")
    frequency = st.text_input("Frequency","Type Here")
    time = st.text_input("Time","Type Here")
    monetary = st.text_input("Monetary","Type Here")
    safe_html="""  
      <div style="background-color:#F4D03F;padding:10px >
       <h2 style="color:white;text-align:center;">Donated</h2>
       </div>
    """
    danger_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;">Not Donated</h2>
       </div>
    """

    if st.button("Predict"):
        output=predict_blood(recency,frequency,time,monetary)
        st.success('The probability of donating blood is {}'.format(output))

        if output > 0.5:
            st.markdown(safe_html,unsafe_allow_html=True)
        else:
            st.markdown(danger_html,unsafe_allow_html=True)

if __name__=='__main__':
    main()

