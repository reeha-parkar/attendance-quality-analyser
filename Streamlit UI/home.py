import streamlit as st
from PIL import Image
import time

def app():
    
    logo = Image.open('../Media/logo_biege.png')
    col1, mid, col2 = st.beta_columns([2, 4, 10])
    with col1:
        st.image(logo, width=250)
    with col2:
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.title("Don't just track students' attendance, analyse it")
    
    
    st.write("**Problem**: Students are often distracted in _online classes_ affecting their grades.")
    st.write("We have its solution! Our app helps teachers not only track the attendance of students, but also analyse if they are often _distracted_ or _focussed_ in class. This will help teachers improve the quality of teaching-learning process.")

    col1, col2 = st.beta_columns([1, 1])
    with col1:
        st.subheader('Team Members!')
        d = {'Aditi Kulkarni': 60002190008, 
        'Dania Juvale': 60001180014, 
        'Nandinee Kaushik': 60001180033, 
        'Reeha Parkar': 60001180046, 
        'Shruti Waghade': 60003190056}
        st.write(d)


    with col2:
        st.write(' ')
        IMAGES = ['online1', 'online2', 'online3', 'online4', 'online5']
        FRAME_WINDOW = st.image([])
        while True:
            for i in range(len(IMAGES)):
                FRAME_WINDOW.image('../Media/'+IMAGES[i]+'.jpg')
                time.sleep(3)   