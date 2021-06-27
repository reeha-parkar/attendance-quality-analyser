import streamlit as st
import pandas as pd

# password hashing for securit
import hashlib
def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()
# DB Management
import sqlite3 
conn = sqlite3.connect('data.db')
c = conn.cursor()
# DB  Functions
def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

st.title("Attendance Quality Analyser")
st.subheader("Teacher Sign Up")
new_user = st.text_input("Username")
new_password = st.text_input("Password",type='password')

if st.button("Signup"):
	create_usertable()
	add_userdata(new_user,make_hashes(new_password))
	st.success("You have successfully created a valid Account")

