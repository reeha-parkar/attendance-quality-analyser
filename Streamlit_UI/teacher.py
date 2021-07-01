from pandas.core.indexes.base import Index
import streamlit as st
import SessionState
import hashlib
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

STUDENTS = ['Aditi', 'Dania', 'Nandinee','Reeha','Shruti']
session_state = SessionState.get(checkboxed=False)

def app():
	def make_hashes(password):
		return hashlib.sha256(str.encode(password)).hexdigest()

	def check_hashes(password,hashed_text):
		if make_hashes(password) == hashed_text:
			return hashed_text
		return False

	# DB Management
	import sqlite3 
	conn = sqlite3.connect('data.db')
	c = conn.cursor()

	# DB  Functions
	def login_user(username,password):
		c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
		data = c.fetchall()
		return data

	
	st.sidebar.subheader("Login to view Dashboard")
	username = st.sidebar.text_input("User Name")
	password = st.sidebar.text_input("Password", type='password')
	hashed_pswd = make_hashes(password)
	login = st.sidebar.button("Login")
	check = False
	try:
		check = session_state.checkboxed
	except:
		pass
	if login or check:
		result = login_user(username,check_hashes(password,hashed_pswd))

		if result:
			session_state.checkboxed = True
			st.sidebar.success("Logged in as {}".format(username))
			st.title("Welcome Teacher👩🏽‍🏫")
			option = st.selectbox('Select Student',tuple(STUDENTS))
			today = datetime.date.today()
			startdate= st.date_input('Start date', today)
			enddate= st.date_input('End date', today)

			if(st.button("Fetch Results")):
				df = pd.read_csv("../eval.csv")
				df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
				df['date'] = df['date'].dt.date
				mask = (df['date'] >=startdate) & (df['date'] <= enddate)
				temp=df.loc[(df['name']==option) & mask]

				if(len(temp)>0):
					st.success('Fetching Results for {}'.format(option))
					st.dataframe(temp)
					plt.style.use('seaborn-pastel')

					#bar plot for per session t_focused and t_distracted
					fig1 = plt.figure()
					x=np.arange(len(temp))
					width = 0.2
					plt.bar(x-0.2, temp['t_focused'], width)
					plt.bar(x,temp['t_distracted'] , width)
					plt.bar(x+0.2,temp['t_total'], width, )
					plt.xticks(x, temp['date'],fontsize=7,rotation='vertical')
					plt.xlabel("Dates")
					plt.ylabel("Session Time")
					plt.legend(["Focused", "Distracted", "Total"])
					st.pyplot(fig1)

					#pie chart for collective focused and distracted in given range
					col1, col2 = st.beta_columns(2)
					fig2 = plt.figure()
					plt.title("Attention Quality for Date Range",y=-0.01)
					plt.pie(temp['Quality'].value_counts())
					plt.legend(labels=['focused','distracted'],loc="center left")
					col1.pyplot(fig2)
					col2.subheader("")
					col2.header("Student Stats")
					col2.subheader("Days Present: {}".format(len(temp)))
					col2.subheader("Days Focused: {}".format(temp['Quality'].value_counts()[0]))
					if(temp['Quality'].value_counts()[0]/len(temp)>=0.7):	
						col2.success("{} is overall Focused".format(option))
					else:
						col2.warning("{} is overall Distracted".format(option))

				else:
					st.warning("No Results found for {} in given range".format(option))				
		else:
			st.sidebar.warning("Incorrect Username/Password")