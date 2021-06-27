# Attendance Quality Analyser

Attendance Quality Analyser is a project deployed using streamlit.
## Team Members:
Aditi Kulkarni<br>
Dania Juvale <br>
Nandinee Kaushik<br>
Reeha Parkar <br>
Shruti Waghade<br>

## Project Overview:

With the increase of distance learning, e-learning faces a number of issues to the teachers when it comes to gauging the engagement level of students. The project identifies the same and implements a system that detects the engagement level of the students. This is carried out by the typical built-in web-camera present in a laptop, computer, and is designed to work in real time. The objective is to reduce manual process errors by providing automation, and a reliable attendance system that uses face recognition technology.

## Project Features:

The app is a holistic web app that can be utilized by both, teachers as well as students.
The main objective of the app is to improve the quality of the teaching-learning process, hence proving useful for educational institutions.

The student only has to turn on their web camera at the start of a lecture and then at the end of it, mark their attendance at the click of a button.

While this is happening, the student's behaviour is analysed and features like closing of eyes, yawning, and emotions during lecture are extracted and quantified. This is converted into a numerical factor that determines if the student is focused or distracted.

The teacher, after authentication with a username and password, can then observe the student's attendance and engagaement level for a day or a range of days. All this is visualised and student's general engagement level is identified.

## General Requirements:

[streamlit](https://streamlit.io/)

[opencv](https://opencv.org/)

[keras](https://keras.io/)

[dlib](http://dlib.net/)

[face_recognition](https://pypi.org/project/face-recognition/)

## To run the app:

Traverse to Streamlit UI, and run the following


```
streamlit app run app.py
```