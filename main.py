import plotly.express as px
import streamlit as st
import pandas as pd

st.title("Getting start")


code = """
def helloWorld():
    print("Hello world!")
"""
show_btn = st.button("Show code!")
if show_btn:
    st.code(code, language="python")


age_int = st.number_input("Input a number")
st.markdown(f"Your number is {age_int}")


df = pd.DataFrame({'col1': [1, 2, 3, 4, 8], 'col2': [5, 6, 7, 8, 10]})
st.dataframe(df)

show_line = st.button("Show linechart!")
if show_line:
    st.line_chart(df, x='col1', y='col2')


df = px.data.iris()  # iris is a pandas DataFrame
fig = px.scatter(df, x="sepal_width", y="sepal_length")

st.plotly_chart(fig, key="iris", on_select="rerun")
