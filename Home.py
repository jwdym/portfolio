import os

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils import nav_page
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Jacob's Website",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Introduction
st.markdown("# My Portfolio")
st.markdown(
    """
    This portfolio contains examples of projects I've worked on, some in a professional environment and others from personal exploration. The examples in this website are light and interactive, so have fun playing with some data science concepts!

    If you want to get in contact with me you can email me <jacobwdym@gmail.com> or reach out on [LinkedIn](https://www.linkedin.com/) 
    """
)

st.header('Forecasting')
forecast_image = Image.open('media/forecast_image.png')
st.image(forecast_image, width=500)
st.markdown(
    """
    I have a background in Bayesian and frequentist time series analysis, and have built out deep neural network systems for various types of forecasting before. 

    The button below will navigate you to an interactive page for forecasting some old stock market data.
    """
)
if st.button("Forecasting"):
    nav_page("Forecasting")

st.header('Natural Language Processing (NLP)')
nlp_image = Image.open('media/chatbot_image.png')
st.image(nlp_image, width=500)
st.markdown(
    """
    Natural Language Processing or NLP has exploded in popularity with the release of Large Language Models (LLMs) such as Chat GPT and others.

    I've worked with NLP in a professional and personal setting off and on over the course of 5 years.
    The button below will navigate you to an interactive page with some LLM capabilities and some other fun NLP uses.
    """
)
if st.button("NLP"):
    nav_page("NLP")

st.header('Data Analysis')
data_analysis_image = Image.open('media/data_analysis_image.png')
st.image(data_analysis_image, width=500)
st.markdown(
    """
    Data analysis is the corner stone of any ML applications, being able to proplerly structure data and gain insights from it is fundamental in any model building process.

    I've been doing data analysis for over 7 years in different applications from prediccting grizzly bear range expansion in Yellostone to building out all sorts of dashboards for teams.
    
    I put together a page to demonstrate some different analysis methods on a home sale data set, along with a simple predictive model to explore how data builds models.
    """
)
if st.button("Data Analysis"):
    nav_page("Data_Analysis")