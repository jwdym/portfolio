import os
import datetime
import prophet
import plotly
import sklearn
import random
import re

import pandas as pd
import numpy as np
import plotly.graph_objects as go

from prophet import Prophet
from streamlit.components.v1 import html
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score


def stock_data_forecast(df: pd.DataFrame, date_col: str, forecast_metric: str, forecast_window: int):
    """
    Function to get the stock forecast for a specific stock
    Inputs:
        df (pd dataframe): a pandas dataframe of stock data
        stock_symbol (str): the specific stock symbol to predict for
        forecast_metric (str): the stock data to forecast on
        forecast_window (int): the look forward prediction window
    Output:
        forecast (pd dataframe): a pandas dataframe of stock forecast
    """

    # get date sequence to fill in gaps
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    date_sequence = pd.date_range(start=min_date,end=max_date)
    date_df = pd.DataFrame({'Date': date_sequence})

    # join data frames
    temp_df = date_df.merge(df, on='Date', how='left')
    # fill in weekend and holidays with last value
    temp_df.fillna(method='ffill', inplace=True)

    # generate forecast
    m = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True, seasonality_mode='multiplicative')
    m.add_seasonality(name='monthly', period=30.5, fourier_order=4)
    m.add_seasonality('quarterly', period=91.25, fourier_order=8)
    m.add_country_holidays(country_name='US')
    model_df = temp_df[['Date', forecast_metric]].rename(columns={"Date": "ds", forecast_metric: "y"})
    m.fit(model_df)
    future = m.make_future_dataframe(periods=forecast_window)
    forecast = m.predict(future)

    # return forecast
    return(m, forecast)

def regression_scores(df: pd.DataFrame, y_true: str, y_pred: str):
    """
    Function to get regression evaluation scores for a model
    Inputs:
        df (Pandas Dataframe): The dataframe with obs and predictions
        y_true (str): The observed metric
        y_pred (str): The predicted metric
    outputs:
        model_scores (dict): The model evaluation scores
    """

    # Get model scores
    model_scores = {
        'explained_variance_score': explained_variance_score(df[y_true], df[y_pred]),
        'max_error': max_error(df[y_true], df[y_pred]),
        'mean_absolute_error': mean_absolute_error(df[y_true], df[y_pred]),
        'mean_squared_error': mean_squared_error(df[y_true], df[y_pred]),
        'r2_score': r2_score(df[y_true], df[y_pred]),
    }

    return(model_scores)

def plot_forecast_vs_actual(forecast_df: pd.DataFrame, df: pd.DataFrame, forecast_metric: str, date_col: str):
    """
    Function to plot the forecasted stock values vs observed
    Inputs:
        forecast_df (Pandas Dataframe): Forecast data set
        df (Pandas Dataframe): The holdout data set
        forecast_metric (str): The metric being forecast
        date_col (str): The date column of the data sets
    outputs:
        fig: A plotly graph object
        model_scores (dict): The model evaluation scores
    """
    # Get data
    temp_df = df.merge(forecast_df, how='inner', left_on='Date', right_on='ds')
    temp_df = temp_df[['Date', forecast_metric, 'yhat_lower', 'yhat', 'yhat_upper']]

    # Create plots
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=temp_df[date_col], y=temp_df.yhat_upper, connectgaps=True, name='Prediction Upper'))
    fig.add_trace(go.Scatter(x=temp_df[date_col], y=temp_df.yhat_lower, connectgaps=True, name='Prediction Lower', fill='tonexty'))
    fig.add_trace(go.Scatter(x=temp_df[date_col], y=temp_df.yhat, connectgaps=True, name='Prediction'))
    fig.add_trace(go.Scatter(x=temp_df[date_col], y=temp_df.Close_Price, connectgaps=True, name='Close Price'))

    fig.update_traces(hoverinfo='text+name', mode='lines+markers')
    fig.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=16))

    return(fig)

def nav_page(page_name, timeout_secs=3):
    nav_script = """
        <script type="text/javascript">
            function attempt_nav_page(page_name, start_time, timeout_secs) {
                var links = window.parent.document.getElementsByTagName("a");
                for (var i = 0; i < links.length; i++) {
                    if (links[i].href.toLowerCase().endsWith("/" + page_name.toLowerCase())) {
                        links[i].click();
                        return;
                    }
                }
                var elasped = new Date() - start_time;
                if (elasped < timeout_secs * 1000) {
                    setTimeout(attempt_nav_page, 100, page_name, start_time, timeout_secs);
                } else {
                    alert("Unable to navigate to page '" + page_name + "' after " + timeout_secs + " second(s).");
                }
            }
            window.addEventListener("load", function() {
                attempt_nav_page("%s", new Date(), %d);
            });
        </script>
    """ % (page_name, timeout_secs)
    html(nav_script)

# Natural Language Toolkit: Chatbot Utilities
#
# Copyright (C) 2001-2023 NLTK Project
# Authors: Steven Bird <stevenbird1@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

# Based on an Eliza implementation by Joe Strout <joe@strout.net>,
# Jeff Epler <jepler@inetnebr.com> and Jez Higgins <jez@jezuk.co.uk>.
reflections = {
    "i am": "you are",
    "i was": "you were",
    "i": "you",
    "i'm": "you are",
    "i'd": "you would",
    "i've": "you have",
    "i'll": "you will",
    "my": "your",
    "you are": "I am",
    "you were": "I was",
    "you've": "I have",
    "you'll": "I will",
    "your": "my",
    "yours": "mine",
    "you": "me",
    "me": "you",
}

class Chat:
    def __init__(self, pairs, reflections={}):
        """
        Initialize the chatbot.  Pairs is a list of patterns and responses.  Each
        pattern is a regular expression matching the user's statement or question,
        e.g. r'I like (.*)'.  For each such pattern a list of possible responses
        is given, e.g. ['Why do you like %1', 'Did you ever dislike %1'].  Material
        which is matched by parenthesized sections of the patterns (e.g. .*) is mapped to
        the numbered positions in the responses, e.g. %1.

        :type pairs: list of tuple
        :param pairs: The patterns and responses
        :type reflections: dict
        :param reflections: A mapping between first and second person expressions
        :rtype: None
        """

        self._pairs = [(re.compile(x, re.IGNORECASE), y) for (x, y) in pairs]
        self._reflections = reflections
        self._regex = self._compile_reflections()

    def _compile_reflections(self):
        sorted_refl = sorted(self._reflections, key=len, reverse=True)
        return re.compile(
            r"\b({})\b".format("|".join(map(re.escape, sorted_refl))), re.IGNORECASE
        )

    def _substitute(self, str):
        """
        Substitute words in the string, according to the specified reflections,
        e.g. "I'm" -> "you are"

        :type str: str
        :param str: The string to be mapped
        :rtype: str
        """

        return self._regex.sub(
            lambda mo: self._reflections[mo.string[mo.start() : mo.end()]], str.lower()
        )

    def _wildcards(self, response, match):
        pos = response.find("%")
        while pos >= 0:
            num = int(response[pos + 1 : pos + 2])
            response = (
                response[:pos]
                + self._substitute(match.group(num))
                + response[pos + 2 :]
            )
            pos = response.find("%")
        return response

    def respond(self, str):
        """
        Generate a response to the user input.

        :type str: str
        :param str: The string to be mapped
        :rtype: str
        """

        # check each pattern
        for (pattern, response) in self._pairs:
            match = pattern.match(str)

            # did the pattern match?
            if match:
                resp = random.choice(response)  # pick a random response
                resp = self._wildcards(resp, match)  # process wildcards

                # fix munged punctuation at the end
                if resp[-2:] == "?.":
                    resp = resp[:-2] + "."
                if resp[-2:] == "??":
                    resp = resp[:-2] + "?"
                return resp

    # Hold a conversation with a chatbot
    def converse(self, user_input, quit="quit"):
        if user_input == quit:
            return('Closing chat')
        else:
            while user_input[-1] in "!.":
                user_input = user_input[:-1]
            return(self.respond(user_input))