import os
import prophet
import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from utils import stock_data_forecast, plot_forecast_vs_actual, regression_scores

# Introduction
st.title('Forecasting')
st.markdown('Forecasting is a type of predictive modeling that takes a sequence of inputs $n_{1}$, $n_{2}$, ..., $n_{k}$ and tries to predict what the next sequence $n_{k+1}$, $n_{k+2}$, ... will be based off learned historic trends.')
st.markdown('This forecasting example will use stock data through 2022 and try to predict what the next $N$ days of closing prices will be.')
st.markdown("Forecasts for this example were generated using [Prophet](https://facebook.github.io/prophet/) (since it's easy)")

st.subheader('Forecasting Projects')
st.markdown(
    f""" The work shown in this page is a very simple example of forecasting.
    I'm working on an application that leverages live stock data to:

    1. Make forecasts on stocks, similar to this page but with time granularity down to the hour and minute.
    2. Summarize the latest news to assess market impacts
    3. Provide portfolio building and analysis tools leveraging advanced analytics

    And more! 

    This is a personal project I work on when I have time and am motivated. If you would like to know more or want to help contribute reach out to me at <jacobwdym@gmail.com>.
"""
)
st.divider()

# Setup base args
data_dir = os.path.join(os.getcwd(), 'data/stocks')

# Specify stock to get forecasts for
stock_symbol = st.selectbox(
    label='Select stock',
    options = [s.split('.')[0] for s in os.listdir(data_dir)]
)

# Forecast period
st.markdown("Choose how far ahead in time you want the model to forecast, longer forecast windows usually yield poorer results.")
forecast_period = st.number_input(
    label='Forecast Window',
    min_value=7,
    max_value=180,
    value=30,
    step=1
)

# Read in data
df = pd.read_csv(os.path.join(data_dir, stock_symbol + '.csv'))
df['Date'] = pd.to_datetime(df['Date'])

# Train/test split
test_cutoff = df['Date'].max() - datetime.timedelta(days = forecast_period)
df['Test_Set'] = (df.Date >= df.loc[forecast_period-1]['Date']).astype(int)
train_df = df.loc[df['Test_Set'] == 0]
test_df = df.loc[df['Test_Set'] == 1]

# Plot data
st.markdown("In order to better evaluate how the forecast model performs, we'll withold some future information to test the forecast with.")
fig = px.line(df, x='Date', y='Close_Price', color='Test_Set')
fig.add_vline(x=df.loc[forecast_period-1]['Date'])
st.plotly_chart(fig)


st.subheader('Making Forecasts')
st.divider()
st.markdown(
    """
    While we can use Prophet out of the box to model time series data, building a good forecasting model requires tuning.
    All models have parameters that affect model performance, for Prophet models we can specify how the model will handle seasonaility effects and how flexible trends are in the model.

    Below are sever parameter options, try fitting a model with different parameter combinations to see how it changes forecast results.
    """
)

# Seasonality effects
yearly_seasonality_on = st.checkbox(
    label='Yearly Seasonality',
    value=True
)
quarterly_seasonality_on = st.checkbox(
    label='Quarterly Seasonality',
    value=True
)
monthly_seasonality_on = st.checkbox(
    label='Montly Seasonality',
    value=True
)
weekly_seasonality_on = st.checkbox(
    label='Weekly Seasonality',
    value=True
)
seasonality_mode = st.selectbox(
    label='Seasonality Mode',
    options=['multiplicative', 'additive'],
    placeholder='multiplicative'
)

# Training parameters
changepoint_prior_scale = st.number_input(
    label='Changepoint Prior Scale',
    min_value=0.001,
    max_value=0.5,
    value=0.05,
    step=0.001
)
seasonality_prior_scale = st.number_input(
    label='Seasonality Prior Scale',
    min_value=0.01,
    max_value=10.0,
    value=10.0,
    step=0.01
)
holidays_prior_scale = st.number_input(
    label='Holidays Prior Scale',
    min_value=0.01,
    max_value=10.0,
    value=10.0,
    step=0.01
)
interval_width = st.number_input(
    label='Interval Width',
    min_value=0.01,
    max_value=0.99,
    value=0.8,
    step=0.01
)

st.markdown(
    """
    > **_NOTE:_** There are many other parameters to be changed on the model, see [Prophet Hyperparameters](https://facebook.github.io/prophet/docs/diagnostics.html#hyperparameter-tuning)
    """
)

# Create forecasts
if st.button(label='Make Forecast'):    
    with st.spinner(text="Forecast in progress..."):
        # get date sequence to fill in gaps
        min_date = train_df['Date'].min()
        max_date = train_df['Date'].max()
        date_sequence = pd.date_range(start=min_date, end=max_date)
        date_df = pd.DataFrame({'Date': date_sequence})

        # join data frames
        temp_df = date_df.merge(train_df, on='Date', how='left')
        # fill in weekend and holidays with last value
        temp_df.fillna(method='ffill', inplace=True)

        # generate forecast

        m = Prophet(
            daily_seasonality=False,
            yearly_seasonality=yearly_seasonality_on,
            weekly_seasonality=weekly_seasonality_on,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            interval_width=interval_width
        )
        if monthly_seasonality_on:
            m.add_seasonality(
                name='monthly', 
                period=30.5, 
                fourier_order=4
            )
        if quarterly_seasonality_on:
            m.add_seasonality(
                'quarterly', 
                period=91.25, 
                fourier_order=8
            )
        m.add_country_holidays(country_name='US')
        model_df = temp_df[['Date', 'Close_Price']].rename(columns={"Date": "ds", 'Close_Price': "y"})
        m.fit(model_df)
        future_window = (test_df['Date'].max() - train_df['Date'].max()).days
        future = m.make_future_dataframe(periods=future_window)
        forecast = m.predict(future)

    st.subheader('Forecast Model Report')
    st.markdown('These are some common metrics used for evaluating how well the model predicted stock prices.')
    temp_df = test_df.merge(forecast, how='inner', left_on='Date', right_on='ds')
    temp_df = temp_df[['Date', 'Close_Price', 'yhat_lower', 'yhat', 'yhat_upper']]
    model_scores = regression_scores(
        df=temp_df, 
        y_true='Close_Price',
        y_pred='yhat'
    )

    st.markdown(
        f"""
        - Explained Variance: {round(model_scores['explained_variance_score'], 3)} 
        - Max Error {round(model_scores['max_error'], 3)}
        - Mean Absolute Error {round(model_scores['mean_absolute_error'], 3)}
        - Mean Squared Error {round(model_scores['mean_squared_error'], 3)}
        - R2 Score {round(model_scores['r2_score'], 3)}
        """
    )
    if 'model_scores' in st.session_state:
        old_model_scores = st.session_state['model_scores']
        if 'forecast_period' in st.session_state:
            st.markdown(
            f"""
                ### Changes over last run
                - Stock: {st.session_state['stock_symbol']} -> {stock_symbol}
                - Forecast Period: {st.session_state['forecast_period']} -> {forecast_period}
                - Yearly Seasonality: {st.session_state['yearly_seasonality_on']} -> {yearly_seasonality_on}
                - Quarterly Seasonality: {st.session_state['quarterly_seasonality_on']} -> {quarterly_seasonality_on}
                - Monthly Seasonality: {st.session_state['monthly_seasonality_on']} -> {monthly_seasonality_on}
                - Weekly Seasonality: {st.session_state['weekly_seasonality_on']} -> {weekly_seasonality_on}
                - Seasonality Mode: {st.session_state['seasonality_mode']} -> {seasonality_mode}
                - Changepoint Prior Scale: {st.session_state['changepoint_prior_scale']} -> {changepoint_prior_scale}
                - Seasonality Prior Scale: {st.session_state['seasonality_prior_scale']} -> {seasonality_prior_scale}
                - Holidays Prior Scale: {st.session_state['holidays_prior_scale']} -> {holidays_prior_scale}
                """
            )
        # Setup records for next run
        st.session_state['stock_symbol'] = stock_symbol
        st.session_state['forecast_period'] = forecast_period
        st.session_state['yearly_seasonality_on'] = yearly_seasonality_on
        st.session_state['quarterly_seasonality_on'] = quarterly_seasonality_on
        st.session_state['monthly_seasonality_on'] = monthly_seasonality_on
        st.session_state['weekly_seasonality_on'] = weekly_seasonality_on
        st.session_state['seasonality_mode'] = seasonality_mode
        st.session_state['changepoint_prior_scale'] = changepoint_prior_scale
        st.session_state['seasonality_prior_scale'] = seasonality_prior_scale
        st.session_state['holidays_prior_scale'] = holidays_prior_scale

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric(
                label="Explained Variance",
                value=round(model_scores['explained_variance_score'], 3),
                delta=round(model_scores['explained_variance_score'] - old_model_scores['explained_variance_score'], 3)
            )
        with col2:
            st.metric(
                label="Max Error",
                value=round(model_scores['max_error'], 3),
                delta=round(model_scores['max_error'] - old_model_scores['max_error'], 3),
                delta_color='inverse'
            )
        with col3:
            st.metric(
                label="Mean Absolute Error",
                value=round(model_scores['mean_absolute_error'], 3),
                delta=round(model_scores['mean_absolute_error'] - old_model_scores['mean_absolute_error'], 3),
                delta_color='inverse'
            )
        with col4:
            st.metric(
                label="Mean Squared Error",
                value=round(model_scores['mean_squared_error'], 3),
                delta=round(model_scores['mean_squared_error'] - old_model_scores['mean_squared_error'], 3),
                delta_color='inverse'
            )
        with col5:
            st.metric(
                label="R2 Score",
                value=round(model_scores['r2_score'], 3),
                delta=round(model_scores['r2_score'] - old_model_scores['r2_score'], 3)
            )

    st.session_state['model_scores'] = model_scores


    st.subheader('Forecast vs Known Data')
    st.markdown('This plot shows how the predicted closing prices compared to the observed, the bands represent the range of error or reasonable values for the model.')
    fig = plot_plotly(m, forecast)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Forecast vs Unkown Data')
    st.markdown("We'll now compare the data that was witheld earlier to the predictions")
    fig = plot_forecast_vs_actual(
        forecast_df=forecast, 
        df=test_df,
        forecast_metric='Close_Price',
        date_col='Date'
    )
    st.plotly_chart(fig)