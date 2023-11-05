import os
import datetime
import sklearn

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from sklearn.impute import KNNImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from utils import regression_scores

################
# Introduction #
################
st.title('Data Analysis')
st.markdown('This page provides some examples of data analysis in conjuction with modeling to show powerful good visualizations can be.')
st.markdown('The data used in this example comes from [kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques), and the modeling is from [my github](https://github.com/jwdym/home-price-regression/tree/main)')
st.divider()

st.markdown(
    """
    This page has several steps, some of which are interactive:
    1. Loading data
    2. Transforming data
    3. Training a model
    4. Evaluating a model

    The data wrangling stages (loading and transforming) require manual investigation to format the data correctly, once the data is formatted any number of algorithms can be trained and evaluated on the data.
    """
)

# Read in data
show_fe = st.toggle('Display Feature Engineering')

base_path = os.getcwd()
df = pd.read_csv(os.path.join(base_path, 'data/homes/train.csv'))

if show_fe:
    # Display dataframe
    st.subheader('Data Set')
    st.markdown(f'Columns: {df.shape[1]}')
    st.markdown(f'Rows: {df.shape[0]}')
    st.dataframe(df.head(5))
    st.divider()

    st.markdown(
        """
        The raw data has lots of features which could prove useful in developing a model to predict sale price, there's also some information which won't be helpful at all that we'll need to remove before modeling.
        To start we'll look at how much information is missing from some of the data.
        """
    )

    # Quantify amount of missing data
    null_values = (np.sum(df.isna()))[np.sum(df.isna()) > 0]
    prop = (np.sum(df.isna()) / df.shape[0])[np.sum(df.isna()) > 0] * 100
    missing_data = pd.concat([null_values, prop], axis=1, keys=['Total', 'Percentage'])
    missing_data.sort_values(by='Total', ascending=False, inplace=True)
    st.dataframe(
        missing_data,
        column_config={
            "Total": st.column_config.NumberColumn(
                "Missing Count",
                help="The number of elements missing from the column",
                format="%d",
            ),
            "Percentage": st.column_config.ProgressColumn(
                "Missing %",
                help="The percentage of elements missing from the column",
                format="%d",
                min_value=0.0,
                max_value=100.0,
            ),
        }
    )

    st.markdown(
        """
        There are methods for handling missing data from removing the data entirely to imputing the missing values. For some of the data we'll have to simply remove the features entirely since there's too much missing.
        """
    )

#######################
# Feature engineering #
#######################
with st.spinner('Feature engineering in progress...'):
    # Identify numerical data type columns
    num_cols = list(df.columns[df.dtypes == np.int64])
    num_cols += list(df.columns[df.dtypes == np.float64])

    # Identify categorical data type columns
    cat_cols = list(set(df.columns) - set(num_cols))

    # Drop columns with lots of missing data and target column
    drops = ['Alley', 'PoolQC', 'Fence', 'FireplaceQu', 'MiscFeature', 'SalePrice']
    na_cols = ['Alley', 'PoolQC', 'Fence', 'FireplaceQu', 'MiscFeature']
    cat_cols = list(set(cat_cols) - set(drops))
    num_cols = list(set(num_cols) - set(drops))

    # Convert date columns to categorical
    df['cat_YrSold'] = df.YrSold.astype(str)
    df['cat_MoSold'] = df.MoSold.astype(str)
    cat_cols += ['cat_YrSold', 'cat_MoSold']

    # Save columns
    cols = {'na_cols': na_cols, 'cat_cols': cat_cols, 'num_cols': num_cols}

    # Setup categorical data encoder
    print('Encoding categorical data')
    cat_categories = []
    for c in cat_cols:
        cat_categories.append(list(set(df[c])))

    enc = OneHotEncoder(categories=cat_categories, handle_unknown='ignore')
    enc.fit(df[cat_cols])

    # Format training data and impute missing values
    df_imp = pd.concat([pd.DataFrame(enc.transform(df[cat_cols]).toarray(), columns=enc.get_feature_names_out(cat_cols)), df[num_cols], df[na_cols].isna() * 1], axis=1)

if show_fe:
    st.markdown(
        """
        There are lots of categorical features in the data set that could prove useful in predicting home sale price, before using categorical data in a model it will need to be formatted in binary dummy columns.
        """
    )
    # One hot encoding
    categorical_column = st.selectbox(
        label='Categorical Column',
        options=cat_cols
    )
    fig = px.histogram(
        df,
        categorical_column
    )
    st.plotly_chart(fig)

if show_fe:
    st.markdown(
        """
        Numeric features are easy to work, but can require scaling and centering depending on the algorithm being used. Luckily for us the tree based algorithm used in this page does not require any adjustment to numeric features.
        """
    )
    numeric_column = st.selectbox(
        label='Numeric Column',
        options=num_cols
    )
    n_bins = st.number_input(
        label='Chart Bins',
        min_value=1,
        max_value=100,
        value=20,
        step=1
    )
    fig = px.histogram(
        df,
        numeric_column,
        nbins=n_bins
    )
    st.plotly_chart(fig)

if show_fe:
    st.subheader('Transformed Data Set')
    st.markdown(f'Columns: {df_imp.shape[1]}')
    st.markdown(f'Rows: {df_imp.shape[0]}')
    st.dataframe(df_imp.head(5))

    st.markdown(
        f"""
        The transformed data has been one hot encoded, had missing data removed, and had missing data imputed if there was enough existing data to work with. 
        """
    )
st.divider()

st.subheader('Model Training')
show_hyperparameters = st.toggle('Display Hyperparameters')
if show_hyperparameters:
    st.markdown(
        """
        Hyperparameter details can be found on [Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor)
        """
    )
    # Hyperparameters
    loss = st.selectbox(
        label='Loss',
        options=['squared_error', 'absolute_error', 'huber', 'quantile']
    )
    learning_rate = st.number_input(
        label='Learning Rate',
        min_value=0.0,
        max_value=100000.0,
        value=0.1,
        step=0.01
    )
    n_estimators = st.number_input(
        label='# Boosting Stages',
        min_value=1,
        max_value=1000,
        value=100,
        step=1
    )
    subsample = st.number_input(
        label='Subsample',
        min_value=0.01,
        max_value=1.0,
        value=1.0,
        step=0.01
    )
    criterion = st.selectbox(
        label='Criterion',
        options=['friedman_mse', 'squared_error']
    )
    min_samples_split = st.number_input(
        label='Min Samples Split',
        min_value=2,
        max_value=1000,
        value=2,
        step=1
    )
    min_samples_leaf = st.number_input(
        label='Min Samples Leaf',
        min_value=1,
        max_value=1000,
        value=1,
        step=1
    )
    min_weight_fraction_leaf = st.number_input(
        label='Min Weight Fraction Leaf',
        min_value=0.0,
        max_value=0.5,
        value=0.0,
        step=0.01
    )
    max_depth = st.number_input(
        label='Max Depth',
        min_value=1,
        max_value=1000,
        value=3,
        step=1
    )
    min_impurity_decrease = st.number_input(
        label='Min Impurity Decrease',
        min_value=0.0,
        max_value=1000.0,
        value=0.0,
        step=0.01
    )
    max_features_auto = st.toggle(
        label='Auto Max Features',
        value=True
    )
    if max_features_auto:
        max_features = st.selectbox(
            label='Max Features',
            options=['sqrt', 'log2', 'None'],
            placeholder='None',
            key='Max Features Auto'
        )
    else:
        max_features = st.number_input(
        label='Max Features',
        min_value=1,
        max_value=df_imp.shape[1] - 1,
        value=100,
        step=1,
        key='Max Features Manual'
    )

show_model = st.toggle(
    label='Display Model Panel',
    value=False
)
if show_model:
    st.markdown(
        f"""
        The holdout set is the proportion of data withheld from the model for later evaluation, the idea being that a model can perfectly learn the inputs and outputs of a data set it is trained on and therefore can provide an unrealistic image of how well the model would actual perform.
        With {df_imp.shape[0]} rows of data we can allocate some to the holdout set for later evaluation.
        Try changing the holdout set size and some of the hyperparameters, if you set the model as a "strong learner" it should perform well on the training data but very poorly on the holdout set, this is called overfitting.
        """
    )
    holdout_proportion = st.number_input(
        label='Holdout Proportion',
        min_value=0.0,
        max_value=0.5,
        value=0.2,
        step=0.01
    )

    if st.button('Run Model'):
        with st.spinner('Training model...'):
            # Creat train/test set
            imputer = KNNImputer(n_neighbors=5)
            x_train = imputer.fit_transform(df_imp)
            y_train = df.SalePrice

            if holdout_proportion > 0:
                x_train, x_test, y_train, y_test = train_test_split(
                    x_train,
                    y_train,
                    random_state=42,
                    test_size=0.2
                )

            # Setup and store params in session state
            if max_features == 'None':
                max_features = None

            params = {
                'loss': loss,
                'learning_rate': learning_rate,
                'n_estimators': n_estimators,
                'subsample': subsample,
                'criterion': criterion,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'min_weight_fraction_leaf': min_weight_fraction_leaf,
                'max_depth': max_depth,
                'min_impurity_decrease': min_impurity_decrease,
                'max_features': max_features
            }

            model = GradientBoostingRegressor(**params)
            model.fit(x_train, y_train)
            model_train_scores = regression_scores(
                    df=pd.DataFrame({'Obs': y_train, 'Pred': model.predict(x_train)}), 
                    y_true='Obs',
                    y_pred='Pred'
                )
            if holdout_proportion > 0:
                model_test_scores = regression_scores(
                    df=pd.DataFrame({'Obs': y_test, 'Pred': model.predict(x_test)}), 
                    y_true='Obs',
                    y_pred='Pred'
                )
            else:
                model_test_scores = {
                    'explained_variance_score': None,
                    'max_error': None,
                    'mean_absolute_error': None,
                    'mean_squared_error': None,
                    'r2_score': None
                }

            if 'model_run_key' in st.session_state:
                model_run_key = st.session_state.model_run_key + 1
                st.session_state.model_run_key = model_run_key
                st.session_state.hp_params.append(
                    {
                        'Model Run': model_run_key,
                        'Model Parameters': params,
                        'Model Train Scores': model_train_scores,
                        'Model Test Scores': model_test_scores,
                        'Test Proportion': holdout_proportion
                    }
                )
            else:
                model_run_key = 1
                st.session_state['model_run_key'] = model_run_key
                st.session_state['hp_params'] = []
                st.session_state.hp_params.append(
                    {
                        'Model Run': model_run_key,
                        'Model Parameters': params,
                        'Model Train Scores': model_train_scores,
                        'Model Test Scores': model_test_scores,
                        'Test Proportion': holdout_proportion
                    }
                )

        st.markdown('#### Model Report')
        st.json(
            st.session_state.hp_params[-1],
            expanded=False
        )
st.divider()
st.subheader('Model Evaluation')
show_model_eval = st.toggle(
    label='Display Model Evaluation',
    value=False
)
if show_model_eval:
    if 'model_run_key' in st.session_state:
        model_run_key = st.session_state.model_run_key
        st.markdown('#### Model Training History')
        st.markdown(f'Trained Models: {st.session_state.model_run_key}')

        # Format metrics and params for report generation
        model_metrics = st.session_state.hp_params[-1]['Model Train Scores'].keys()
        model_params = st.session_state.hp_params[-1]['Model Parameters'].keys()
        model_eval_json = {}
        model_eval_json[f'Model Run'] = []
        model_eval_json[f'Test Proportion'] = []
        for param in model_params:
            model_eval_json[param] = []
        for metric in model_metrics:
            model_eval_json[f'Training {metric}'] = []
            model_eval_json[f'Testing {metric}'] = []
        for report in st.session_state.hp_params:
            model_eval_json[f'Model Run'].append(report['Model Run'])
            model_eval_json[f'Test Proportion'].append(report['Test Proportion'])
            for metric in model_metrics:
                model_eval_json[f'Training {metric}'].append(report['Model Train Scores'][metric])
                model_eval_json[f'Testing {metric}'].append(report['Model Test Scores'][metric])
            for param in model_params:
                model_eval_json[param].append(report['Model Parameters'][param])
        model_eval_df = pd.DataFrame(model_eval_json)

        # Best Model
        st.markdown(f"Best Model: {model_eval_df['Testing mean_squared_error'].argmin() + 1}")

        # Model training history
        st.dataframe(model_eval_df, hide_index=True)

        # Plot Testing metrics over time
        eval_metric = st.selectbox(
            label='Evaluation Metric',
            options=model_metrics,
            placeholder='mean_squared_error'
        )
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=model_eval_df['Model Run'],
                y=model_eval_df[f'Testing {eval_metric}'],
                mode='lines',
                name='Test Performance'
            )
        )
        fig.add_trace(
            go.Scatter(
                x=model_eval_df['Model Run'],
                y=model_eval_df[f'Training {eval_metric}'],
                mode='lines',
                name='Training Performance'
            )
        )
        fig.update_layout(
            title=f'Training/Testing {eval_metric} Over Model Runs',
            xaxis_title='Model Runs',
            yaxis_title=eval_metric
        )
        st.plotly_chart(fig, use_container_width=True)

        # Plot parameter ranges and objective metric
        model_param = st.selectbox(
            label='Model Parameter',
            options=model_params
        )
        fig = px.scatter(model_eval_df, x=model_param, y=f'Testing {eval_metric}')
        st.plotly_chart(fig, use_container_width=True)

        if st.toggle('Show Subplots', value=False):
            fig = make_subplots(
                rows=4,
                cols=3,
                specs=[
                    [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
                    [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
                    [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
                    [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]
                ]
            )
            counter = 0
            row_col = [[r, c] for r in [1, 2, 3, 4] for c in [1, 2, 3]]
            for param in model_params:
                fig.add_trace(
                    go.Scatter(
                        x=model_eval_df[param],
                        y=model_eval_df[f'Testing {eval_metric}'],
                        mode='markers'
                    ),
                    row=row_col[counter][0], col=row_col[counter][1]
                )
                counter+=1
            fig.update_layout(
                title=f'Parameter vs Test {eval_metric} ',
                xaxis_title='Parameter',
                yaxis_title=eval_metric
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown('No model training history yet.')