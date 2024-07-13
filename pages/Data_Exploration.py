import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Load and preprocess data
data = pd.read_csv('C:\\Users\\Moritus Peters\\Documents\\Datasets\\bank-additional-full.csv', delimiter=';')
month_Matching = {
    'jan': 'January',
    'feb': 'February',
    'mar': 'March',
    'apr': 'April',
    'may': 'May',
    'jun': 'June',
    'jul': 'July',
    'aug': 'August',
    'sep': 'September',
    'oct': 'October',
    'nov': 'November',
    'dec': 'December'
}
data['month'] = data['month'].replace(month_Matching)
data['month'] = pd.Categorical(data['month'], categories=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], ordered=True)
data['day_of_week'] = pd.Categorical(data['day_of_week'], categories=['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'], ordered=True)
data = data.rename(columns={
    'previous': "Previous_contacts_distribution",
    'poutcome': 'Outcome_of_previous_marketing_campaign',
    'emp.var.rate': 'Employment_variation_rate',
    'cons.price.idx': 'Consumer_price_index',
    'cons.conf.idx': 'Consumer_confidence_index',
    'euribor3m': 'Euribor 3 month rate',
    'nr.employed': 'Number of employees',
    'y': 'Deposit'
})

available_indicators = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
                        'pdays', 'Previous_contacts_distribution', 'Outcome_of_previous_marketing_campaign',
                        'Employment_variation_rate', 'Consumer_price_index', 'Consumer_confidence_index',
                        'Euribor 3 month rate', 'Number of employees', 'Deposit']



layout = html.Div([
    html.H2('Data Exploration'),
    # Add your overview content here
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Label("Month Checklist", className='dropdown-label text'),
                dbc.Checklist(
                    id='month_checklist',
                    options=[{'label': str(month), 'value': month} for month in sorted(data['month'].unique(), key=lambda x: data['month'].cat.categories.tolist().index(x))],
                    value=[],
                    inline=True,
                    className='text-center px-2'
                )
            ], className='metric-container')
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H5('Conversion Rate by Age Group and Loan', className='text-center '),
                    dcc.Graph(id='age_group_and_loan_distribution', figure={}),
                ])
            ), width=6
        ),
        dbc.Col(
            dbc.Card([
               # dbc.CardHeader("Conversion Rate by Job and Marital Status",  className='text-center fs-2'),
                dbc.CardBody([
                    html.H5('Conversion Rate by Job and Marital Status', className='text-center'),
                    dcc.Graph(id='job_and_marital', figure={}),
                ]),
                #dbc.CardFooter("Thia bar chart shows the conversion rate of the percentage of 'yes' deposits based on job type and marital status. more insight is gotten by selecting each monthor more for actionable descision making "),
    ]), width=6
        ),
    ]),
    dbc.Row([
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H5('Correlation of Features with Deposit', 
                            className='text-center',),
                    dcc.Graph(id='Features_Deposit', figure={}),
                ])
            ),width=7
        ),
        dbc.Col([
        dbc.Card(
            dbc.CardBody([
                html.H5('Conversion Rate by Month and Week',
                        className='text-center'),
                dcc.Graph(id='Month_and_week_conversion', figure={}),
            ])
        ),
    ], width=5),
    ])
])

# Callbacks for charts
@callback(Output('age_group_and_loan_distribution', 'figure'),
          Input('month_checklist', 'value'),
          suppress_callback_exceptions=True)

def update_age_group_and_loan_distribution(selected_month=None):
    if not selected_month:
        return {}
    filtered_data = data[data['month'].isin(selected_month)]                                #                                                    
    filtered_data['age_group'] = pd.cut(filtered_data['age'], bins=[0, 20, 30, 40, 50, 60, np.inf], labels=['<20', '20-30', '30-40', '40-50', '50-60', '60+'])
    conversion_rate_by_age_housing = filtered_data.groupby(['age_group', 'loan'])['Deposit'].apply(lambda x: (x == 'yes').mean() * 100).unstack(fill_value=0)
    fig = px.bar(conversion_rate_by_age_housing, barmode='group')
    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    return fig

@callback(Output('job_and_marital', 'figure'),
          Input('month_checklist', 'value'),
          suppress_callback_exceptions=True)

def update_job_and_marital(selected_month=None):
    if not selected_month:
        return {}
    filtered_data = data[data['month'].isin(selected_month)]
    conversion_rate_by_job_marital = filtered_data.groupby(['job', 'marital'])['Deposit'].apply(lambda x: (x == 'yes').mean() * 100).unstack(fill_value=0)
    fig = px.bar(conversion_rate_by_job_marital, barmode='group')
    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    return fig

@callback(Output('Features_Deposit', 'figure'),
          Input('month_checklist', 'value'),
          suppress_callback_exceptions=True)

def update_features_deposit(selected_month=None):
    if not selected_month:
        return {}
    filtered_data = data[data['month'].isin(selected_month)]
    
    # Encode the Deposit column
    filtered_data['Deposit'] = filtered_data['Deposit'].map({'yes': 1, 'no': 0})

    categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'Outcome_of_previous_marketing_campaign']
    
    # Convert categorical variables to numerical for correlation calculation
    data_encoded = pd.get_dummies(filtered_data, columns=categorical_columns, drop_first=True)

    # Calculate the correlation matrix
    correlation_matrix = data_encoded.corr()

    # Extract correlations with 'Deposit'
    corr_with_deposit = correlation_matrix['Deposit'].sort_values(ascending=False)

    # Create a table to display the correlation values with 'Deposit'
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Feature', 'Correlation with Deposit'],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[corr_with_deposit.index, corr_with_deposit.values],
                   fill_color='lavender',
                   align='left'))
    ])
    return fig

@callback(Output('Month_and_week_conversion', 'figure'),
              Input('month_checklist', 'value'),
              suppress_callback_exceptions=True)

def update_month_week_conversion(selected_month=None):
    if not selected_month:
        return {}
    filtered_data = data[data['month'].isin(selected_month)]
    conversion_rate_by_month_week = filtered_data.groupby(['month', 'day_of_week'])['Deposit'].apply(lambda x: (x == 'yes').mean() * 100).unstack(fill_value=0)
    fig = go.Figure(data=go.Heatmap(
        z=conversion_rate_by_month_week.values,
        x=conversion_rate_by_month_week.columns.tolist(),
        y=conversion_rate_by_month_week.index.tolist(),
        colorscale='Viridis'))
    fig.update_layout( xaxis_title='Weekday', yaxis_title='Month')
    return fig