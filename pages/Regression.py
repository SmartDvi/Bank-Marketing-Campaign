import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from dash import dash_table
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

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



layout = html.Div([
    html.H2('Regression Analysis'),
    
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
                    html.H5('Model Performance Chart', 
                            className='text-center',),
                    dcc.Graph(id='Models_Performance_chart', figure={}),
                ])
            ), width=4
        ),
        dbc.Col([
            dbc.Card(
                dbc.CardBody([
                    html.H5('Actual vs Predicted Values (xgboost)',
                            className='text-center'),
                    dcc.Graph(id='Actual_vs_Predicted', figure={}),
                ])
            ),
        ], width=4),
        dbc.Col([
            dbc.Card(
                dbc.CardBody([
                    html.H5('Top 10 Feature Importances',
                            className='text-center'),
                    dcc.Graph(id='Feature_Importances', figure={}),
                ])
            ),
        ], width=4),
    ]),
    dbc.Row([
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H5('Hyperparameter Tuning Chart', 
                            className='text-center',),
                    dcc.Graph(id='Hyperparameter_Tuning', figure={}),
                ])
            ), width=4
        ),
        dbc.Col([
            dbc.Card(
                dbc.CardBody([
                    html.H5('Model Evaluation Results (Cross-Validation Scores)',
                            className='text-center'),
                    dcc.Graph(id='Model_Evaluation', figure={}),
                ])
            ),
        ], width=8),
    ]),
    dbc.Row([
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H5('Model Performance Test Table', 
                            className='text-center',),
                    html.Div(id='model_performance_test_table')
                ])
            ), width=12
        ),
    ])
])

@callback(
    Output('Models_Performance_chart', 'figure'),
    Output('Actual_vs_Predicted', 'figure'),
    Output('Feature_Importances', 'figure'),
    Output('Hyperparameter_Tuning', 'figure'),
    Output('Model_Evaluation', 'children'),
    Output('model_performance_test_table', 'children'),
    Input('month_checklist', 'value')
)
def model_performance_chart(selected_month):
    if not selected_month:
        return {}, {}, {}, {}, {}, {}

    filtered_data = data.copy()
    filtered_data = filtered_data[filtered_data['month'].isin(selected_month)]
    filtered_data.reset_index(drop=True, inplace=True)

    le = LabelEncoder()
    filtered_data['Deposit'] = le.fit_transform(filtered_data['Deposit'].values.ravel())

    filtered_data['Outcome_of_previous_marketing_campaign'] = pd.to_numeric(filtered_data['Outcome_of_previous_marketing_campaign'], errors='coerce')
    filtered_data['duration'] = pd.to_numeric(filtered_data['duration'], errors='coerce')

    filtered_data['Outcome_of_previous_marketing_campaign'].fillna(0, inplace=True)
    filtered_data['duration'].fillna(0, inplace=True)

    filtered_data['Outcome_of_previous_marketing_campaign_squared'] = filtered_data['Outcome_of_previous_marketing_campaign'] ** 2
    filtered_data['Deposit'] = filtered_data['duration'] ** 2

    filtered_data = pd.get_dummies(filtered_data, columns=['job', 'marital', 'education', 'contact', 'month', 'day_of_week'])

    X = pd.get_dummies(filtered_data.drop(columns=['Deposit']), drop_first=True)
    y = filtered_data['Deposit']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(max_iter=10000), 
        'XGBoost': XGBRegressor(random_state=42)
    }

    # Hyperparameter tuning parameters for XGBoost
    xgb_params = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
    }

    # Perform GridSearchCV for XGBoost
    xgb = XGBRegressor(random_state=42)
    grid_search_xgb = GridSearchCV(estimator=xgb, param_grid=xgb_params, scoring='r2', cv=10)
    grid_search_xgb.fit(X_train_scaled, y_train)
    best_xgb_model = grid_search_xgb.best_estimator_
    best_score_xgb = grid_search_xgb.best_score_

    # Evaluate model performance
    model_performance = {}
    cv_results = {}
    y_pred_xgb = best_xgb_model.predict(X_test_scaled)

    for name, model in models.items():
        pipeline = Pipeline([('scaler', StandardScaler()), (name, model)])
        cv_results[name] = cross_val_score(pipeline, X_train, y_train, cv=10, scoring='r2')
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        model_performance[name] = {'R2': r2, 'MSE': mse}

    # Create figure for Models Performance Chart (Bar chart)
    model_performance_data = pd.DataFrame(model_performance).T.reset_index()
    fig_models_performance = px.bar(model_performance_data, x='index', y='R2', labels={'index': 'Model', 'R2': 'R² Score'},
                                    title='Model Performance')

    # Example of creating a Plotly Express scatter plot for actual vs predicted values (Scatter plot)
    fig_actual_vs_predicted = px.scatter(x=y_pred_xgb, y=y_test, labels={'x': 'Predicted Values', 'y': 'Actual Values'},
                                         title='Actual vs Predicted Values')

    # Define the feature_importance_data based on the XGBoost model
    feature_importance_data = pd.DataFrame({
        'Feature': X.columns,
        'Importance': grid_search_xgb.best_estimator_.feature_importances_
    }).sort_values(by='Importance', ascending=False).head(10)

    fig_feature_importances = px.bar(feature_importance_data, x='Importance', y='Feature', orientation='h', title='Top 10 Feature Importances')

    # Example of creating a Plotly Express bar chart for hyperparameter tuning results (if applicable)
    hyperparam_table_data = pd.DataFrame({
        'Model': ['XGBoost'],
        'Best CV R² Score': [best_score_xgb]
    })
    fig_hyperparameter_tuning = px.bar(hyperparam_table_data, x='Model', y='Best CV R² Score', title='Hyperparameter Tuning Results')

    # Creating a Dash DataTable for model performance on test set (Dash DataTable)
    #cv_table_data = pd.DataFrame(cv_results)
    #cv_table_data = cv_table_data.iloc[:, :20]  # Display first 20 columns, adjust as needed

    cv_table_data = [{'Model': name, **{f'Fold {i+1}': score for i, score in enumerate(scores)}, 'Mean CV R² Score': scores.mean()} for name, scores in cv_results.items()]

    model_table_data = [{'Model': name, 'R²': metrics['R2'], 'MSE': metrics['MSE']} for name, metrics in model_performance.items()]


    fig4 = dash_table.DataTable(
        columns=[{'name': i, 'id': i} for i in cv_table_data[0].keys()],
        data=cv_table_data,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '5px'},
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
    ),

    fig5 = dash_table.DataTable(
        id='model-table',
        columns=[{"name": i, "id": i} for i in model_table_data[0].keys()],
        data=model_table_data,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '5px'},
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
    )

    return fig_models_performance, fig_actual_vs_predicted, fig_feature_importances, fig_hyperparameter_tuning, fig5 ,fig4


