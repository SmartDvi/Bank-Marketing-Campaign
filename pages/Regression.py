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
            ),width=4
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
                dcc.Graph(id=' Feature_Importances', figure={}),
            ])
        ),
    ], width=4),
    ]),
    dbc.Row([
          dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H5('Model Performance Chart', 
                            className='text-center',),
                    dcc.Graph(id='Model_Performance', figure={}),
                ])
            ),width=4
        ),
        dbc.Col([
        dbc.Card(
            dbc.CardBody([
                html.H5('Model Evaluation Results (Cross-Validation Scores)',
                        className='text-center'),
                dcc.Graph(id='Hyperparameter_Tuning', figure={}),
            ])
        ),
    ], width=8),
    ]),
    dbc.Row([
         dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H5('Model Performance Chart', 
                            className='text-center',),
                    dcc.Graph(id='model_performance_test_table', figure={}),
                ])
            ),width=12
        ),
    ])
    
])

@callback(Output('Models_Performance_chart', 'figure'),
          Input('month_checklist', 'value'))

def model_performance_chart(selected_month):
    if not selected_month:
        return {}
    
    
    filtered_data = data[data('month')].isin(selected_month)

    filtered_data.reset_index(drop=True, inplace = True)

    le = LabelEncoder()
    filtered_data['Deposit'] = le.fit_transform(filtered_data['Deposit'].values.ravel())

    filtered_data['Outcome_of_previous_marketing_campaign'] = pd.to_numeric(filtered_data['Outcome_of_previous_marketing_campaign'],errors='coerce' )
    filtered_data['duration'] = pd.to_numeric(filtered_data['duration'], errors='coerce')

    filtered_data['Outcome_of_previous_marketing_campaign'].fillna(0, inplace=True)
    filtered_data['duration'].fillna(0, inplace=True)

    filtered_data['Outcome_of_previous_marketing_campaign_squared'] = filtered_data['Outcome_of_previous_marketing_campaign'] **2
    filtered_data['Deposit'] = data['duration'] **2

    filtered_data = pd.get_dummies(filtered_data, columns=['job', 'marital', 'education', 'contact', 'month', 'day_of_week'])

    X = pd.get_dummies(filtered_data.drop(columns=['Deposit']), drop_first=True)
    y = filtered_data['Deposit']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # using minMaxScaler 
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
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
    grid_search_xgb = GridSearchCV(XGBRegressor(random_state=42), xgb_params, cv=5, scoring='r2')
    grid_search_xgb.fit(X_train_scaled, y_train)

    best_params_xgb = grid_search_xgb.best_params_
    best_score_xgb = grid_search_xgb.best_score_

    print(f"Best XGBoost Parameters: {best_params_xgb}")
    print(f"Best XGBoost R² Score: {best_score_xgb}")

    # Evaluate on test set
    y_pred_xgb = grid_search_xgb.best_estimator_.predict(X_test_scaled)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    print(f"XGBoost - R²: {r2_xgb}, MSE: {mse_xgb}")

    cv_results = {}
    model_performance = {}

    for name, model in models.items():
        if name == 'XGBoost':
            model = grid_search_xgb.best_estimator_  # Use best XGBoost model from GridSearchCV for evaluation
        pipeline = Pipeline([('scaler', MinMaxScaler()), ('model', model)])
        
        # Cross-validation scores
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
        cv_results[name] = cv_scores
        
        # Model performance on test set
        pipeline.fit(X_train, y_train)  # Fit on unscaled data
        y_pred = pipeline.predict(X_test)  # Predict on unscaled data
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        model_performance[name] = {'R2': r2, 'MSE': mse}

    model_performance_data = pd.DataFrame(model_performance).T.reset_index.rename(columns={'index': 'Model'})
    fig =px.bar(model_performance_data,
            x='Model', y='R2',  # Use 'R2' to match the column name
            labels={'R2': 'R² Score'})
        

    return fig

@callback(Output('Actual_vs_Predicted', 'figure'),
          Input('month_checklist', 'value'))
           
def model_performance_chart(selected_month):
    if not selected_month:
        return {}
    
    
    filtered_data = data[data('month').isin(selected_month)]

    filtered_data.reset_index(drop=True, inplace = True)

    le = LabelEncoder()
    filtered_data['Deposit'] = le.fit_transform(filtered_data['Deposit'].values.ravel())

    filtered_data['Outcome_of_previous_marketing_campaign'] = pd.to_numeric(filtered_data['Outcome_of_previous_marketing_campaign'],errors='coerce' )
    filtered_data['duration'] = pd.to_numeric(filtered_data['duration'], errors='coerce')

    filtered_data['Outcome_of_previous_marketing_campaign'].fillna(0, inplace=True)
    filtered_data['duration'].fillna(0, inplace=True)

    filtered_data['Outcome_of_previous_marketing_campaign_squared'] = filtered_data['Outcome_of_previous_marketing_campaign'] **2
    filtered_data['Deposit'] = data['duration'] **2

    filtered_data = pd.get_dummies(filtered_data, columns=['job', 'marital', 'education', 'contact', 'month', 'day_of_week'])

    X = pd.get_dummies(filtered_data.drop(columns=['Deposit']), drop_first=True)
    y = filtered_data['Deposit']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # using minMaxScaler 
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
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
    grid_search_xgb = GridSearchCV(XGBRegressor(random_state=42), xgb_params, cv=5, scoring='r2')
    grid_search_xgb.fit(X_train_scaled, y_train)

    best_params_xgb = grid_search_xgb.best_params_
    best_score_xgb = grid_search_xgb.best_score_

    print(f"Best XGBoost Parameters: {best_params_xgb}")
    print(f"Best XGBoost R² Score: {best_score_xgb}")

    # Evaluate on test set
    y_pred_xgb = grid_search_xgb.best_estimator_.predict(X_test_scaled)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    residuals = y_test - y_pred_xgb

    fig = px.scatter(x=y_pred_xgb, y=residuals,
                      labels={'x': 'Predicted Values', 'y': 'Residuals'})
    fig.add_hline(y=0, line_dash='dash',
                  line_color='red')
    return fig






@callback(Output('Feature_Importances', 'figure'),
          Input('month_checklist', 'value'))


def model_performance_chart(selected_month):
    if not selected_month:
        return {}
    
    filtered_data = data.copy()
    filtered_data = filtered_data[filtered_data('month').isin(selected_month)]

    filtered_data.reset_index(drop=True, inplace = True)

    le = LabelEncoder()
    filtered_data['Deposit'] = le.fit_transform(filtered_data['Deposit'].values.ravel())

    filtered_data['Outcome_of_previous_marketing_campaign'] = pd.to_numeric(filtered_data['Outcome_of_previous_marketing_campaign'],errors='coerce' )
    filtered_data['duration'] = pd.to_numeric(filtered_data['duration'], errors='coerce')

    filtered_data['Outcome_of_previous_marketing_campaign'].fillna(0, inplace=True)
    filtered_data['duration'].fillna(0, inplace=True)

    filtered_data['Outcome_of_previous_marketing_campaign_squared'] = filtered_data['Outcome_of_previous_marketing_campaign'] **2
    filtered_data['Deposit'] = data['duration'] **2

    filtered_data = pd.get_dummies(filtered_data, columns=['job', 'marital', 'education', 'contact', 'month', 'day_of_week'])

    X = pd.get_dummies(filtered_data.drop(columns=['Deposit']), drop_first=True)
    y = filtered_data['Deposit']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # using minMaxScaler 
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
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
    grid_search_xgb = GridSearchCV(XGBRegressor(random_state=42), xgb_params, cv=5, scoring='r2')
    grid_search_xgb.fit(X_train_scaled, y_train)

    best_params_xgb = grid_search_xgb.best_params_
    best_score_xgb = grid_search_xgb.best_score_

    print(f"Best XGBoost Parameters: {best_params_xgb}")
    print(f"Best XGBoost R² Score: {best_score_xgb}")

    # Evaluate on test set
    y_pred_xgb = grid_search_xgb.best_estimator_.predict(X_test_scaled)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    print(f"XGBoost - R²: {r2_xgb}, MSE: {mse_xgb}")

    cv_results = {}
    model_performance = {}

    for name, model in models.items():
        if name == 'XGBoost':
            model = grid_search_xgb.best_estimator_  # Use best XGBoost model from GridSearchCV for evaluation
        pipeline = Pipeline([('scaler', MinMaxScaler()), ('model', model)])
        
        # Cross-validation scores
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
        cv_results[name] = cv_scores
        
        # Model performance on test set
        pipeline.fit(X_train, y_train)  # Fit on unscaled data
        y_pred = pipeline.predict(X_test)  # Predict on unscaled data
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        model_performance[name] = {'R2': r2, 'MSE': mse}

    # Train Random forest  for feature importance
    xgb_model = XGBRegressor(random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    feature_importances = xgb_model.feature_importance_
    features = X.columns
    feature_importance_data = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)


    fig =px.bar(feature_importance_data.head(10),
                 x='Importance',
                   y='Feature',
                     orientation='h', 
                     )

    return fig


@callback(Output('model_performance_test', 'figure'),
          Input('month_checklist', 'figure'),
          allow_duplicate=True)


def model_performance_test(selected_month):
    if not selected_month:
        return {}
    
    filtered_data = data.copy()
    filtered_data = filtered_data[filtered_data('month').isin(selected_month)]

    filtered_data.reset_index(drop=True, inplace = True)

    le = LabelEncoder()
    filtered_data['Deposit'] = le.fit_transform(filtered_data['Deposit'].values.ravel())

    filtered_data['Outcome_of_previous_marketing_campaign'] = pd.to_numeric(filtered_data['Outcome_of_previous_marketing_campaign'],errors='coerce' )
    filtered_data['duration'] = pd.to_numeric(filtered_data['duration'], errors='coerce')

    filtered_data['Outcome_of_previous_marketing_campaign'].fillna(0, inplace=True)
    filtered_data['duration'].fillna(0, inplace=True)

    filtered_data['Outcome_of_previous_marketing_campaign_squared'] = filtered_data['Outcome_of_previous_marketing_campaign'] **2
    filtered_data['Deposit'] = data['duration'] **2

    filtered_data = pd.get_dummies(filtered_data, columns=['job', 'marital', 'education', 'contact', 'month', 'day_of_week'])

    X = pd.get_dummies(filtered_data.drop(columns=['Deposit']), drop_first=True)
    y = filtered_data['Deposit']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # using minMaxScaler 
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
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
    grid_search_xgb = GridSearchCV(XGBRegressor(random_state=42), xgb_params, cv=5, scoring='r2')
    grid_search_xgb.fit(X_train_scaled, y_train)

    best_params_xgb = grid_search_xgb.best_params_
    best_score_xgb = grid_search_xgb.best_score_

    print(f"Best XGBoost Parameters: {best_params_xgb}")
    print(f"Best XGBoost R² Score: {best_score_xgb}")

    # Evaluate on test set
    y_pred_xgb = grid_search_xgb.best_estimator_.predict(X_test_scaled)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    print(f"XGBoost - R²: {r2_xgb}, MSE: {mse_xgb}")

    cv_results = {}
    model_performance = {}

    for name, model in models.items():
        if name == 'XGBoost':
            model = grid_search_xgb.best_estimator_  # Use best XGBoost model from GridSearchCV for evaluation
        pipeline = Pipeline([('scaler', MinMaxScaler()), ('model', model)])
        
        # Cross-validation scores
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
        cv_results[name] = cv_scores
        
        # Model performance on test set
        pipeline.fit(X_train, y_train)  # Fit on unscaled data
        y_pred = pipeline.predict(X_test)  # Predict on unscaled data
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        model_performance[name] = {'R2': r2, 'MSE': mse}

        # Create data for the tables
        cv_table_data = []
        for name, scores in cv_results.items():
            cv_table_data.append({
                'Model': name,
                'Fold 1': scores[0],
                'Fold 2': scores[1],
                'Fold 3': scores[2],
                'Fold 4': scores[3],
                'Fold 5': scores[4],
                'Mean CV R² Score': scores.mean()
            })

        model_table_data = []
        for name, metrics in model_performance.items():
            model_table_data.append({
                'Model': name,
                'R²': metrics['R2'],
                'MSE': metrics['MSE']
            })

        hyperparam_table_data = []
        for name, result in {'XGBoost': {'best_params': best_params_xgb, 'best_score': best_score_xgb}}.items():
            hyperparam_table_data.append({
                'Model': name,
                'Best Parameters': result['best_params'],
                'Best CV R² Score': result['best_score']
            })

        fig = dash_table.DataTable(
            columns =[{'name': i, 'id':i} for i in model_table_data[0].keys()],
            data=model_table_data,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '5px'},
            style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
            },
            ),
        return fig

@callback(Output('model_performance_test_table', 'figure'),
          Input('month_checklist', 'figure'))


def model_performance_test(selected_month):
    if not selected_month:
        return {}
    
    filtered_data = data.copy()
    filtered_data = filtered_data[filtered_data('month').isin(selected_month)]

    filtered_data.reset_index(drop=True, inplace = True)

    le = LabelEncoder()
    filtered_data['Deposit'] = le.fit_transform(filtered_data['Deposit'].values.ravel())

    filtered_data['Outcome_of_previous_marketing_campaign'] = pd.to_numeric(filtered_data['Outcome_of_previous_marketing_campaign'],errors='coerce' )
    filtered_data['duration'] = pd.to_numeric(filtered_data['duration'], errors='coerce')

    filtered_data['Outcome_of_previous_marketing_campaign'].fillna(0, inplace=True)
    filtered_data['duration'].fillna(0, inplace=True)

    filtered_data['Outcome_of_previous_marketing_campaign_squared'] = filtered_data['Outcome_of_previous_marketing_campaign'] **2
    filtered_data['Deposit'] = data['duration'] **2

    filtered_data = pd.get_dummies(filtered_data, columns=['job', 'marital', 'education', 'contact', 'month', 'day_of_week'])

    X = pd.get_dummies(filtered_data.drop(columns=['Deposit']), drop_first=True)
    y = filtered_data['Deposit']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # using minMaxScaler 
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
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
    grid_search_xgb = GridSearchCV(XGBRegressor(random_state=42), xgb_params, cv=5, scoring='r2')
    grid_search_xgb.fit(X_train_scaled, y_train)

    best_params_xgb = grid_search_xgb.best_params_
    best_score_xgb = grid_search_xgb.best_score_

    print(f"Best XGBoost Parameters: {best_params_xgb}")
    print(f"Best XGBoost R² Score: {best_score_xgb}")

    # Evaluate on test set
    y_pred_xgb = grid_search_xgb.best_estimator_.predict(X_test_scaled)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    print(f"XGBoost - R²: {r2_xgb}, MSE: {mse_xgb}")

    cv_results = {}
    model_performance = {}

    for name, model in models.items():
        if name == 'XGBoost':
            model = grid_search_xgb.best_estimator_  # Use best XGBoost model from GridSearchCV for evaluation
        pipeline = Pipeline([('scaler', MinMaxScaler()), ('model', model)])
        
        # Cross-validation scores
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
        cv_results[name] = cv_scores
        
        # Model performance on test set
        pipeline.fit(X_train, y_train)  # Fit on unscaled data
        y_pred = pipeline.predict(X_test)  # Predict on unscaled data
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        model_performance[name] = {'R2': r2, 'MSE': mse}

        # Create data for the tables
        cv_table_data = []
        for name, scores in cv_results.items():
            cv_table_data.append({
                'Model': name,
                'Fold 1': scores[0],
                'Fold 2': scores[1],
                'Fold 3': scores[2],
                'Fold 4': scores[3],
                'Fold 5': scores[4],
                'Mean CV R² Score': scores.mean()
            })

        model_table_data = []
        for name, metrics in model_performance.items():
            model_table_data.append({
                'Model': name,
                'R²': metrics['R2'],
                'MSE': metrics['MSE']
            })

        hyperparam_table_data = []
        for name, result in {'XGBoost': {'best_params': best_params_xgb, 'best_score': best_score_xgb}}.items():
            hyperparam_table_data.append({
                'Model': name,
                'Best Parameters': result['best_params'],
                'Best CV R² Score': result['best_score']
            })

        fig = dash_table.DataTable(
            columns =[{'name': i, 'id':i} for i in cv_table_data[0].keys()],
            data=model_table_data,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '5px'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            ),
        return fig
        
        