import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dcc.Markdown('''
                         
                            Please Detials information to pivote actionable insights and project documentation will be added to the app after my examination.
            ## Project Overview

            ### Introduction
            The Bank Marketing Campaign Dashboard provides an in-depth analysis of a real-world marketing dataset from a Portuguese banking institution. This dataset includes a range of demographic, contact, and campaign-specific information used to predict whether a client will subscribe to a term deposit.

            ### Objective
            The primary goal is to explore, analyze, and model this data to derive actionable insights and predictive models that can improve the bank's marketing strategies.

            ### Dataset Details
            The dataset consists of the following columns:
            - **Age:** The age of the client.
            - **Job:** The type of job of the client.
            - **Marital:** Marital status of the client.
            - **Education:** Level of education of the client.
            - **Default:** Indicates if the client has credit in default.
            - **Housing:** Indicates if the client has a housing loan.
            - **Loan:** Indicates if the client has a personal loan.
            - **Contact:** Type of communication contact (e.g., cellular, telephone).
            - **Month:** The last contact month of the year.
            - **Day_of_week:** The last contact day of the week.
            - **Campaign:** The number of contacts performed during this campaign.
            - **Pdays:** Number of days since the client was last contacted from a previous campaign.
            - **Previous_contacts_distribution:** Number of contacts performed before this campaign.
            - **Outcome_of_previous_marketing_campaign:** Outcome of the previous marketing campaign.
            - **Employment_variation_rate:** Quarterly employment variation rate.
            - **Consumer_price_index:** Monthly consumer price index.
            - **Consumer_confidence_index:** Monthly consumer confidence index.
            - **Euribor 3 month rate:** Euribor 3 month rate.
            - **Number of employees:** Number of employees in the bank.
            - **Deposit:** Indicates if the client subscribed to a term deposit.

            ### Features and Insights
            The dashboard includes several key features:
            - **Overview:** This is the Highligh of the project adding a presentation slide of key insight linked to dashbard source of the Insight for detail information
            - **Data Exploration:** Visualizes the distribution and relationships of various features, providing insights into client demographics and campaign outcomes.
            - **Regression Analysis:** Uses various regression models to predict the likelihood of a client subscribing to a term deposit based on the provided features.

            ### Visual Insights
            The dashboard contains interactive visualizations to engage users, including bar charts, heatmaps, and more. Below are some example images from the dashboard:

            ### Project Slide Presentation
            '''),

            

            # Placeholder for image carousel or slideshow
            dbc.Carousel(
                items=[
                    {"key": "1", "src": "/assets/images/data_exploration.png", "caption": "Data Exploration"},
                    {"key": "2", "src": "/assets/images/age_group_and_loan_distribution.png", "caption": "Age Group and Loan Distribution"},
                    {"key": "3", "src": "/assets/images/job_and_marital.png", "caption": "Job and Marital Status"},
                    {"key": "4", "src": "/assets/images/features_deposit.png", "caption": "Features Correlation with Deposit"},
                    {"key": "5", "src": "/assets/images/month_and_week_conversion.png", "caption": "Conversion Rate by Month and Week"}
                ],
                controls=True,
                indicators=True,
                interval=2000,
                ride="carousel",
                className="mb-4"
            ),

            dcc.Markdown('''
            ### Conclusion
            This dashboard serves as a powerful tool for data scientists, analysts, and marketing professionals to understand and leverage client data effectively. By exploring the various features and insights provided, users can make data-driven decisions to optimize marketing strategies and improve client acquisition rates.

            ''')
        ])
    ])
], fluid=True)
