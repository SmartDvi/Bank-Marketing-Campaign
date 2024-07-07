import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from pages import Overview, Table, Data_Exploration, Regression

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, '/assets/style2.css'])

# Define navigation links
navbar = dbc.Navbar(
    children=[
        dbc.Nav(
            [
                dbc.NavItem(dbc.NavLink("Overview", href="/")),
                dbc.NavItem(dbc.NavLink("Dataset Details", href="/page-2")),
                dbc.NavItem(dbc.NavLink("Data Exploration", href="/page-3")),
                dbc.NavItem(dbc.NavLink("Regression", href="/page-4")),
            ],
            pills=True,
            className="ml-auto text-center",
        ),
    ],
    color="light",
    dark=False,
    className="mb-4 text-center",
)

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Bank Marketing Campaign", className="dashboard-title text-center"), width=12)
    ]),
    navbar,
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
], fluid=True)

# Callback to update page content based on URL
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-2':
        return Table.layout
    elif pathname == '/page-3':
        return Data_Exploration.layout
    elif pathname == '/page-4':
        return Regression.layout
    else:
        return Overview.layout

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=5050)
