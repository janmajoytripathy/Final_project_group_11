import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from dash import dcc, html, Input, Output
import dash
import plotly.express as px
import statsmodels.api as sm


# Function to fetch stock prices for the last 60 days
def fetch_last_60_days_prices(api_key, symbols):
    today = datetime.today ()
    start_date = today - timedelta ( days=60 )
    date_to = today.strftime ( '%Y-%m-%d' )
    date_from = start_date.strftime ( '%Y-%m-%d' )
    base_url = "http://api.marketstack.com/v1/eod"
    all_data = []

    for symbol in symbols:
        print ( f"Fetching data for symbol: {symbol}" )
        params = {
            'access_key': api_key,
            'symbols': symbol,
            'date_from': date_from,
            'date_to': date_to,
            'limit': 100
        }

        try:
            response = requests.get ( base_url, params=params )
            if response.status_code == 200:
                data = response.json ()
                if 'data' in data and data['data']:
                    df = pd.DataFrame ( data['data'] )
                    all_data.append ( df )
                else:
                    print ( f"No data found for symbol: {symbol}" )
            else:
                print ( f"Error fetching data for {symbol}: {response.status_code}" )
        except Exception as e:
            print ( f"Exception occurred while fetching data for {symbol}: {e}" )
        time.sleep ( 2 )  # To avoid hitting API rate limits

    if all_data:
        non_empty_data = [df for df in all_data if not df.empty]
        raw_data = pd.concat ( non_empty_data, ignore_index=True ) if non_empty_data else pd.DataFrame ()
        raw_data.to_csv ( "last_60_days_prices.csv", index=False )  # Save raw API data
        print ( "Raw API data saved to last_60_days_prices.csv" )
        return raw_data
    else:
        print ( "No data processed." )
        return pd.DataFrame ()


# Function to analyze stock data and recommend the top 5 stocks
def analyze_and_recommend(data):
    if data.empty:
        print ( "No data available for analysis." )
        return pd.DataFrame (), pd.DataFrame ()

    data['date'] = pd.to_datetime ( data['date'] )
    data.sort_values ( by=['symbol', 'date'], inplace=True )
    data['percent_change'] = ((data['adj_close'] - data['adj_open']) / data['adj_open']) * 100

    performance = data.groupby ( 'symbol' ).agg (
        avg_percent_change=('percent_change', 'mean'),
        volatility=('percent_change', 'std'),
        last_closing_price=('adj_close', 'last'),
        avg_dividend=('dividend', 'mean')
    ).reset_index ()

    performance['score'] = performance['avg_percent_change'] / performance['volatility']
    top_5_stocks = performance.sort_values ( by='score', ascending=False ).head ( 5 )

    top_5_stocks.to_csv ( "top_5_recommendations.csv", index=False )  # Save top 5 recommendations
    print ( "Top 5 recommendations saved to top_5_recommendations.csv" )

    return performance, top_5_stocks


# Function to perform regression analysis
def perform_regression(data):
    stock_data_cleaned = data.dropna ()
    stock_symbols = stock_data_cleaned['symbol'].unique ()
    stock_metrics = pd.DataFrame ()

    for symbol in stock_symbols:
        symbol_data = stock_data_cleaned[stock_data_cleaned['symbol'] == symbol]
        X = symbol_data[['adj_open']]
        y = symbol_data['adj_close']
        X = sm.add_constant ( X )
        model = sm.OLS ( y, X ).fit ()

        performance_score = model.rsquared_adj
        coefficients = model.params
        std_errors = model.bse
        t_values = model.tvalues
        p_values = model.pvalues
        confidence_intervals = model.conf_int ()

        stock_results = pd.DataFrame ( {
            'symbol': [symbol] * len ( coefficients ),
            'variable': coefficients.index,
            'coef': coefficients.values,
            'std_err': std_errors.values,
            't': t_values.values,
            'P>|t|': p_values.values,
            '[0.025': confidence_intervals[0].values,
            '0.975]': confidence_intervals[1].values,
            'score': [performance_score] * len ( coefficients )
        } )

        stock_metrics = pd.concat ( [stock_metrics, stock_results], ignore_index=True )

    stock_metrics.to_csv ( "stock_regression_metrics.csv", index=False )  # Save regression metrics
    print ( "Regression metrics saved to stock_regression_metrics.csv" )

    top_stocks = stock_metrics[stock_metrics['variable'] == 'const'].sort_values ( by='score', ascending=False ).head (
        5 )
    top_stocks.to_csv ( "top_5_stocks_regression.csv", index=False )  # Save top 5 stocks by regression
    print ( "Top 5 stocks by regression saved to top_5_stocks_regression.csv" )

    return stock_metrics, top_stocks


# Initialize Dash app
app = dash.Dash ( __name__ )
app.title = "Stock Analysis Dashboard"

# API key and stock symbols
API_KEY = "f5d84e530e2a3d3397ab7608d49619da"
SYMBOLS = [
    "NKE", "AAPL", "AMZN", "AXP", "BA", "CSCO", "IBM", "JPM", "MSFT",
    "V", "MA", "NVDA", "TSLA", "VZ", "NFLX", "CRM", "PG", "UNH", "WMT", "GS", "DJI.INDX"
]

# Fetch data
data = fetch_last_60_days_prices ( API_KEY, SYMBOLS )
if data.empty:
    print ( "No data available from API. Exiting..." )
    exit ()

data['date'] = pd.to_datetime ( data['date'] )

# Dashboard layout
app.layout = html.Div ( [
    html.H1 ( "Stock Data Dashboard", style={'text-align': 'center', 'color': '#2c3e50'} ),
    html.Div ( [
        html.Label ( "Select Stock Symbol:" ),
        dcc.Dropdown (
            id='symbol-dropdown',
            options=[{'label': symbol, 'value': symbol} for symbol in data['symbol'].unique ()],
            value=data['symbol'].unique ()[0],
            multi=False
        )
    ], style={'width': '50%', 'margin': '20px auto'} ),
    html.Div ( [
        dcc.Graph ( id='adj-close-line-chart' ),
        dcc.Graph ( id='volume-bar-chart' ),
        dcc.Graph ( id='adj-close-histogram' ),
        html.Div ( id='descriptive-stats-table' )
    ], style={'padding': '20px'} ),
] )


@app.callback (
    [
        Output ( 'adj-close-line-chart', 'figure' ),
        Output ( 'volume-bar-chart', 'figure' ),
        Output ( 'adj-close-histogram', 'figure' ),
        Output ( 'descriptive-stats-table', 'children' )
    ],
    [Input ( 'symbol-dropdown', 'value' )]
)
def update_dashboard(selected_symbol):
    filtered_data = data[data['symbol'] == selected_symbol]

    line_chart_fig = px.line ( filtered_data, x='date', y='adj_close',
                               title=f"Adjusted Close Prices for {selected_symbol}" )
    bar_chart_fig = px.bar ( filtered_data, x='date', y='volume', title=f"Trading Volume for {selected_symbol}" )
    histogram_fig = px.histogram ( filtered_data, x='adj_close', nbins=20,
                                   title=f"Distribution of Adjusted Close Prices for {selected_symbol}" )

    stats = filtered_data['adj_close'].describe ().reset_index ()
    stats.columns = ['Statistic', 'Value']
    stats_table = dcc.Graph (
        figure=px.bar ( stats, x='Statistic', y='Value', title=f"Descriptive Statistics for {selected_symbol}" )
    )

    return line_chart_fig, bar_chart_fig, histogram_fig, stats_table


# Analyze data and generate files
performance, top_5_stocks = analyze_and_recommend ( data )
stock_metrics, top_stocks = perform_regression ( data )

if __name__ == "__main__":
    app.run_server ( debug=True )
