import yfinance as yf

# Define the ticker symbol
ticker = 'BRK-B'

# Download historical data
data = yf.download(ticker, start='2000-01-01', end='2024-08-28')

# save the data to a CSV file
data = data[['Date', 'Adj Close']]

data.to_csv('./resources/BRK_B_stock_price.csv')
