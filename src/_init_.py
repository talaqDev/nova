import pandas as pd
import matplotlib as plt
import os

# Function to load and prepare stock data
def load_data(filepath):
    """Load stock price data from a CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    data = pd.read_csv(filepath)
    # Ensure necessary columns are present
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    return data

# Function to calculate simple moving average (SMA)
def calculate_sma(data, period):
    """Calculate Simple Moving Average (SMA)."""
    data[f'SMA_{period}'] = data['Close'].rolling(window=period).mean()
    return data

# Function to calculate relative strength index (RSI)
def calculate_rsi(data, period):
    """Calculate Relative Strength Index (RSI)."""
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    rs = avg_gain / avg_loss
    data[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    return data

# Function to calculate MACD
def calculate_macd(data):
    """Calculate Moving Average Convergence Divergence (MACD)."""
    short_ema = data['Close'].ewm(span=12, adjust=False).mean()
    long_ema = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = short_ema - long_ema
    data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    return data

# Function to visualize stock price and indicators
def visualize_data(data):
    """Create visualizations for stock prices and indicators."""
    plt.figure(figsize=(14, 10))

    # Plot Close Price and SMA
    plt.subplot(3, 1, 1)
    plt.plot(data['Close'], label='Close Price', color='blue')
    if 'SMA_20' in data:
        plt.plot(data['SMA_20'], label='SMA (20)', color='orange')
    plt.title('Stock Price with Simple Moving Average')
    plt.legend()

    # Plot RSI
    plt.subplot(3, 1, 2)
    if 'RSI_14' in data:
        plt.plot(data['RSI_14'], label='RSI (14)', color='green')
        plt.axhline(y=70, color='red', linestyle='--', label='Overbought')
        plt.axhline(y=30, color='blue', linestyle='--', label='Oversold')
        plt.title('Relative Strength Index (RSI)')
        plt.legend()

    # Plot MACD
    plt.subplot(3, 1, 3)
    if 'MACD' in data and 'MACD_signal' in data:
        plt.plot(data['MACD'], label='MACD', color='purple')
        plt.plot(data['MACD_signal'], label='MACD Signal', color='orange')
        plt.title('MACD and Signal Line')
        plt.legend()

    plt.tight_layout()
    plt.show()

# Main execution block
def main():
    """Main function to execute Task 2 steps."""
  
    try:
        # Load data
        filepath = "./data/raw_analyst_ratings.csv/raw_analyst_ratings.csv"  # Replace with your dataset path
        stock_data = load_data(filepath)

        # Calculate indicators
        stock_data = calculate_sma(stock_data, period=20)
        stock_data = calculate_rsi(stock_data, period=14)
        stock_data = calculate_macd(stock_data)

        # Save enriched data to a new file
        enriched_filepath = 'enriched_stock_data.csv'
        stock_data.to_csv(enriched_filepath, index=False)
        print(f"Enriched data saved to: {enriched_filepath}")

        # Visualize data
        visualize_data(stock_data)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
