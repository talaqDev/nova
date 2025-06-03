import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from task1 import date_conv

# Load datasets
news_df = pd.read_csv("C:\\Users\\nadew\\10x\\week1\\Nova_Financial_Solutions\\data_file\\raw_analyst_ratings.csv\\raw_analyst_ratings.csv")  # Columns: ['headline', 'date', 'stock']
stock_df = pd.read_csv("C:\\Users\\nadew\\10x\\week1\\Nova_Financial_Solutions\\data_file\\yfinance_data\\yfinance_data\\AAPL_historical_data.csv")  # Columns: ['date', 'stock', 'close']
news_df.columns = news_df.columns.str.strip().str.lower()
print(news_df.head())
# Convert 'date' columns to datetime
news_df['date'] = pd.to_datetime(news_df['date'], format="%Y-%m-%d %H:%M:%S", errors='coerce')
# Step 1: Perform Sentiment Analysis on Headlines
def get_sentiment(text):
    """Calculates sentiment polarity (-1 to 1) using TextBlob."""
    return TextBlob(text).sentiment.polarity

# Check and clean column names
print("Original Columns:", news_df.columns)
news_df.columns = news_df.columns.str.strip().str.lower()

# Verify column existence
if 'headline' in news_df.columns:
    news_df['sentiment'] = news_df['headline'].apply(get_sentiment)
else:
    print("Error: 'headline' column not found in the dataset.")
    print("Available Columns:", news_df.columns)
    exit()


# Step 2: Aggregate Sentiments by Date and Stock
average_sentiment = news_df.groupby(['date', 'stock'])['sentiment'].mean().reset_index()

# Step 3: Calculate Daily Stock Returns
stock_df['daily_return'] = stock_df.groupby('stock')['close'].pct_change()

# Step 4: Merge Sentiment Data with Stock Returns
merged_df = pd.merge(average_sentiment, stock_df, on=['date', 'stock'])

# Step 5: Correlation Analysis
def calculate_correlation(df):
    """Calculates Pearson correlation between sentiment and daily returns."""
    sentiment = df['sentiment']
    returns = df['daily_return']
    correlation, p_value = pearsonr(sentiment.dropna(), returns.dropna())
    return correlation, p_value

# Group by stock to calculate correlations for each stock
results = []
for stock, group in merged_df.groupby('stock'):
    corr, p_val = calculate_correlation(group)
    results.append({'stock': stock, 'correlation': corr, 'p_value': p_val})

results_df = pd.DataFrame(results)

# Display correlation results
print("Correlation Results:")
print(results_df)

# Step 6: Visualization
# Plot correlation for each stock
plt.figure(figsize=(10, 6))
plt.bar(results_df['stock'], results_df['correlation'], color='blue', alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.title('Correlation between Sentiment and Stock Returns by Stock')
plt.xlabel('Stock')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=45)
plt.show()

# Save results
results_df.to_csv('correlation_results.csv', index=False)
