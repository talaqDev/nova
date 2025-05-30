import unittest
import pandas as pd
from io import StringIO

# Mock dataset for testing
MOCK_DATA = """headline,publisher,date
Stock hits record high,Publisher A,2024-12-10
Earnings report shows growth,Publisher B,2024-12-10
CEO steps down,Publisher A,2024-12-11
Company files for bankruptcy,Publisher C,2024-12-11
New product launch excites investors,Publisher A,2024-12-12
"""

class TestEDAFunctions(unittest.TestCase):

    def setUp(self):
        """Set up a mock DataFrame for testing."""
        self.data = pd.read_csv(StringIO(MOCK_DATA))

    def test_load_data(self):
        """Test if the data loads correctly."""
        self.assertEqual(len(self.data), 5, "The dataset should have 5 rows.")
        self.assertIn('headline', self.data.columns, "Column 'headline' should exist.")

    def test_descriptive_statistics(self):
        """Test the descriptive statistics calculation."""
        self.data['headline_length'] = self.data['headline'].apply(len)
        avg_length = self.data['headline_length'].mean()
        self.assertGreater(avg_length, 10, "The average headline length should be greater than 10.")

    def test_publisher_analysis(self):
        """Test the publisher analysis."""
        publisher_counts = self.data['publisher'].value_counts()
        self.assertEqual(publisher_counts['Publisher A'], 3, "Publisher A should have 3 articles.")

    def test_publication_trends(self):
        """Test the publication trends analysis."""
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data['publication_date'] = self.data['date'].dt.date
        publication_counts = self.data['publication_date'].value_counts()
        self.assertEqual(publication_counts[2024-12-10], 2, "There should be 2 articles published on 2024-12-10.")

    def test_sentiment_analysis(self):
        """Test the sentiment analysis functionality."""
        from textblob import TextBlob

        self.data['sentiment'] = self.data['headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
        self.assertTrue(self.data['sentiment'].between(-1, 1).all(), "Sentiment values should be between -1 and 1.")

if __name__ == '__main__':
    unittest.main()
