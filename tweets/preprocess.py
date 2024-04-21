# flake8: noqa
import pandas as pd

from utils import beautify_tweet_field_names, beautify_tweet_items_with_lists

# Read the CSV file into a pandas DataFrame
tweets_df = pd.read_csv('dataset/raw_tweets.csv')

# Fill all NaN values with an empty string
tweets_df = tweets_df.fillna('')

beautify_tweet_field_names(tweets_df)
beautify_tweet_items_with_lists(tweets_df)

# Drop unnecessary columns
tweets_df = tweets_df.drop(
    ['Unnamed: 0', 'lang', 'attachments.poll_ids'], axis=1)

# Normalize numerical columns
numeric_cols = ['possibly_sensitive', 'retweet_count', 'reply_count', 'like_count', 'quote_count',
                'impression_count', 'media_keys', 'urls', 'context_annotations', 'mentions', 'hashtags', 'annotations', 'text_len']
tweets_df[numeric_cols] = (tweets_df[numeric_cols] /
                           tweets_df[numeric_cols].max()) * 100

# Write normalized df to csv
tweets_df.to_csv('tweets/normalized-tweets.csv')
