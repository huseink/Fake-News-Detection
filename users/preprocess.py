
# flake8: noqa
import pandas as pd

from utils import beautify_user_field_names, beautify_user_items_with_lists

# Read the CSV file into a pandas DataFrame
users_df = pd.read_csv('dataset/raw_users.csv')

# Fill all NaN values with an empty string
users_df = users_df.fillna('')
users_df.rename(columns={"id": "author_id"}, inplace=True)
users_df = users_df.loc[users_df.astype(
    str).drop_duplicates(subset='author_id', keep="last").index]

beautify_user_field_names(users_df)
beautify_user_items_with_lists(users_df)

# Drop unnecessary columns
users_df = users_df.drop(
    ['Unnamed: 0'], axis=1)

cols = users_df.dtypes[users_df.dtypes ==
                       "int64"].index.values.tolist()

# Normalize numerical columns
numeric_cols = ['verified', 'location', 'url', 'followers_count', 'following_count', 'tweet_count',
                'listed_count', 'url_urls', 'description_urls', 'mentions', 'hashtags', 'days_on_twitter', 'names_not_equal']
users_df[numeric_cols] = (users_df[numeric_cols] /
                          users_df[numeric_cols].max()) * 100

# Write normalized df to csv
users_df = users_df.drop(
    ['created_at', 'today'], axis=1)
users_df.to_csv('users/normalized-users.csv')
