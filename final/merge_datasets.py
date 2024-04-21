# flake8: noqa
import pandas as pd

tweets_df = pd.read_csv('tweets/tweet_social_score.csv')
users_df = pd.read_csv('users/author_cred_score.csv')

merged_real_tweets = pd.merge(users_df, tweets_df, on='author_id')


merged_real_tweets['overall_cred_score'] = merged_real_tweets['tweet_text_cred_score'] * \
    0.33 + merged_real_tweets['author_cred_score'] * 0.33 + \
    merged_real_tweets['tweet_social_cred_score'] * 0.33

merged_real_tweets = merged_real_tweets.fillna(0)
merged_real_tweets = merged_real_tweets.drop(
    ['Unnamed: 0_x', 'Unnamed: 0_y'], axis=1)

cols_to_drop = ['verified', 'location', 'url', 'followers_count', 'following_count', 'tweet_count', 'listed_count', 'url_urls', 'description_urls',
                'mentions_x', 'mentions_y', 'hashtags_x', 'hashtags_y', 'days_on_twitter', 'names_not_equal', 'Unnamed: 0.1', 'possibly_sensitive',
                'retweet_count', 'reply_count', 'like_count', 'quote_count', 'impression_count', 'media_keys', 'urls', 'context_annotations',
                'annotations', 'text_len', 'has_bad_words', 'has_subjective_words']

stripped_merge_real_tweets = merged_real_tweets.drop(
    cols_to_drop, axis=1)
merged_real_tweets.to_csv('final/merged_data.csv')
stripped_merge_real_tweets.to_csv('final/stripped_merged_data.csv')
