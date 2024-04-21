import pandas as pd

tweets_df = pd.read_csv('tweets/tweet_text_cred_score.csv')


WEIGHTS = {
    'retweet_count': 0.30,
    'impression_count': 0.25,
    'like_count': 0.20,
    'quote_count': 0.15,
    'reply_count': 0.10,
}


def calculate_score(row):
    score_sum = 0
    for weight in WEIGHTS:
        score_sum += WEIGHTS[weight]*row[weight]
    return score_sum


tweets_df['tweet_social_cred_score'] = tweets_df.apply(calculate_score, axis=1)
tweets_df = tweets_df.drop(['Unnamed: 0'], axis=1)
tweets_df.to_csv('tweets/tweet_social_score.csv')
