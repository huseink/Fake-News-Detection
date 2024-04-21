import pandas as pd

tweets_df = pd.read_csv('tweets/analyzed-tweets.csv')

tweets_df['has_bad_words'] = tweets_df['has_bad_words'] * 100
tweets_df['has_subjective_words'] = tweets_df['has_subjective_words'] * 100
tweets_df['sentiment_analysis'] = tweets_df['sentiment_analysis'] * 100

WEIGHTS = {
    'urls': 0.29,
    'context_annotations': 0.25,
    'text_len': 0.18,
    'media_keys': 0.18,
    'has_subjective_words': 0.04,
    'has_bad_words': 0.03,
    'possibly_sensitive': 0.03,
}


def calculate_score(row):
    score_sum = 0
    for weight in WEIGHTS:
        score_sum += WEIGHTS[weight]*row[weight]
    return score_sum


tweets_df['tweet_text_cred_score'] = tweets_df.apply(calculate_score, axis=1)
tweets_df = tweets_df.drop(['Unnamed: 0'], axis=1)
tweets_df.to_csv('tweets/tweet_text_cred_score.csv')
