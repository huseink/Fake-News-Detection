import pandas as pd

user_df = pd.read_csv('users/normalized-users.csv')

WEIGHTS = {
    'VERIFIED': 0.25,
    'FOLLOWERS_COUNT': 0.20,
    'LISTED_COUNT': 0.015,
    'DAYS_ON_TWITTER': 0.10,
    'TWEET_COUNT': 0.10,
    'LOCATION': 0.075,
    'URL': 0.075,
    'FOLLOWING_COUNT': 0.025,
    'NAMES_NOT_EQUAL': 0.015,
    'DESCRIPTION_URLS': 0.01,
    'MENTIONS': 0.005,
}

total_weights = sum(WEIGHTS.values())
if total_weights > 1 or total_weights < 0:
    print('Total weights should equal to 1')


def calculate_score(row):
    score_sum = 0
    for weight in WEIGHTS:
        score_sum += WEIGHTS[weight]*row[weight.lower()]
    return score_sum


user_df['author_cred_score'] = user_df.apply(calculate_score, axis=1)
user_df = user_df.drop(['Unnamed: 0'], axis=1)
user_df.to_csv('users/author_cred_score.csv')
