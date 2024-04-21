# flake8: noqa
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from utils import word_list_includes_text_words

# Read the CSV file into a pandas DataFrame
tweets_df = pd.read_csv('tweets/normalized-tweets.csv')

with open('tweets/tr-swear-words.txt', 'r', encoding='utf-8') as f:
    swear_words = [line.strip() for line in f]

with open('tweets/tr-subjective-words.txt', 'r', encoding='utf-8') as f:
    subjective_words = [line.strip() for line in f]

# Bad words filter
tweets_df['has_bad_words'] = tweets_df['text'].apply(
    word_list_includes_text_words, args=(swear_words,))

# Subjective words filter
tweets_df['has_subjective_words'] = tweets_df['text'].apply(
    word_list_includes_text_words, args=(subjective_words,))

# Sentiment analysis filter
model = AutoModelForSequenceClassification.from_pretrained(
    "savasy/bert-base-turkish-sentiment-cased")
tokenizer = AutoTokenizer.from_pretrained(
    "savasy/bert-base-turkish-sentiment-cased")
sa = pipeline("sentiment-analysis", tokenizer=tokenizer, model=model)


def sentiment_analysis(row):
    result = sa(row["text"])
    print(result)
    if result[0]['label'] == 'positive':
        return 1
    else:
        return 0


tweets_df['sentiment_analysis'] = tweets_df.apply(
    sentiment_analysis, axis=1)

tweets_df.to_csv('tweets/text-scored-tweets.csv')
