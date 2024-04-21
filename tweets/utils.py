# flake8: noqa
import pandas as pd


def beautify_tweet_field_names(data):
    data = data.rename(columns={"entities.hashtags": "hashtags",
                                "entities.mentions": "mentions",
                                "entities.urls": "urls",
                                "public_metrics.retweet_count": "retweet_count",
                                "public_metrics.reply_count": "reply_count",
                                "public_metrics.like_count": "like_count",
                                "public_metrics.quote_count": "quote_count",
                                "attachments.media_keys": "media_keys",
                                "id": "tweet_id",
                                "public_metrics.impression_count": "impression_count",
                                "entities.annotations": "annotations", }, inplace=True)


def beautify_tweet_items_with_lists(data):
    if "location" in data:
        data["location"] = data["location"].fillna("")
    if "hashtags" in data:
        data["hashtags"] = data["hashtags"].fillna("").apply(list)
        data['hashtags'] = data["hashtags"].map(len)
    if "mentions" in data:
        data["mentions"] = data["mentions"].fillna("").apply(list)
        data['mentions'] = data["mentions"].map(len)
    if "urls" in data:
        data["urls"] = data["urls"].fillna("").apply(list)
        data['urls'] = data["urls"].map(len)
    if "media_keys" in data:
        data["media_keys"] = data["media_keys"].fillna("").apply(list)
        data['media_keys'] = data["media_keys"].map(len)
    if "annotations" in data:
        data["annotations"] = data["annotations"].fillna("").apply(list)
        data['annotations'] = data["annotations"].map(len)
    if "context_annotations" in data:
        data["context_annotations"] = data["context_annotations"].fillna(
            "").apply(list)
        data['context_annotations'] = data["context_annotations"].map(len)
    if (data['lang'].eq('tr')).any():
        data["lang"] = 1
    else:
        data["lang"] = 0

    if "possibly_sensitive" in data:
        data["possibly_sensitive"] = data["possibly_sensitive"].astype(int)
    data["text_len"] = data["text"].str.len()


def word_list_includes_text_words(text, word_list):
    # Convert the text to lowercase and split into words
    words = text.lower().split()

    # Check if any of the words appear in the swear words list
    for word in words:
        if word in word_list:
            return 1

    return 0
