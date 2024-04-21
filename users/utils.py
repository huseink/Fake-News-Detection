# flake8: noqa
import pandas as pd
import regex as re


def beautify_user_field_names(data):
    data = data.rename(columns={"entities.description.urls": "description_urls",
                                "entities.description.mentions": "mentions",
                                "entities.description.hashtags": "hashtags",
                                "entities.url.urls": "url_urls",
                                "public_metrics.followers_count": "followers_count",
                                "public_metrics.following_count": "following_count",
                                "public_metrics.tweet_count": "tweet_count",
                                "public_metrics.listed_count": "listed_count",
                                "id": "author_id",
                                "entities.description.annotations": "annotations", }, inplace=True)


def beautify_user_items_with_lists(data):
    if "location" in data:
        data["location"] = 1
    else:
        data["location"] = 0
    if "created_at" in data:
        from datetime import datetime, timezone
        data['today'] = datetime.now(timezone.utc)
        data[['created_at', 'today']] = data[['created_at', 'today']].apply(
            pd.to_datetime)  # if conversion required
        data['days_on_twitter'] = (data['today'] - data['created_at']).dt.days
    if "hashtags" in data:
        data["hashtags"] = data["hashtags"].fillna("").apply(list)
        data['hashtags'] = data["hashtags"].map(len)
    if "mentions" in data:
        data["mentions"] = data["mentions"].fillna("").apply(list)
        data['mentions'] = data["mentions"].map(len)
    if "name" in data and "username" in data:
        if data['name'].equals(data['username']):
            data["names_not_equal"] = 0
        else:
            data["names_not_equal"] = 1
    if "description_urls" in data:
        data["description_urls"] = data["description_urls"].fillna(
            "").apply(list)
        data['description_urls'] = data["description_urls"].map(len)
    if "url_urls" in data:
        data["url_urls"] = data["url_urls"].fillna("").apply(list)
        data['url_urls'] = data["url_urls"].map(len)
    if "media_keys" in data:
        data["media_keys"] = data["media_keys"].fillna("").apply(list)
        data['media_keys'] = data["media_keys"].map(len)
    if "annotations" in data:
        data["annotations"] = data["annotations"].fillna("").apply(list)
        data['annotations'] = data["annotations"].map(len)
    if "url" in data:
        data["url"] = 1
    else:
        data["url"] = 0
    if "verified" in data:
        data["verified"] = data["verified"].astype(int)


def contains_bad_words(text):
    # Define a list of bad words
    bad_words = ['küfür', 'argo', 'alçak', 'sapık', 'zina']

    # Convert the text to lowercase and remove non-alphanumeric characters
    clean_text = re.sub(r'\W+', ' ', text.lower())

    # Check if any of the bad words appear in the clean text
    for word in bad_words:
        if word in clean_text:
            return 1

    return 0
