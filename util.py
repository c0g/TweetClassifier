from textblob import TextBlob
import pandas as pd
from random import sample


def find_tweet(tweet, place_list):
    t = TextBlob(unicode(tweet))
    tweet_loc = []
    for word in t.lower().tokenize():
        if word in place_list:
            tweet_loc = tweet_loc + [word]


def ignore_word(word):
    ignore = False
    ignore |= 't.co' in word
    ignore |= 'http' in word
    ignore |= any([True for char in word if char.isdigit()])
    return ignore


def find_features(cat_text):
    features = set()
    for text_list in cat_text.itervalues():
        for text in text_list:
            if type(text) == unicode:
                features = features | set((word for word in TextBlob(text).lower().tokenize()))
    return set((word.lemma for word in features if not ignore_word(word)))


def place_names(file_name):
    places = pd.read_excel(file_name, 'Copy of Philippines Place Names', encoding='UTF-8')
    place_list = [name.lower() for name in places["NAME"]]
    return place_list


def read_tweets(file_name, sheets):
    tweet_list = []
    for sheet in sheets:
        tweet_list += [pd.read_excel(file_name, sheet, encoding='UTF-8')]
    tweets = pd.concat(tweet_list, keys=["Tweet", "Category"])
    cat_text = dict()
    for text, cat in zip(tweets["Tweet"], tweets["Category"]):
        if cat in cat_text:
            cat_text[cat] += [text]
        else:
            cat_text[cat] = [text]
    return cat_text


def sub_cat(cat):
    return cat.split(" ")[0]


def make_labels(cat_text):
    labels = {cat: i for i, cat in enumerate(cat_text) if type(cat) is unicode}
    rough_labels = {sub_cat(cat): i for i, cat in enumerate(cat_text) if type(cat) is unicode}
    return labels, rough_labels


def find_good_trainers(cat_text, min_length):
    good_keys = [(len(cat_text[key]), key) for key in cat_text if len(cat_text[key]) > min_length]
    shortest = min(good_keys)[0]
    return {cat: sample(cat_text[cat], shortest) for cat in cat_text if len(cat_text[cat]) >= shortest}


def tweet_to_feat(tweet, features):
    words = [word.lemma for word in TextBlob(tweet).lower().tokenize()]
    return [words.count(feature) for feature in features]


def make_training_data(cat_text, fine_labels, rough_labels, features):
    x = []
    y = []
    y_rough = []
    for key in cat_text:
        if type(key) is unicode:
            for text in cat_text[key]:
                if type(text) is unicode:
                    y.append(fine_labels[key])
                    y_rough.append(rough_labels[sub_cat(key)])
                    x.append(tweet_to_feat(text, features))
    return x, y, y_rough
