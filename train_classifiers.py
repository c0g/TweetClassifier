import pandas as pd
from textblob import TextBlob
from sklearn.naive_bayes import MultinomialNB
import random
import numpy as np
from util import *
import pickle


class TweetClassifier:
    def __init__(self):
        self.x, self.x_rich, self.y, self.y_rich, self.y_approx, self.y_approx_rich, self.features, \
            self.classifiers, self.labels, self.rough_labels = [None] * 10

    def read_excel(self, sheet_name):
        category_text = read_tweets(sheet_name, ['URGENT', 'Tweets Located DONE'])
        self.features = find_features(category_text)
        self.labels, self.rough_labels = make_labels(category_text)
        rich_data = find_good_trainers(category_text, 100)
        all_data = make_training_data(category_text, self.labels, self.rough_labels, self.features)
        rich_data = make_training_data(rich_data, self.labels, self.rough_labels, self.features)
        self.x = all_data[0] if self.x is None else self.x + all_data[0]
        self.y = all_data[1] if self.y is None else self.y + all_data[1]
        self.y_approx = all_data[2] if self.y_approx is None else self.y_approx + all_data[2]
        self.x_rich = rich_data[0] if self.x_rich is None else self.x_rich + rich_data[0]
        self.y_rich = rich_data[1] if self.y_rich is None else self.y_rich + rich_data[1]
        self.y_approx_rich = rich_data[2] if self.y_approx_rich is None else self.y_approx_rich + rich_data[2]

    def train_classifiers(self):
        self.classifiers = {'all_data': MultinomialNB().fit(self.x, self.y),
                            'all_data_restricted': MultinomialNB().fit(self.x, self.y_approx),
                            'rich_data': MultinomialNB().fit(self.x_rich, self.y_rich),
                            'rich_data_restricted': MultinomialNB().fit(self.x_rich, self.y_approx_rich)}

    def read_pickle(self, pickle_file):
        self.x, self.x_rich, self.y, self.y_rich, self.y_approx, self.y_approx_rich, self.classifiers = \
            pickle.load(open(pickle_file, 'rb'))

    def dump_pickle(self, pickle_file):
        pickle.dump([self.x, self.x_rich, self.y, self.y_rich, self.y_approx, self.y_approx_rich, self.classifiers],
                    open(pickle_file, 'wb'))

    def classify(self, tweet):
        f_vec = tweet_to_feat(tweet, self.features)
        return {key: (self.classifiers[key].predict(f_vec), self.classifiers[key].predict_proba(f_vec))
                for key in self.classifiers}


if __name__ == "__main__":
    import redis
    from pprint import pprint as pretty
    tc = TweetClassifier()
    tc.read_excel('/Users/tom/Desktop/phillipines/sheets.xlsx')
    tc.train_classifiers()
    r = redis.StrictRedis()
    tweet = "water needed at boggle city"
    pretty(tweet)
    pretty(tc.classify(tweet))
    pretty(tc.labels)
    pretty(tc.rough_labels)



##urgent_tweets = pd.read_excel('./sheets.xlsx', 'URGENT', encoding='UTF-8')
##tweets = pd.read_excel('./sheets.xlsx', 'Tweets Located DONE', encoding='UTF-8')
##
##all_tweets = pd.concat([urgent_tweets, tweets], )
##
##cats = dict()
##for tweet in tweet_cat:
##    cat = tweet_cat[tweet]
##    if cat in cats:
##        cats[cat] = cats[cat] + [tweet]
##    else:
##        cats[cat] = [tweet]
#
#
#good_keys = [(len(cats[key]), key) for key in cats if len(cats[key]) > 100]
#length_min = min(good_keys)[0]
#sub_cat = {cat[1]: random.sample(cats[cat[1]], length_min) for cat in good_keys}
#
#
#
#
#
#for i, tweet in enumerate(tweet_cat):
#    if type(tweet) == unicode:
#        features = features | set((word for word in TextBlob(tweet).lower().tokenize()))
#
#features = set((word.lemma for word in features if not ignore(word)))
#
#labels = {name: num for num, name in enumerate(set(all_tweets["Category"]))}
#labels_approx = {name: num for num, name in enumerate(set([cat[:10] if
#                        type(cat) != float else cat for cat in all_tweets["Category"]]))}
#
#
#y = []
#y_approx = []
#X = []
#for n, tweet in enumerate(tweet_cat):
#    if type(tweet) is unicode:
#        cat = tweet_cat[tweet]
#        if type(cat) is unicode:
#            y.append(labels[cat])
#            y_approx.append(labels_approx[cat[:10]])
#            words = [word.lemma for word in TextBlob(tweet).lower().tokenize()]
#            X.append([words.count(feature) for feature in features])
#X = np.array(X)
#y = np.array(y)
#y_approx = np.array(y_approx)
#
#
#
#
#X_sub = np.array(X_sub)
#y_sub = np.array(y_sub)
#y_sub_approx = np.array(y_sub_approx)
#
#mnb = MultinomialNB()
#mnb.fit(X, y)
#
#mnb_a = MultinomialNB()
#mnb.fit(X, y_approx)
#
#mnb_s = MultinomialNB()
#mnb.fit(X_sub, y_sub)
#
#mnb_sa = MultinomialNB()
#mnb.fit(X_sub, y_sub_approx)
#
#
#
#def train_classifier(x, y):
#    return MultinomialNB().fit(x, y)