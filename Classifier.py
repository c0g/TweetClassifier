from sklearn.naive_bayes import MultinomialNB
from util import *
import pickle
import numpy as np


class TweetClassifier:
    def __init__(self):
        self.x, self.x_rich, self.y, self.y_rich, self.y_approx, self.y_approx_rich, self.features, \
            self.classifiers, self.labels, self.rough_labels = [None] * 10
        self.file_name = 'classifiers.p'
        self._load_classifiers()

    def _load_classifiers(self):
        print "Loading classifiers"
        try:
            self._read_pickle()
        except IOError:
            print "No classifiers - please wait, learning"
            self.make_classifiers()

    def make_classifiers(self):
        print "Loading tweets"
        self._read_excel()
        print "Training..."
        self._train_classifiers()
        try:
            print "Pickling classifiers"
            self._dump_pickle()
        except IOError:
            print "Failed to save classifiers to file"

    def _read_excel(self):
        category_text = read_tweets('sheets.xlsx', ['URGENT', 'Tweets Located DONE'])
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

    def _train_classifiers(self):
        self.classifiers = {'all_data': MultinomialNB().fit(self.x, self.y),
                            'all_data_restricted': MultinomialNB().fit(self.x, self.y_approx),
                            'rich_data': MultinomialNB().fit(self.x_rich, self.y_rich),
                            'rich_data_restricted': MultinomialNB().fit(self.x_rich, self.y_approx_rich)}

    def _read_pickle(self):
        self.classifiers, self.features, self.labels, self.rough_labels = pickle.load(open(self.file_name, 'rb'))

    def _dump_pickle(self):
        pickle.dump([self.classifiers, self.features, self.labels, self.rough_labels], open(self.file_name, 'wb'))

    def classify(self, tweet):
        f_vec = tweet_to_feat(tweet, self.features)
        cat_names = dict_inverse(self.labels)
        rough_cat_names = dict_inverse(self.rough_labels)
        results = dict()
        for key in self.classifiers:
            result = self.classifiers[key].predict(f_vec)
            if key[-3:] == 'ted':
                results[key] = rough_cat_names[result[0]]
            else:
                results[key] = cat_names[result[0]]
        return results