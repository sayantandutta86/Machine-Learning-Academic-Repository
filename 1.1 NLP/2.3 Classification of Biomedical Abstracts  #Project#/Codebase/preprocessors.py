#/usr/bin/env python

"""
    IS590ML - Final Project
    Preprocessing for comparison of 'Pubmed' against 'BiorXiv' article abstracts

    Run from main directory
        > python ./scripts/preproc.py
"""
from pathlib import Path
import re
import pickle

from helpers import DATA_DIR, OUTPUT_DIR, ABSTRACT_DIR, PICKLE_DIR, TSV_DIR, pickle_object

import nltk

# download necessary libraries
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

from nltk import PorterStemmer, word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import numpy as np
import pandas as pd


# Only look at terms which occur in MAX_DF or fewer documents (ignoring extremely common words)
# 65% tends to be a good starting point


class LSAPreprocessor():
    def __init__(self, corpus_path=None, corpus_frac=None, max_df=0.65, min_word_len=3, max_tfidf_features=10000, n_svd_components=100, label_col=None, id_col='id', append_pos_tags=False):
        self.max_df = max_df
        self.min_word_len = min_word_len
        self.max_tfidf_features = max_tfidf_features
        self.n_svd_components = n_svd_components
        self.append_pos_tags = append_pos_tags
        
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

        self.stopwords = []
        with (DATA_DIR / 'misc' / 'stopwords.txt').open('r') as fd:
            self.stopwords = [line.strip() for line in fd]

        self.regex = {
            'number': re.compile(r'[0-9]+?'),
            'web_email': re.compile(r'((www.+?\s)|(http.+?\s)|([a-z]+?\@.+?s)|(\.[a-z]{2,3}))'),
            'spacer': re.compile(r'[\_\-]'),
            'punct': re.compile(r'[\[\]\'\.,\/\#\!\?\$\%\^\&\*;\:{}=\_`~\(\)\n\r�\<\>\@\\]+?')
        }

        self.tokenizer = str.split

        self.corpus = self.load_corpus(corpus_path, label_col=label_col, corpus_frac=corpus_frac)
        self.training_corpus, self.testing_corpus = self.split_corpus()

        self.vectorizer_params = {
            # 'lowercase': True,  # Covered by preprocessor
            # 'stop_words': self.stopwords,  # Covered by preprocessor
            'analyzer': 'word', 
            'preprocessor': self.preprocess,
            'tokenizer': self.tokenizer,
            'max_df': self.max_df, 
            'max_features': self.max_tfidf_features,
        }

        self.svd_params = {
            'n_components': self.n_svd_components,
            'n_iter': 5
        }

        self.count_vectorizer = None
        self.tfidf_transformer = TfidfTransformer()

        self.tfidf_vectorizer = None
        self.svd = None
        self.lsa = None

        self.set_vectorizers()
        self.set_svd()

    def set_vectorizers(self, **kwargs):
        if kwargs:
            self.vectorizer_params.update(**kwargs)

            if 'max_df' in kwargs:
                self.max_df = kwargs.get('max_df')

            if 'max_features' in kwargs:
                self.max_features = kwargs.get('max_features')
    
        self.count_vectorizer = CountVectorizer(self.vectorizer_params)
        # https://scikit-learn.org/stable/modules/decomposition.html#truncated-singular-value-decomposition-and-latent-semantic-analysis
        self.tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, **self.vectorizer_params)

        # set LSA params if needed
        if self.lsa:
            self.lsa.set_params(**{'tfidf__{}'.format(k): v for k,v in self.vectorizer_params.items()})

    def set_svd(self, **kwargs):
        if kwargs:
            self.svd_params.update(**kwargs)

            if 'n_components' in kwargs:
                self.n_svd_components = kwargs.get('n_components')

        self.svd = TruncatedSVD(self.svd_params)

        # set LSA params if needed
        if self.lsa:
            self.lsa.set_params(**{'svd__{}'.format(k): v for k,v in self.svd_params.items()})

    def load_corpus(self, filepath, label_col=None, id_col='id', corpus_frac=None, to_pickle=True):
        self.corpus = pd.read_csv(
            filepath, 
            sep='\t', 
            header=0,
            na_values=['None', ''],
            keep_default_na=False,
            dtype=str)

        self.corpus.drop_duplicates(id_col, keep='first', inplace=True)    
        self.corpus.dropna(inplace=True)

        self.corpus.set_index(id_col, drop=True, inplace=True)

        if label_col:
            self.corpus.rename(columns={label_col: 'label'}, inplace=True)

        if to_pickle:
            self.corpus.to_pickle(PICKLE_DIR / 'raw_corpus.gz')

        if corpus_frac:
            original_size = self.corpus.shape[0]
            self.corpus = self.corpus.sample(frac=corpus_frac, random_state=17)
            print('Reducing corpus to {:02%} of original size: {} to {}'.format(corpus_frac, original_size, self.corpus.shape[0]))

            if to_pickle:
                self.corpus.to_pickle(PICKLE_DIR / 'reduced_corpus.gz')

        return self.corpus

    def split_corpus(self, test_size=0.2, to_tsv=True, to_pickle=True):
        self.training_corpus, self.testing_corpus = train_test_split(self.corpus, test_size=test_size, random_state=17)
        
        if to_tsv:
            self.training_corpus.to_csv(TSV_DIR / 'training.tsv', sep='\t')
            self.testing_corpus.to_csv(TSV_DIR / 'testing_final.tsv', sep='\t')

        if to_pickle:
            self.training_corpus.to_pickle(PICKLE_DIR / 'training.pkl.gz')
            self.testing_corpus.to_pickle(PICKLE_DIR / 'testing_final.pkl.gz')

        return self.training_corpus, self.testing_corpus

    def get_lsa_pipeline(self):
        self.lsa = Pipeline(steps=[('tfidf', self.tfidf_vectorizer), ('svd', self.svd), ('normalizer', Normalizer(copy=False))])
        return self.lsa

    def get_wordnet_pos(self, treebank_tag):
        """ Annoying conversions to wordnet abbreviations. """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return ''

    def preprocess(self, doc):
        """ Convert doc to lowercase, remove apostrophes and numbers. """
        new_words = []

        doc = doc.lower()
        doc = self.regex.get('spacer').sub(' ', doc)  # add spacing
        doc = self.regex.get('number').sub('', doc)
        doc = self.regex.get('web_email').sub('', doc)
        doc = self.regex.get('punct').sub('', doc)
        doc = pos_tag(word_tokenize(doc))

        for word, tag in doc:
            tag = self.get_wordnet_pos(tag)
            
            if word in self.stopwords or len(word) < self.min_word_len:
                # ignore words with 2 or fewer characters
                continue

            try:
                float(word) # ignore numbers
            except ValueError as e:
                pass
            else:
                continue

            if tag and tag != '':
                word = self.lemmatizer.lemmatize(word, tag)
            
            if self.append_pos_tags:
                word = word + '(' + tag + ')'
            
            new_words.append(word)
        
        return ' '.join(new_words)

