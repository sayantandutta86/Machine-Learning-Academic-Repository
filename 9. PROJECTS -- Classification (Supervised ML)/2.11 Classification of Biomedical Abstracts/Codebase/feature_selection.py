from pathlib import Path

from helpers import pickle_object, unpickle_object, ABSTRACT_DIR, TSV_DIR, PICKLE_DIR, Logger
from preprocessors import LSAPreprocessor

import pandas as pd
import numpy as np

from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.decomposition import TruncatedSVD

N_SPLITS = 2
logger = Logger()

def test_svd_components():
    logger.plog('Started Initial Feature Selection.')

    # preproc = LSAPreprocessor(ABSTRACT_DIR / 'abstracts.tsv', label_col='source', max_tfidf_features=10000)
    preproc = LSAPreprocessor(ABSTRACT_DIR / 'abstracts.tsv', label_col='source', max_tfidf_features=10000)
    
    logger.plog("Corpus Loaded")
    X = preproc.training_corpus['abstract']
    y = preproc.training_corpus['label']
    preproc.get_lsa_pipeline()

    # component_set =  (10, 75, 100, 500, 1000, 2500, 5000)
    # component_set = (50, 100, 250, 500, 1000, 2500, 5000)
    component_set = (7500, 9000)

    n_splits = N_SPLITS
    n_repeats = len(component_set)

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=5861)
    
    explained_variances = {}

    logger.plog('Beginning LSA CV with {} instances.'.format(X.shape[0]))

    tfidf_components = 0

    for i, (train_indices, test_indices) in enumerate(rskf.split(X, y)):
        split_num = i % n_splits
        repeat_num = int(i / n_splits)
        n_components = component_set[repeat_num]

        # logger.plog('#{:02d} ({:02d} | {:02d} | {:04d}): {} ...] | {} ...]'.format(i, split_num, repeat_num, n_components, str(train_indices[:5])[:-1], str(test_indices[:5])[:-1]))
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

        preproc.set_svd(n_components=n_components)

        logger.plog('Fitting {} components (split #{}):'.format(n_components, split_num))
        
        X_lsa = preproc.lsa.fit_transform(X_train, y_train)

        tfidf_components = len(preproc.lsa.named_steps['tfidf'].vocabulary_)

        key = '{:d}:{:d}'.format(split_num, n_components)
        explained_variances[key] = preproc.lsa.named_steps['svd'].explained_variance_ratio_.sum()

        logger.plog('Total explained variance ratio with {} components (from {} components), split #{}: {:.02%} ...'.format(n_components, tfidf_components, split_num, explained_variances[key]))
        print('')


    logger.plog(explained_variances)
    logger.plog('')

    pickle_object(preproc, 'trained_preprocessor.pkl')


def run_lsa(n_components_list):
    logger.plog('Starting LSA.')

    preproc = LSAPreprocessor(ABSTRACT_DIR / 'abstracts.tsv', label_col='source', max_tfidf_features=10000)
    preproc.get_lsa_pipeline()
    
    X_train = preproc.training_corpus['abstract']
    y_train = preproc.training_corpus['label']
    y_train.to_pickle(PICKLE_DIR / 'training_labels.gz', 'gzip')

    X_test = preproc.testing_corpus['abstract']
    y_test = preproc.testing_corpus['label']
    y_test.to_pickle(PICKLE_DIR / 'testing_labels.gz', 'gzip')

    for n_components in n_components_list:
        logger.plog('Starting SVD for {} components.'.format(n_components))
        preproc.set_svd(n_components=n_components)

        logger.plog('Beginning fit on training data.')
        X_train_lsa = preproc.lsa.fit_transform(X_train, y_train)
        X_train_lsa = pd.DataFrame(X_train_lsa, index=X_train.index)
        pickle_object(preproc, 'trained_preprocessor_{:04d}_components.pkl'.format(n_components))

        explained_variance = preproc.lsa.named_steps['svd'].explained_variance_ratio_.sum()
        logger.plog('Total explained variance ratio for {} components: {:.02%} ...'.format(n_components, explained_variance))

        X_train_lsa.to_pickle(PICKLE_DIR / 'training_X_lsa_{:04d}_components.gz'.format(n_components), 'gzip')

        logger.plog('Beginning transform on testing data.')
        X_test_lsa = preproc.lsa.transform(X_test)
        X_test_lsa = pd.DataFrame(X_test_lsa, index=X_test.index)
        X_test_lsa.to_pickle(PICKLE_DIR / 'testing_X_lsa_{:04d}_components.gz'.format(n_components), 'gzip')

        logger.plog('LSA Complete.')


if __name__ == "__main__":
    run_lsa([5000, 6000, 7000])
