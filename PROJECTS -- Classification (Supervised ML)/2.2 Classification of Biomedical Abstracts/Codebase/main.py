import pandas as pd
from sklearn.model_selection import train_test_split

from helpers import ABSTRACT_DIR, TSV_DIR, PICKLE_DIR, pickle_object, unpickle_object, Logger
from preprocessors import LSAPreprocessor

ABSTRACT_FILE = ABSTRACT_DIR / 'abstracts.tsv'

# def preprocess():
#     ### BEGIN LSA HERE
#     logger = Logger()

#     # result: 5000 - 6000 components
#     logger.plog('Starting LSA.')

#     preproc = LSAPreprocessor(ABSTRACT_DIR / 'abstracts.tsv', label_col='source', max_tfidf_features=7000)
#     X_train = preproc.training_corpus['abstract']
#     y_train = preproc.training_corpus['label']
#     preproc.get_lsa_pipeline()  # sets preproc.lsa 

#     preproc.set_svd(n_components=6000)

#     logger.plog('Beginning fit on training data.')
#     X_train_lsa = preproc.lsa.fit_transform(X_train, y_train)

#     explained_variance = preproc.lsa.named_steps['svd'].explained_variance_ratio_.sum()
#     logger.plog('Total explained variance ratio: {:.02%} ...'.format(explained_variance))

#     pickle_object(X_train_lsa, 'training_X_lsa.gz')
#     pickle_object(y_train, 'training_labels.gz')

#     X_test = preproc.testing_corpus['abstract']
#     y_test = preproc.testing_corpus['label']

#     logger.plog('Beginning transform on testing data.')
#     X_test_lsa = preproc.lsa.transform(X_test)
#     pickle_object(X_test_lsa, 'testing_X_lsa.gz')
#     pickle_object(y_test,'testing_labels.gz')

#     logger.plog('LSA Complete.')

#     pickle_object(preproc, 'trained_preprocessor.pkl')


if __name__ == "__main__":
    preproc = unpickle_object('trained_preprocessor.pkl')
    print(preproc.lsa.named_steps['svd'].components_)