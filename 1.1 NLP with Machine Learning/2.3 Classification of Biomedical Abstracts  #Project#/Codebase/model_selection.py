from sklearn.linear_model import LassoCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import classification_report, cohen_kappa_score as kappa, confusion_matrix, roc_auc_score, roc_curve

from helpers import PICKLE_DIR, pickle_object
import pandas as pd

if __name__ == '__main__':
    binarizer = LabelBinarizer()

    X = pd.read_pickle(PICKLE_DIR / 'training_X_lsa_6000_components.gz', 'gzip')
    y = pd.read_pickle(PICKLE_DIR / 'training_labels.gz', 'gzip')
    # y = binarizer.fit_transform(y)

    skf = StratifiedKFold(4)
    gnb = GaussianNB()
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        gnb.fit(X_train, y_train)
        y_pred = gnb.predict(X_test)
        score = gnb.score(X_test, y_test)

        print('Score: {}'.format(score))
        print('Classification report for fold {}:'.format(i))
        print(classification_report(y_test, y_pred))
        print('\n---\n')


    X_test_final = pd.read_pickle(PICKLE_DIR / 'testing_X_lsa_6000_components.gz', 'gzip')
    y_test_final = pd.read_pickle(PICKLE_DIR / 'testing_labels.gz', 'gzip')
    # y_test_final = binarizer.transform(y_test_final)
    y_pred_final = gnb.predict(X_test_final)

    score = gnb.score(X_test_final, y_test_final)
    print('Score: {}'.format(score))

    print('Final Classification Report:')
    print(classification_report(y_test_final, y_pred_final))