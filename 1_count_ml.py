import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

from utils import get_dataset
from preprocessing import extract_features, get_lemma_labels

# Future warning for coef_, says to use importance_getter but this function doesn't exists
import warnings

warnings.filterwarnings("ignore")


def filter_features(dataset):
    """
    POSTag filtering of noun, adjective, verb and adverb.
    """
    filtered_dataset = []
    for entry in dataset:
        valid_tokens = []
        for n, tag in enumerate(entry["features"]["tags"]):
            if tag in ["NOUN", "ADJ", "VERB", "ADV"]:
                valid_tokens.append(n)
        for feature_name, value in entry["features"].items():
            entry["features"][feature_name] = [value[i] for i in valid_tokens]
        for label_name, value in entry["labels"].items():
            entry["labels"][label_name] = [value[i] for i in valid_tokens]
        filtered_dataset.append(entry)
    return filtered_dataset


raw_train = get_dataset("train")
train = extract_features(raw_train)
train = filter_features(train)
raw_test = get_dataset("test")
test = extract_features(raw_test)
test = filter_features(test)

# Use lemmatized tokens

labels_lemmas_train = get_lemma_labels(train)
labels_train = list(set([entry for sub_list in labels_lemmas_train for entry in sub_list]))
labels_lemmas_test = get_lemma_labels(test)
labels_test = set([entry for sub_list in labels_lemmas_test for entry in sub_list])
total_test = len(labels_test)
common_labels = labels_test.intersection(labels_train)
labels_test = list(labels_test)
max_accuracy = len(common_labels) / total_test
print("Max accuracy : ", max_accuracy)

# All the preprocessing steps are already done  (by filter_features)

one_hot_labels = []
# Extract one_hot train
for entry, label_train in zip(train, labels_lemmas_train):
    row_data = {key: 0 for key in labels_train}
    for label in label_train:
        row_data[label] = 1
    row_data["X"] = entry["features"]["lemmas"]
    one_hot_labels.append(row_data)
X_df_train = pd.DataFrame(one_hot_labels)
X_train = X_df_train["X"]
y_train = X_df_train.drop(columns="X", axis=1)
# Extract one_hot test, ignoring labels that we don't have in train
one_hot_labels = []
for entry, labels in zip(test, labels_lemmas_test):
    row_data = {key: 0 for key in labels_train}
    for label in labels:
        if label in labels_train:
            row_data[label] = 1
    row_data["X"] = entry["features"]["lemmas"]
    one_hot_labels.append(row_data)
X_test_df = pd.DataFrame(one_hot_labels).fillna(0)
X_test = X_test_df["X"]
y_test = X_test_df.drop(columns="X", axis=1)
# Vectorize text using counts
y_train = np.asarray(y_train, dtype=np.int64)
y_test = np.asarray(y_test, dtype=np.int64)
vect = CountVectorizer(max_df=0.7, min_df=1, analyzer=lambda x: x, preprocessor=lambda x: x)
X_train_cnt = vect.fit_transform(X_train)
X_test_cnt = vect.transform(X_test)
# Create different models. These are multi-label models

nb_classif = OneVsRestClassifier(MultinomialNB()).fit(X_train_cnt, y_train)
C = 1.0
lin_svc = OneVsRestClassifier(LinearSVC(C=C)).fit(X_train_cnt, y_train)
# Predict the test data using classifiers, no grid search as it's just a weak example.

y_pred_class = nb_classif.predict(X_test_cnt)
print(" Accuracy in terms that exist in train dataset for Naive Bayes classifier")
print(metrics.accuracy_score(y_test, y_pred_class))
print(" True accuracy for Naive Bayes classifier")
print(metrics.accuracy_score(y_test, y_pred_class) * max_accuracy)

y_pred_class_lin_svc = lin_svc.predict(X_test_cnt)
print(" Accuracy in terms that exist in train dataset for Linear Support Vector Classification")
print(metrics.accuracy_score(y_test, y_pred_class_lin_svc))
print(" True accuracy for Linear Support Vector Classification")
print(metrics.accuracy_score(y_test, y_pred_class_lin_svc) * max_accuracy)
print(" Recall (per term) for Linear Support Vector Classification")
print(metrics.recall_score(y_test, y_pred_class_lin_svc, average="micro"))
print("True Recall for Linear Support Vector Classification")
print(metrics.recall_score(y_test, y_pred_class_lin_svc, average="micro") * max_accuracy)
# Importance linear svc
# For example, for the labels "lid", "service" and "window operating system" we have that
# following tokens are the most important for identifying these labels:
features_name = np.array(vect.get_feature_names_out())
for feature_to_analyse in ["lid", "service", "window operating system"]:
    label = labels_train.index(feature_to_analyse)
    top_importance_idx = lin_svc.coef_[label].argsort()[::-1]
    top_features = [features_name[idx] for idx in top_importance_idx[:5] if lin_svc.coef_[label][idx] > 0]
    top_neg_features = [features_name[idx] for idx in top_importance_idx[::-1][:5] if lin_svc.coef_[label][idx] < 0]
    print("Top contributing tokens  {}".format(feature_to_analyse))
    print(top_features, "\n")
    print("Top negatin token {}".format(top_neg_features))
