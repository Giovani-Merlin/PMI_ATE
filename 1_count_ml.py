import numpy as np
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from utils import get_dataset
from preprocessing import extract_features, BIO_MAP,INVERSE_BIO_MAP, parse_dependencies

# Future warning for coef_, says to use importance_getter but this function doesn't exists
import warnings
from sklearn.feature_extraction import DictVectorizer
warnings.filterwarnings("ignore")

SEED = 0
def filter_features(dataset):
    """
    POSTag filtering of noun, adjective, verb and adverb.
    """
    filtered_dataset = []
    label_tags = []
    for entry in dataset:
        valid_tokens = []
        for n, tag in enumerate(entry["features"]["tags"]):
            if tag in ["NOUN", "ADJ", "VERB", "ADV"]:
                valid_tokens.append(n)
            if entry['labels']['bio_map'][n]!='O':
                label_tags.append(entry['features']['tags'][n])


        for feature_name, value in entry["features"].items():
            entry["features"][feature_name] = [value[i] for i in valid_tokens]
        for label_name, value in entry["labels"].items():
            entry["labels"][label_name] = [value[i] for i in valid_tokens]
        filtered_dataset.append(entry)
    return filtered_dataset, label_tags


raw_train = get_dataset("train")
train = extract_features(raw_train)#, punctuation=True,lowercase=True,keep_quote=False)
train = parse_dependencies(train, use_lemmas=False)
# Filter labels too
#train, label_tags = filter_features(train)
raw_test = get_dataset("test")
test = extract_features(raw_test)#, punctuation=True,lowercase=True,keep_quote=False)
test = parse_dependencies(test, use_lemmas=False)
#test, label_tags = filter_features(test)

def extract_train_dataset(dataset):
    x_train = []
    y_train = []
    for n, entry in enumerate(dataset):
        for token_n in range(len(entry['features']['tokens'])):
            row = {}
            features = entry['features']
            row["tag"] = features['tags'][token_n]
            row["token"] = features['tokens'][token_n]
            row["dependency"] = features['dependency_map'][token_n]
            row["head_tag"] = features['head_tag'][token_n]
            #row['id'] = entry['id']
            y_train.append(BIO_MAP[entry["labels"]['bio_map'][token_n]])
            #row['sentence'] = n
            x_train.append(row)
    return x_train, y_train

# Create features
v = DictVectorizer(sparse=True)
x_train, y_train = extract_train_dataset(train)
x_test, y_test = extract_train_dataset(test)
X_train_vec = v.fit_transform(x_train)
X_test_vec = v.transform(x_test)
y_test = np.array(y_test)
# Train models
nb_classif = MultinomialNB().fit(X_train_vec, y_train)
y_pred = nb_classif.predict(X_test_vec)
print("NB report")
print(metrics.classification_report(y_test, y_pred, labels=[1,2]))
#
lin_svc = LinearSVC(class_weight="balanced",random_state=SEED).fit(X_train_vec, y_train)
y_pred = lin_svc.predict(X_test_vec)
print("Linear SVC report")
print(metrics.classification_report(y_test, y_pred, labels=[1,2]))
#
decision_tree = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=30, class_weight='balanced',random_state=SEED).fit(X_train_vec,y_train)
y_pred = decision_tree.predict(X_test_vec)
print("Decision Tree report")
print(metrics.classification_report(y_test, y_pred, labels=[1,2]))
#
logistic_regression = LogisticRegression(class_weight="balanced",random_state=SEED,C=1000.0).fit(X_train_vec, y_train)
y_pred = logistic_regression.predict(X_test_vec)
print("Logistic Regression report")
print(metrics.classification_report(y_test, y_pred, labels=[1,2]))


# Importance analysis
def get_importance_str(coefs, feature_names, important_factors_idx):
    top_features_strs = [f"{feature_names[idx]}:{coefs[idx]:0.3f}" for idx in important_factors_idx[:5] if coefs[idx] > 0]
    top_neg_features_strs = [f"{feature_names[idx]}:{coefs[idx]:0.3f}" for idx in important_factors_idx[::-1][:5] if coefs[idx] < 0]
    return top_features_strs+top_neg_features_strs

feature_names = np.array(v.get_feature_names_out())
for n , coefs in enumerate(lin_svc.coef_):
    print("###############################")
    print(f"Most important factors do determine that a token is of type {INVERSE_BIO_MAP[n]}")
    #
    tokens_idx = np.array([n for n, tag in enumerate(feature_names) if tag.startswith("token")])
    tokens_coefs = coefs[tokens_idx]
    important_factors_idx = tokens_coefs.argsort()[::-1]
    for feature_str in get_importance_str(tokens_coefs, feature_names, important_factors_idx):
        print(feature_str+"\n")
    #
    dependencies_idx = np.array([n for n, dependency in enumerate(feature_names) if dependency.startswith("dependency")])
    dependencies_coefs = coefs[dependencies_idx]
    important_factors_idx = dependencies_coefs.argsort()[::-1]
    for feature_str in get_importance_str(dependencies_coefs[dependencies_idx], feature_names, important_factors_idx):
        print(feature_str+"\n")
    #
    head_tags_idx = np.array([n for n, head_tag in enumerate(feature_names) if head_tag.startswith("head_tag")])
    head_tags_coefs = coefs[head_tags_idx]
    important_factors_idx = head_tags_coefs.argsort()[::-1]
    for feature_str in get_importance_str(head_tags_coefs, feature_names[head_tags_idx], important_factors_idx):
        print(feature_str+"\n")
    #
    tags_idx = np.array([n for n, tag in enumerate(feature_names) if tag.startswith("tag")])
    tags_coefs = coefs[tags_idx]
    important_factors_idx = tags_coefs.argsort()[::-1]
    for feature_str in get_importance_str(tags_coefs, feature_names[tags_idx], important_factors_idx):
        print(feature_str+"\n")

# More details about the best model
#
y_pred = lin_svc.predict(X_test_vec)
print("Linear SVC report")
print(metrics.classification_report(y_test, y_pred))
# Partial match and full match
tagged_idx = y_test>0
y_test_tagged = y_test[tagged_idx]
y_pred = y_pred[tagged_idx]
mentions_gold = []
mentions_pred = []
mention_pred = [y_pred[0]]
mention_gold = [1]
# First, extract all the mentions results
for pred, gold in zip(y_pred, y_test_tagged):
    if gold == 1:
        mentions_gold.append(mention_gold)
        mentions_pred.append(mention_pred)
        mention_gold = [gold]
        mention_pred = [pred]
    else:
        mention_gold.append(gold)
        mention_pred.append(pred)

# Using only the mentions, extract the full and partial match results
full_match = 0
partial_match = 0
for mention_gold, mention_pred in zip(mentions_gold, mentions_pred):
    if mention_gold == mention_pred:
        full_match += 1
    if sum(mention_pred) > 0:
        partial_match +=1
print(f"Full match score: {full_match}/{len(mentions_gold)} - {full_match/len(mentions_gold):0.3f}")
print(f"Partial match score: {partial_match}/{len(mentions_gold)} - {partial_match/len(mentions_gold):0.3f}")
