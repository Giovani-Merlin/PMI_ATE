from time import time
from collections import Counter
import sklearn_crfsuite
from functools import wraps
from sklearn import metrics

from sklearn_crfsuite.utils import flatten
from utils import get_dataset
from preprocessing import extract_features, parse_dependencies


# Crfsuite is only compatible with sklearn < 0.24 - https://github.com/TeamHG-Memex/sklearn-crfsuite/issues/60
# So, small hacks to make it work with sklearn >= 0.24


def _flattens_y(func):
    @wraps(func)
    def wrapper(y_true, y_pred, *args, **kwargs):
        y_true_flat = flatten(y_true)
        y_pred_flat = flatten(y_pred)
        return func(y_true_flat, y_pred_flat, *args, **kwargs)

    return wrapper


@_flattens_y
def flat_classification_report(y_true, y_pred, labels=None, **kwargs):
    """
    Return classification report for sequence items.
    """

    return metrics.classification_report(y_true, y_pred, labels=labels, **kwargs)


@_flattens_y
def flat_f1_score(y_true, y_pred, **kwargs):
    """
    Return F1 score for sequence items.
    """
    return metrics.f1_score(y_true, y_pred, **kwargs)


###############################################################################
def features2crf(features, i):
    """
    Extract features from dataset to crf format, including previous and next words (window of size 3).
    """
    tokens = features["tokens"]

    def _get_features(features, i):
        tag = features["tags"][i]
        head = features["head"][i]
        head_tag = features["head_tag"][i]
        token_pos = features["tokens_id"][i]
        token_dependency = features["dependency_map"][i]
        token = tokens[i]
        return tag, head, head_tag, token_pos, token_dependency, token

    tag, head, head_tag, token_pos, token_dependency, token = _get_features(features, i)

    features_crf = {
        "bias": 1.0,
        "tag": tag,
        "head": head,
        "head_tag": head_tag,
        "token_pos": token_pos,
        "token_dependency": token_dependency,
        "token": token,

    }
    if i > 0:
        tag, head, head_tag, token_pos, token_dependency, token = _get_features(features, i - 1)

        features_crf.update(
            {
                "-1:tag": tag,
                "-1:head": head,
                "-1:head_tag": head_tag,
                "-1:token_pos": token_pos,
                "-1:token_dependency": token_dependency,
                "-1:token": token,

            }
        )
    else:
        # Beginning of sequence.
        features_crf["BOS"] = True

    if i < len(tokens) - 1:
        tag, head, head_tag, token_pos, token_dependency, token = _get_features(features, i + 1)

        features_crf.update(
            {
                "+1:tag": tag,
                "+1:head": head,
                "+1:head_tag": head_tag,
                "+1:token_pos": token_pos,
                "+1:token_dependency": token_dependency,
                "+1:token": token,

            }
        )
    else:
        features_crf["EOS"] = True

    return features_crf


### Train
raw_train = get_dataset("train")
train = extract_features(raw_train, lowercase=False)
parsed_train_dataset = parse_dependencies(train, use_lemmas=True)
X_train = [
    [features2crf(entry["features"], i) for i in range(len(entry["features"]["tokens"]))]
    for entry in parsed_train_dataset
]
y_train = [entry["labels"]["bio_map"] for entry in train]
### Test
raw_test = get_dataset("test")
test = extract_features(raw_test, lowercase=False)
parsed_test_dataset = parse_dependencies(test, use_lemmas=True)
X_test = [
    [features2crf(entry["features"], i) for i in range(len(entry["features"]["tokens"]))]
    for entry in parsed_test_dataset
]
y_test = [entry["labels"]["bio_map"] for entry in test]

###
crf = sklearn_crfsuite.CRF(
    algorithm="lbfgs",
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True,
)
print("Training started...")
t0 = time()
crf.fit(X_train, y_train)  ### Error message when try to train
print(f"Training completed in {(time() - t0) / 1000}")
###
y_pred = crf.predict(X_test)
labels = list(crf.classes_)
labels.remove("O")  # remove 'O' label from evaluation
print(flat_classification_report(y_test, y_pred, labels=labels, digits=3))


def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))


print("Top likely transitions:")
print_transitions(Counter(crf.transition_features_).most_common(20))



def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))


print("Top positive:")
print_state_features(Counter(crf.state_features_).most_common(30))

print("\nTop negative:")
print_state_features(Counter(crf.state_features_).most_common()[-30:])
