import json
import spacy
import regex as re
import numpy as np
import unicodedata
import xml.etree.cElementTree as ET
from spacy import displacy

POLARITY_MAP = {"negative": 0, "positive": 1, "neutral": 2, "conflict": 3}
BIO_MAP = {"O": 0, "B": 1, "I": 2}
INVERSE_BIO_MAP = {v: k for k, v in BIO_MAP.items()}


def parse_laptops_xml(xml_path):
    """
    Parses xml laptops dataset to JSON format.
    """
    dataset = []
    root = ET.parse(xml_path).getroot()

    # Each "sentence" object is one entry in the dataset
    for sentence in root.findall("sentence"):
        text = sentence.find("text").text
        id_ = sentence.attrib["id"]
        # Labels branch
        aspectTerms = sentence.find("aspectTerms")
        # If we have labels to this sentence
        polarities = []
        terms_position = []
        if aspectTerms is not None:
            for aspectTerm in aspectTerms.findall("aspectTerm"):

                start_char = int(aspectTerm.get("from"))
                end_char = int(aspectTerm.get("to"))
                terms_position.append((start_char, end_char))
                # Define polarity of each term token
                polarities.append(POLARITY_MAP[aspectTerm.get("polarity")])
                # Assert correct term extraction
                assert aspectTerm.get("term") == text[start_char:end_char]
        if terms_position:
            # Sort terms_position and polarities to make easier to extract BIO tagging afterwards.
            # Just sort it by start_char
            polarities = np.array(polarities)
            terms_position = np.array(terms_position)
            sorted_terms = np.apply_along_axis(lambda x: x[0], axis=1, arr=terms_position).argsort()
            terms_position = terms_position[sorted_terms].tolist()
            polarities = polarities[sorted_terms].tolist()
        dataset.append(
            {
                "id": id_,
                "text": text,
                "terms_position": terms_position,
                "polarity": polarities,
            }
        )

    return dataset


def preprocess_with_terms_position(text, terms_position, punctuation=False, lowercase=False, keep_quote=True):
    """
    Preprocesses text to normalize it for tokenization while keeping labels positions.
    """
    if terms_position:
        final_text = ""
        final_positions = []
        last_end = 0
        for start, end in terms_position:
            # Preprocess text
            processed_text = preprocess_data(
                text[last_end:start], punctuation=punctuation, lowercase=lowercase, keep_quote=keep_quote
            )
            # Preprocess labels
            processed_term = preprocess_data(
                text[start:end], punctuation=punctuation, lowercase=lowercase, keep_quote=keep_quote
            )
            # Keeps track of already processed text
            last_end = end
            final_text += processed_text + " "
            # Saves new labels positions.
            final_positions.append((len(final_text), len(final_text) + len(processed_term)))
            final_text += processed_term + " "
            # Be sure to have label at the correct position
            assert processed_term == final_text[final_positions[-1][0] : final_positions[-1][1]]
        # Preprocess text after last label
        processed_text = preprocess_data(
            text[last_end:], punctuation=punctuation, lowercase=lowercase, keep_quote=keep_quote
        )

        final_text += processed_text
        return final_text, final_positions
    else:
        return preprocess_data(text, punctuation=punctuation, lowercase=lowercase, keep_quote=keep_quote), []


def preprocess_data(text, punctuation=False, lowercase=False, keep_quote=True):
    """
    Preprocesses text to normalize it for tokenization.
    """
    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)
    # Remove leading and trailing spaces
    text = text.strip()
    if lowercase:
        text = text.lower()
    if punctuation and keep_quote:
        text = re.sub(r"[^\w\s\']", "", text)
    elif punctuation and not keep_quote:
        text = re.sub(r"[^\w\s]", "", text)
    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)
    # Split text in spaces
    tokens = text.split(" ")
    # Split in beginning/ending punctuation
    if not punctuation:
        expanded_tokens = []
        for token in tokens:
            # Regex to get multiple beggining or ending punctuation. As I use the "|" operator, I need to filter empty matches.
            # Also, no problem in groupping special characters lie ")." as spacy tokenizer handles them.
            expanded_tokens.extend(
                list(
                    filter(
                        lambda x: x != "",
                        re.split(r"([^\w\']+$|^[^\w\']+)", token),
                    )
                )
            )
        text = " ".join(expanded_tokens)
    return text


def label_tokens(tokens, terms_position, polarity):
    """
    Labels tokens with BIO tags and extend polarity labels to the tokens.
    """
    bio_map = []
    polarities = []
    inside = False
    init_chars = [term_position[0] for term_position in terms_position]
    end_chars = [term_position[1] for term_position in terms_position]
    for token in tokens:
        if token.idx in init_chars and not inside:
            # Starts token labelling
            bio_map.append("B")
            polarities.append(polarity.pop())
            init_chars.pop(0)
            # If it's an unique token, just label it
            if token.idx + len(token) in end_chars:
                end_chars.pop(0)
            else:
                inside = True
        elif token.idx + len(token) in end_chars and inside:
            # Finishes inside token labelling
            bio_map.append("I")
            polarities.append(polarities[-1])
            end_chars.pop(0)
            inside = False
        elif inside:
            # Keeps inside = true and label token as inside
            bio_map.append("I")
            polarities.append(polarities[-1])
        else:
            # Normal case
            bio_map.append("O")
            polarities.append(None)
    # Verifies that we extracted all labels
    assert len(init_chars) == 0 and len(end_chars) == 0
    return {"bio_map": bio_map, "polarities": polarities}


def tokenize_nouns(text, spacy_nlp):
    """
    Recover only nouns from text.
    """
    doc = spacy_nlp(text)
    nouns_doc = []
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            nouns_doc.append(token.text)
    return nouns_doc


def extract_features(dataset, punctuation=False, lowercase=True, keep_quote=True):
    """
    Preprocess and extract features from the dataset. Features are tokenized text, lemmatized, POS tags and dependency tags and map.
    """
    nlp = spacy.load("en_core_web_sm")

    processed_dataset = []
    for entry in dataset:
        # Preprocess text
        text, labels = preprocess_with_terms_position(
            entry["text"], entry["terms_position"], punctuation=punctuation, lowercase=lowercase, keep_quote=keep_quote
        )

        processed_text = nlp(text)
        # displacy.serve(processed_text, style="dep")
        tokens = list(processed_text)
        entry["labels"] = label_tokens(tokens, labels, entry["polarity"])
        entry.pop("polarity")
        # Preprocess terms position
        tokenized_text = []
        dependency_map = []
        dependency_ref = []
        tags_map = []
        tokens_id = []
        lemmas = []
        for n, token in enumerate(tokens):
            tokenized_text.append(token.text)
            dependency_map.append(token.dep_)
            dependency_ref.append(token.head.i)
            lemmas.append(token.lemma_)
            tags_map.append(token.pos_)
            # For mapping tokens to labels and dependency if further processing is needed
            tokens_id.append(n)
        features = {}
        features["tokens"] = tokenized_text
        features["dependency_map"] = dependency_map
        features["tags"] = tags_map
        features["dependency_ref"] = dependency_ref
        features["tokens_id"] = tokens_id
        features["lemmas"] = lemmas
        entry["features"] = features
        entry["text"] = text
        entry["terms_position"] = labels
        processed_dataset.append(entry)

    return processed_dataset


def get_lemma_labels(dataset):
    """
    Extract lemma labels from the dataset using BIO tags.
    """

    lemmas = []
    for entry in dataset:
        entry_lemmas = []
        composite_label = []
        last_value = 10

        for value, lemma in zip(entry["labels"]["bio_map"] + [0], entry["features"]["lemmas"] + ["dumb"]):
            # Finished joining label
            if last_value == "I" and value != "I":
                entry_lemmas.append(" ".join(composite_label))
            # Single token label
            elif last_value == "B" and value != "I":
                entry_lemmas.append(composite_label[0])
            if value == "I":
                composite_label = [lemma]
            elif value == "B":
                composite_label.append(lemma)

            last_value = value
        lemmas.append(entry_lemmas)
    return lemmas

def parse_dependencies(dataset, use_lemmas=False):
    """
    Recovers Head word, head word TAG and dependencies.
    """
    new_dataset = []
    for entry in dataset:
        if use_lemmas:
            to_use = "lemmas"
        else:
            to_use = "tokens"
        mapped_head = [entry["features"][to_use][i] for i in entry["features"]["dependency_ref"]]
        mapped_head_tag = [entry["features"]["tags"][i] for i in entry["features"]["dependency_ref"]]
        # In the case we're using lemmas, our tokens will be the lemmas...
        entry["features"]["tokens"] = entry["features"][to_use]
        entry["features"].pop("dependency_ref"), entry["features"].pop("lemmas")
        entry["dependency_map"] = [
            dependency
            for dependency in entry["features"]["dependency_map"]
            if dependency in ["amod", "dep", "nsubj", "dobj"]
        ]
        entry["features"]["head"] = mapped_head
        entry["features"]["head_tag"] = mapped_head_tag
        new_dataset.append(entry)
    return new_dataset


if __name__ == "__main__":
    train_path = "data/Laptops_Train_v2.xml"
    with open("data/laptops_train.json", "w") as f:
        json.dump(parse_laptops_xml(train_path), f)
    test_path = "data/Laptops_Test_Gold.xml"
    with open("data/laptops_test.json", "w") as f:
        json.dump(parse_laptops_xml(test_path), f)
    # for entry in parse_laptops_xml(train_path):
    #     print(entry["text"])
    #     print(preprocess_with_terms_position(entry["text"], entry["terms_position"])[0] + "\n")
