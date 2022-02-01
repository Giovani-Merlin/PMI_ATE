import spacy
from utils import *
from preprocessing import preprocess_with_terms_position, label_tokens


def extract_features(dataset):
    nlp = spacy.load("en_core_web_sm")

    processed_dataset = []
    for entry in dataset:
        # Preprocess text
        text, labels = preprocess_with_terms_position(
            entry["text"], entry["terms_position"], punctuation=False, lowercase=False
        )
        processed_text = nlp(text)
        tokens = list(processed_text)
        labels = label_tokens(tokens, labels, entry["polarity"])
        # Preprocess terms position
        tokenized_text = []
        dependency_map = []
        dependency_ref = []
        tags_map = []
        lemmas = []
        chunks = []
        for chunk_ in processed_text.noun_chunks:
            chunk = chunk_.text
            chunks.append(chunk)
        for token in tokens:
            tokenized_text.append(token.text)
            dependency_map.append(token.dep_)
            dependency_ref.append(token.head.idx)
            lemmas.append(token.lemma_)
            tags_map.append(token.pos_)

        entry["tokenized_text"] = tokenized_text
        entry["dependency_map"] = dependency_map
        entry["tags_map"] = tags_map
        entry["dependency_ref"] = dependency_ref
        entry["chunks"] = chunks
        processed_dataset.append(entry)


raw_train = get_dataset("train")
train = extract_features(raw_train)
test = get_dataset("test")
