import xml.etree.cElementTree as ET
import numpy as np
import unicodedata
import json
import regex as re

POLARITY_MAP = {"negative": 0, "positive": 1, "neutral": 2, "conflict": 3}
BIO_MAP = {"O": 0, "B": 1, "I": 2}


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


def preprocess_with_terms_position(text, terms_position, punctuation=False, lowercase=False):
    """
    Preprocesses text to normalize it for tokenization while keeping labels positions.
    """
    if terms_position:
        final_text = ""
        final_positions = []
        last_end = 0
        for start, end in terms_position:
            # Preprocess text
            processed_text = preprocess_data(text[last_end:start], punctuation=punctuation, lowercase=lowercase)
            # Preprocess labels
            processed_term = preprocess_data(text[start:end], punctuation=punctuation, lowercase=lowercase)
            # Keeps track of already processed text
            last_end = end
            final_text += processed_text + " "
            # Saves new labels positions.
            final_positions.append((len(final_text), len(final_text) + len(processed_term)))
            final_text += processed_term + " "
            # Be sure to have label at the correct position
            assert processed_term == final_text[final_positions[-1][0] : final_positions[-1][1]]
        # Preprocess text after last label
        processed_text = preprocess_data(text[last_end:], punctuation=punctuation, lowercase=lowercase)

        final_text += processed_text
        return final_text, final_positions
    else:
        return preprocess_data(text, punctuation=punctuation, lowercase=lowercase), []


def preprocess_data(text, punctuation=False, lowercase=False):
    """
    Preprocesses text to normalize it for tokenization.
    """
    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)
    # Remove leading and trailing spaces
    text = text.strip()
    if lowercase:
        text = text.lower()
    if punctuation:
        text = re.sub(r"[^\w\s\']", "", text)
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
            bio_map.append(BIO_MAP["B"])
            polarities.append(polarity.pop())
            init_chars.pop(0)
            # If it's an unique token, just label it
            if token.idx + len(token) in end_chars:
                end_chars.pop(0)
            else:
                inside = True
        elif token.idx + len(token) in end_chars and inside:
            # Finishes inside token labelling
            bio_map.append(BIO_MAP["I"])
            polarities.append(polarities[-1])
            end_chars.pop(0)
            inside = False
        elif inside:
            # Keeps inside = true and label token as inside
            bio_map.append(BIO_MAP["I"])
            polarities.append(polarities[-1])
        else:
            # Normal case
            bio_map.append(BIO_MAP["O"])
            polarities.append(None)
    # Verifies that we extracted all labels
    assert len(init_chars) == 0 and len(end_chars) == 0
    return bio_map, polarities


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
