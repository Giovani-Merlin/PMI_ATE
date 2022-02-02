# PMI_ATE

Automatic Term Extraction (ATE) algorithms, and analysis, for the Philipp Morris interview challenge

## Setup

```bash
pip install virtualenv
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Create a "data" directory, download the dataset (internal link) and unzip it in the data directory.
The dataset consists in laptops reviews in XML format, first we will convert this format to JSON format to make it easier to use.

The format is:

1. text: sentence review
2. term_pos: position of each identified term in the sentence, in chars unit.
3. polarity: encoded polarity of each identified term - 0: negative, 1: positive, 2: neutral.
BIO encoding will be made in the next step, as it depends on the tokenization form.

To do so, just run `python3 preprocessing.py`.

In a quickly check we can see that the dataset is in the right format. That is, first lines of the xml files are:

```xlm
<sentence id="2339">
    <text>I charge it at night and skip taking the cord with me because of the good battery life.</text>
    <aspectTerms>
        <aspectTerm term="cord" polarity="neutral" from="41" to="45"/>
        <aspectTerm term="battery life" polarity="positive" from="74" to="86"/>
    </aspectTerms>
</sentence>
<sentence id="812">
    <text>I bought a HP Pavilion DV4-1222nr laptop and have had so many problems with the computer.</text>
</sentence>
```

And first lines of the dataset files are:

```json
{
    "id": "2339",
    "text": "I charge it at night and skip taking the cord with me because of the good battery life.",
    "terms_position": [
        [41, 45],
        [74, 86]
    ],
    "polarity": [2, 1]
}, {
    "id": "812",
    "text": "I bought a HP Pavilion DV4-1222nr laptop and have had so many problems with the computer.",
    "terms_position": [],
    "polarity": []
}, {
```

## Algorithms

I have some experience in the field of entity recognition and entity linking, it is easy to observe that the ATE algorithms are very similar in this respect, but we have to pay attention to the following aspects:

* Term is a domain-specific and entity is a general concept. Thus, we can think of the term as a sub-domain of entity recognition.
* Term maps to a specific onthology, and entity recognition is a general process. Thus, we need to know (or create using the training set) our onthology in term extraction.
Entity recognition, being a broad process, is harder to do without deep-learning. ATM can be done using simpler rules.
For this rapport, I will use the following algorithms:

1. "Simple" machine-learning  approach, no training (no use of labels). Just for having a baseline.
2. Words embeddings (word2vec) with CRF (Conditional Random Field) layer. Still explainable as word2vec keeps the semantic meaning of the words and CRF just adds a layer of conditional probabilities.

### Tokenization & pre-processing

**For all algorithms I will normalize accents, strip text and remove multiple spaces, other steps are additional steps**

We need to be consistent with our data-preprocessing/tokenization and with our labels. For example, our dataset contains the terms "sales" and ""sales"" with quotes, therefore, if we use Spacy to tokenize the text we will have the tokens [",sales,"] for the term "sales", which could be mapped to (B-I-I) - so, without problem but maybe over tagging. Also, we have the term humans with the form "humans in the text, when it is tokenized by Spacy it results in a token ["humans], which is not recoverable from the terms/labels. One solution would be to remove all special characters, but this would remove some information (as "sales" is prejorative and just sales is a neutral term). Therefore, I will test both cases, but when I keep the special characters, I will put a space before and after each token to tokenize them correctly - always keeping track of the labels

### Simple ML-approach

Preprocessing by using only nouns, adjectives, adverbs and verbs tokens (naturally removes punctuation).
This process is done as follows:
Automatic Term Extraction algorithms for the Philipp Morris interview challenge 
