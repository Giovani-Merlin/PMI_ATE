# PMI_ATE

Automatic Term Extraction (ATE) algorithms, and analysis, for the Philipp Morris interview challenge

## Setup

```bash
pip install virtualenv
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Dataset

Create a "data" directory, download the dataset (internal link) and unzip it in the data directory.
The dataset consists of laptop reviews in XML format, first, we will convert this format to JSON format to make it easier to use.

The format is:

1. text: sentence review
2. term_pos: position of each identified term in the sentence, in chars unit.
3. polarity: encoded polarity of each identified term - 0: negative, 1: positive, 2: neutral.
BIO encoding will be made in the next step, as it depends on the tokenization form.

To do so, just run `python3 preprocessing.py`.

In a quick check, we can see that the dataset is in the right format. That is, the first lines of the XML files are:

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
}
```

## Report

All project/algorithms explanation in the report_pmi.pdf.
