import math
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download stopwords if not already downloaded
# nltk.download("stopwords")
# nltk.download("punkt")


def stem_sentence(sentence: str) -> str:
    """Stem the input sentence by reducing each word to its root form using the Porter Stemmer.

    :param sentence: The input sentence to be stemmed.
    :return: A string representing the stemmed version of the input sentence,
             consisting of stemmed tokens joined by spaces.
    """
    stemmer = PorterStemmer()
    tokens = word_tokenize(sentence.lower())
    stemmed_tokens = [stemmer.stem(token) for token in tokens if token.isalpha()]
    stemmed_sentence = " ".join(stemmed_tokens)
    return stemmed_sentence


def stem_sentence_exclude_stopwords(sentence: str) -> str:
    """Stem the input sentence by removing English stopwords and reducing each remaining word to its root form using the Porter Stemmer.

    :param sentence: The input sentence to be processed.
    :return: A string of stemmed tokens excluding stopwords, joined by spaces.
    """
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(sentence.lower())
    stemmed_tokens = [stemmer.stem(token) for token in tokens if token.isalpha() and token not in stop_words]
    stemmed_sentence = " ".join(stemmed_tokens)
    return stemmed_sentence


def tokenize(doc: str) -> list[str]:
    """Convert input text to lowercase and extracts tokens consisting of word characters.

    :param doc: The document or string to tokenize.
    :return: A list of tokens consisting of sequences of word characters (letters, digits, underscore).
    """
    return re.findall(r"\b\w+\b", doc.lower())


def term_frequency(word: str, doc: str) -> float:
    """Calculate the term frequency of a word in a given document.

    Term frequency (TF) is the ratio of the number of times a word appears in the document
    to the total number of words in the document.

    :param word: The word whose frequency is being calculated.
    :param doc: The document in which to calculate the term frequency.
    :return: The term frequency as a float value between 0 and 1.
    """
    tokenized_doc = tokenize(doc=doc)
    if not tokenized_doc:
        return 0.0
    return tokenized_doc.count(word) / len(tokenized_doc)


def inverse_document_frequency(word: str, docs: list[str]) -> float:
    """Calculate the inverse document frequency (IDF) of a word in a collection of documents.

    IDF measures how unique or rare a word is across all documents. It is computed as
    the logarithm of the ratio of the total number of documents to the number of documents
    containing the word.

    :param word: The word for which to calculate the IDF.
    :param docs: A list of documents (strings) constituting the corpus.
    :return: The IDF score as a float.
    """
    N = len(docs)

    # Count how many documents contain the word as a token
    doc_count = sum(1 for doc in docs if word in tokenize(doc))
    if doc_count == 0:
        return 0.0

    return math.log10(N / doc_count)


def tf_idf(word: str, docs: list[str]) -> list[float]:
    """Calculate the TF-IDF score for a given word across multiple documents.

    TF-IDF is the product of term frequency (TF) and inverse document frequency (IDF),
    measuring the importance of a word in a document relative to a collection of documents.

    :param word: The word for which to calculate TF-IDF scores.
    :param docs: A list of documents (strings) in which to calculate the scores.
    :return: A list of TF-IDF scores for the word, one score per document.
    """
    idf = inverse_document_frequency(word, docs)  # compute IDF once
    result = []
    for doc in docs:
        tf = term_frequency(word=word, doc=doc)
        tfidf_score = round(tf * idf, 4)
        result.append(tfidf_score)
    return result


def bm25(word: str, docs: list[str], k: float = 1.2, b: float = 0.75) -> list[float]:
    """Compute BM25 scores for a given word across a list of documents.

    BM25 is a ranking function used to estimate the relevance of documents based on the frequency of a query term
    and document length normalization. The parameters k and b control term frequency saturation and length normalization.

    :param word: The target word to score.
    :param docs: The document corpus as a list of strings.
    :param k: Term frequency saturation parameter (default 1.2).
    :param b: Document length normalization parameter (default 0.75).
    :return: List of BM25 scores for the word in each document.
    """
    N = len(docs)

    tokenized_docs = [tokenize(doc) for doc in docs]
    avgdl = sum(len(doc_tokens) for doc_tokens in tokenized_docs) / N

    # Document frequency: number of docs containing the word
    N_q = sum(1 for doc_tokens in tokenized_docs if word in doc_tokens)

    result = []
    for doc_tokens in tokenized_docs:
        freq = doc_tokens.count(word)
        denom = freq + k * (1 - b + b * len(doc_tokens) / avgdl)
        tf = (freq * (k + 1)) / denom if denom != 0 else 0

        idf = math.log(((N - N_q + 0.5) / (N_q + 0.5)) + 1)
        score = round(tf * idf, 4)
        result.append(score)
    return result
