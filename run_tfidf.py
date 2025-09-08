from common import filter_positive_scores_by_index, print_table
from search import (
    stem_sentence_exclude_stopwords,
    tf_idf,
)


def run_tfidf(docs: list[str]) -> None:
    """Compute and print the TF-IDF scores for unique words in a list of documents.

    The input documents are first processed by stemming and stopword exclusion. Then, the TF-IDF score
    is calculated for each unique word across the documents. Only positive scores are kept, and the results
    are printed in a formatted table.

    :param docs: A list of document strings to process.
    :return: None
    """
    docs = [stem_sentence_exclude_stopwords(doc) for doc in docs]
    unique_words = list(set([word for doc in docs for word in doc.split()]))
    tfidf_scores = {word: tf_idf(word=word, docs=docs) for word in unique_words}
    tfidf_table = filter_positive_scores_by_index(scores=tfidf_scores)
    print_table(tfidf_table)


if __name__ == "__main__":
    a = "The cat sat on the mat"
    b = "The dog played in the park"
    c = "Cats and dogs are great pets"
    docs = [a, b, c]
    print(f"Documents:\n{docs}\n")
    print("TF-IDF Scores")
    run_tfidf(docs=docs)
