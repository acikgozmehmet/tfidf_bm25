from common import filter_positive_scores_by_index, print_table
from search import (
    bm25,
    stem_sentence_exclude_stopwords,
)


def run_bm25(docs: list[str]) -> None:
    """Compute and display BM25 scores for terms across a collection of documents.

    This function preprocesses the input documents by stemming and removing stopwords,
    then calculates BM25 scores for each unique word. Only terms with positive BM25
    scores are retained, and the results are displayed in a formatted table.

    :param docs: A list of raw text documents to analyze.
    :return: This function does not return anything, it prints the BM25 score table.
    """
    docs = [stem_sentence_exclude_stopwords(doc) for doc in docs]
    unique_words = list(set([word for doc in docs for word in doc.split()]))
    bm25_scores = {word: bm25(word=word, docs=docs) for word in unique_words}
    bm25_table = filter_positive_scores_by_index(scores=bm25_scores)
    print_table(bm25_table)


if __name__ == "__main__":
    a = "The cat sat on the mat"
    b = "The dog played in the park"
    c = "Cats and dogs are great pets"
    docs = [a, b, c]
    print(f"Documents:\n{docs}\n")
    print("BM25 Scores")
    run_bm25(docs=docs)
