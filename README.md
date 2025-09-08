# TF-IDF vs BM25: Understanding and Implementing Text Ranking Algorithms in Python


**Author:** Mehmet Acikgoz

## Description
This project implements and compares TF-IDF and BM25, which are two popular text ranking algorithms. The goal is to understand their differences and provide a working implementation in Python. The project includes functionalities to compute and print the term frequency-inverse document frequency (TF-IDF) and BM25 scores for documents, helping users analyze the relevance of terms within textual data.

For a detailed step-by-step guide, see the accompanying Medium article on [Medium](https://medium.com/@macikgozm/creating-and-deploying-a-databricks-app-with-asset-bundles-f9395eb46f91)

## Installation
To install the necessary dependencies for this project, run:

```bash
pip install nltk==3.9.1 rich==14.1.0
```

Ensure that you have Python 3.12 or higher installed.

## Usage
The main scripts, `run_tfidf.py` and `run_bm25.py`, can be executed to compute the respective scores for documents. Each script takes a list of text documents as input. For example:

```python
if __name__ == "__main__":
    a = "The cat sat on the mat"
    b = "The dog played in the park"
    c = "Cats and dogs are great pets"
    docs = [a, b, c]
    print(f"Documents:\n{docs}\n")
    print("BM25 Scores")
    run_bm25(docs=docs)
```  

You can replace the document strings with your own text data.

## Configuration
No specific configuration is required, but make sure to have the `nltk` stopwords downloaded if you are running for the first time. You can do this by uncommenting the following lines in the `search.py` file:

```python
# nltk.download("stopwords")
# nltk.download("punkt")
```

## Authors/Credits
The project is developed by the contributors who are listed in the `pyproject.toml` file.

## Project Status
Currently, this project is stable and usable. Future updates may introduce additional features or optimizations.