"""
Methods for cleaning and preprocessing documents, including computation of
similarity between chunks and saving of intermediate results.
"""

from typing import (Callable, Dict, Generator, List, Literal, Optional, Set,
                    Tuple, Union)

import numpy as np
import pandas as pd
from haystack.schema import Document
from pandas import DataFrame
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.lang.en import English


# --------------------------------- CLEANING --------------------------------- #
def clean_pagetext(pagetext: str) -> str:
    """
    Takes text for a single page of a document and returns cleaned text.
    """
    # NOTE: This is currently not implemented.
    return pagetext


def clean_fulltext(fulltext: str) -> str:
    """
    Takes text for a full document and returns cleaned text.
    """
    # NOTE: This is currently not implemented.
    return fulltext


def clean_and_combine_pages(text_by_page: List[str]) -> str:
    """
    Takes a document defined as a list of strings for each page and returns a
    single cleaned string for the full document.
    """

    # Perform cleaning on pages before they get concatenated
    cleaned_pages = (clean_pagetext(pagetext) for pagetext in text_by_page)

    # Concatenate
    fulltext = ' '.join(cleaned_pages)

    # Perform cleaning on full text after concatenation
    cleaned = clean_fulltext(fulltext)

    return cleaned


def convert_frame_to_haystack(
        df: DataFrame,
        content_col: str = 'fulltext',
        meta_cols: List[str] = ('corpus', 'title', 'file_name', 'url', 'id'),
) -> List[Document]:
    """
    Convert a dataframe of text documents and metadata into Haystack Document
    objects. The output can be passed to a Haystack PreProcessor node.
    """

    content = df[content_col].to_list()
    meta = df[list(meta_cols)].to_dict(orient='records')

    docs = []
    for i in range(df.shape[0]):
        new_doc = Document(content=content[i], meta=meta[i])
        docs.append(new_doc)

    return docs


def clean_sentence_splits(
        sentences: List[str],
        toc_period_threshold: int = 5,
        length_minimum: int = 0,
        length_maximum: int = 1000,
) -> List[str]:
    """
    Take a list of sentences and remove ones which are potentially part of a
    table of contents or are suspiciously long (may be part of a data table).
    """
    cleaned_sentences = [
        s for s in sentences if (
            # Remove table of contents
            ('.' * toc_period_threshold not in s)
            # Remove too short sentences
            and
            (len(s) > length_minimum)
            # Remove overly long sentences
            and
            (len(s) < length_maximum)
        )
    ]
    return cleaned_sentences


# -------------------------------- SIMILARITY -------------------------------- #
def compute_chunk_embeddings(
        chunks: List[Document],
        model_name: str,
        **kwargs,
) -> List[List[float]]:
    """
    Compute vector embeddings for each document in a list of Haystack Documents,
    using the sentence-transformers model specified by `model_name`.
    """
    model = SentenceTransformer(model_name)
    # NOTE: May run into issues with performing this computation all-at-once
    # when working with much larger datasets. Consider batching.
    embeddings = model.encode([chunk.content for chunk in chunks], **kwargs)
    return embeddings


def compute_chunk_similarity(
        chunks: List[Document],
) -> np.ndarray:
    """
    Compute upper-triangle cosine similarity scores between chunk embeddings.
    """
    sim = cosine_similarity([chunk.embedding for chunk in chunks])
    scores = np.triu(sim, 1)  # Remove lower triangle and diagonal
    return scores


def get_top_n_similar_chunk_pair_indices(
        scores: np.ndarray,
        n: int,
) -> List[List[int]]:
    """
    Returns indices for the highest k values in a similarity matrix between
    chunk pairs.
    """
    # Partitioning guarantees that the first k values are the smallest k values
    # in the array (or, in our case that the *last* k values are the *largest*).
    # Using argpartition we get the indices, then we just take the last k.
    top_k_indices = np.argpartition(scores, -n, axis=None)[-n:]
    # Since the resulting indices are flattened, we need to put it back into
    # row,col form.
    row_indices, col_indices = np.unravel_index(top_k_indices, scores.shape)
    top_k_pair_indices = list(zip(row_indices, col_indices))
    return top_k_pair_indices


def split_chunks_to_sentences(
        chunks: List[Document],
        split_cleaner: Optional[Callable[[List[str]], List[str]]] = None,
) -> List[List[str]]:
    # Potential improvement: replace with SpaCy 'senter' from en_web_core_sm
    nlp = English()
    nlp.add_pipe('sentencizer')
    nlp.select_pipes(enable=['sentencizer'])
    sentences_docs = list(nlp.pipe([chunk.content for chunk in chunks]))
    sentences = [[s.text for s in sd.sents] for sd in sentences_docs]
    if split_cleaner is not None:
        sentences = [split_cleaner(ss) for ss in sentences]
    return sentences
