"""
Script to perform full pipeline from raw data to contradiction candidates.
"""

from functools import partial
from typing import Dict, List, Tuple

from haystack.schema import Document
import numpy as np
from pandas import DataFrame

from src import custom_preprocessors, loading, processing, scoring
from config import *


def load_documents() -> DataFrame:
    print("[▶] Loading data...")
    df = loading.load_dataset_from_json(DATASET_FILEPATH)
    print("[✓] Finished loading!")
    return df


def preprocess_documents(df: DataFrame) -> List[Document]:
    print("[▶] Processing data...")
    df['fulltext'] = df.text_by_page.apply(processing.clean_and_combine_pages)

    if SUBSET_SIZE:
        df = df.iloc[:SUBSET_SIZE]
    docs = processing.convert_frame_to_haystack(df)

    print(" | [+] Splitting into chunks")
    # A chunk is constructed using a sliding window. It will be N sentences long
    # if there are that many sentences remaining in the document. The next chunk
    # will include the last K sentences of the previous chunk if a previous
    # chunk exists. Chunks will not span documents.
    chunk_sentence_cleaning_func = partial(processing.clean_sentence_splits,
        toc_period_threshold = CHUNK_CLEANING_TOC_PERIOD_THRESHOLD,
        length_minimum = CHUNK_CLEANING_LENGTH_MINIMUM,
        length_maximum = CHUNK_CLEANING_LENGTH_MAXIMUM,
    )
    chunker = custom_preprocessors.SplitCleanerPreProcessor(
        language='en',
        split_by='sentence',
        split_cleaner=chunk_sentence_cleaning_func,
        split_length=CHUNK_LENGTH,
        split_overlap=CHUNK_OVERLAP,
        split_respect_sentence_boundary=False, # incompatible with 'passage' or 'sentence'
    )
    doc_chunks = chunker.process(docs)

    # NOTE: We might consider split into sentences before chunking. This would
    # make it easy to remove duplicate sentences (such as headers, footers,
    # disclaimers, legal text) prior to chunk embedding.
    doc_chunks = processing.remove_identical_chunks(doc_chunks)
    print("[✓] Finished processing!")
    return doc_chunks


def preselect_similar_chunks(doc_chunks: List[Document]) -> Tuple[Dict[str, Document], List[Tuple[str, str]]]:
    print("[▶] Pre-selecting similar chunks...")
    print(" | [+] Computing chunk embeddings")
    embeddings = processing.compute_chunk_embeddings(
        chunks=doc_chunks,
        model_name=EMBEDDING_MODEL_NAME,
        show_progress_bar=True,
    )
    # Enrich our chunks with embeddings
    for chunk, embedding in zip(doc_chunks, embeddings):
        chunk.embedding = embedding
    # TODO: Save intermediate chunks with their embeddings?

    print(" | [+] Selecting similar chunks")
    similarity_matrix = processing.compute_chunk_similarity(doc_chunks)
    # TODO: Save intermediate similarity scores between chunks? (only makes
    #       sense if we also save the chunks themselves)

    # Get rid of chunks that have similarity scores that are too high. This
    # value was fine-tuned by trial and error to remove similar chunks that were
    # just common header/disclaimer text.
    top_n_pair_indices = processing.get_top_n_similar_chunk_pair_indices(
        scores=similarity_matrix,
        n=CHUNK_SIMILARITY_TOP_N,
        max_similarity_threshold=MAX_CHUNK_SIMILARITY_THRESHOLD,
    )
    similar_chunk_id_pairs = [
        (doc_chunks[c1].id, doc_chunks[c2].id) for c1, c2 in top_n_pair_indices
    ]
    desired_indices = np.unique(np.array(top_n_pair_indices).flat)
    # NOTE: Storing our desired chunks to a new variable should allow us to
    # dispose of the full set of doc_chunks from memory, which we may need for
    # the contradiction scoring model! If we need this, additional code will be
    # required here.
    desired_chunks = {
        doc_chunks[i].id: doc_chunks[i]
        for i in desired_indices
    }

    print(" | [+] Splitting chunks into sentences")
    sentence_cleaning_func = partial(processing.clean_sentence_splits,
        length_minimum = SENTENCE_CLEANING_LENGTH_MINIMUM,
    )
    desired_chunk_sentences = processing.split_chunks_to_sentences(
        chunks=desired_chunks.values(),
        split_cleaner=sentence_cleaning_func,
    )
    # Enrich our desired chunks with sentences
    for chunk, sentences in zip(desired_chunks.values(), desired_chunk_sentences):
        chunk.sentences = sentences
    print("[✓] Finished pre-selection!")
    return desired_chunks, similar_chunk_id_pairs


def find_contradictions(desired_chunks, similar_chunk_id_pairs) -> DataFrame:
    print("[▶] Selecting contradiction candidates...")
    print(" | [+] Loading models")
    tokenizer, contradiction_model = scoring.load_contradiction_model()
    print(" | [+] Computing contradiction scores")
    contradiction_scores = scoring.compute_sentence_contradiction_scores(
        chunks=desired_chunks,
        chunk_id_pairs=similar_chunk_id_pairs,
        tokenizer=tokenizer,
        model=contradiction_model,
    )
    print(" | [+] Selecting candidates")
    candidates = scoring.get_top_k_contradictive_candidates(
        contradiction_scores=contradiction_scores,
        k=CANDIDATE_SELECTION_TOP_K,
    )
    candidate_info = scoring.retrieve_candidate_info(
        candidates=candidates,
        chunks=desired_chunks
    )
    print("[✓] Finished candidate selection!")
    return candidate_info


if __name__ == "__main__":

    df = load_documents()

    doc_chunks = preprocess_documents(df)

    desired_chunks, similar_chunk_id_pairs = preselect_similar_chunks(doc_chunks)

    candidates = find_contradictions(desired_chunks, similar_chunk_id_pairs)
    print(f".... Saving candidates to {SAVED_CANDIDATES_FILEPATH}")
    loading.save_candidates_csv(candidates, SAVED_CANDIDATES_FILEPATH)

    print(f"\n\n\n\nExample of candidates:")
    for idx, candidate in candidates.iloc[:10].iterrows():
        print(f"({idx})")
        scoring.pretty_print_candidate(candidate)
        print('\n\n')
