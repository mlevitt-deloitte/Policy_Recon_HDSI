"""
Script to perform full pipeline from raw data to contradiction candidates.
"""

from functools import partial

import numpy as np

from . import custom_preprocessors, data, processing, scoring


# TODO: Provide pipeline methods. Maybe a quick argparse or sys.argv for run
#       targets.


if __name__ == "__main__":

    # -------------------------------- LOADING ------------------------------- #
    df = data.load_dataset_from_json("../data/02. Data Sets/DoD Issuances/contradictions_datasets_dod_issuances.zip")

    # ------------------------------ PREPROCESSING --------------------------- #
    df['fulltext'] = df.text_by_page.apply(processing.clean_and_combine_pages)

    SUBSET_SIZE = 100
    if SUBSET_SIZE:
        df = df.iloc[:SUBSET_SIZE]
    docs = processing.convert_frame_to_haystack(df)

    # A chunk is constructed using a sliding window. It will be N sentences long
    # if there are that many sentences remaining in the document. The next chunk
    # will include the last K sentences of the previous chunk if a previous
    # chunk exists. Chunks will not span documents.
    CHUNK_SENTENCE_CLEANING_FUNC1 = partial(processing.clean_sentence_splits,
        toc_period_threshold = 5,
        length_minimum = 0,
        length_maximum = 1000,
    )
    CHUNK_LENGTH = 8
    CHUNK_OVERLAP = 4
    chunker = custom_preprocessors.SplitCleanerPreProcessor(
        split_by="sentence",
        split_cleaner=CHUNK_SENTENCE_CLEANING_FUNC1,
        split_length=CHUNK_LENGTH,
        split_overlap=CHUNK_OVERLAP,
        split_respect_sentence_boundary=False, # incompatible with 'passage' or 'sentence'
    )

    doc_chunks = chunker.process(docs)

    # -------------------- CHUNK SIMILARITY (PRESELECTION) ------------------- #
    MODEL_NAME = 'all-MiniLM-L6-v2'
    embeddings = processing.get_chunk_embeddings(
        doc_chunks,
        model_name=MODEL_NAME,
        show_progress_bar=True,
    )
    for chunk, embedding in zip(doc_chunks, embeddings):
        chunk.embedding = embedding
    # TODO: Save intermediate chunks with their embeddings?

    similarity_matrix = processing.get_chunk_similarity_scores(doc_chunks)
    chunk_ids = [chunk.id for chunk in doc_chunks]
    # TODO: Save intermediate similarity scores between chunks? (only makes
    #       sense if we also save the chunks themselves)

    TOP_K = 10
    top_k_pair_indices = processing.get_top_k_similar_chunk_pair_indices(
        similarity_matrix,
        k=TOP_K,
    )
    similar_chunk_id_pairs = [
        (doc_chunks[c1].id, doc_chunks[c2].id) for c1, c2 in top_k_pair_indices
    ]
    desired_indices = np.unique(np.array(top_k_pair_indices).flat)
    # NOTE: Storing our desired chunks to a new variable should allow us to
    # dispose of the full set of doc_chunks from memory, which we may need for
    # the contradiction scoring model! If we need this, additional code will be
    # required here.
    desired_chunks = {
        doc_chunks[i].id: doc_chunks[i]
        for i in desired_indices
    }

    CHUNK_SENTENCE_CLEANING_FUNC2 = partial(processing.clean_sentence_splits,
        toc_period_threshold = 5,
        length_minimum = 20,
        length_maximum = 1000,  # FIXME: Should support None as default arg
    )
    desired_chunk_sentences = processing.split_chunks_to_sentences(
        desired_chunks.values(),
        split_cleaner=CHUNK_SENTENCE_CLEANING_FUNC2,
    )
    for chunk, sentences in zip(desired_chunks.values(), desired_chunk_sentences):
        chunk.sentences = sentences

    # ------------------------- CONTRADICTION SCORING ------------------------ #
    # TODO:
    # - Create sentence combinations
    # - Run pretrained contradiction model on sentence combinations
    # - Select top k most contradictive sentence combinations as "candidates"
    # - Print document details for candidates -- e.g. title and url -- so that
    #   humans can verify
