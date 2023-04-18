"""
Script to perform full pipeline from raw data to contradiction candidates.
"""

from functools import partial

import numpy as np

from . import custom_preprocessors, data, processing, scoring


# TODO: Provide pipeline methods. Maybe a quick argparse or sys.argv for run
# targets and pull from a config file or the command line. For now, at least a
# wrapper function that runs the full pipeline with support for each of the
# hyperparameters.

# TODO: I just realized, if we split into sentences before we perform our
# sliding window, then we could easily remove sentences that are exactly
# duplicated in multiple documents (such as headers, footers, disclaimers, legal
# text, etc) by using the same easy approach as below based on counting the hash
# id of the resulting sentence docs. This may drastically speed up our runtime
# by eliminating the need to perform embeddings on chunks that we'd like to
# discard anyway.
#
# We may need to take care since we'll be dealing with hundreds of thousands of
# sentences. But I think the performance gain (to both runtime and utility)
# should be worth it!


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
    CHUNK_CLEANING_TOC_PERIOD_THRESHOLD = 5
    CHUNK_CLEANING_LENGTH_MINIMUM = 0
    CHUNK_CLEANING_LENGTH_MAXIMUM = 1000
    chunk_sentence_cleaning_func = partial(processing.clean_sentence_splits,
        toc_period_threshold = CHUNK_CLEANING_TOC_PERIOD_THRESHOLD,
        length_minimum = CHUNK_CLEANING_LENGTH_MINIMUM,
        length_maximum = CHUNK_CLEANING_LENGTH_MAXIMUM,
    )
    CHUNK_LENGTH = 8
    CHUNK_OVERLAP = 4
    chunker = custom_preprocessors.SplitCleanerPreProcessor(
        split_by="sentence",
        split_cleaner=chunk_sentence_cleaning_func,
        split_length=CHUNK_LENGTH,
        split_overlap=CHUNK_OVERLAP,
        split_respect_sentence_boundary=False, # incompatible with 'passage' or 'sentence'
    )

    doc_chunks = chunker.process(docs)

    doc_chunks = processing.remove_identical_chunks(doc_chunks)

    # -------------------- CHUNK SIMILARITY (PRESELECTION) ------------------- #
    MODEL_NAME = 'all-MiniLM-L6-v2'
    embeddings = processing.compute_chunk_embeddings(
        chunks=doc_chunks,
        model_name=MODEL_NAME,
        show_progress_bar=True,
    )
    # Enrich our chunks with embeddings
    for chunk, embedding in zip(doc_chunks, embeddings):
        chunk.embedding = embedding
    # TODO: Save intermediate chunks with their embeddings?

    similarity_matrix = processing.compute_chunk_similarity(doc_chunks)
    chunk_ids = [chunk.id for chunk in doc_chunks]
    # TODO: Save intermediate similarity scores between chunks? (only makes
    #       sense if we also save the chunks themselves)

    # Get rid of chunks that have similarity scores that are too high. This
    # value was fine-tuned by trial and error to remove similar chunks that were
    # just common header/disclaimer text.
    MAX_CHUNK_SIMILARITY_THRESHOLD = 0.87
    CHUNK_SIMILARITY_TOP_N = 500
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

    SENTENCE_CLEANING_LENGTH_MINIMUM = 40
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

    SAVED_CHUNKS_FILEPATH = 'desired-chunks.pkl'
    data.save_chunks_pickle(desired_chunks, SAVED_CHUNKS_FILEPATH)

    # ------------------------- CONTRADICTION SCORING ------------------------ #
    tokenizer, contradiction_model = scoring.load_contradiction_model()
    contradiction_scores = scoring.compute_sentence_contradiction_scores(
        chunks=desired_chunks,
        chunk_id_pairs=similar_chunk_id_pairs,
    )
    CANDIDATE_SELECTION_TOP_K = 100
    candidates = scoring.get_top_k_contradictive_candidates(
        contradiction_scores=contradiction_scores,
        k=CANDIDATE_SELECTION_TOP_K,
    )
    candidate_info = scoring.retrieve_candidate_info(
        candidates=candidates,
        chunks=desired_chunks
    )

    SAVED_CANDIDATES_FILEPATH = 'candidates.csv'
    data.save_candidates_csv(candidate_info, SAVED_CANDIDATES_FILEPATH)
