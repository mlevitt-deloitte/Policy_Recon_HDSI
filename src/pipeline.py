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

    # ///-- ADDITIONAL CLEANING (SHOULD BE POTENTIALLY MOVED TO MODULES) --\\\ #
    # Get rid of chunks that have the same content (and therefore the same hash
    # id)
    from collections import Counter
    doc_chunk_id_counts = Counter([c.id for c in doc_chunks])
    doc_chunks = [c for c in doc_chunks if doc_chunk_id_counts[c.id] == 1]
    # \\\------------------------------------------------------------------/// #

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

    # ///-- ADDITIONAL CLEANING (SHOULD BE POTENTIALLY MOVED TO MODULES) --\\\ #
    # Get rid of chunks that have similarity scores that are too high
    # This value was fine-tuned by trial and error to remove similar chunks that
    # were just common header/disclaimer text.
    MAX_CHUNK_SIMILARITY_SCORE = 0.87

    similarity_matrix = np.where(
        similarity_matrix < MAX_CHUNK_SIMILARITY_SCORE,
        similarity_matrix, 0
    )
    # Get rid of adjacent chunks
    similarity_matrix = np.triu(similarity_matrix, 2)
    # \\\------------------------------------------------------------------/// #

    CHUNK_SIMILARITY_TOP_N = 10  # Should be much much larger!
    top_n_pair_indices = processing.get_top_n_similar_chunk_pair_indices(
        scores=similarity_matrix,
        n=CHUNK_SIMILARITY_TOP_N,
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

    CHUNK_SENTENCE_CLEANING_FUNC2 = partial(processing.clean_sentence_splits,
        toc_period_threshold = 5,
        length_minimum = 20,
        length_maximum = 1000,  # FIXME: Should support None as default arg
    )
    desired_chunk_sentences = processing.split_chunks_to_sentences(
        chunks=desired_chunks.values(),
        split_cleaner=CHUNK_SENTENCE_CLEANING_FUNC2,
    )
    # Enrich our desired chunks with sentences
    for chunk, sentences in zip(desired_chunks.values(), desired_chunk_sentences):
        chunk.sentences = sentences

    # ------------------------- CONTRADICTION SCORING ------------------------ #
    tokenizer, contradiction_model = scoring.load_contradiction_model()
    contradiction_scores = scoring.compute_sentence_contradiction_scores(
        chunks=desired_chunks,
        chunk_id_pairs=similar_chunk_id_pairs,
    )
    CANDIDATE_SELECTION_TOP_K = 10
    candidates = scoring.get_top_k_contradictive_candidates(
        contradiction_scores=contradiction_scores,
        k=CANDIDATE_SELECTION_TOP_K,
    )
    candidate_info = scoring.retrieve_candidate_info(
        candidates=candidates,
        chunks=desired_chunks
    )
