# Which dataset to load.
DATASET_FILEPATH = "./data/DoD Issuances/contradictions_datasets_dod_issuances.zip"
# How many documents to include in the pipeline run.
SUBSET_SIZE = 10                # DEMO VALUE ::: Recommend `None` for full pipeline.
# Parameters for cleaning sentences in initial chunking
CHUNK_CLEANING_TOC_PERIOD_THRESHOLD = 5
CHUNK_CLEANING_LENGTH_MINIMUM = 15
CHUNK_CLEANING_LENGTH_MAXIMUM = 1000
# Parameters for creating chunks
CHUNK_LENGTH = 8
CHUNK_OVERLAP = 2
# Parameters for pre-selecting similar chunks
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
MAX_CHUNK_SIMILARITY_THRESHOLD = 0.87
CHUNK_SIMILARITY_TOP_N = 10     # DEMO VALUE ::: Recommend `500` for full pipeline.
# Parameters for selecting sentences as contradiction candidates
SENTENCE_CLEANING_LENGTH_MINIMUM = 40
CANDIDATE_SELECTION_TOP_K = 10  # DEMO VALUE ::: Recommend `500` for full pipeline.
SAVED_CANDIDATES_FILEPATH = 'output/candidates.csv'
