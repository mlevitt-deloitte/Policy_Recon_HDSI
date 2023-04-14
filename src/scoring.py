"""
Methods for selecting potentially contradictive sentence candidates.
"""

import itertools
from typing import Callable, List, Optional

from haystack.schema import Document


def get_sentence_combinations(
        chunk_pairs: List[List[Document]],
) -> List[List[str]]:
    # Split chunks into sentences
    raise NotImplementedError
