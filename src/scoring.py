"""
Methods for selecting potentially contradictive sentence candidates.
"""

import itertools
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import torch
from haystack.schema import Document
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          PreTrainedModel, PreTrainedTokenizer)
from tqdm.auto import tqdm


def load_contradiction_model(
        model_name: str = 'ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli',
) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """
    Loads a tokenizer and sequence classification model from HuggingFace. Meant
    for https://huggingface.co/ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model


# TODO: FIXME: This should be optimized so that we can create sentence
# embeddings once, then pass these embeddings to the model for contradiction
# scoring. We shouldn't compute the tokenization for the same sentence multiple
# times.
def _evaluate_contradictions(
        premise,
        hypothesis,
        tokenizer,
        model,
) -> List[float]:
    """
    From https://github.com/facebookresearch/anli/blob/main/src/hg_api/interactive.py
    """
    max_length = 256

    tokenized_input_seq_pair = tokenizer.encode_plus(premise, hypothesis,
                                                     max_length=max_length,
                                                     return_token_type_ids=True, truncation=True)

    input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
    # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
    token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
    attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)

    outputs = model(input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=None)

    predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()  # batch_size only one

    # Note:
    # "id2label": {
    #     "0": "entailment",
    #     "1": "neutral",
    #     "2": "contradiction"
    # },
    return predicted_probability


def compute_sentence_contradiction_scores(
        chunks: Dict[str, Document],
        chunk_id_pairs: List[Tuple[str, str]],
        tokenizer: PreTrainedTokenizer,
        model: PreTrainedModel,
) -> DataFrame:
    """
    Given a collection of chunks with a .sentences property, and the IDs of
    pairs to evaluate, compute NLI probabilities (entailment, neutral,
    contradiction) between every combination of sentences between the two chunks
    in each pair.
    """
    columns = ['chunk_A', 'chunk_B', 'sentence_A', 'sentence_B',
               'entailment', 'neutral', 'contradiction']
    contradiction_scores = pd.DataFrame(columns=columns)

    for chunk_id_A, chunk_id_B in tqdm(chunk_id_pairs[:5]):
        sentences_chunk_A = chunks[chunk_id_A].sentences
        sentences_chunk_B = chunks[chunk_id_B].sentences
        sentence_combination_indices = list(
            itertools.product(
                range(len(sentences_chunk_A)),
                range(len(sentences_chunk_B))
            )
        )
        outputs = []
        for idx_A, idx_B in sentence_combination_indices:
            probs = _evaluate_contradictions(
                sentences_chunk_A[idx_A], sentences_chunk_B[idx_B],
                tokenizer=tokenizer, model=model,
            )
            outputs.append([chunk_id_A, chunk_id_B, idx_A, idx_B, *probs])

        contradiction_scores = pd.concat([
            contradiction_scores,
            pd.DataFrame(columns=columns, data=outputs)
        ])

    return contradiction_scores


def get_top_k_contradictive_candidates(
        contradiction_scores: DataFrame,
        k: int,
) -> DataFrame:
    """
    Return the top k rows from the contradiction scores table which are
    considered the most contradictive, sorted with greatest first.
    """
    # The heuristic is currently to just select the 'contradiction' output
    # column.
    top_k_rows = (
        contradiction_scores
        .sort_values('contradiction', ascending=False)
        .iloc[:k]
    )
    return top_k_rows


def retrieve_candidate_info(
        candidates: DataFrame,
        chunks: Dict[str, Document],
) -> DataFrame:
    """
    Convert a dataframe with chunk IDs and sentence indices into a frame with
    the contents and metadata of each chunk and the candidate sentence text.
    The resulting frame contains all information needed for a human to verify
    the candidates.
    """

    chunk_A_infos = (
        pd.json_normalize([{
            'id': cid,
            'content': chunks[cid].content,
            'meta': chunks[cid].meta
        } for cid in candidates.chunk_A])
        .add_prefix('chunk_A.')
    )
    chunk_B_infos = (
        pd.json_normalize([{
            'id': cid,
            'content': chunks[cid].content,
            'meta': chunks[cid].meta
        } for cid in candidates.chunk_B])
        .add_prefix('chunk_B.')
    )
    sentence_A_texts = pd.Series(name='sentence_A', data=[
        chunks[cid].sentences[sidx]
        for (cid, sidx) in zip(candidates.chunk_A, candidates.sentence_A)
    ])
    sentence_B_texts = pd.Series(name='sentence_B', data=[
        chunks[cid].sentences[sidx]
        for (cid, sidx) in zip(candidates.chunk_B, candidates.sentence_B)
    ])

    candidate_info = pd.concat(
        [
            chunk_A_infos, sentence_A_texts,
            chunk_B_infos, sentence_B_texts,
            candidates[['entailment', 'neutral', 'contradiction']].reset_index(drop=True)
        ],
        axis=1,
    )
    return candidate_info


def pretty_print_candidate(candidate: Series):
    raise NotImplementedError
