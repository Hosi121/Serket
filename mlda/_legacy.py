import random
import math

import numpy as np

from ._common import ALPHA, BETA, legacy_payload, load_model_payload, normalize_bias
from ._common import normalize_rows, prepare_modalities, save_model_payload, save_outputs

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def decorator(function):
            return function

        return decorator


@njit(cache=True)
def _sample_topic(doc_index, word_index, n_dz, n_zw, n_z, vocab_size, bias_row):
    num_topics = n_dz.shape[1]
    total = 0.0

    for topic in range(num_topics):
        total += (
            (n_dz[doc_index, topic] + ALPHA)
            * (n_zw[topic, word_index] + BETA)
            / (n_z[topic] + vocab_size * BETA)
            * bias_row[topic]
        )

    threshold = total * np.random.random()
    cumulative = 0.0
    for topic in range(num_topics):
        cumulative += (
            (n_dz[doc_index, topic] + ALPHA)
            * (n_zw[topic, word_index] + BETA)
            / (n_z[topic] + vocab_size * BETA)
            * bias_row[topic]
        )
        if cumulative >= threshold:
            return topic

    return num_topics - 1


@njit(cache=True)
def _init_counts(doc_offsets, words, topics, n_dz, n_zw, n_z):
    num_docs = len(doc_offsets) - 1
    for doc_index in range(num_docs):
        for token_index in range(doc_offsets[doc_index], doc_offsets[doc_index + 1]):
            topic = topics[token_index]
            word_index = words[token_index]
            n_dz[doc_index, topic] += 1
            n_zw[topic, word_index] += 1
            n_z[topic] += 1


@njit(cache=True)
def _seed_numba(seed):
    np.random.seed(seed)


@njit(cache=True)
def _gibbs_sweep_modality(doc_offsets, words, topics, n_dz, n_zw, n_z, vocab_size, bias, update_global):
    num_docs = len(doc_offsets) - 1

    for doc_index in range(num_docs):
        bias_row = bias[doc_index]
        for token_index in range(doc_offsets[doc_index], doc_offsets[doc_index + 1]):
            word_index = words[token_index]
            topic = topics[token_index]

            n_dz[doc_index, topic] -= 1
            if update_global:
                n_zw[topic, word_index] -= 1
                n_z[topic] -= 1

            topic = _sample_topic(doc_index, word_index, n_dz, n_zw, n_z, vocab_size, bias_row)

            topics[token_index] = topic
            n_dz[doc_index, topic] += 1
            if update_global:
                n_zw[topic, word_index] += 1
                n_z[topic] += 1


def conv_to_word_list(data):
    counts = np.asarray(data, dtype=np.int64)
    return np.repeat(np.arange(len(counts), dtype=np.int32), counts)


@njit(cache=True)
def calc_liklihood(data, n_dz, n_zw, n_z, num_topics, vocab_size):
    likelihood = 0.0
    num_docs = data.shape[0]
    topic_denom = np.empty(num_topics, dtype=np.float64)
    for topic in range(num_topics):
        topic_denom[topic] = n_z[topic] + vocab_size * BETA

    for doc_index in range(num_docs):
        doc_total = 0.0
        for topic in range(num_topics):
            doc_total += n_dz[doc_index, topic]
        doc_denom = doc_total + num_topics * ALPHA

        for word_index in range(vocab_size):
            count = data[doc_index, word_index]
            if count == 0:
                continue

            prob = 0.0
            for topic in range(num_topics):
                prob += (
                    (n_dz[doc_index, topic] + ALPHA)
                    / doc_denom
                    * (n_zw[topic, word_index] + BETA)
                    / topic_denom[topic]
                )
            likelihood += count * math.log(prob + 1e-6)

    return likelihood


def _calc_lda_param(docs_mdn, topics_mdn, num_topics, dims):
    num_modalities = len(docs_mdn)
    num_docs = len(docs_mdn[0])

    n_dz = np.zeros((num_docs, num_topics), dtype=np.float64)
    n_mzw = [np.zeros((num_topics, dims[index]), dtype=np.float64) for index in range(num_modalities)]
    n_mz = [np.zeros(num_topics, dtype=np.float64) for _ in range(num_modalities)]

    for doc_index in range(num_docs):
        for modality_index in range(num_modalities):
            words = docs_mdn[modality_index][doc_index]
            if words is None:
                continue

            topics = topics_mdn[modality_index][doc_index]
            for word_index, topic in zip(words, topics):
                n_dz[doc_index, topic] += 1
                n_mzw[modality_index][topic, word_index] += 1
                n_mz[modality_index][topic] += 1

    return n_dz, n_mzw, n_mz


def _build_token_state(matrix, num_topics):
    doc_words = [conv_to_word_list(row) for row in matrix]
    lengths = np.array([len(words) for words in doc_words], dtype=np.int64)
    doc_offsets = np.zeros(len(matrix) + 1, dtype=np.int64)
    doc_offsets[1:] = np.cumsum(lengths)

    if lengths.sum() == 0:
        words = np.empty(0, dtype=np.int32)
    else:
        words = np.concatenate(doc_words).astype(np.int32, copy=False)

    topics = np.random.randint(0, num_topics, size=len(words)).astype(np.int32, copy=False)
    return doc_offsets, words, topics


def _load_legacy_counts(load_dir):
    payload = load_model_payload(load_dir)
    if isinstance(payload, dict) and payload.get("backend") == "legacy_gibbs":
        return payload["n_mzw"], payload["n_mz"]
    if isinstance(payload, (list, tuple)) and len(payload) == 2:
        return payload
    raise ValueError("The legacy Gibbs backend can only load legacy Gibbs models.")


def train(
    data,
    K,
    num_itr=100,
    save_dir="model",
    bias_dz=None,
    categories=None,
    load_dir=None,
    num_restarts=1,
    random_state=None,
):
    del num_restarts

    matrices, dims, num_docs = prepare_modalities(data)
    matrices = [
        None if matrix is None else np.asarray(matrix, dtype=np.int32)
        for matrix in matrices
    ]
    bias = normalize_bias(bias_dz, num_docs, K).astype(np.float64)

    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
        _seed_numba(random_state)

    token_state = []
    n_dz = np.zeros((num_docs, K), dtype=np.int64)
    n_mzw = []
    n_mz = []
    for modality_index, matrix in enumerate(matrices):
        if matrix is None:
            token_state.append(None)
            n_mzw.append(None)
            n_mz.append(None)
            continue

        doc_offsets, words, topics = _build_token_state(matrix, K)
        token_state.append((doc_offsets, words, topics))
        topic_word = np.zeros((K, dims[modality_index]), dtype=np.int64)
        topic_count = np.zeros(K, dtype=np.int64)
        _init_counts(doc_offsets, words, topics, n_dz, topic_word, topic_count)
        n_mzw.append(topic_word)
        n_mz.append(topic_count)

    if load_dir is not None:
        n_mzw, n_mz = _load_legacy_counts(load_dir)
        n_mzw = [
            None if count is None else np.asarray(count, dtype=np.int64)
            for count in n_mzw
        ]
        n_mz = [
            None if count is None else np.asarray(count, dtype=np.int64)
            for count in n_mz
        ]

    liks = []
    for _ in range(num_itr):
        for modality_index, matrix in enumerate(matrices):
            if matrix is None:
                continue
            doc_offsets, words, topics = token_state[modality_index]
            _gibbs_sweep_modality(
                doc_offsets,
                words,
                topics,
                n_dz,
                n_mzw[modality_index],
                n_mz[modality_index],
                dims[modality_index],
                bias,
                load_dir is None,
            )

        likelihood = 0.0
        for modality_index, matrix in enumerate(matrices):
            if matrix is None:
                continue
            likelihood += calc_liklihood(
                matrix,
                n_dz,
                n_mzw[modality_index],
                n_mz[modality_index],
                K,
                dims[modality_index],
            )
        liks.append(likelihood)

    Pdz = normalize_rows(n_dz + ALPHA)
    Pmdw = []
    for modality_index, matrix in enumerate(matrices):
        if matrix is None:
            Pmdw.append(None)
            continue
        phi = normalize_rows(np.asarray(n_mzw[modality_index], dtype=np.float64) + BETA)
        Pmdw.append(Pdz.dot(phi))

    save_outputs(save_dir, Pdz, Pmdw, categories, liks)
    if load_dir is None:
        save_model_payload(save_dir, legacy_payload(n_mzw, n_mz, K))

    return Pdz, Pmdw
