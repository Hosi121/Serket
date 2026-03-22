import random

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
    cumulative = np.empty(num_topics, dtype=np.float64)
    total = 0.0

    for topic in range(num_topics):
        total += (
            (n_dz[doc_index, topic] + ALPHA)
            * (n_zw[topic, word_index] + BETA)
            / (n_z[topic] + vocab_size * BETA)
            * bias_row[topic]
        )
        cumulative[topic] = total

    threshold = total * np.random.random()
    for topic in range(num_topics):
        if cumulative[topic] >= threshold:
            return topic

    return num_topics - 1


def conv_to_word_list(data):
    counts = np.asarray(data, dtype=np.int64)
    return np.repeat(np.arange(len(counts), dtype=np.int32), counts)


def calc_liklihood(data, n_dz, n_zw, n_z, num_topics, vocab_size):
    likelihood = 0.0

    P_wz = (n_zw.T + BETA) / (n_z + vocab_size * BETA)
    for doc_index in range(len(data)):
        Pz = (n_dz[doc_index] + ALPHA) / (np.sum(n_dz[doc_index]) + num_topics * ALPHA)
        Pwz = Pz * P_wz
        Pw = np.sum(Pwz, axis=1) + 1e-6
        likelihood += float(np.sum(data[doc_index] * np.log(Pw)))

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

    docs_mdn = [[None for _ in range(num_docs)] for _ in range(len(matrices))]
    topics_mdn = [[None for _ in range(num_docs)] for _ in range(len(matrices))]
    for doc_index in range(num_docs):
        for modality_index, matrix in enumerate(matrices):
            if matrix is None:
                continue
            docs_mdn[modality_index][doc_index] = conv_to_word_list(matrix[doc_index])
            topics_mdn[modality_index][doc_index] = np.random.randint(
                0, K, len(docs_mdn[modality_index][doc_index])
            )

    n_dz, n_mzw, n_mz = _calc_lda_param(docs_mdn, topics_mdn, K, dims)

    if load_dir is not None:
        n_mzw, n_mz = _load_legacy_counts(load_dir)

    liks = []
    for _ in range(num_itr):
        for doc_index in range(num_docs):
            for modality_index, matrix in enumerate(matrices):
                if matrix is None:
                    continue

                words = docs_mdn[modality_index][doc_index]
                topics = topics_mdn[modality_index][doc_index]
                for token_index in range(len(words)):
                    word_index = words[token_index]
                    topic = topics[token_index]

                    n_dz[doc_index, topic] -= 1
                    if load_dir is None:
                        n_mzw[modality_index][topic, word_index] -= 1
                        n_mz[modality_index][topic] -= 1

                    topic = _sample_topic(
                        doc_index,
                        word_index,
                        n_dz,
                        n_mzw[modality_index],
                        n_mz[modality_index],
                        dims[modality_index],
                        bias[doc_index],
                    )

                    topics[token_index] = topic
                    n_dz[doc_index, topic] += 1
                    if load_dir is None:
                        n_mzw[modality_index][topic, word_index] += 1
                        n_mz[modality_index][topic] += 1

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
