import os
import pickle

import numpy as np


ALPHA = 1.0
BETA = 1.0
EPS = 1e-12
MODEL_FORMAT = "serket.mlda.v2"


def as_modality_array(modality):
    if modality is None:
        return None

    array = np.asarray(modality)
    if array.ndim == 0 and array.dtype == object and array.item() is None:
        return None
    return array


def prepare_modalities(data):
    matrices = []
    dims = []
    num_docs = None

    for modality in data:
        array = as_modality_array(modality)
        if array is None:
            matrices.append(None)
            dims.append(0)
            continue

        array = np.asarray(array)
        if array.ndim != 2:
            raise ValueError("Each modality must be a 2D matrix or None.")

        if num_docs is None:
            num_docs = array.shape[0]
        elif num_docs != array.shape[0]:
            raise ValueError("All modalities must contain the same number of documents.")

        matrices.append(array)
        dims.append(array.shape[1])

    if num_docs is None:
        raise ValueError("At least one modality is required.")

    return matrices, dims, num_docs


def normalize_rows(matrix):
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim == 1:
        total = array.sum()
        if total <= 0:
            return np.ones_like(array, dtype=np.float64) / len(array)
        return array / total

    total = array.sum(axis=1, keepdims=True)
    total[total <= 0] = 1.0
    return array / total


def normalize_bias(bias_dz, num_docs, num_topics):
    if bias_dz is None:
        return np.ones((num_docs, num_topics), dtype=np.float64) / num_topics

    bias = np.asarray(bias_dz, dtype=np.float64)
    if bias.shape != (num_docs, num_topics):
        raise ValueError(
            "bias_dz must have shape ({}, {}) but got {}.".format(
                num_docs, num_topics, bias.shape
            )
        )

    bias = np.clip(bias, EPS, None)
    return normalize_rows(bias)


def calc_acc(results, correct):
    num_topics = int(np.max(correct) + 1)
    num_docs = len(results)
    max_acc = 0
    changed = True
    results = np.asarray(results, dtype=np.float64)

    while changed:
        changed = False
        for i in range(num_topics):
            for j in range(num_topics):
                swapped = np.array(results, copy=True)
                swapped[results == i] = j
                swapped[results == j] = i

                acc = (swapped == correct).sum() / float(num_docs)
                if acc > max_acc:
                    max_acc = acc
                    results = swapped
                    changed = True

    return max_acc, results


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_outputs(save_dir, Pdz, Pmdw, categories, liks):
    ensure_dir(save_dir)

    np.savetxt(os.path.join(save_dir, "Pdz.txt"), Pdz, fmt="%f")
    np.savetxt(os.path.join(save_dir, "liklihood.txt"), liks, fmt="%f")

    for index, modality_prob in enumerate(Pmdw):
        if modality_prob is None:
            continue
        np.savetxt(
            os.path.join(save_dir, "Pmdw[{}].txt".format(index)),
            modality_prob,
            fmt="%f",
        )

    results = np.argmax(Pdz, axis=-1)
    if categories is not None:
        acc, results = calc_acc(results, categories)
        np.savetxt(os.path.join(save_dir, "categories.txt"), results, fmt="%d")
        np.savetxt(os.path.join(save_dir, "acc.txt"), [acc], fmt="%f")
    else:
        np.savetxt(os.path.join(save_dir, "categories.txt"), results, fmt="%d")


def save_model_payload(save_dir, payload):
    ensure_dir(save_dir)
    with open(os.path.join(save_dir, "model.pickle"), "wb") as file_obj:
        pickle.dump(payload, file_obj)


def load_model_payload(load_dir):
    with open(os.path.join(load_dir, "model.pickle"), "rb") as file_obj:
        return pickle.load(file_obj)


def phi_from_payload(payload, dims=None):
    if isinstance(payload, dict):
        backend = payload.get("backend")
        if backend == "dense_em":
            phi_mw = []
            for phi in payload["phi_mw"]:
                if phi is None:
                    phi_mw.append(None)
                else:
                    phi_mw.append(np.asarray(phi, dtype=np.float64))
            return phi_mw
        if backend == "legacy_gibbs":
            counts = payload["n_mzw"]
        else:
            raise ValueError("Unsupported model payload backend: {}".format(backend))
    elif isinstance(payload, (list, tuple)) and len(payload) == 2:
        counts = payload[0]
    else:
        raise ValueError("Unsupported model payload format.")

    phi_mw = []
    for count in counts:
        count = np.asarray(count, dtype=np.float64)
        phi_mw.append(normalize_rows(count + BETA))
    return phi_mw


def dense_payload(phi_mw, num_topics):
    return {
        "format": MODEL_FORMAT,
        "backend": "dense_em",
        "K": num_topics,
        "phi_mw": [
            None if phi is None else np.asarray(phi, dtype=np.float64)
            for phi in phi_mw
        ],
    }


def legacy_payload(n_mzw, n_mz, num_topics):
    return {
        "format": MODEL_FORMAT,
        "backend": "legacy_gibbs",
        "K": num_topics,
        "n_mzw": [np.asarray(count, dtype=np.float64) for count in n_mzw],
        "n_mz": [np.asarray(count, dtype=np.float64) for count in n_mz],
    }
