import math

import numpy as np

from ._common import ALPHA, BETA, EPS, dense_payload, load_model_payload
from ._common import normalize_bias, normalize_rows, phi_from_payload, prepare_modalities
from ._common import save_model_payload, save_outputs

try:
    import torch
    from fastkmeans import FastKMeans

    FASTKMEANS_AVAILABLE = True
except ImportError:
    torch = None
    FastKMeans = None
    FASTKMEANS_AVAILABLE = False


def _concat_features(matrices):
    features = []
    for matrix in matrices:
        if matrix is None:
            continue
        features.append(normalize_rows(matrix))
    return np.concatenate(features, axis=1)


def _kmeanspp_init(features, num_topics, rng):
    num_docs = features.shape[0]
    centers = [int(rng.integers(num_docs))]
    min_dist = np.sum((features - features[centers[0]]) ** 2, axis=1)

    for _ in range(1, num_topics):
        total = min_dist.sum()
        if total <= 0:
            centers.append(int(rng.integers(num_docs)))
        else:
            centers.append(int(rng.choice(num_docs, p=min_dist / total)))
        dist = np.sum((features - features[centers[-1]]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, dist)

    centers = features[np.asarray(centers, dtype=np.int32)].copy()
    labels = np.full(num_docs, -1, dtype=np.int32)

    for _ in range(30):
        distances = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        updated = np.argmin(distances, axis=1)
        if np.array_equal(updated, labels):
            break
        labels = updated
        for topic in range(num_topics):
            mask = labels == topic
            if mask.any():
                centers[topic] = features[mask].mean(axis=0)
            else:
                centers[topic] = features[int(rng.integers(num_docs))]

    return labels


def _fastkmeans_init(features, num_topics, rng):
    if not FASTKMEANS_AVAILABLE:
        raise RuntimeError("fastkmeans is not installed.")

    seed = int(rng.integers(0, 2**31 - 1))
    data = np.asarray(features, dtype=np.float32)
    use_gpu = bool(torch.cuda.is_available())

    def _run(gpu):
        model = FastKMeans(
            d=data.shape[1],
            k=num_topics,
            niter=25,
            tol=1e-8,
            gpu=gpu,
            seed=seed,
            max_points_per_centroid=None,
            verbose=False,
            use_triton=False,
        )
        return model.fit_predict(data).astype(np.int32, copy=False)

    if use_gpu:
        try:
            return _run(True)
        except Exception:
            pass

    return _run(False)


def _choose_init_method(init_method, num_docs, num_topics):
    if init_method != "auto":
        return init_method

    if FASTKMEANS_AVAILABLE and num_docs >= 1000 and num_topics >= 16:
        return "fastkmeans"
    return "native"


def _theta_from_labels(labels, bias):
    num_docs, num_topics = bias.shape
    theta = np.full((num_docs, num_topics), 0.05 / max(num_topics - 1, 1), dtype=np.float64)
    if num_topics == 1:
        theta[:, 0] = 1.0
    else:
        theta[np.arange(num_docs), labels] = 0.95
    return normalize_rows(theta * (0.25 + bias))


def _phi_from_labels(matrices, labels, num_topics):
    phi_mw = []
    for matrix in matrices:
        if matrix is None:
            phi_mw.append(None)
            continue
        topic_word = np.ones((num_topics, matrix.shape[1]), dtype=np.float64)
        for doc_index, label in enumerate(labels):
            topic_word[label] += matrix[doc_index]
        phi_mw.append(normalize_rows(topic_word))
    return phi_mw


def _compute_doc_topic_and_phi(matrices, effective_theta, phi_mw, update_phi):
    num_docs, num_topics = effective_theta.shape
    doc_topic = np.zeros((num_docs, num_topics), dtype=np.float64)
    next_phi = []
    lik = 0.0

    for matrix, phi in zip(matrices, phi_mw):
        if matrix is None:
            next_phi.append(None)
            continue

        denom = np.clip(effective_theta.dot(phi), EPS, None)
        scaled = matrix / denom
        doc_topic += effective_theta * scaled.dot(phi.T)
        if update_phi:
            topic_word = phi * effective_theta.T.dot(scaled)
            next_phi.append(normalize_rows(topic_word + BETA))
        else:
            next_phi.append(phi)

        lik += float(np.sum(matrix * np.log(denom)))

    return doc_topic, next_phi, lik


def _train_chain(matrices, num_topics, num_itr, bias, rng, fixed_phi=None, init_method="auto"):
    num_docs = bias.shape[0]
    if fixed_phi is None:
        features = _concat_features(matrices)
        chosen_init = _choose_init_method(init_method, num_docs, num_topics)
        if chosen_init == "fastkmeans":
            try:
                labels = _fastkmeans_init(features, num_topics, rng)
            except Exception:
                labels = _kmeanspp_init(features, num_topics, rng)
        else:
            labels = _kmeanspp_init(features, num_topics, rng)
        theta = _theta_from_labels(labels, bias)
        phi_mw = _phi_from_labels(matrices, labels, num_topics)
        update_phi = True
    else:
        theta = normalize_rows(bias + rng.random((num_docs, num_topics)) * 1e-3)
        phi_mw = fixed_phi
        update_phi = False

    liks = []
    best_lik = -math.inf
    best_theta = theta
    best_phi = phi_mw
    stale = 0

    for _ in range(max(1, num_itr)):
        effective_theta = normalize_rows(theta * bias)
        doc_topic, phi_candidate, lik = _compute_doc_topic_and_phi(
            matrices, effective_theta, phi_mw, update_phi
        )
        theta = normalize_rows(doc_topic + ALPHA)
        if update_phi:
            phi_mw = phi_candidate
        liks.append(lik)

        if lik > best_lik + 1e-9:
            best_lik = lik
            best_theta = theta.copy()
            best_phi = [
                None if phi is None else np.array(phi, copy=True)
                for phi in phi_mw
            ]
            stale = 0
        else:
            stale += 1
            if stale >= 8:
                break

    return best_lik, best_theta, best_phi, liks


def train(
    data,
    K,
    num_itr=100,
    save_dir="model",
    bias_dz=None,
    categories=None,
    load_dir=None,
    num_restarts=4,
    random_state=None,
    init_method="auto",
):
    matrices, dims, num_docs = prepare_modalities(data)
    matrices = [
        None if matrix is None else np.asarray(matrix, dtype=np.float64)
        for matrix in matrices
    ]
    bias = normalize_bias(bias_dz, num_docs, K)

    if load_dir is None:
        restarts = max(1, int(num_restarts))
        fixed_phi = None
    else:
        restarts = 1
        fixed_phi = phi_from_payload(load_model_payload(load_dir), dims)

    seed_sequence = np.random.SeedSequence(random_state)
    child_seeds = seed_sequence.spawn(restarts)

    best = None
    for child_seed in child_seeds:
        rng = np.random.default_rng(child_seed)
        candidate = _train_chain(
            matrices,
            K,
            num_itr,
            bias,
            rng,
            fixed_phi=fixed_phi,
            init_method=init_method,
        )
        if best is None or candidate[0] > best[0]:
            best = candidate

    _, theta, phi_mw, liks = best
    Pdz = normalize_rows(theta * bias)
    Pmdw = []
    for phi in phi_mw:
        if phi is None:
            Pmdw.append(None)
        else:
            Pmdw.append(Pdz.dot(phi))

    save_outputs(save_dir, Pdz, Pmdw, categories, liks)
    if load_dir is None:
        save_model_payload(save_dir, dense_payload(phi_mw, K))

    return Pdz, Pmdw
