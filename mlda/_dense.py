import math

import numpy as np

from ._common import ALPHA, BETA, EPS, dense_payload, load_model_payload
from ._common import normalize_bias, phi_from_payload, prepare_modalities
from ._common import save_model_payload, save_outputs

try:
    import torch
    from fastkmeans import FastKMeans

    FASTKMEANS_AVAILABLE = True
except ImportError:
    torch = None
    FastKMeans = None
    FASTKMEANS_AVAILABLE = False


DENSE_DTYPE = np.float32
DENSE_EPS = DENSE_DTYPE(EPS)


def _normalize_rows32(matrix):
    array = np.asarray(matrix, dtype=DENSE_DTYPE)
    if array.ndim == 1:
        total = float(array.sum(dtype=np.float64))
        if total <= 0:
            return np.full_like(array, 1.0 / len(array), dtype=DENSE_DTYPE)
        return np.ascontiguousarray(array / DENSE_DTYPE(total))

    total = array.sum(axis=1, keepdims=True, dtype=np.float64)
    total[total <= 0] = 1.0
    return np.ascontiguousarray(array / total.astype(DENSE_DTYPE))


def _copy_phi(phi_mw):
    return [
        None if phi is None else np.ascontiguousarray(np.array(phi, dtype=DENSE_DTYPE, copy=True))
        for phi in phi_mw
    ]


def _concat_features(matrices):
    features = []
    for matrix in matrices:
        if matrix is None:
            continue
        features.append(_normalize_rows32(matrix))
    return np.concatenate(features, axis=1).astype(DENSE_DTYPE, copy=False)


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


def _restart_settings(chosen_init, max_restarts):
    if max_restarts <= 1:
        return max_restarts, 0, 0.0
    if chosen_init == "fastkmeans":
        return min(2, max_restarts), 1, 1e-4
    return max_restarts, max_restarts + 1, 0.0


def _theta_from_labels(labels, bias):
    num_docs, num_topics = bias.shape
    theta = np.full((num_docs, num_topics), 0.05 / max(num_topics - 1, 1), dtype=DENSE_DTYPE)
    if num_topics == 1:
        theta[:, 0] = 1.0
    else:
        theta[np.arange(num_docs), labels] = 0.95
    return _normalize_rows32(theta * (DENSE_DTYPE(0.25) + bias))


def _phi_from_labels(matrices, labels, num_topics):
    phi_mw = []
    for matrix in matrices:
        if matrix is None:
            phi_mw.append(None)
            continue
        topic_word = np.ones((num_topics, matrix.shape[1]), dtype=DENSE_DTYPE)
        for doc_index, label in enumerate(labels):
            topic_word[label] += matrix[doc_index]
        phi_mw.append(_normalize_rows32(topic_word))
    return phi_mw


def _compute_doc_topic_and_phi(matrices, effective_theta, phi_mw, update_phi):
    num_docs, num_topics = effective_theta.shape
    doc_topic = np.zeros((num_docs, num_topics), dtype=DENSE_DTYPE)
    next_phi = []
    lik = 0.0
    effective_theta_t = np.ascontiguousarray(effective_theta.T)

    for matrix, phi in zip(matrices, phi_mw):
        if matrix is None:
            next_phi.append(None)
            continue

        phi_t = np.ascontiguousarray(phi.T)
        denom = np.clip(effective_theta.dot(phi), DENSE_EPS, None)
        scaled = matrix / denom
        doc_topic += effective_theta * scaled.dot(phi_t)
        if update_phi:
            topic_word = phi * effective_theta_t.dot(scaled)
            next_phi.append(_normalize_rows32(topic_word + DENSE_DTYPE(BETA)))
        else:
            next_phi.append(phi)

        lik += float(np.sum(matrix * np.log(denom), dtype=np.float64))

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
        theta = _normalize_rows32(
            bias + rng.random((num_docs, num_topics), dtype=DENSE_DTYPE) * DENSE_DTYPE(1e-3)
        )
        phi_mw = fixed_phi
        update_phi = False

    liks = []
    best_lik = -math.inf
    best_theta = theta
    best_phi = phi_mw
    stale = 0

    for _ in range(max(1, num_itr)):
        effective_theta = _normalize_rows32(theta * bias)
        doc_topic, phi_candidate, lik = _compute_doc_topic_and_phi(
            matrices, effective_theta, phi_mw, update_phi
        )
        theta = _normalize_rows32(doc_topic + DENSE_DTYPE(ALPHA))
        if update_phi:
            phi_mw = phi_candidate
        liks.append(lik)

        if lik > best_lik + 1e-9:
            best_lik = lik
            best_theta = np.array(theta, dtype=DENSE_DTYPE, copy=True)
            best_phi = _copy_phi(phi_mw)
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
        None if matrix is None else np.asarray(matrix, dtype=DENSE_DTYPE)
        for matrix in matrices
    ]
    bias = np.asarray(normalize_bias(bias_dz, num_docs, K), dtype=DENSE_DTYPE)

    if load_dir is None:
        restarts = max(1, int(num_restarts))
        fixed_phi = None
        chosen_init = _choose_init_method(init_method, num_docs, K)
    else:
        restarts = 1
        fixed_phi = _copy_phi(phi_from_payload(load_model_payload(load_dir), dims))
        chosen_init = init_method

    seed_sequence = np.random.SeedSequence(random_state)
    child_seeds = seed_sequence.spawn(restarts)
    min_restarts, stale_restart_patience, restart_tol = _restart_settings(chosen_init, restarts)

    best = None
    stale_restarts = 0
    for restart_index, child_seed in enumerate(child_seeds, start=1):
        rng = np.random.default_rng(child_seed)
        candidate = _train_chain(
            matrices,
            K,
            num_itr,
            bias,
            rng,
            fixed_phi=fixed_phi,
            init_method=chosen_init,
        )
        if best is None:
            best = candidate
            stale_restarts = 0
        else:
            improvement_floor = max(1.0, abs(best[0])) * restart_tol
            if candidate[0] > best[0] + improvement_floor:
                best = candidate
                stale_restarts = 0
            else:
                stale_restarts += 1

        if (
            restarts > 1
            and fixed_phi is None
            and restart_index >= min_restarts
            and stale_restarts >= stale_restart_patience
        ):
            break

    _, theta, phi_mw, liks = best
    Pdz = _normalize_rows32(theta * bias)
    Pmdw = []
    for phi in phi_mw:
        if phi is None:
            Pmdw.append(None)
        else:
            Pmdw.append(Pdz.dot(phi).astype(DENSE_DTYPE, copy=False))

    save_outputs(save_dir, Pdz, Pmdw, categories, liks)
    if load_dir is None:
        save_model_payload(save_dir, dense_payload(phi_mw, K))

    return Pdz, Pmdw
