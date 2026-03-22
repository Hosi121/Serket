import numpy as np

from . import _dense
from . import _legacy
from ._common import calc_acc, prepare_modalities


def _active_modalities(data):
    matrices, _, _ = prepare_modalities(data)
    return sum(1 for matrix in matrices if matrix is not None)


def _total_mass(data):
    matrices, _, _ = prepare_modalities(data)
    total = 0
    for matrix in matrices:
        if matrix is not None:
            total += int(np.asarray(matrix).sum())
    return total


def choose_backend(data, K, backend="auto"):
    if backend != "auto":
        return backend

    if K > 100:
        return "dense_em"
    if _active_modalities(data) > 1:
        return "dense_em"
    if not _legacy.NUMBA_AVAILABLE:
        return "dense_em"
    if _total_mass(data) > 20000:
        return "dense_em"
    return "legacy_gibbs"


def train(
    data,
    K,
    num_itr=100,
    save_dir="model",
    bias_dz=None,
    categories=None,
    load_dir=None,
    backend="auto",
    num_restarts=None,
    random_state=None,
):
    backend = choose_backend(data, K, backend=backend)

    if backend == "dense_em":
        if num_restarts is None:
            num_restarts = 4 if load_dir is None else 1
        return _dense.train(
            data,
            K,
            num_itr=num_itr,
            save_dir=save_dir,
            bias_dz=bias_dz,
            categories=categories,
            load_dir=load_dir,
            num_restarts=num_restarts,
            random_state=random_state,
        )

    if backend == "legacy_gibbs":
        return _legacy.train(
            data,
            K,
            num_itr=num_itr,
            save_dir=save_dir,
            bias_dz=bias_dz,
            categories=categories,
            load_dir=load_dir,
            num_restarts=1,
            random_state=random_state,
        )

    raise ValueError("Unknown MLDA backend: {}".format(backend))


BACKENDS = ("auto", "dense_em", "legacy_gibbs")
