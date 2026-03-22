from . import _dense
from . import _legacy
from ._common import calc_acc, load_model_payload


def _backend_from_saved_model(load_dir):
    payload = load_model_payload(load_dir)
    if isinstance(payload, dict):
        backend = payload.get("backend")
        if backend in ("dense_em", "legacy_gibbs"):
            return backend
    if isinstance(payload, (list, tuple)) and len(payload) == 2:
        return "legacy_gibbs"
    return "dense_em"


def choose_backend(data, K, backend="auto", load_dir=None):
    if backend != "auto":
        return backend

    if load_dir is not None:
        return _backend_from_saved_model(load_dir)

    return "dense_em"


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
    backend = choose_backend(data, K, backend=backend, load_dir=load_dir)

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
