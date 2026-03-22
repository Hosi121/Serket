#!/usr/bin/env python3
import argparse
import functools
import os
import sys
import tempfile
import time

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mlda import _dense, _legacy, mlda  # noqa: E402


BREAKDOWN_TARGETS = {
    "dense_em": [
        "_concat_features",
        "_kmeanspp_init",
        "_phi_from_labels",
        "_compute_doc_topic_and_phi",
        "save_outputs",
    ],
    "legacy_gibbs": [
        "_build_token_state",
        "_gibbs_sweep_modality",
        "calc_liklihood",
        "save_outputs",
    ],
}


def parse_int_list(text):
    return [int(value) for value in text.split(",") if value]


def make_synthetic_data(num_docs, dims, nnz_per_doc, seed):
    rng = np.random.default_rng(seed)
    data = []
    for dim, nnz in zip(dims, nnz_per_doc):
        matrix = np.zeros((num_docs, dim), dtype=np.int32)
        for doc_index in range(num_docs):
            word_index = rng.choice(dim, size=nnz, replace=False)
            matrix[doc_index, word_index] = rng.integers(1, 4, size=nnz)
        data.append(matrix)
    return data


def patch_breakdown(module, backend):
    names = BREAKDOWN_TARGETS[backend]
    totals = {name: 0.0 for name in names}
    counts = {name: 0 for name in names}
    originals = {}

    for name in names:
        fn = getattr(module, name)
        originals[name] = fn

        def make_wrapper(fn, name):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = fn(*args, **kwargs)
                totals[name] += time.perf_counter() - start
                counts[name] += 1
                return result

            return wrapper

        setattr(module, name, make_wrapper(fn, name))

    return originals, totals, counts


def restore_breakdown(module, originals):
    for name, fn in originals.items():
        setattr(module, name, fn)


def main():
    parser = argparse.ArgumentParser(description="Profile MLDA on synthetic large sparse count data.")
    parser.add_argument("--backend", choices=("dense_em", "legacy_gibbs"), required=True)
    parser.add_argument("--docs", type=int, required=True)
    parser.add_argument("--dims", type=parse_int_list, required=True)
    parser.add_argument("--nnz", type=parse_int_list, required=True)
    parser.add_argument("--topics", type=int, default=32)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--breakdown", action="store_true")
    args = parser.parse_args()

    if len(args.dims) != len(args.nnz):
        raise ValueError("--dims and --nnz must have the same length.")

    data = make_synthetic_data(args.docs, args.dims, args.nnz, args.seed)
    bias = np.ones((args.docs, args.topics), dtype=np.float64) / args.topics

    if args.breakdown:
        module = _dense if args.backend == "dense_em" else _legacy
        originals, totals, counts = patch_breakdown(module, args.backend)
    else:
        module = None
        originals = totals = counts = None

    try:
        with tempfile.TemporaryDirectory(prefix="mlda_profile_") as save_dir:
            start = time.perf_counter()
            Pdz, _ = mlda.train(
                data,
                args.topics,
                num_itr=args.iters,
                save_dir=save_dir,
                bias_dz=bias,
                backend=args.backend,
                num_restarts=args.restarts,
                random_state=args.seed,
            )
            elapsed = time.perf_counter() - start
    finally:
        if args.breakdown:
            restore_breakdown(module, originals)

    print(
        "backend={} docs={} topics={} dims={} nnz={} elapsed={:.3f}s shape={}".format(
            args.backend,
            args.docs,
            args.topics,
            args.dims,
            args.nnz,
            elapsed,
            Pdz.shape,
        )
    )

    if args.breakdown:
        for name in BREAKDOWN_TARGETS[args.backend]:
            print(
                "{} time={:.3f}s calls={} share={:.3f}".format(
                    name,
                    totals[name],
                    counts[name],
                    totals[name] / elapsed if elapsed else 0.0,
                )
            )


if __name__ == "__main__":
    main()
