#!/usr/bin/env python3
import argparse
import os
import sys
import tempfile
import time

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mlda import mlda  # noqa: E402


def preprocess_modalities(paths, weights):
    matrices = []
    for path, weight in zip(paths, weights):
        matrix = np.loadtxt(path)
        divider = np.where(matrix.sum(axis=1) == 0, 1, matrix.sum(axis=1))
        matrix = (matrix.T / divider).T * weight
        matrices.append(np.asarray(matrix, dtype=np.int32))
    return matrices


def evaluate_task(name, paths, weights, categories, K, backend, seeds, num_itr, num_restarts, init_method):
    data = preprocess_modalities(paths, weights)
    bias = np.ones((len(categories), K), dtype=np.float64) / K
    accs = []
    times = []

    for seed in range(seeds):
        with tempfile.TemporaryDirectory(prefix="mlda_bench_") as work_dir:
            start = time.perf_counter()
            Pdz, _ = mlda.train(
                data,
                K,
                num_itr=num_itr,
                save_dir=work_dir,
                bias_dz=bias,
                categories=categories,
                backend=backend,
                num_restarts=num_restarts,
                random_state=seed,
                init_method=init_method,
            )
            times.append(time.perf_counter() - start)
            acc, _ = mlda.calc_acc(np.argmax(Pdz, axis=-1), categories)
            accs.append(acc)

    return {
        "task": name,
        "backend": backend,
        "init_method": init_method,
        "mean_time": float(np.mean(times)),
        "mean_acc": float(np.mean(accs)),
        "best_acc": float(np.max(accs)),
        "worst_acc": float(np.min(accs)),
        "accs": accs,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark MLDA backends on the bundled mMLDA example.")
    parser.add_argument(
        "--backend",
        action="append",
        dest="backends",
        choices=mlda.BACKENDS,
        help="Backend(s) to benchmark. Defaults to legacy_gibbs, dense_em, auto.",
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--init-method", choices=("auto", "native", "fastkmeans"), default="auto")
    args = parser.parse_args()

    backends = args.backends or ["legacy_gibbs", "dense_em", "auto"]
    base_dir = os.path.join(ROOT, "examples", "mMLDA")
    object_categories = np.loadtxt(os.path.join(base_dir, "Object_Category.txt"))
    motion_categories = np.loadtxt(os.path.join(base_dir, "Motion_Category.txt"))

    tasks = [
        (
            "object",
            [
                os.path.join(base_dir, "dsift.txt"),
                os.path.join(base_dir, "mfcc.txt"),
                os.path.join(base_dir, "tactile.txt"),
            ],
            [200, 200, 200],
            object_categories,
        ),
        (
            "motion",
            [os.path.join(base_dir, "angle.txt")],
            [200],
            motion_categories,
        ),
    ]

    for backend in backends:
        for task_name, paths, weights, categories in tasks:
            result = evaluate_task(
                task_name,
                paths,
                weights,
                categories,
                10,
                backend,
                args.seeds,
                args.iters,
                args.restarts,
                args.init_method,
            )
            print(
                "{task} {backend} init={init_method} mean_acc={mean_acc:.3f} best={best_acc:.3f} "
                "worst={worst_acc:.3f} mean_time={mean_time:.3f}s".format(**result)
            )


if __name__ == "__main__":
    main()
