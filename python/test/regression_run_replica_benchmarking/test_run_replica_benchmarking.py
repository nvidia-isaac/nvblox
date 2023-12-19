#!/bin/env python3
"""Regression test for benchmarking KPIs

This test computes KPIs from a limited dataset and compares them to baseline KPIs stored in the
repo. This allow us to

1. Ensure that end-to-end benchmarking is functional.
2. Detect unexpected changes in performance.

If the change was expected (e.g. the algorithm has changed) we can update the baseline KPIs by
invoking this test with '--generate_baseline' argument
"""

import tempfile
import json
import argparse
from pathlib import Path

import numpy

from nvblox_run_replica_benchmarking.__main__ import main

# Repo path to dataset used in regression tests
DATASET_PATH = "nvblox/tests/data/replica/office0"

# Repo path to directory containing baseline kpis
BASELINE_KPI_PATH = "python/test/regression_run_replica_benchmarking/baseline_kpi.json"

# Binary path to fuser
FUSE_REPLICA_BINARY_PATH = "nvblox/build/executables/fuse_replica"


def read_json(path):
    """Load and return a json file."""
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def write_baseline_kpis(kpis):
    """Update the baseline KPIs used for regression."""
    with open(BASELINE_KPI_PATH, "w", encoding="utf-8") as fp:
        json.dump(kpis, fp, indent=4)


def compare_kpis(baseline_kpis, computed_kpis):
    """Compare two KPI dicts and assert if they differ."""
    assert len(baseline_kpis) > 0

    for key in baseline_kpis:
        assert key in computed_kpis
        baseline = baseline_kpis[key]
        computed = computed_kpis[key]
        assert numpy.isclose(baseline, computed, rtol=1E-3, atol=1E-4), (
            f"{key} has changed: {baseline} -> {computed}"
            "run this script with --generate_baseline to update the KPIs")


def parse_args():
    """Return parsed args."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--generate_baseline",
        action='store_const',
        const=True,
        default=False,
        help="Flag indicating if we should re-generate the baseline kpis.")

    args, _ = parser.parse_known_args()
    return args


def test_run_replica_benchmark():
    """End-to-end test of replica benchmark."""

    args = parse_args()
    with tempfile.TemporaryDirectory() as benchmark_output_dir:

        # Run benchmarking
        benchmark_args = argparse.Namespace(
            dataset_path=Path(DATASET_PATH),
            do_coverage_visualization=False,
            do_error_visualization=False,
            do_slice_animation=False,
            do_slice_visualization=False,
            do_display_error_histogram=False,
            fuse_replica_binary_path=Path(FUSE_REPLICA_BINARY_PATH),
            output_root_path=Path(benchmark_output_dir),
            kpi_namespace=None)
        main(benchmark_args)

        # Either compare kpis or write updated ones to disk
        baseline_kpis = read_json(BASELINE_KPI_PATH)
        computed_kpis = read_json(
            Path(benchmark_output_dir) / "office0" / "kpi.json")
        if args.generate_baseline:
            write_baseline_kpis(computed_kpis)
        else:
            compare_kpis(baseline_kpis, computed_kpis)


if __name__ == "__main__":
    test_run_replica_benchmark()
