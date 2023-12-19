# Copyright 2023 NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Script for running end-to-end replica benchmarking"""

from typing import Optional
import argparse
from pathlib import Path
import json

# pylint:disable=line-too-long
from nvblox_evaluation.replica_reconstruction_evaluation.replica_reconstruction import replica_reconstruction
from nvblox_evaluation.replica_reconstruction_evaluation.replica_surface_evaluation import evaluate_mesh
from nvblox_evaluation.replica_reconstruction_evaluation.replica_esdf_evaluation import evaluate_esdf


def parse_args():
    """Return parsed args."""
    parser = argparse.ArgumentParser(
        description="""Reconstruct a mesh from the replica dataset.""")
    parser.add_argument("--dataset_path",
                        type=Path,
                        required=True,
                        help="Path to the dataset root folder.")
    parser.add_argument("--output_root_path",
                        type=Path,
                        required=True,
                        help="Path to the directory in which to save results.")
    parser.add_argument(
        "--do_visualize_error_mesh",
        dest="do_error_visualization",
        action='store_const',
        const=True,
        default=False,
        help="Flag indicating if we should visualize the error mesh.")
    parser.add_argument(
        "--do_visualize_slice",
        dest="do_slice_visualization",
        action='store_const',
        const=True,
        default=False,
        help="Flag indicating if we should visualize an ESDF slice in 3D.")
    parser.add_argument(
        "--do_animate_slice",
        dest="do_slice_animation",
        action='store_const',
        const=True,
        default=False,
        help="Flag indicating if we should animate an ESDF slice.")
    parser.add_argument(
        "--do_coverage_visualization",
        dest="do_coverage_visualization",
        action='store_const',
        const=True,
        default=False,
        help="Flag indicating if we should display the coverage mesh.")
    parser.add_argument(
        "--do_display_error_histogram",
        dest="do_display_error_histogram",
        action='store_const',
        const=True,
        default=False,
        help="Flag indicating if we should display the error histogram.")
    parser.add_argument(
        "--fuse_replica_binary_path",
        type=Path,
        help=
        "Path to the fuse_replica binary. If not passed we search the standard build folder location."
    )
    parser.add_argument(
        "--kpi_namespace",
        type=str,
        help=
        "If passed, the KPIs in the output JSON will be preceeded by this string."
    )

    args = parser.parse_args()

    return args


def get_output_file_path(output_root_path, file_name):
    """Check if a given file exists in the output dir and return its path."""
    file_path = Path(output_root_path) / "office0" / file_name
    if not file_path.is_file():
        raise FileNotFoundError(f"Output file not found::{file_path}")

    return file_path


def get_groundtruth_mesh_path(dataset_path):
    """Check if the ground truth mesh file exists and return its path."""
    sequence_name = dataset_path.stem
    groundtruth_mesh_path = Path(
        dataset_path) / ".." / f"{sequence_name}_mesh.ply"

    if not groundtruth_mesh_path.is_file():
        raise FileNotFoundError(
            f"Groundtruth mesh does not exist:{groundtruth_mesh_path}")

    return groundtruth_mesh_path


def write_merged_kpis(dataset_path,
                      output_root_path,
                      kpi_namespace: Optional[str] = None):
    """Write a dict with merged kpis."""
    merged_kpis = {}
    sequence_name = dataset_path.stem
    for kpi_file in [
            "esdf_error_statistics.json", "surface_error_statistics.json",
            "timing.json"
    ]:
        with open(Path(output_root_path) / sequence_name / kpi_file,
                  "r",
                  encoding="utf-8") as fp:
            merged_kpis.update(json.load(fp))

    # Push KPIs into a namespace if requested
    if kpi_namespace:
        kpi_namespace = kpi_namespace.rstrip('/')
        merged_kpis = {
            (kpi_namespace + '/' + key): val
            for key, val in merged_kpis.items()
        }

    merged_kpis_path = Path(output_root_path / sequence_name / "kpi.json")
    with open(merged_kpis_path, "w", encoding="utf-8") as fp:
        json.dump(merged_kpis, fp, indent=4)

    print(f"Wrote kpis to: {merged_kpis_path}")
    print(json.dumps(merged_kpis, indent=4))


def main(args):
    """Main entry function."""

    # Create reconstruction
    replica_reconstruction(
        args.dataset_path,
        output_root_path=args.output_root_path,
        fuse_replica_binary_path=args.fuse_replica_binary_path)

    groundtruth_mesh_path = get_groundtruth_mesh_path(args.dataset_path)

    # Evaluate surface reconstruction
    reconstructed_mesh_path = get_output_file_path(args.output_root_path,
                                                   "reconstructed_mesh.ply")

    evaluate_mesh(reconstructed_mesh_path,
                  groundtruth_mesh_path,
                  output_root_path=args.output_root_path,
                  do_error_visualization=args.do_error_visualization,
                  do_coverage_visualization=args.do_coverage_visualization)

    # Evaluate ESDF reconstruction
    reconstructed_esdf_path = get_output_file_path(args.output_root_path,
                                                   "reconstructed_esdf.ply")

    evaluate_esdf(reconstructed_esdf_path,
                  groundtruth_mesh_path,
                  output_root_path=args.output_root_path,
                  reconstructed_mesh_path=reconstructed_mesh_path,
                  do_slice_visualization=args.do_slice_visualization,
                  do_slice_animation=args.do_slice_animation,
                  do_display_error_histogram=args.do_display_error_histogram)

    # Merge kpi files into one
    write_merged_kpis(args.dataset_path, args.output_root_path,
                      args.kpi_namespace)


if __name__ == "__main__":
    main(parse_args())
