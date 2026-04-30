#
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import sys

import torch
import matplotlib
import numpy.typing as npt
import open3d as o3d

from nvblox_torch.mapper import QueryType
from nvblox_torch.visualization import get_sphere_mesh
from nvblox_torch.examples.utils.scenes import get_single_sphere_scene_mapper
from nvblox_torch.examples.utils.interrupt_handling import run_with_graceful_interrupt

# Scene parameters
SCENE_SIZE = 8.0
VOXEL_SIZE = 0.05
CENTER = [0.0, 0.0, 0.0]
RADIUS = 2.0

# Num points on the trajectory
NUM_POINTS = 100

# Cost parameter - distance closer than which we incure distance cost
# Within this distance we incur a cost proportional to the distance (relu)
COST_START_DISTANCE = 1.0
# Cost parameter - scale for the collision/distance cost
ALPHA_COLLISION = 20.0
# Cost parameter - how far points can stretch before a cost is incurred
# For a free stretch ratio of 0.1, the points can stretch 10% beyond the initial spacing
# without incurring a cost. After that the cost increases linearly (relu).
FREE_STRETCH_RATIO = 0.2

# Optimizer parameters
LEARNING_RATE = 0.20
LEARNING_RATE_DECAY = 0.995
NUM_ITERATIONS = 500


def points_to_zero_radius_spheres(points: torch.Tensor) -> torch.Tensor:
    """Converts a tensor of points to a tensor of points with zero radius spheres."""
    sphere_radii = torch.zeros((points.shape[0], 1), device='cuda')
    query_spheres = torch.cat((points, sphere_radii), dim=1)
    return query_spheres


def get_interpoint_distances(points: torch.Tensor) -> torch.Tensor:
    """Gets the distance between consecutive points."""
    return torch.norm(torch.diff(points, dim=0), dim=1)


def distance_to_cost(distance: torch.Tensor) -> torch.Tensor:
    """Penalizes points for being too close to the objects."""
    return torch.relu(COST_START_DISTANCE - distance)


def points_to_stretching_cost(points: torch.Tensor,
                              initial_interpoint_distances: torch.Tensor) -> torch.Tensor:
    """Penalizes points from deviating from their initial spacing."""
    return torch.relu(
        get_interpoint_distances(points) - initial_interpoint_distances *
        (1.0 + FREE_STRETCH_RATIO))


def points_to_boundary_cost(points: torch.Tensor, start_point: torch.Tensor,
                            end_point: torch.Tensor) -> torch.Tensor:
    """Adds constraints that forces the first/last points to stay close to
       the start/end positions.
    """
    start_cost = torch.norm(points[0, :] - start_point)
    end_cost = torch.norm(points[-1, :] - end_point)
    return start_cost + end_cost


def get_stretching_cost_per_point(stretching_cost_per_point_pair: torch.Tensor) -> torch.Tensor:
    """Converts cost per point-pair to cost per point."""
    device = stretching_cost_per_point_pair.device
    return (torch.cat((torch.tensor([0], device=device), stretching_cost_per_point_pair)) + \
            torch.cat((stretching_cost_per_point_pair, torch.tensor([0], device=device)))) / 2.0


def convert_cost_to_color(cost_per_point: torch.Tensor,
                          max_cost_for_coloring: torch.Tensor) -> npt.NDArray:
    """Converts per-point cost to a color."""
    scaled_cost_per_point = torch.clamp(cost_per_point / max_cost_for_coloring, min=0.0, max=1.0)
    colors = matplotlib.cm.plasma(scaled_cost_per_point.detach().cpu().numpy())[:, :3]
    return colors


class Path(torch.nn.Module):
    """An optimizable path represented as an Nx3 tensor of points.

    The path contains the points in a torch.nn.Module so that
    we can use the optimizer to optimize it.
    """

    def __init__(self, points: torch.Tensor):
        super().__init__()
        self.points = torch.nn.Parameter(points)

    def forward(self) -> torch.Tensor:
        return self.points


def main(visualize: bool = True) -> int:
    """Main function for the trajectory optimization example."""

    # CUDA les go.
    device = 'cuda'

    # Trajectory start and end points
    start_point = torch.tensor([-SCENE_SIZE / 2 - 1.0, -SCENE_SIZE / 2, -SCENE_SIZE / 2],
                               device=device)
    end_point = torch.tensor([SCENE_SIZE / 2, SCENE_SIZE / 2, SCENE_SIZE / 2], device=device)

    # Get a scene containing a single sphere
    mapper = get_single_sphere_scene_mapper(
        scene_size_m=SCENE_SIZE,
        voxel_size_m=VOXEL_SIZE,
        center=CENTER,
        radius_m=RADIUS,
    )

    # Initial points are just uniformly spaced along a line joining start and end points
    initial_points = (end_point - start_point).repeat(NUM_POINTS, 1) * torch.linspace(
        0.0, 1.0, NUM_POINTS, device=device).repeat(3, 1).T + start_point.repeat(NUM_POINTS, 1)
    initial_interpoint_distances = get_interpoint_distances(initial_points).detach()

    # Create an optimizable path
    path = Path(initial_points)

    # Optimizer
    optimizer = torch.optim.Adam(path.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LEARNING_RATE_DECAY)

    # Visualizer
    if visualize:
        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window('nvblox Path Optimization')

    # Get a mesh of the environment
    mapper.update_color_mesh()
    environment_mesh_o3d = mapper.get_color_mesh().to_open3d()
    environment_mesh_o3d.compute_vertex_normals()

    # Optimize the path
    max_cost_for_coloring = None
    for i in range(NUM_ITERATIONS):

        # Forward pass
        points = path()

        # Cost
        query_spheres = points_to_zero_radius_spheres(points)
        distances = mapper.query_differentiable_layer(QueryType.ESDF, query_spheres)
        stretching_cost_per_point_pair = points_to_stretching_cost(points,
                                                                   initial_interpoint_distances)
        collision_cost_per_point = ALPHA_COLLISION * distance_to_cost(distances) / NUM_POINTS
        stretching_cost = torch.sum(stretching_cost_per_point_pair)
        collision_cost = torch.sum(collision_cost_per_point)
        boundary_cost = points_to_boundary_cost(points, start_point, end_point)
        cost = stretching_cost + collision_cost + boundary_cost
        print(f'Iteration {i} / {NUM_ITERATIONS}: '
              f'Stretching cost: {stretching_cost:0.2f}, '
              f'Collision cost: {collision_cost:0.2f}, '
              f'Boundary cost: {boundary_cost:0.2f}')

        # Backward pass
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        # Visualize
        if visualize:
            visualizer.clear_geometries()
            # Cost to color
            stretching_cost_per_point = get_stretching_cost_per_point(
                stretching_cost_per_point_pair)
            cost_per_point = stretching_cost_per_point + collision_cost_per_point
            if max_cost_for_coloring is None:
                max_cost_for_coloring = 0.25 * torch.max(cost_per_point)
            colors = convert_cost_to_color(cost_per_point, max_cost_for_coloring)
            # Add to visualizer
            sphere_meshes = o3d.geometry.TriangleMesh()
            for point, color in zip(points, colors):
                sphere_meshes += get_sphere_mesh(point.detach(), 0.05, color)
            visualizer.add_geometry(sphere_meshes)
            visualizer.add_geometry(environment_mesh_o3d)
            visualizer.poll_events()
            visualizer.update_renderer()

    print('Done')
    if visualize:
        print('Close the visualize window to continue...')
        visualizer.run()
        visualizer.destroy_window()

    return 0


if __name__ == '__main__':
    sys.exit(run_with_graceful_interrupt(main, visualize=True))
