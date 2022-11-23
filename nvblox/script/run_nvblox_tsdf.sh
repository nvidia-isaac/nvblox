# /bin/bash
cd build;
make && \
./executables/fuse_fusionportable \
	/Spy/dataset/mapping_results/nvblox/20220216_garden_day/ \
	-tsdf_integrator_max_integration_distance_m 70.0 \
	-num_frames 500 \
	-voxel_size 0.1 \
	-mesh_output_path /Spy/dataset/mapping_results/nvblox/20220216_garden_day_mesh_test.ply \
	-mesh_frame_subsampling 500 \
	-esdf_output_path /Spy/dataset/mapping_results/nvblox/20220216_garden_day_esdf_test.ply \
	-esdf_mode 1 \
	-esdf_zmin 0.5 \
	-esdf_zmax 1.0 \
	-obstacle_output_path /Spy/dataset/mapping_results/nvblox/20220216_garden_day_obs_test.ply \

