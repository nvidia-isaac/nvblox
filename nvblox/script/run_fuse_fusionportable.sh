# /bin/bash
cd build;
make && \
./executables/fuse_fusionportable \
	/Spy/dataset/mapping_results/nvblox/20220216_garden_day/ \
	--tsdf_integrator_max_integration_distance_m 70.0 \
	--color_integrator_max_integration_distance_m 30.0 \
	--num_frames 10 \
	--voxel_size 0.1 \
	--mesh_frame_subsampling 20 \
	--color_frame_subsampling -1 \
	--esdf_frame_subsampling 10 \
	--esdf_mode 1 \
	--esdf_zmin 0.5 \
	--esdf_zmax 1.0 \
	--mesh_output_path \
	/Spy/dataset/mapping_results/nvblox/20220216_garden_day_mesh_test.ply \
	--esdf_output_path \
	/Spy/dataset/mapping_results/nvblox/20220216_garden_day_esdf_test.ply \
	--obstacle_output_path \
	/Spy/dataset/mapping_results/nvblox/20220216_garden_day_obs_test.ply \

