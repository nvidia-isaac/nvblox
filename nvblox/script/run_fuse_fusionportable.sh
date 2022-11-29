# /bin/bash
cd build;
make && \
./executables/fuse_fusionportable \
	/Spy/dataset/mapping_results/nvblox/20220216_garden_day/ \
	-tsdf_integrator_max_integration_distance_m 70.0 \
	-num_frames 2000 \
	-voxel_size 0.1 \
	-tsdf_integrator_truncation_distance_vox 4.0 \
	-mesh_output_path \
	/Spy/dataset/mapping_results/nvblox/20220216_garden_day_mesh_2000_weightmethod6.ply \
	# -mesh_frame_subsampling 20 \
	# -esdf_output_path \
	# /Spy/dataset/mapping_results/nvblox/20221126_lab_static_esdf_test.ply \
	# -esdf_mode 1 \
	# -esdf_zmin 0.5 \
	# -esdf_zmax 1.0 \
	# -esdf_frame_subsampling 10 \
	# -obstacle_output_path \
	# /Spy/dataset/mapping_results/nvblox/20221126_lab_static_obs_test.ply \

