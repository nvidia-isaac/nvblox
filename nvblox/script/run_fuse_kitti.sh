# /bin/bash
cd build;
make && \
./executables/fuse_kitti \
	/Spy/dataset/mapping_results/nvblox/2011_09_30_drive_0027_sync/ \
	--tsdf_integrator_max_integration_distance_m 70.0 \
	--color_integrator_max_integration_distance_m 30.0 \
	--num_frames 1000 \
	--voxel_size 0.1 \
	--mesh_frame_subsampling 20 \
	--mesh_output_path \
	/Spy/dataset/mapping_results/nvblox/2011_09_30_drive_0027_sync_mesh_1000_weightmethod6.ply \
	--color_frame_subsampling 1
	# -esdf_output_path \
	# /Spy/dataset/mapping_results/nvblox/2011_09_30_drive_0027_sync_esdf_1000_weightmethod1_0.1.ply \
	# -esdf_mode 1 \
	# -esdf_zmin 0.5 \
	# -esdf_zmax 1.0 \
	# -esdf_frame_subsampling 10 \
	# -obstacle_output_path \
	# /Spy/dataset/mapping_results/nvblox/2011_09_30_drive_0027_sync_obs_1000_weightmethod1_0.1.ply \

