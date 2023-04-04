# /bin/bash
cd build;
make && \
./executables/fuse_kitti \
	/Spy/dataset/mapping_results/nvblox/2011_09_30_drive_0027_sync/ \
	--tsdf_integrator_max_integration_distance_m 70.0 \
	--color_integrator_max_integration_distance_m 30.0 \
	--num_frames 100 \
	--voxel_size 0.2 \
	--mesh_frame_subsampling 20 \
	--color_frame_subsampling 1 \
	--esdf_frame_subsampling 10 \
	--esdf_mode 1 \
	--esdf_zmin 0.5 \
	--esdf_zmax 1.0 \
	--mesh_output_path \
	/Spy/dataset/mapping_results/nvblox/2011_09_30_drive_0027_sync_mesh_100_weightmethod6.ply \
	--esdf_output_path \
	/Spy/dataset/mapping_results/nvblox/2011_09_30_drive_0027_sync_esdf_100_weightmethod6.ply \
	--obstacle_output_path \
	/Spy/dataset/mapping_results/nvblox/2011_09_30_drive_0027_sync_obs_100_weightmethod6.ply \

