# /bin/bash
cd build;
make && \
./executables/fuse_kitti \
	/Spy/dataset/mapping_results/nvblox/2011_09_30_drive_0027_sync/ \
	-tsdf_integrator_max_integration_distance_m 200.0 \
	-num_frames 1000 \
	-voxel_size 0.25 \
	-mesh_output_path \
	/Spy/dataset/mapping_results/nvblox/2011_09_30_drive_0027_sync_mesh_1000_weightmethod3_0.25.ply \
	-mesh_frame_subsampling 20 \
	-esdf_output_path \
	/Spy/dataset/mapping_results/nvblox/2011_09_30_drive_0027_sync_esdf_1000_weightmethod3_0.25.ply \
	-esdf_mode 1 \
	-esdf_zmin 0.5 \
	-esdf_zmax 1.0 \
	-esdf_frame_subsampling 10 \
	-obstacle_output_path \
	/Spy/dataset/mapping_results/nvblox/2011_09_30_drive_0027_sync_obs_1000_weightmethod3_0.25.ply \

