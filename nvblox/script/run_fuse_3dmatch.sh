# /bin/bash
cd build;
make && \
./executables/fuse_3dmatch \
	/Spy/dataset/3DMatch/sun3d-mit_76_studyroom-76-1studyroom2 \
	--voxel_size 0.05 \
	--mesh_output_path /Spy/dataset/3DMatch/sun3d-mit_76_studyroom-76-1studyroom2/mesh_test.ply \
	--esdf_frame_subsampling 3000 \

