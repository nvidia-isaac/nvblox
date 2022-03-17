add_executable(layer_cake_interface_dynamic 
  main_dynamic.cpp
  src/cuda/user_defined_block.cu
)
target_include_directories(layer_cake_interface_dynamic PUBLIC
include
)
target_link_libraries(layer_cake_interface_dynamic nvblox_lib)

add_executable(layer_cake_interface_static 
  main_static.cpp
)
target_include_directories(layer_cake_interface_static PUBLIC
include
)
target_link_libraries(layer_cake_interface_static nvblox_lib)

add_executable(layer_cake_interface_type_erasure 
  main_type_erasure.cpp
)
target_include_directories(layer_cake_interface_type_erasure PUBLIC
include
)
target_link_libraries(layer_cake_interface_type_erasure nvblox_lib)

