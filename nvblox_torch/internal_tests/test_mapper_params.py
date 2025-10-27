#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
#%%

from typing import List, Any, Callable, Set
import inspect
import pathlib
import re

from nvblox_torch.mapper import Mapper
from nvblox_torch.mapper_params import (ProjectiveIntegratorParams, MeshIntegratorParams,
                                        DecayIntegratorBaseParams, TsdfDecayIntegratorParams,
                                        OccupancyDecayIntegratorParams, EsdfIntegratorParams,
                                        MapperParams, BlockMemoryPoolParams, ViewCalculatorParams)
from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType

NVBLOX_TORCH_DIR = pathlib.Path(pathlib.Path(__file__).parents[1])
NVBLOX_CORE_DIR = NVBLOX_TORCH_DIR.parents[0] / 'nvblox'
PROJECTIVE_INTEGRATOR_PARAMS_PATH = NVBLOX_CORE_DIR \
    / 'include' / 'nvblox' / 'integrators' / 'projective_integrator_params.h'
MESH_INTEGRATOR_PARAMS_PATH = NVBLOX_CORE_DIR \
    / 'include' / 'nvblox' / 'mesh' / 'mesh_integrator_params.h'
DECAY_INTEGRATOR_BASE_PARAMS_PATH = NVBLOX_CORE_DIR \
    / 'include' / 'nvblox' / 'integrators' / 'internal' / 'decay_integrator_base_params.h'
TSDF_DECAY_INTEGRATOR_PARAMS_PATH = NVBLOX_CORE_DIR \
    / 'include' / 'nvblox' / 'integrators' / 'tsdf_decay_integrator_params.h'
OCCUPANCY_DECAY_INTEGRATOR_PARAMS_PATH = NVBLOX_CORE_DIR \
    / 'include' / 'nvblox' / 'integrators' / 'occupancy_decay_integrator_params.h'
ESDF_DECAY_INTEGRATOR_PARAMS_PATH = NVBLOX_CORE_DIR \
    / 'include' / 'nvblox' / 'integrators' / 'esdf_integrator_params.h'
VIEW_CALCULATOR_PARAMS_PATH = NVBLOX_CORE_DIR \
    / 'include' / 'nvblox' / 'integrators' / 'view_calculator_params.h'


def get_attributes(mapper_params: Any) -> List[str]:
    attribute_names = [
        attribute for attribute in dir(mapper_params) if not attribute.startswith('_')
    ]
    non_method_attribute_names = []
    for attribute_name in attribute_names:
        attr = getattr(mapper_params, attribute_name)
        if not inspect.ismethod(attr):
            non_method_attribute_names.append(attribute_name)
    return non_method_attribute_names


# These are values we have to hardcode because they're enums which cant
# just be set to a generated integer value
special_test_values = {
    'projective_integrator_weighting_mode': 'kInverseSquareTsdfDistancePenalty',
    'workspace_bounds_type': 'kUnbounded'
}


def get_test_value(attribute_name: str, idx: int, params: Any) -> Any:
    if attribute_name in special_test_values:
        return special_test_values[attribute_name]
    else:
        type_converter = type(getattr(params, attribute_name))
        return type_converter(idx)


def assert_getting_and_setting(parameter_class: Callable) -> None:
    params = parameter_class()
    # Gather the attributes from the object
    attribute_names = get_attributes(params)
    print(f'Parameter class has attributes: {attribute_names}')

    # Loop through all the attributes and set them to a random value (the index)
    # We also convert the index into the type of the attribute (we got errors
    # using the raw index).
    for idx, attribute_name in enumerate(attribute_names):
        test_value = get_test_value(attribute_name, idx, params)
        print(f'Setting attribute {attribute_name} = {test_value}')
        setattr(params, attribute_name, test_value)

    # Loop through all the attributes and check them
    for idx, attribute_name in enumerate(attribute_names):
        value = getattr(params, attribute_name)
        print(f'Getting attribute {attribute_name} = {value}')
        test_value = get_test_value(attribute_name, idx, params)
        assert value == test_value


def get_parameter_names_from_cpp(header_file_path: pathlib.Path) -> List[str]:
    param_names = []
    with open(header_file_path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            if 'Param<' in line and 'constexpr' not in line:
                line.split()
                param_name = re.split(r'[{>]+', line)[1].strip()
                param_names.append(param_name)
    return param_names


def get_unwrapped_parameters(parameter_class: Callable, header_file_path: pathlib.Path) -> Set[str]:
    parameter_names_cpp = get_parameter_names_from_cpp(header_file_path)
    mapper_params = parameter_class()
    attribute_names = get_attributes(mapper_params)
    unwrapped = set(parameter_names_cpp) - set(attribute_names)
    return unwrapped


def assert_wrapped_exist(parameter_class: Callable, header_file_path: pathlib.Path) -> None:
    # Tests that each of the parameters/attributes existing in python
    # actually exist in the underlying cpp struct.
    parameter_names_cpp = get_parameter_names_from_cpp(header_file_path)
    mapper_params = parameter_class()
    attribute_names = get_attributes(mapper_params)
    wrapped_non_existing = set(attribute_names) - set(parameter_names_cpp)
    assert len(
        wrapped_non_existing
    ) == 0, f'Some attributes correspond to non-existing parameters: {wrapped_non_existing}'
    # Print the correctly wrapped for encouragement
    correctly_wrapped = set(parameter_names_cpp).intersection(set(attribute_names))
    print(f'Correctly wrapped the following params: {list(correctly_wrapped)}')
    unwrapped = get_unwrapped_parameters(parameter_class, header_file_path)
    print(f'The following parameters are unwrapped: {list(unwrapped)}')


def assert_no_unwrapped_params(parameter_class: Callable, header_file_path: pathlib.Path) -> None:
    unwrapped = get_unwrapped_parameters(parameter_class, header_file_path)
    print(f'The following parameters are unwrapped: {list(unwrapped)}')
    assert len(unwrapped) == 0, 'Some MapperParams are unwrapped in python'


def test_getting_and_setting() -> None:
    assert_getting_and_setting(ProjectiveIntegratorParams)
    assert_getting_and_setting(MeshIntegratorParams)
    assert_getting_and_setting(DecayIntegratorBaseParams)
    assert_getting_and_setting(TsdfDecayIntegratorParams)
    assert_getting_and_setting(EsdfIntegratorParams)
    assert_getting_and_setting(ViewCalculatorParams)
    assert_getting_and_setting(BlockMemoryPoolParams)


def test_all_wrapped() -> None:

    assert_wrapped_exist(ProjectiveIntegratorParams, PROJECTIVE_INTEGRATOR_PARAMS_PATH)
    assert_no_unwrapped_params(ProjectiveIntegratorParams, PROJECTIVE_INTEGRATOR_PARAMS_PATH)

    assert_wrapped_exist(MeshIntegratorParams, MESH_INTEGRATOR_PARAMS_PATH)
    assert_no_unwrapped_params(MeshIntegratorParams, MESH_INTEGRATOR_PARAMS_PATH)

    assert_wrapped_exist(DecayIntegratorBaseParams, DECAY_INTEGRATOR_BASE_PARAMS_PATH)
    assert_no_unwrapped_params(DecayIntegratorBaseParams, DECAY_INTEGRATOR_BASE_PARAMS_PATH)

    assert_wrapped_exist(TsdfDecayIntegratorParams, TSDF_DECAY_INTEGRATOR_PARAMS_PATH)
    assert_no_unwrapped_params(TsdfDecayIntegratorParams, TSDF_DECAY_INTEGRATOR_PARAMS_PATH)

    assert_wrapped_exist(OccupancyDecayIntegratorParams, OCCUPANCY_DECAY_INTEGRATOR_PARAMS_PATH)
    assert_no_unwrapped_params(OccupancyDecayIntegratorParams,
                               OCCUPANCY_DECAY_INTEGRATOR_PARAMS_PATH)

    assert_wrapped_exist(EsdfIntegratorParams, ESDF_DECAY_INTEGRATOR_PARAMS_PATH)
    assert_no_unwrapped_params(EsdfIntegratorParams, ESDF_DECAY_INTEGRATOR_PARAMS_PATH)

    assert_wrapped_exist(ViewCalculatorParams, VIEW_CALCULATOR_PARAMS_PATH)
    assert_no_unwrapped_params(ViewCalculatorParams, VIEW_CALCULATOR_PARAMS_PATH)


def test_mapper_params_sub_params() -> None:
    mapper_params = MapperParams()

    # Projective integrator sub-params test
    projective_integrator_params = mapper_params.get_projective_integrator_params()
    projective_integrator_params.projective_integrator_max_weight = 6.0
    mapper_params.set_projective_integrator_params(projective_integrator_params)
    # projective_integrator_params_2 = mapper_params.get_projective_integrator_params()
    assert projective_integrator_params.projective_integrator_max_weight == 6.0

    # Get and set all subparams
    getter_names = [method for method in dir(mapper_params) if method[:len('get_')] == 'get_']
    for getter_name in getter_names:
        # Get
        sub_params = getattr(mapper_params, getter_name)()
        # Set
        setter_name = 'set_' + getter_name[len('get_'):]
        getattr(mapper_params, setter_name)(sub_params)


def test_parameter_setting_in_mapper() -> None:
    test_value = 1.0

    # Create a new parameter struct
    projective_integrator_params = ProjectiveIntegratorParams()
    projective_integrator_params.projective_integrator_max_integration_distance_m = 1.0
    mapper_params = MapperParams()
    mapper_params.set_projective_integrator_params(projective_integrator_params)

    # Recreate a mapper with modified/non-default params
    voxel_size_m = 0.1
    new_mapper = Mapper(voxel_sizes_m=[voxel_size_m],
                        integrator_types=[ProjectiveIntegratorType.TSDF],
                        mapper_parameters=mapper_params)
    new_mapper_params = new_mapper.params()

    new_projective_integrator_params = new_mapper_params.get_projective_integrator_params()
    new_value = new_projective_integrator_params.projective_integrator_max_integration_distance_m
    assert new_value == test_value
