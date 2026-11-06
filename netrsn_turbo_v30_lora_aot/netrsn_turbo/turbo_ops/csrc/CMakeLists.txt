cmake_minimum_required(VERSION 3.19)
project(turbo_ops_lora LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
if(NOT SOC_VERSION MATCHES "^(ascend910b[1-4](-1|c)?|ascend910_93[0-9][0-9])$")
  message(FATAL_ERROR "MoE LoRA direct kernels support A2/A3 only; got ${SOC_VERSION}")
endif()
if(NOT ASCEND_CANN_PACKAGE_PATH)
  set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
endif()
set(ASCEND_HOME_PATH "${ASCEND_CANN_PACKAGE_PATH}")
set(RUN_MODE npu)
foreach(location tools/tikcpp/ascendc_kernel_cmake compiler/tikcpp/ascendc_kernel_cmake ascendc_devkit/tikcpp/samples/cmake)
  if(EXISTS "${ASCEND_CANN_PACKAGE_PATH}/${location}/ascendc.cmake")
    set(ASCENDC_CMAKE_DIR "${ASCEND_CANN_PACKAGE_PATH}/${location}")
    break()
  endif()
endforeach()
if(NOT ASCENDC_CMAKE_DIR)
  message(FATAL_ERROR "CANN AscendC development CMake package is missing")
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(COMMAND "${Python3_EXECUTABLE}" -c
  "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_PREFIX}")
find_package(Torch REQUIRED)
execute_process(COMMAND "${Python3_EXECUTABLE}" -c
  "import pathlib,torch_npu; print(pathlib.Path(torch_npu.__file__).parent)"
  OUTPUT_VARIABLE TORCH_NPU_PATH OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include("${ASCENDC_CMAKE_DIR}/ascendc.cmake")
set(KERNEL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/kernels/lora")
file(GLOB KERNEL_SOURCES CONFIGURE_DEPENDS "${KERNEL_ROOT}/*.cpp")
if(NOT KERNEL_SOURCES)
  message(FATAL_ERROR "MoE LoRA kernel sources missing in ${KERNEL_ROOT}")
endif()
ascendc_library(turbo_ops_lora_kernels SHARED ${KERNEL_SOURCES})
file(GLOB_RECURSE HOST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/lora/*.cpp")
add_library(_C SHARED ${HOST_SOURCES})
set_target_properties(_C PROPERTIES PREFIX "" INSTALL_RPATH "$ORIGIN")
target_include_directories(_C PRIVATE
  ${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS}
  "${TORCH_NPU_PATH}/include" "${ASCEND_HOME_PATH}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/lora" "${KERNEL_ROOT}")
target_link_directories(_C PRIVATE "${TORCH_NPU_PATH}/lib" "${ASCEND_HOME_PATH}/lib64")
target_link_libraries(_C PRIVATE turbo_ops_lora_kernels ${TORCH_LIBRARIES}
  torch_npu ascendcl opapi dl)
# CANN's imported build helpers can suppress CMake's generated rpath flags.
# Keep the package-local dependency resolvable without LD_LIBRARY_PATH edits.
target_link_options(_C PRIVATE "-Wl,-rpath,$ORIGIN")
install(TARGETS _C turbo_ops_lora_kernels LIBRARY DESTINATION . COMPONENT turbo_ops)
