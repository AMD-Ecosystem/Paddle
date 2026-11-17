function(add_thrust_patches_if_necessary)
  # The vendored patches/thrust overlay is a backport of thrust::shuffle for old thrust
  # releases that lacked it. On ROCm the nvcc "--run" probe below can never succeed (there
  # is no working nvcc in a HIP build), so it would ALWAYS conclude shuffle is missing and
  # overlay the backport -- which COLLIDES with rocThrust's own native thrust::shuffle
  # (ROCm ships it), producing __thrust_exec_check_disable__ / iterator_value_t errors.
  # rocThrust provides shuffle natively, so detect the real header and skip the overlay.
  if(WITH_ROCM)
    find_file(
      ROCM_THRUST_SHUFFLE_H thrust/shuffle.h
      PATHS "${ROCM_PATH}/include" "$ENV{ROCM_PATH}/include" /opt/rocm/include
      NO_DEFAULT_PATH)
    if(ROCM_THRUST_SHUFFLE_H)
      message(STATUS "rocThrust provides thrust::shuffle natively (${ROCM_THRUST_SHUFFLE_H}); skipping vendored thrust patches")
      return()
    endif()
    message(STATUS "rocThrust thrust/shuffle.h not found; falling back to vendored thrust patches")
    set(thrust_patches "${PADDLE_SOURCE_DIR}/patches/thrust")
    include_directories(${thrust_patches})
    return()
  endif()

  set(thrust_detect_file ${PROJECT_BINARY_DIR}/detect_thrust.cu)
  file(
    WRITE ${thrust_detect_file}
    ""
    "#include \"thrust/version.h\"\n"
    "#include \"thrust/shuffle.h\"\n"
    "#include \"stdio.h\"\n"
    "int main() {\n"
    "  int version = THRUST_VERSION;\n"
    "  printf(\"%d\", version);\n"
    "  return 0;\n"
    "}\n")

  execute_process(
    COMMAND "${CUDA_NVCC_EXECUTABLE}" "--run" "${thrust_detect_file}"
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/CMakeFiles/"
    RESULT_VARIABLE nvcc_res
    ERROR_QUIET)
  if(NOT nvcc_res EQUAL 0)
    set(thrust_patches "${PADDLE_SOURCE_DIR}/patches/thrust")
    message(STATUS "Add thrust patches: ${thrust_patches}")
    include_directories(${thrust_patches})
  endif()
endfunction()

add_thrust_patches_if_necessary()
