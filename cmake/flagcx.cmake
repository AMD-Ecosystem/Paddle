set(CMAKE_FIND_DEBUG_MODE ON)
# flagcx.cmake
if(NOT WITH_FLAGCX)
  return()
endif()

if(WITH_FLAGCX)
  set(FLAGCX_ROOT
      $ENV{FLAGCX_ROOT}
      CACHE PATH "FLAGCX_ROOT")
  message(STATUS "FLAGCX_ROOT is ${FLAGCX_ROOT}")
  find_path(
    FLAGCX_INCLUDE_DIR flagcx.h
    PATHS ${FLAGCX_ROOT}/flagcx/include
    NO_DEFAULT_PATH)
  message(STATUS "FLAGCX_INCLUDE_DIR is ${FLAGCX_INCLUDE_DIR}")
  include_directories(SYSTEM ${FLAGCX_INCLUDE_DIR})
  set(FLAGCX_LIB
      "${FLAGCX_ROOT}/build/lib/libflagcx.so"
      CACHE FILEPATH "flagcx library." FORCE)
  generate_dummy_static_lib(LIB_NAME "flagcx" GENERATOR "flagcx.cmake")
  target_link_libraries(flagcx ${FLAGCX_LIB})
  message(STATUS "FLAGCX_LIB is ${FLAGCX_LIB}")
endif()
