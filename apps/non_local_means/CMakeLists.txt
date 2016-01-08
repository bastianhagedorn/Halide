# Generator
halide_project(non_local_means "apps" non_local_means.cpp)

set(non_local_means_h "${CMAKE_CURRENT_BINARY_DIR}/non_local_means.h")
set(non_local_means_obj "${CMAKE_CURRENT_BINARY_DIR}/non_local_means${CMAKE_CXX_OUTPUT_EXTENSION}")

# Final executable
add_executable(filter filter.cpp ${non_local_means_h})
target_link_libraries(filter PRIVATE ${non_local_means_obj} ${PNG_LIBRARIES})
target_include_directories(filter PRIVATE "${CMAKE_CURRENT_BINARY_DIR}"
                           "${PNG_INCLUDE_DIRS}")
target_compile_definitions(filter PRIVATE ${PNG_DEFINITIONS})

if (NOT WIN32)
  target_link_libraries(filter PRIVATE dl pthread)
endif()

if (NOT MSVC)
  target_compile_options(filter PRIVATE "-std=c++11")
endif()

# FIXME: Cannot use halide_add_generator_dependency() because
# non_local_means doesn't handle the commandline args passed.
add_custom_command(OUTPUT "${non_local_means_h}" "${non_local_means_obj}"
                   COMMAND non_local_means 7 7
                   WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                   COMMENT "Generating non_local_means"
                  )
