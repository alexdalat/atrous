
file(GLOB SRC_FILES *.cpp *.c)
add_executable(${PROJECT_NAME} ${SRC_FILES}) # add files to exectuable

target_include_directories(${PROJECT_NAME} PUBLIC ../include)
target_link_libraries(${PROJECT_NAME} ${CONAN_LIBS})

set_target_properties( # change output paths
        ${PROJECT_NAME} PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        )