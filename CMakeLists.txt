    cmake_minimum_required(VERSION 3.5)

    set(PROJECT_NAME AerialCarDetector)
    project(${PROJECT_NAME} VERSION 0.0.1 LANGUAGES CXX)

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS ON)


    # -------------- OpenCV  ------------------#
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})


    # -------------- Compile CUDA for FP16 inference if needed  ------------------#
    # option(USE_CUDA "Enable CUDA support" ON)
    # if (NOT APPLE AND USE_CUDA)
    #     find_package(CUDA REQUIRED)
    #     include_directories(${CUDA_INCLUDE_DIRS})
    #     add_definitions(-DUSE_CUDA)
    # else ()
    #     set(USE_CUDA OFF)
    # endif ()

    # -------------- ONNXRUNTIME  ------------------#
    add_executable(${PROJECT_NAME} main.cpp src/inference.cpp)

    target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/include )

    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${CMAKE_SOURCE_DIR}/libonnxruntime.so.1.17.1)

    # Download https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco.yaml
    # and put it in the same folder of the executable file
    configure_file(coco.yaml ${CMAKE_CURRENT_BINARY_DIR}/coco.yaml COPYONLY)

    # Copy yolov8n.onnx file to the same folder of the executable file
    configure_file(yolov8n.onnx ${CMAKE_CURRENT_BINARY_DIR}/yolov8n.onnx COPYONLY)

    # Create folder name images in the same folder of the executable file
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/images
    )
