include(opentrack-opencv)
find_package(OpenCV QUIET)
find_package(Eigen3 QUIET)
set(SDK_ONNX_LIBPATH "" CACHE FILEPATH "Full path of onnx library")

if(OpenCV_FOUND AND EIGEN3_FOUND AND SDK_ONNX_LIBPATH)
    set(modules "${SDK_ONNX_LIBPATH}" opencv_imgproc opencv_core opencv_imgcodecs opencv_highgui)

    get_filename_component(ONNX_INCLUDE_DIR "${SDK_ONNX_LIBPATH}" DIRECTORY)
    get_filename_component(ONNX_INCLUDE_DIR "${ONNX_INCLUDE_DIR}" ABSOLUTE)
    set(ONNX_INCLUDE_DIR "${ONNX_INCLUDE_DIR}/../include")
    # message (STATUS "ONNX_INCLUDE_DIR is ${ONNX_INCLUDE_DIR}")

    set(includes ${EIGEN3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} "${ONNX_INCLUDE_DIR}")

    # try_compile(tracker-aruco_has-working-abi "${CMAKE_CURRENT_BINARY_DIR}"
    #                    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/compile-test/abi.cpp"
    #                    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${dir}"
    #                                "-DCXX_STANDARD=17" "-DCXX_STANDARD_REQUIRED=1"
    #                    OUTPUT_VARIABLE krap)
    # if(NOT tracker-aruco_has-working-abi)
    #     message(FATAL_ERROR "${krap}\n" "Must use Aruco fork from <https://github.com/opentrack/aruco>")
    # endif()

    # maybe_add_static_define()
    # otr_install_lib("${SDK_ARUCO_LIBPATH}" "${opentrack-libexec}")

    otr_module(tracker-neuralnet)
    target_include_directories(${self} SYSTEM PUBLIC ${includes})
    target_link_libraries(${self} opentrack-cv ${modules})

    install(
        FILES "models/head-localizer.onnx" "models/head-pose.onnx"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/${opentrack-i18n}/../models/"
        PERMISSIONS ${opentrack-perms-file})
endif()