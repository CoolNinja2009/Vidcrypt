cmake_minimum_required(VERSION 3.16)
project(vidcrypt VERSION 4.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(ENABLE_PROFILING "Enable per-stage profiling instrumentation" OFF)
option(ENABLE_AVX2     "Enable AVX2 SIMD paths for calibration reader" ON)
option(USE_CUDA        "Enable CUDA-based GPU acceleration" OFF)
option(ENABLE_LOGGING  "Enable detailed session logging to log/ directory" OFF)

if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:C>:/W3>
        $<$<COMPILE_LANGUAGE:C>:/utf-8>
    )
else()
    add_compile_options(
        $<$<COMPILE_LANGUAGE:C>:-Wall>
        $<$<COMPILE_LANGUAGE:C>:-Wextra>
        $<$<COMPILE_LANGUAGE:C>:-Wpedantic>
        $<$<COMPILE_LANGUAGE:C>:-Wconversion>
        $<$<COMPILE_LANGUAGE:C>:-Wno-unused-parameter>
    )
endif()

if(ENABLE_PROFILING)
    add_compile_definitions(ENABLE_PROFILING=1)
endif()

if(ENABLE_LOGGING)
    add_compile_definitions(ENABLE_LOGGING=1)
endif()

# If AVX2 is enabled, apply flags ONLY to files with AVX2 paths (not globally).
# Global /arch:AVX2 on MSVC forces VEX-prefixed instructions everywhere,
# adding ~10% overhead to scalar code that doesn't benefit from SIMD.
if(ENABLE_AVX2)
    if(MSVC)
        set_source_files_properties(
            src/simd_decode.c
            src/bitstream.c
            PROPERTIES COMPILE_FLAGS "/arch:AVX2")
        # /arch:AVX2 on MSVC implies __AVX2__ automatically
    else()
        set_source_files_properties(
            src/simd_decode.c
            src/bitstream.c
            PROPERTIES COMPILE_FLAGS "-mavx2 -D__AVX2__")
    endif()
endif()

set(VIDCRYPT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
include_directories(${VIDCRYPT_INCLUDE_DIR})

set(CORE_SOURCES
    src/backend.c
    src/crc16.c
    src/bitstream.c
    src/calibration.c
    src/reedsolomon.c
    src/simd_decode.c
    src/framegen.c
    src/framedecode.c
    src/threadpool.c
    src/encoder.c
    src/decoder.c
    src/sha256.c
    src/ffmpeg_pipe.c
    src/profiling.c
    src/logutil.c
    src/gpu_backend.c
    src/libav_decoder.c
)

add_library(vidcrypt STATIC ${CORE_SOURCES})
target_include_directories(vidcrypt PUBLIC ${VIDCRYPT_INCLUDE_DIR})

find_package(Threads REQUIRED)
target_link_libraries(vidcrypt PRIVATE Threads::Threads)
target_compile_definitions(vidcrypt PRIVATE _GNU_SOURCE)

if(UNIX)
    target_link_libraries(vidcrypt PRIVATE m)
endif()

# ── CUDA GPU Acceleration ─────────────────────────────────────────────
if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    # Target architectures: SASS for all consumer GPU generations,
    # PTX virtual arch for forward compatibility on newer CUDA versions.
    # Override with: -DVIDCRYPT_CUDA_ARCHS="61;75;86;89;120"
    if(NOT DEFINED VIDCRYPT_CUDA_ARCHS)
        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
            set(VIDCRYPT_CUDA_ARCHS "61;75;86;89;120"
                CACHE STRING "CUDA target architectures: 61=Pascal 75=Turing 86=Ampere 89=Ada 120=Blackwell")
        elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.8")
            set(VIDCRYPT_CUDA_ARCHS "61;75;86;89;90-virtual"
                CACHE STRING "CUDA target architectures (90-virtual=PTX for Blackwell JIT)")
        else()
            set(VIDCRYPT_CUDA_ARCHS "61;75;86;90-virtual"
                CACHE STRING "CUDA target architectures")
        endif()
    endif()
    message(STATUS "CUDA architectures: ${VIDCRYPT_CUDA_ARCHS}")
    set(GPU_SOURCES
        src/gpu_kernels.cu
        src/gpu_presets.c
        src/gpu_nvenc.c
        src/ffmpeg_hwdecoder.c
    )

    target_sources(vidcrypt PRIVATE ${GPU_SOURCES})
    # Apply USE_CUDA per-source (not globally) so that files without
    # CUDA dependency compile as plain C.
    set_source_files_properties(
        src/gpu_backend.c
        src/gpu_nvenc.c
        src/ffmpeg_hwdecoder.c
        src/decoder.c
    PROPERTIES COMPILE_DEFINITIONS "USE_CUDA=1")
    set_source_files_properties(src/gpu_kernels.cu PROPERTIES COMPILE_DEFINITIONS "USE_CUDA=1")
    # gpu_presets.c has no CUDA dependency — compiled as-is.
    set_property(TARGET vidcrypt PROPERTY CUDA_ARCHITECTURES "${VIDCRYPT_CUDA_ARCHS}")

    # Use fast math for CUDA threshold kernels (no precision requirement)
    set_source_files_properties(src/gpu_kernels.cu PROPERTIES COMPILE_FLAGS "--use_fast_math")

    # NVENC requires nvEncodeAPI.h from NVIDIA Video Codec SDK.
    # Include it only if available; otherwise compile stubs.
    find_path(NVENC_INCLUDE_DIR nvEncodeAPI.h
        HINTS
            "${CUDAToolkit_INCLUDE_DIRS}"
            "$ENV{CUDA_PATH}/include"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include"
    )

    if(NVENC_INCLUDE_DIR)
        target_compile_definitions(vidcrypt PRIVATE HAS_NVENC=1)
        target_include_directories(vidcrypt PRIVATE ${NVENC_INCLUDE_DIR})
        message(STATUS "NVENC header found: ${NVENC_INCLUDE_DIR}")
    else()
        message(STATUS "NVENC header (nvEncodeAPI.h) not found — NVENC encoder disabled")
        message(STATUS "  Install NVIDIA Video Codec SDK to enable NVENC")
    endif()

    target_include_directories(vidcrypt PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(vidcrypt PRIVATE CUDA::cudart CUDA::cuda_driver)


    # ── FFmpeg (NVDEC / libavcodec) ───────────────────────────────────
    # On Windows: use BtbN shared build with .lib files
    # On macOS/Linux: use system FFmpeg via pkg-config or manual find
    if(WIN32)
        set(FFMPEG_DIR "C:/Users/cooln/Documents/ffmpeg-master-latest-win64-lgpl-shared"
            CACHE PATH "Path to BtbN FFmpeg shared build root")
        if(EXISTS "${FFMPEG_DIR}/include/libavformat/avformat.h")
            target_include_directories(vidcrypt PRIVATE "${FFMPEG_DIR}/include")
            target_link_directories(vidcrypt PRIVATE "${FFMPEG_DIR}/lib")
            target_link_libraries(vidcrypt PRIVATE
                "${FFMPEG_DIR}/lib/avformat.lib"
                "${FFMPEG_DIR}/lib/avcodec.lib"
                "${FFMPEG_DIR}/lib/avutil.lib"
                "${FFMPEG_DIR}/lib/swscale.lib"
            )
            target_compile_definitions(vidcrypt PRIVATE HAS_FFMPEG=1)
            message(STATUS "FFmpeg NVDEC found at ${FFMPEG_DIR}")

            set(FFMPEG_DLLS
                "${FFMPEG_DIR}/bin/avformat-62.dll"
                "${FFMPEG_DIR}/bin/avcodec-62.dll"
                "${FFMPEG_DIR}/bin/avutil-60.dll"
                "${FFMPEG_DIR}/bin/swscale-9.dll"
                "${FFMPEG_DIR}/bin/swresample-6.dll"
                CACHE INTERNAL "FFmpeg DLLs for NVDEC runtime"
            )
        else()
            message(WARNING "FFmpeg dev headers not found at ${FFMPEG_DIR}")
            message(WARNING "  NVDEC zero-copy decode disabled")
        endif()
    else()
        # macOS / Linux: use pkg-config or find_library
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(FFMPEG QUIET
                libavformat libavcodec libavutil libswscale)
        endif()
        if(FFMPEG_FOUND)
            target_include_directories(vidcrypt PRIVATE ${FFMPEG_INCLUDE_DIRS})
            target_link_directories(vidcrypt PRIVATE ${FFMPEG_LIBRARY_DIRS})
            target_link_libraries(vidcrypt PRIVATE ${FFMPEG_LIBRARIES})
            target_compile_definitions(vidcrypt PRIVATE HAS_FFMPEG=1)
            message(STATUS "FFmpeg found via pkg-config")
        else()
            # Try find_library fallback
            find_path(AVFORMAT_INCLUDE_DIR libavformat/avformat.h)
            find_library(AVFORMAT_LIB avformat)
            find_library(AVCODEC_LIB avcodec)
            find_library(AVUTIL_LIB avutil)
            find_library(SWSCALE_LIB swscale)
            if(AVFORMAT_INCLUDE_DIR AND AVFORMAT_LIB AND AVCODEC_LIB AND AVUTIL_LIB)
                target_include_directories(vidcrypt PRIVATE ${AVFORMAT_INCLUDE_DIR})
                target_link_libraries(vidcrypt PRIVATE
                    ${AVFORMAT_LIB} ${AVCODEC_LIB} ${AVUTIL_LIB} ${SWSCALE_LIB})
                target_compile_definitions(vidcrypt PRIVATE HAS_FFMPEG=1)
                message(STATUS "FFmpeg found via find_library")
            else()
                message(WARNING "FFmpeg dev libraries not found — NVDEC disabled")
                message(WARNING "  Install ffmpeg dev package (e.g. 'brew install ffmpeg' or 'apt install libavformat-dev')")
            endif()
        endif()
    endif()

    message(STATUS "CUDA GPU acceleration enabled")
    message(STATUS "  CUDA architectures: ${VIDCRYPT_CUDA_ARCHS}")
    message(STATUS "  GPU sources: ${GPU_SOURCES}")
endif()

# ── Tile Decode Acceleration (3 tiers) ────────────────────────────────
# Tier 1: USE_NATIVE_TILES — pre-built native lib (GraalVM, no runtime Java)
# Tier 2: USE_JNI_TILES — JVM at runtime (JAR auto-loaded next to exe)
# Tier 3: neither — C SIMD fallback (always works)

option(USE_NATIVE_TILES "Link pre-built GraalVM native tile decoder (no runtime Java)" OFF)
option(USE_JNI_TILES "Enable Java JNI-accelerated tile decoding (requires JVM at runtime)" OFF)

if(USE_NATIVE_TILES)
    # Pre-built native shared library from GraalVM native-image
    # Built via: cd vidcrypt-java && native-image --shared -o vidcrypt_tiles ...
    find_library(NATIVE_TILES_LIB vidcrypt_tiles
        HINTS "${CMAKE_SOURCE_DIR}/vidcrypt-java" "${CMAKE_SOURCE_DIR}")
    if(NATIVE_TILES_LIB)
        target_sources(vidcrypt PRIVATE src/tile_decode_jni.c)
        target_link_libraries(vidcrypt PRIVATE ${NATIVE_TILES_LIB})
        target_compile_definitions(vidcrypt PRIVATE USE_NATIVE_TILES=1)
        target_link_libraries(vidcrypt-decoder PRIVATE ${NATIVE_TILES_LIB})
        message(STATUS "Native tile decoder: ${NATIVE_TILES_LIB}")
    else()
        message(WARNING "USE_NATIVE_TILES=ON but vidcrypt_tiles lib not found. Falling back.")
        set(USE_NATIVE_TILES OFF)
    endif()
endif()

if(USE_JNI_TILES AND NOT USE_NATIVE_TILES)
    if(WIN32 AND JAVA_HOME)
        # Convert to CMake-style paths and resolve \.. to avoid escape issues in FindJNI
        # (Windows paths C:\Program Files\... break foreach parsing in FindJNI.cmake)
        file(TO_CMAKE_PATH "${JAVA_HOME}" JAVA_HOME)
        get_filename_component(JAVA_HOME "${JAVA_HOME}" REALPATH)
    endif()
    find_package(JNI REQUIRED)
    include_directories(${JNI_INCLUDE_DIRS})
    if(WIN32)
        find_library(JVM_LIBRARY jvm HINTS "${JAVA_HOME}/lib" "${JAVA_HOME}/lib/server")
    elseif(APPLE)
        find_library(JVM_LIBRARY jvm HINTS "${JAVA_HOME}/lib/server" "${JAVA_HOME}/jre/lib/server")
    else()
        find_library(JVM_LIBRARY jvm HINTS "${JAVA_HOME}/lib/server" "${JAVA_HOME}/jre/lib/server")
    endif()
    if(NOT JVM_LIBRARY)
        message(FATAL_ERROR "JVM library not found. Set JAVA_HOME or disable USE_JNI_TILES.")
    endif()
    target_sources(vidcrypt PRIVATE src/tile_decode_jni.c)
    target_link_libraries(vidcrypt PRIVATE ${JVM_LIBRARY})
    target_compile_definitions(vidcrypt PRIVATE USE_JNI_TILES=1)
    message(STATUS "Java JNI tile decode enabled (JVM at runtime)")
    message(STATUS "  JVM library: ${JVM_LIBRARY}")
endif()

add_executable(vidcrypt-encoder main_encoder.c)
target_link_libraries(vidcrypt-encoder PRIVATE vidcrypt)

add_executable(vidcrypt-decoder main_decoder.c)
target_link_libraries(vidcrypt-decoder PRIVATE vidcrypt)

# Link JVM or native tiles to decoder (after target is defined)
if(USE_NATIVE_TILES)
    target_link_libraries(vidcrypt-decoder PRIVATE ${NATIVE_TILES_LIB})
endif()
if(USE_JNI_TILES AND NOT USE_NATIVE_TILES)
    target_link_libraries(vidcrypt-decoder PRIVATE ${JVM_LIBRARY})
endif()

# Copy JAR to output dir if it exists
if(EXISTS "${CMAKE_SOURCE_DIR}/vidcrypt-tiles.jar")
    add_custom_command(TARGET vidcrypt-decoder POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/vidcrypt-tiles.jar"
            "$<TARGET_FILE_DIR:vidcrypt-decoder>/vidcrypt-tiles.jar"
        COMMENT "Copying vidcrypt-tiles.jar to output dir"
    )
endif()

enable_testing()

add_executable(test-crc16 tests/test_crc16.c)
target_link_libraries(test-crc16 PRIVATE vidcrypt)

add_executable(test-bitstream tests/test_bitstream.c)
target_link_libraries(test-bitstream PRIVATE vidcrypt)

add_executable(test-reedsolomon tests/test_reedsolomon.c)
target_link_libraries(test-reedsolomon PRIVATE vidcrypt)

add_test(NAME test-crc16 COMMAND test-crc16)
add_test(NAME test-bitstream COMMAND test-bitstream)
add_executable(test-decode tests/test_decode.c)
target_link_libraries(test-decode PRIVATE vidcrypt)
add_test(NAME test-decode COMMAND test-decode)

add_test(NAME test-reedsolomon COMMAND test-reedsolomon)

# ── FFmpeg DLL copy for runtime NVDEC ───────────────────────────────
# Copy BtbN FFmpeg DLLs to each executable's output directory so
# Windows-only: copy FFmpeg DLLs alongside executables for NVDEC runtime
if(WIN32 AND USE_CUDA AND DEFINED FFMPEG_DLLS)
    foreach(TGT vidcrypt-decoder vidcrypt-encoder)
        add_custom_command(TARGET ${TGT} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${FFMPEG_DLLS}
                $<TARGET_FILE_DIR:${TGT}>
            COMMENT "Copying FFmpeg DLLs to ${TGT} output dir"
        )
    endforeach()
endif()

install(TARGETS vidcrypt vidcrypt-encoder vidcrypt-decoder
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY src/ DESTINATION include/vidcrypt
    FILES_MATCHING PATTERN "*.h"
)
