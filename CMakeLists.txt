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

# If AVX2 is enabled, apply flags ONLY to simd_decode.c (not globally).
# Global /arch:AVX2 on MSVC forces VEX-prefixed instructions everywhere,
# adding ~10% overhead to scalar code that doesn't benefit from SIMD.
if(ENABLE_AVX2)
    if(MSVC)
        set_source_files_properties(src/simd_decode.c PROPERTIES COMPILE_FLAGS "/arch:AVX2")
        # /arch:AVX2 on MSVC implies __AVX2__ automatically
    else()
        set_source_files_properties(src/simd_decode.c PROPERTIES COMPILE_FLAGS "-mavx2 -D__AVX2__")
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

    # Target architectures: SASS for modern GPUs + PTX for backward compat
    # RTX 5070 = sm_120 (Blackwell consumer). CUDA 12.6 supports up to sm_90,
    # so we include sm_90 PTX for driver JIT-compilation on Blackwell.
    set(VIDCRYPT_CUDA_ARCHS "50-virtual;75;86;89;90;90-virtual")

    set(GPU_SOURCES
        src/gpu_backend.c
        src/gpu_kernels.cu
        src/gpu_presets.c
        src/gpu_nvenc.c
        src/ffmpeg_hwdecoder.c
        src/libav_decoder.c
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

    # ── FFmpeg (NVDEC) ────────────────────────────────────────────────
    # BtbN FFmpeg LGPL shared build with CUDA hwaccel support.
    # Provides libavformat, libavcodec, libavutil for NVDEC zero-copy.
    set(FFMPEG_DIR "C:/Users/cooln/Documents/ffmpeg-master-latest-win64-lgpl-shared")
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
        message(STATUS "  libavformat: ${FFMPEG_DIR}/lib/avformat.lib")
        message(STATUS "  libavcodec:  ${FFMPEG_DIR}/lib/avcodec.lib")
        message(STATUS "  CUDA hwaccel: enabled (BtbN shared build)")

        # Save FFMPEG_DIR and DLL list for post-build copy (after executables are created)
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
        message(WARNING "  Download BtbN FFmpeg shared build to ffmpeg_shared/ to enable")
    endif()

    message(STATUS "CUDA GPU acceleration enabled")
    message(STATUS "  CUDA architectures: ${VIDCRYPT_CUDA_ARCHS}")
    message(STATUS "  GPU sources: ${GPU_SOURCES}")
endif()

add_executable(vidcrypt-encoder main_encoder.c)
target_link_libraries(vidcrypt-encoder PRIVATE vidcrypt)

add_executable(vidcrypt-decoder main_decoder.c)
target_link_libraries(vidcrypt-decoder PRIVATE vidcrypt)

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
# Windows can find them at runtime (searches: exe dir, CWD, PATH).
if(USE_CUDA AND DEFINED FFMPEG_DLLS)
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
