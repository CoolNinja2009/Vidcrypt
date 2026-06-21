cmake_minimum_required(VERSION 3.16)
project(vidcrypt VERSION 4.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(ENABLE_PROFILING "Enable per-stage profiling instrumentation" OFF)

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
        $<$<COMPILE_LANGUAGE:C>:-Wno-unused-function>
    )
endif()

if(ENABLE_PROFILING)
    add_compile_definitions(ENABLE_PROFILING=1)
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
)

add_library(vidcrypt STATIC ${CORE_SOURCES})
target_include_directories(vidcrypt PUBLIC ${VIDCRYPT_INCLUDE_DIR})

find_package(Threads REQUIRED)
target_link_libraries(vidcrypt PRIVATE Threads::Threads)
target_compile_definitions(vidcrypt PRIVATE _GNU_SOURCE)

if(UNIX)
    target_link_libraries(vidcrypt PRIVATE m)
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
add_test(NAME test-reedsolomon COMMAND test-reedsolomon)

install(TARGETS vidcrypt vidcrypt-encoder vidcrypt-decoder
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY src/ DESTINATION include/vidcrypt
    FILES_MATCHING PATTERN "*.h"
)
