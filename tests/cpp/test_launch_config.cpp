#include "cudaforge/launch_config.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>

using namespace cudaforge;

// These run on a host with no CUDA toolkit, which is the point of the header
// existing separately. A wrong grid size is a real bug class — a kernel that
// silently processes only part of its input — and it would otherwise only be
// catchable on hardware.

TEST_CASE("ceil_div rounds up", "[launch]") {
    STATIC_REQUIRE(ceil_div(0, 4) == 0);
    STATIC_REQUIRE(ceil_div(1, 4) == 1);
    STATIC_REQUIRE(ceil_div(4, 4) == 1);
    STATIC_REQUIRE(ceil_div(5, 4) == 2);
    STATIC_REQUIRE(ceil_div(1023, 256) == 4);
    STATIC_REQUIRE(ceil_div(1024, 256) == 4);
    STATIC_REQUIRE(ceil_div(1025, 256) == 5);
}

TEST_CASE("ceil_div covers every element", "[launch]") {
    // The property that matters: blocks * block_size must reach the count, or
    // the tail is silently dropped.
    for (int count = 1; count <= 2000; ++count) {
        for (int block : {32, 64, 128, 256, 1024}) {
            REQUIRE(ceil_div(count, block) * block >= count);
            REQUIRE((ceil_div(count, block) - 1) * block < count);
        }
    }
}

TEST_CASE("the size_t overload behaves identically", "[launch]") {
    STATIC_REQUIRE(ceil_div(std::size_t{1025}, std::size_t{256}) == 5);
    // Well past the 32-bit range, where an int overload would wrap.
    REQUIRE(ceil_div(std::size_t{1} << 40, std::size_t{256}) == (std::size_t{1} << 32));
}

TEST_CASE("a row block size is a power of two", "[launch]") {
    for (int length = 1; length <= 4096; ++length) {
        const int block = block_size_for_row(length);
        INFO("row length " << length << " gave block " << block);
        // A tree reduction's halving loop only terminates cleanly at one active
        // thread if the block size is a power of two.
        REQUIRE((block & (block - 1)) == 0);
    }
}

TEST_CASE("a row block size is at least one warp", "[launch]") {
    // The warp primitives need a full warp to be present.
    for (int length : {1, 2, 8, 31, 32}) {
        REQUIRE(block_size_for_row(length) == kWarpSize);
    }
}

TEST_CASE("a row block size covers the row until the cap", "[launch]") {
    for (int length : {33, 64, 100, 256, 513, 1024}) {
        INFO("row length " << length);
        REQUIRE(block_size_for_row(length) >= length);
    }
}

TEST_CASE("a row block size is capped at the hardware maximum", "[launch]") {
    // Beyond the cap the kernel relies on its grid-stride loop instead.
    REQUIRE(block_size_for_row(4096) == kMaxBlockSize);
    REQUIRE(block_size_for_row(1'000'000) == kMaxBlockSize);
}

TEST_CASE("warps per block rounds up", "[launch]") {
    STATIC_REQUIRE(warps_per_block(32) == 1);
    STATIC_REQUIRE(warps_per_block(33) == 2);
    STATIC_REQUIRE(warps_per_block(256) == 8);
    STATIC_REQUIRE(warps_per_block(1024) == 32);
}

TEST_CASE("a grid-stride grid is never zero", "[launch]") {
    REQUIRE(grid_size_for_stride_loop(0, 256, 80) >= 1);
    REQUIRE(grid_size_for_stride_loop(1, 256, 80) >= 1);
    REQUIRE(grid_size_for_stride_loop(1'000'000, 256, 0) >= 1);
}

TEST_CASE("a small input does not launch idle blocks", "[launch]") {
    // 100 elements at 256 per block needs exactly one block, however many SMs
    // the device has.
    REQUIRE(grid_size_for_stride_loop(100, 256, 80) == 1);
    REQUIRE(grid_size_for_stride_loop(256, 256, 108) == 1);
    REQUIRE(grid_size_for_stride_loop(257, 256, 108) == 2);
}

TEST_CASE("a large input is capped by occupancy, not by size", "[launch]") {
    // A grid-stride loop means each block handles many elements. One block per
    // tile would create far more blocks than can be resident.
    constexpr int kSMs = 80;
    constexpr int kBlocksPerSM = 8;
    REQUIRE(grid_size_for_stride_loop(1UL << 30, 256, kSMs) == kSMs * kBlocksPerSM);
}

TEST_CASE("the grid scales with the multiprocessor count", "[launch]") {
    const int small = grid_size_for_stride_loop(1UL << 30, 256, 20);
    const int large = grid_size_for_stride_loop(1UL << 30, 256, 108);
    REQUIRE(large > small);
}

TEST_CASE("blocks per sm is configurable", "[launch]") {
    REQUIRE(grid_size_for_stride_loop(1UL << 30, 256, 40, 4) == 160);
    REQUIRE(grid_size_for_stride_loop(1UL << 30, 256, 40, 16) == 640);
}

TEST_CASE("an invalid block size does not produce a zero grid", "[launch]") {
    REQUIRE(grid_size_for_stride_loop(1000, 0, 80) == 1);
    REQUIRE(grid_size_for_stride_loop(1000, -1, 80) == 1);
}

TEST_CASE("the default block size is a whole number of warps", "[launch]") {
    STATIC_REQUIRE(kDefaultBlockSize % kWarpSize == 0);
    STATIC_REQUIRE(kDefaultBlockSize <= kMaxBlockSize);
}
