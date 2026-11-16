#pragma once

#include <cstdint>
#include <cstddef>
#include <limits>
#include <vector>
#include <string>
#include <cstdlib>
#include <new>
#include <cstring>

namespace pitmem {

constexpr uint32_t INF_TICK = std::numeric_limits<uint32_t>::max();

enum class MemoryTier : uint8_t {
    RAW = 0,
    REFLECT = 1,
    COMPACT = 2
};

enum class OpKind : uint8_t {
    INSERT = 0,
    EVOLVE = 1,
    RESTATE = 2
};

// 64-byte cache-line aligned allocator for AVX2 / AVX-512 SoA buffers
template <typename T, std::size_t Alignment = 64>
struct AlignedBuffer {
    T* data = nullptr;
    std::size_t size = 0;
    std::size_t capacity = 0;

    AlignedBuffer() = default;
    ~AlignedBuffer() { release(); }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& other) noexcept
        : data(other.data), size(other.size), capacity(other.capacity) {
        other.data = nullptr;
        other.size = 0;
        other.capacity = 0;
    }

    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            release();
            data = other.data;
            size = other.size;
            capacity = other.capacity;
            other.data = nullptr;
            other.size = 0;
            other.capacity = 0;
        }
        return *this;
    }

    void release() {
        if (data) {
            std::free(data);
            data = nullptr;
        }
        size = 0;
        capacity = 0;
    }

    void reserve(std::size_t new_cap) {
        if (new_cap <= capacity) return;
        std::size_t cap = capacity == 0 ? 64 : capacity;
        while (cap < new_cap) cap *= 2;
        std::size_t bytes = ((cap * sizeof(T) + Alignment - 1) / Alignment) * Alignment;
        T* next = static_cast<T*>(std::aligned_alloc(Alignment, bytes));
        if (!next) throw std::bad_alloc();
        if (data && size > 0) {
            std::memcpy(next, data, size * sizeof(T));
        }
        if (data) std::free(data);
        data = next;
        capacity = cap;
    }

    void push_back(const T& val) {
        if (size == capacity) reserve(size + 1);
        data[size++] = val;
    }

    void resize(std::size_t new_size, const T& fill_val = T{}) {
        if (new_size > capacity) reserve(new_size);
        for (std::size_t i = size; i < new_size; ++i) {
            data[i] = fill_val;
        }
        size = new_size;
    }

    inline T& operator[](std::size_t i) { return data[i]; }
    inline const T& operator[](std::size_t i) const { return data[i]; }
};

struct SearchStats {
    double latency_us = 0.0;
    uint64_t visited_cells = 0;
    uint64_t graph_cells = 0;
    uint64_t scan_cells = 0;
    uint64_t distance_computations = 0;
    uint64_t predicate_checks = 0;
};

struct QueryResult {
    std::vector<uint64_t> ids;
    std::vector<float> distances;
    SearchStats stats;
};

// Exact 2D Bi-Temporal Stabbing Predicate:
// tv_s <= tv_q < tv_e  AND  tk_s <= tk_q < tk_e
inline bool bitemp_visible(
    uint32_t tv_s, uint32_t tv_e,
    uint32_t tk_s, uint32_t tk_e,
    uint32_t tv_q, uint32_t tk_q) noexcept {
    return (tv_s <= tv_q) && (tv_q < tv_e) && (tk_s <= tk_q) && (tk_q < tk_e);
}

} // namespace pitmem
