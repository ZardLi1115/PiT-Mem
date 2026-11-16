#pragma once

#include "pit_types.hpp"
#include <cmath>
#include <algorithm>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace pitmem {
namespace simd {

// Exact FP32 Squared L2 distance with AVX2 + FMA
inline float l2_sqr_fp32(const float* __restrict a, const float* __restrict b, std::size_t dim) noexcept {
#if defined(__AVX2__)
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    std::size_t i = 0;
    for (; i + 32 <= dim; i += 32) {
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i));
        __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8));
        __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16));
        __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24));
    #if defined(__FMA__)
        sum0 = _mm256_fmadd_ps(d0, d0, sum0);
        sum1 = _mm256_fmadd_ps(d1, d1, sum1);
        sum2 = _mm256_fmadd_ps(d2, d2, sum2);
        sum3 = _mm256_fmadd_ps(d3, d3, sum3);
    #else
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(d0, d0));
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(d1, d1));
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(d2, d2));
        sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(d3, d3));
    #endif
    }
    for (; i + 8 <= dim; i += 8) {
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    #if defined(__FMA__)
        sum0 = _mm256_fmadd_ps(d0, d0, sum0);
    #else
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(d0, d0));
    #endif
    }
    sum0 = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
    __m128 lo = _mm256_castps256_ps128(sum0);
    __m128 hi = _mm256_extractf128_ps(sum0, 1);
    __m128 s128 = _mm_add_ps(lo, hi);
    s128 = _mm_hadd_ps(s128, s128);
    s128 = _mm_hadd_ps(s128, s128);
    float res = _mm_cvtss_f32(s128);
    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        res += diff * diff;
    }
    return res;
#else
    float res = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        res += diff * diff;
    }
    return res;
#endif
}

// Quantize database vector into signed int8 [-127, 127], returning scale, norm_sq, and sum_s8
struct SQ8VectorMeta {
    float scale;
    float norm_sq;
    int32_t sum_s8;
};

inline SQ8VectorMeta quantize_db_sq8(const float* __restrict vec, int8_t* __restrict out_s8, std::size_t dim) noexcept {
    float max_abs = 0.0f;
    float norm_sq = 0.0f;
    for (std::size_t j = 0; j < dim; ++j) {
        float v = vec[j];
        norm_sq += v * v;
        float av = std::fabs(v);
        if (av > max_abs) max_abs = av;
    }
    float scale = (max_abs > 1e-12f) ? (max_abs / 127.0f) : 1.0f;
    float inv_scale = (max_abs > 1e-12f) ? (127.0f / max_abs) : 0.0f;
    int32_t sum_s8 = 0;
    for (std::size_t j = 0; j < dim; ++j) {
        int q = static_cast<int>(std::lrintf(vec[j] * inv_scale));
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        out_s8[j] = static_cast<int8_t>(q);
        sum_s8 += q;
    }
    return {scale, norm_sq, sum_s8};
}

// Quantize query vector into unsigned uint8 [1, 127] (bias = 64, range [-63, 63])
// Guarantees 2 * 127 * 127 = 32,258 <= 32,767: ZERO int16 saturation in _mm256_maddubs_epi16!
struct SQ8Query {
    AlignedBuffer<uint8_t, 64> q_u8;
    float scale_q = 1.0f;
    float norm_sq_q = 0.0f;
    std::size_t padded_dim = 0;
};

inline SQ8Query prepare_sq8_query(const float* __restrict q, std::size_t dim) {
    SQ8Query res;
    res.padded_dim = ((dim + 31) / 32) * 32;
    res.q_u8.resize(res.padded_dim, 64); // 64 represents 0 for padded tail
    float max_abs = 0.0f;
    float norm_sq = 0.0f;
    for (std::size_t j = 0; j < dim; ++j) {
        float v = q[j];
        norm_sq += v * v;
        float av = std::fabs(v);
        if (av > max_abs) max_abs = av;
    }
    res.norm_sq_q = norm_sq;
    res.scale_q = (max_abs > 1e-12f) ? (max_abs / 63.0f) : 1.0f;
    float inv_scale = (max_abs > 1e-12f) ? (63.0f / max_abs) : 0.0f;
    for (std::size_t j = 0; j < dim; ++j) {
        int qi = static_cast<int>(std::lrintf(q[j] * inv_scale));
        if (qi > 63) qi = 63;
        if (qi < -63) qi = -63;
        res.q_u8[j] = static_cast<uint8_t>(qi + 64);
    }
    return res;
}

// Ultra-fast AVX2 INT8 dot product (32 dimensions per loop step) -> approximate L2^2
inline float sq8_approx_l2_sqr(
    const SQ8Query& q,
    const int8_t* __restrict db_s8,
    const SQ8VectorMeta& meta) noexcept {
#if defined(__AVX2__)
    __m256i acc = _mm256_setzero_si256();
    const __m256i ones16 = _mm256_set1_epi16(1);
    const uint8_t* qu = q.q_u8.data;
    for (std::size_t j = 0; j < q.padded_dim; j += 32) {
        __m256i vq = _mm256_load_si256(reinterpret_cast<const __m256i*>(qu + j));
        __m256i vx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(db_s8 + j));
        __m256i prod16 = _mm256_maddubs_epi16(vq, vx); // exact, no saturation!
        __m256i prod32 = _mm256_madd_epi16(prod16, ones16);
        acc = _mm256_add_epi32(acc, prod32);
    }
    __m128i lo = _mm256_castsi256_si128(acc);
    __m128i hi = _mm256_extracti128_si256(acc, 1);
    __m128i sum128 = _mm_add_epi32(lo, hi);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    int32_t raw_dot = _mm_cvtsi128_si32(sum128) - 64 * meta.sum_s8;
#else
    int32_t raw_dot = 0;
    for (std::size_t j = 0; j < q.padded_dim; ++j) {
        raw_dot += (static_cast<int32_t>(q.q_u8[j]) - 64) * static_cast<int32_t>(db_s8[j]);
    }
#endif
    float dot = static_cast<float>(raw_dot) * (q.scale_q * meta.scale);
    return q.norm_sq_q + meta.norm_sq - 2.0f * dot;
}

// AVX2 8-wide unsigned right-endpoint filter for Fully-Born Cells:
// Returns 8-bit mask where bit m is 1 iff (tv_q < tv_end[i+m]) && (tk_q < tk_end[i+m])
inline uint8_t filter8_death_only(
    const uint32_t* __restrict tv_end,
    const uint32_t* __restrict tk_end,
    uint32_t tv_q,
    uint32_t tk_q) noexcept {
#if defined(__AVX2__)
    const __m256i v_tvq = _mm256_set1_epi32(static_cast<int>(tv_q));
    const __m256i v_tkq = _mm256_set1_epi32(static_cast<int>(tk_q));
    __m256i v_tve = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tv_end));
    __m256i v_tke = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tk_end));

    // max_epu32(q, end) == q  <=>  q >= end  <=>  DEAD
    __m256i dead_v = _mm256_cmpeq_epi32(_mm256_max_epu32(v_tvq, v_tve), v_tvq);
    __m256i dead_k = _mm256_cmpeq_epi32(_mm256_max_epu32(v_tkq, v_tke), v_tkq);
    __m256i dead = _mm256_or_si256(dead_v, dead_k);
    int dead_mask = _mm256_movemask_ps(_mm256_castsi256_ps(dead));
    return static_cast<uint8_t>((~dead_mask) & 0xFF);
#else
    uint8_t mask = 0;
    for (int m = 0; m < 8; ++m) {
        if (tv_q < tv_end[m] && tk_q < tk_end[m]) mask |= static_cast<uint8_t>(1u << m);
    }
    return mask;
#endif
}

// AVX2 8-wide full 4-sided bi-temporal stabbing filter for Boundary Cells:
// Returns 8-bit mask where bit m is 1 iff tv_s <= tv_q < tv_e && tk_s <= tk_q < tk_e
inline uint8_t filter8_full_bitemp(
    const uint32_t* __restrict tv_start,
    const uint32_t* __restrict tv_end,
    const uint32_t* __restrict tk_start,
    const uint32_t* __restrict tk_end,
    uint32_t tv_q,
    uint32_t tk_q) noexcept {
#if defined(__AVX2__)
    const __m256i v_tvq = _mm256_set1_epi32(static_cast<int>(tv_q));
    const __m256i v_tkq = _mm256_set1_epi32(static_cast<int>(tk_q));

    __m256i v_tvs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tv_start));
    __m256i v_tve = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tv_end));
    __m256i v_tks = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tk_start));
    __m256i v_tke = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tk_end));

    // born: min_epu32(start, q) == start <=> start <= q
    __m256i born_v = _mm256_cmpeq_epi32(_mm256_min_epu32(v_tvs, v_tvq), v_tvs);
    __m256i born_k = _mm256_cmpeq_epi32(_mm256_min_epu32(v_tks, v_tkq), v_tks);
    __m256i born = _mm256_and_si256(born_v, born_k);

    // dead: max_epu32(q, end) == q <=> q >= end
    __m256i dead_v = _mm256_cmpeq_epi32(_mm256_max_epu32(v_tvq, v_tve), v_tvq);
    __m256i dead_k = _mm256_cmpeq_epi32(_mm256_max_epu32(v_tkq, v_tke), v_tkq);
    __m256i dead = _mm256_or_si256(dead_v, dead_k);

    // alive = born & (~dead) = andnot(dead, born)
    __m256i alive = _mm256_andnot_si256(dead, born);
    return static_cast<uint8_t>(_mm256_movemask_ps(_mm256_castsi256_ps(alive)) & 0xFF);
#else
    uint8_t mask = 0;
    for (int m = 0; m < 8; ++m) {
        if (bitemp_visible(tv_start[m], tv_end[m], tk_start[m], tk_end[m], tv_q, tk_q)) {
            mask |= static_cast<uint8_t>(1u << m);
        }
    }
    return mask;
#endif
}

} // namespace simd
} // namespace pitmem
