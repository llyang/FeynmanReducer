#pragma once

#include <cstdint>
#include <limits>
#include <span>
#include <vector>

/// 在当前有限域下已求值的各类系数
template <typename T> struct EvaluatedCoeffs {
  T minus_half_d;                  ///< minus_half_d = -d/2
  std::span<const T> basis_lp_inv; ///< LP 表示下基底积分的整体因子的逆
  std::span<const T> targets_lp;   ///< LP 表示下目标积分的整体因子
  std::span<const T> top_lp_coefficients;
};

/// 单项式：表示被积函数展开中的一个单项式
///
/// c[polynomial_term_index] ·
/// (coeff_int + coeff_minus_half_d · minus_half_d) ·
/// x^{powers[1:]} · G^{powers[0]}
///
/// powers[0] = Lee–Pomeransky 多项式 G 的幂次
/// powers[k] = 变量 x_k 的幂次（k=1,...,n）；powers[k]=-1-r 对应 δ^(r)(x_k)
/// polynomial_term_index = G 中对应项的系数索引；是否使用该索引由
/// with_polynomial_coefficient 标记。
/// coeff_int = 整数部分系数；当 coefficient_expression 有效时，
///             它是该预编译表达式的整数乘子
/// coeff_minus_half_d = -d/2 的系数
struct Monomial {
  std::vector<int> powers;          ///< 幂次列表: [G的幂次, x_1的幂次, ..., x_n的幂次]
  bool with_polynomial_coefficient; ///< 是否要乘以 G 中某一项的系数
  unsigned polynomial_term_index;   ///< G 的项索引
  std::int64_t coeff_int;           ///< 整数系数
  std::int64_t coeff_minus_half_d;  ///< minus_half_d = -d/2 的系数
  // Top-LP terms may use a precompiled nonlinear coefficient.
  // Denominator equation terms keep the sentinel value.
  std::uint32_t coefficient_expression = std::numeric_limits<std::uint32_t>::max();
};

/// vector<int> 的哈希函数（用于 unordered_map 的键）
/// 采用经典的 hash_combine 策略，对小整数数组非常高效
struct VectorHash {
  std::size_t operator()(const std::vector<int>& v) const noexcept
  {
    std::size_t seed = v.size();
    for (const auto& i : v) {
      seed ^= std::hash<int>{}(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};
