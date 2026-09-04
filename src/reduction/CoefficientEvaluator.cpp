#include "reduction/BlackBoxFeynman.hpp"
#include "reduction/ParameterEvaluation.hpp"
#include "reduction/detail/BatchInverse.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

EvaluatedCoeffs<firefly::FFInt>
BlackBoxFeynman::evaluate_coefficients(const std::vector<firefly::FFInt>& values,
                                       const CoefficientSelection* selection) const
{
  using T = firefly::FFInt;

  EvaluatedCoeffs<T> res;
  const T d_val = reduction::detail::evaluate_dimension(cfg, values);
  const T inverse_two((firefly::FFInt::p >> 1U) + 1U);
  const T half_d = d_val * inverse_two;
  res.minus_half_d = T(0) - half_d;
  const T d0 = T(cfg.loop_count + 1) * half_d - T(lp_variable_count);

  // 持久化的线程局部内存池（只在首次调用时分配内存）
  // 正确性依赖以下前提，若将来改变需重新评估：
  // bunch_size = 1，T 始终为 FFInt，由 operator() 中的拦截保证
  static thread_local std::vector<T> local_target_lp;
  static thread_local std::vector<T> local_basis_lp_inv;
  static thread_local std::vector<T> local_lp_numerators;
  static thread_local std::vector<T> local_lp_denominators;
  static thread_local std::vector<T> local_target_denominator_inverses;
  static thread_local std::vector<T> local_basis_numerator_inverses;
  static thread_local std::vector<T> local_reciprocal_values;
  static thread_local std::vector<T> local_reciprocal_inverses;
  static thread_local std::vector<T> local_reciprocal_prefix;
  static thread_local std::vector<T> local_positive_delta_products;
  static thread_local std::vector<T> local_negative_delta_products;
  static thread_local std::vector<T> local_top_lp_coefficients;
  static thread_local std::vector<T> local_extended_polynomial_values;
  static thread_local std::vector<T> local_top_lp_products;
  static thread_local std::vector<T> local_top_lp_falling_factors;

  if (local_target_lp.size() != cfg.targets.size())
    local_target_lp.resize(cfg.targets.size());
  if (local_basis_lp_inv.size() != cfg.basis.size())
    local_basis_lp_inv.resize(cfg.basis.size());
  if (local_lp_numerators.size() != lp_programs.size()) {
    local_lp_numerators.resize(lp_programs.size());
    local_lp_denominators.resize(lp_programs.size());
    local_target_denominator_inverses.resize(lp_programs.size());
    local_basis_numerator_inverses.resize(lp_programs.size());
  }

  if (local_top_lp_coefficients.size() != top_lp_expression_programs.size()) {
    local_top_lp_coefficients.resize(top_lp_expression_programs.size());
  }
  if (!top_lp_expression_programs.empty()) {
    if (local_extended_polynomial_values.size() !=
        cfg.extended_lp.polynomial_terms.size()) {
      local_extended_polynomial_values.resize(cfg.extended_lp.polynomial_terms.size());
    }
    const auto evaluate_polynomial_factor = [&](std::size_t term) {
      local_extended_polynomial_values[term] =
          reduction::detail::evaluate_polynomial_coefficient(
              cfg, cfg.extended_lp.polynomial_terms[term], values);
    };
    if (selection == nullptr) {
      for (std::size_t term = 0; term < local_extended_polynomial_values.size(); ++term)
        evaluate_polynomial_factor(term);
    } else {
      for (const std::uint32_t term : selection->extended_polynomial_factors) {
        if (term >= local_extended_polynomial_values.size())
          throw std::logic_error("selected extended LP factor is out of range");
        evaluate_polynomial_factor(term);
      }
    }

    if (local_top_lp_products.size() != top_lp_product_nodes.size() + 1)
      local_top_lp_products.resize(top_lp_product_nodes.size() + 1);
    local_top_lp_products[0] = T(1);
    const auto evaluate_product_node = [&](std::size_t node) {
      const auto& program = top_lp_product_nodes[node];
      local_top_lp_products[node + 1] =
          local_top_lp_products[program.parent] *
          local_extended_polynomial_values[program.polynomial_factor];
    };
    if (selection == nullptr) {
      for (std::size_t node = 0; node < top_lp_product_nodes.size(); ++node)
        evaluate_product_node(node);
    } else {
      for (const std::uint32_t node : selection->top_lp_product_nodes) {
        if (node >= top_lp_product_nodes.size())
          throw std::logic_error("selected top-LP product node is out of range");
        evaluate_product_node(node);
      }
    }

    if (local_top_lp_falling_factors.size() !=
        static_cast<std::size_t>(top_lp_maximum_falling_degree) + 1) {
      local_top_lp_falling_factors.resize(
          static_cast<std::size_t>(top_lp_maximum_falling_degree) + 1);
    }
    local_top_lp_falling_factors[0] = T(1);
    const std::size_t maximum_falling_degree = selection == nullptr
                                                   ? top_lp_maximum_falling_degree
                                                   : selection->maximum_falling_degree;
    for (std::size_t degree = 1; degree <= maximum_falling_degree; ++degree) {
      local_top_lp_falling_factors[degree] =
          local_top_lp_falling_factors[degree - 1] * (res.minus_half_d - T(degree - 1));
    }

    const auto evaluate_expression = [&](std::uint32_t expression) {
      if (expression >= top_lp_expression_programs.size())
        throw std::logic_error("top-LP expression program is out of range");
      const auto& program = top_lp_expression_programs[expression];
      T value(0);
      const std::size_t end =
          static_cast<std::size_t>(program.term_begin) + program.term_count;
      for (std::size_t term = program.term_begin; term < end; ++term) {
        const auto& atom = top_lp_expression_terms[term];
        value = value + T(atom.weight) *
                            local_top_lp_falling_factors[atom.falling_degree] *
                            local_top_lp_products[atom.product];
      }
      local_top_lp_coefficients[expression] = value;
    };
    if (selection == nullptr) {
      for (std::uint32_t expression = 0; expression < top_lp_expression_programs.size();
           ++expression) {
        evaluate_expression(expression);
      }
    } else {
      for (const std::uint32_t expression : selection->top_lp_expressions)
        evaluate_expression(expression);
    }
  }

  const std::size_t positive_degree = selection == nullptr
                                          ? maximum_positive_lp_delta
                                          : selection->maximum_positive_lp_delta;
  const std::size_t negative_degree = selection == nullptr
                                          ? maximum_negative_lp_delta
                                          : selection->maximum_negative_lp_delta;
  local_positive_delta_products.resize(positive_degree + 1);
  local_negative_delta_products.resize(negative_degree + 1);
  local_positive_delta_products[0] = T(1);
  for (std::size_t degree = 1; degree <= positive_degree; ++degree) {
    local_positive_delta_products[degree] =
        local_positive_delta_products[degree - 1] * (d0 - T(degree));
  }
  local_negative_delta_products[0] = T(1);
  for (std::size_t degree = 1; degree <= negative_degree; ++degree) {
    local_negative_delta_products[degree] =
        local_negative_delta_products[degree - 1] * (d0 + T(degree - 1));
  }

  const auto evaluate_lp_program = [&](const std::size_t program) {
    if (program >= lp_programs.size())
      throw std::logic_error("selected LP normalization program is out of range");
    const auto fraction =
        evaluate_lp_fraction<T>(lp_programs[program], local_positive_delta_products,
                                local_negative_delta_products);
    local_lp_numerators[program] = fraction.numerator;
    local_lp_denominators[program] = fraction.denominator;
  };
  if (selection == nullptr) {
    for (size_t program = 0; program < lp_programs.size(); ++program)
      evaluate_lp_program(program);
  } else {
    for (const std::size_t program : selection->lp_programs)
      evaluate_lp_program(program);
  }

  const auto& reciprocal_requests =
      selection == nullptr ? full_lp_reciprocals : selection->lp_reciprocals;
  local_reciprocal_values.resize(reciprocal_requests.size());
  local_reciprocal_inverses.resize(reciprocal_requests.size());
  for (std::size_t index = 0; index < reciprocal_requests.size(); ++index) {
    const auto request = reciprocal_requests[index];
    if (request.program >= lp_programs.size())
      throw std::logic_error("selected LP reciprocal program is out of range");
    local_reciprocal_values[index] =
        request.side == CoefficientSelection::LpReciprocalSide::TargetDenominator
            ? local_lp_denominators[request.program]
            : local_lp_numerators[request.program];
  }
  if (reduction::detail::batch_inverse<T>(
          local_reciprocal_values, local_reciprocal_inverses, local_reciprocal_prefix)
          .has_value()) [[unlikely]] {
    throw std::runtime_error("LP coefficient denominator is zero");
  }
  for (std::size_t index = 0; index < reciprocal_requests.size(); ++index) {
    const auto request = reciprocal_requests[index];
    if (request.side == CoefficientSelection::LpReciprocalSide::TargetDenominator) {
      local_target_denominator_inverses[request.program] =
          local_reciprocal_inverses[index];
    } else {
      local_basis_numerator_inverses[request.program] =
          local_reciprocal_inverses[index];
    }
  }

  if (selection == nullptr) {
    for (std::size_t target = 0; target < cfg.targets.size(); ++target) {
      const std::size_t program = target_lp_program_ids[target];
      local_target_lp[target] =
          local_lp_numerators[program] * local_target_denominator_inverses[program];
    }
    for (std::size_t master = 0; master < cfg.basis.size(); ++master) {
      const std::size_t program = basis_lp_program_ids[master];
      local_basis_lp_inv[master] =
          local_lp_denominators[program] * local_basis_numerator_inverses[program];
    }
  } else {
    for (const std::uint32_t target : selection->targets) {
      if (target >= target_lp_program_ids.size())
        throw std::logic_error("selected target normalization is out of range");
      const std::size_t program = target_lp_program_ids[target];
      local_target_lp[target] =
          local_lp_numerators[program] * local_target_denominator_inverses[program];
    }
    for (const std::uint32_t master : selection->basis) {
      if (master >= basis_lp_program_ids.size())
        throw std::logic_error("selected basis normalization is out of range");
      const std::size_t program = basis_lp_program_ids[master];
      local_basis_lp_inv[master] =
          local_lp_denominators[program] * local_basis_numerator_inverses[program];
    }
  }

  res.targets_lp = std::span<const T>(local_target_lp.data(), local_target_lp.size());
  res.basis_lp_inv =
      std::span<const T>(local_basis_lp_inv.data(), local_basis_lp_inv.size());
  res.top_lp_coefficients = std::span<const T>(local_top_lp_coefficients.data(),
                                               local_top_lp_coefficients.size());

  return res;
}
