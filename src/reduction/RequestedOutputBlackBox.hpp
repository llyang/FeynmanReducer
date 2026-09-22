#pragma once

#include "core/Config.hpp"
#include "core/FlintRational.hpp"

#include <firefly/BlackBoxBase.hpp>
#include <firefly/FFInt.hpp>

#include <flint/fmpz_mpoly_q.h>
#include <flint/nmod.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace reduction::detail {

inline std::int64_t output_scale_offset(const Config& config, const Integral& integral)
{
  std::int64_t sum = 0;
  for (const int index : integral.indices) {
    if (__builtin_add_overflow(sum, static_cast<std::int64_t>(index), &sum) ||
        sum == std::numeric_limits<std::int64_t>::min())
      throw std::overflow_error("integral index sum overflow");
  }
  std::int64_t shift = 0;
  if (__builtin_mul_overflow(static_cast<std::int64_t>(config.loop_count),
                             static_cast<std::int64_t>(integral.dimension_shift / 2),
                             &shift) ||
      __builtin_sub_overflow(shift, sum, &shift))
    throw std::overflow_error("integral scale degree overflow");
  return shift;
}

inline std::vector<ReductionRequest> resolved_reduction_requests(const Config& config)
{
  if (!config.reduction_requests.empty()) return config.reduction_requests;
  std::vector<ReductionRequest> result;
  result.reserve(config.targets.size());
  for (std::size_t target = 0; target < config.targets.size(); ++target) {
    if (target > std::numeric_limits<std::uint32_t>::max())
      throw std::runtime_error("target index exceeds 32-bit request ids");
    ReductionRequest request;
    request.output.integral = config.targets[target];
    request.output.scale_offset = output_scale_offset(config, config.targets[target]);
    request.terms.push_back({static_cast<std::uint32_t>(target), std::string("1")});
    result.push_back(std::move(request));
  }
  return result;
}

template <typename Native>
class RequestedOutputBlackBox
    : public firefly::BlackBoxBase<RequestedOutputBlackBox<Native>> {
public:
  RequestedOutputBlackBox(const Config& config, std::unique_ptr<Native> source)
      : native(std::move(source)), basis_size_(config.basis.size()),
        requests_(resolved_reduction_requests(config)),
        coefficient_context_(std::make_shared<FlintRationalContext>(config.parameters))
  {
    if (!native) throw std::invalid_argument("requested-output source is missing");
    if (basis_size_ == 0 || requests_.empty())
      throw std::invalid_argument("requested-output shape must be non-empty");
    std::vector<const char*> names;
    names.reserve(config.parameters.size());
    for (const auto& name : config.parameters)
      names.push_back(name.c_str());
    coefficients_.reserve(requests_.size());
    for (const auto& request : requests_) {
      if (request.terms.empty() && !request.output.differential)
        throw std::invalid_argument("reduction request has no terms");
      auto& parsed = coefficients_.emplace_back();
      parsed.reserve(request.terms.size());
      for (const auto& term : request.terms) {
        if (term.target >= config.targets.size())
          throw std::out_of_range("reduction request target is out of range");
        FlintRational value(coefficient_context_);
        if (fmpz_mpoly_q_set_str_pretty(
                value.raw(), term.coefficient.c_str(), names.data(),
                const_cast<fmpz_mpoly_ctx_struct*>(coefficient_context_->raw())) != 0)
          throw std::runtime_error("cannot parse compiled reduction coefficient: " +
                                   term.coefficient);
        value.canonicalise();
        parsed.push_back(std::move(value));
      }
    }

    const auto source_outputs = native->reconstructed_outputs();
    source_position_to_output_.assign(native->total_output_count(), invalid_output);
    for (std::size_t output = 0; output < source_outputs.size(); ++output) {
      const auto position = source_outputs[output];
      if (position >= source_position_to_output_.size() ||
          output > std::numeric_limits<std::uint32_t>::max())
        throw std::logic_error("native reconstructed output is out of range");
      source_position_to_output_[position] = static_cast<std::uint32_t>(output);
    }
    const std::size_t total = total_output_count();
    if (total > std::numeric_limits<std::uint32_t>::max())
      throw std::runtime_error("requested output layout exceeds 32-bit ids");
    reconstructed_outputs_.reserve(total);
    for (std::size_t position = 0; position < total; ++position) {
      const std::size_t request = position / basis_size_;
      const std::size_t basis = position % basis_size_;
      const bool possible =
          std::ranges::any_of(requests_[request].terms, [&](const auto& term) {
            const std::size_t source_position =
                static_cast<std::size_t>(term.target) * basis_size_ + basis;
            return source_position_to_output_.at(source_position) != invalid_output;
          });
      if (possible)
        reconstructed_outputs_.push_back(static_cast<std::uint32_t>(position));
    }
  }

  void prime_changed()
  {
    native->prime_changed();
  }

  [[nodiscard]] std::span<const std::uint32_t> reconstructed_outputs() const noexcept
  {
    return reconstructed_outputs_;
  }

  [[nodiscard]] std::size_t total_output_count() const noexcept
  {
    return requests_.size() * basis_size_;
  }

  [[nodiscard]] const std::vector<ReductionRequest>& requests() const noexcept
  {
    return requests_;
  }

  template <typename T> std::vector<T> operator()(const std::vector<T>& values)
  {
    if constexpr (!std::is_same_v<T, firefly::FFInt>) {
      throw std::runtime_error("requested-output reconstruction is scalar only");
    } else {
      std::vector<std::uint32_t> outputs(reconstructed_outputs_.size());
      std::iota(outputs.begin(), outputs.end(), std::uint32_t{0});
      return eval_selected_compact(values, outputs);
    }
  }

  std::vector<firefly::FFInt>
  eval_selected_compact(const std::vector<firefly::FFInt>& values,
                        const std::vector<std::uint32_t>& active_outputs)
  {
    if (!std::ranges::is_sorted(active_outputs) ||
        std::ranges::adjacent_find(active_outputs) != active_outputs.end())
      throw std::invalid_argument("active requested outputs must be sorted and unique");
    if (!active_outputs.empty() &&
        active_outputs.back() >= reconstructed_outputs_.size())
      throw std::out_of_range("active requested output is out of range");
    if (active_outputs.empty()) return {};

    std::vector<std::uint32_t> source_outputs;
    for (const auto active : active_outputs) {
      const auto position = reconstructed_outputs_[active];
      const std::size_t request = position / basis_size_;
      const std::size_t basis = position % basis_size_;
      for (const auto& term : requests_[request].terms) {
        const std::size_t source_position =
            static_cast<std::size_t>(term.target) * basis_size_ + basis;
        const auto source_output = source_position_to_output_.at(source_position);
        if (source_output != invalid_output) source_outputs.push_back(source_output);
      }
    }
    std::ranges::sort(source_outputs);
    source_outputs.erase(std::unique(source_outputs.begin(), source_outputs.end()),
                         source_outputs.end());
    const auto source_values = native->eval_selected_compact(values, source_outputs);
    if (!source_outputs.empty() && source_values.empty()) return {};
    if (source_values.size() != source_outputs.size())
      throw std::logic_error("native selected output shape is inconsistent");

    std::vector<ulong> coordinates;
    coordinates.reserve(values.size());
    for (const auto& value : values)
      coordinates.push_back(value.n);
    std::vector<std::vector<firefly::FFInt>> weights(requests_.size());
    std::vector<std::uint8_t> evaluated(requests_.size(), 0);
    std::vector<firefly::FFInt> result;
    result.reserve(active_outputs.size());
    for (const auto active : active_outputs) {
      const auto position = reconstructed_outputs_[active];
      const std::size_t request = position / basis_size_;
      const std::size_t basis = position % basis_size_;
      if (evaluated[request] == 0) {
        auto& request_weights = weights[request];
        request_weights.reserve(coefficients_[request].size());
        for (const auto& coefficient : coefficients_[request]) {
          const auto value =
              evaluate_coefficient(coefficient, coordinates, firefly::FFInt::p);
          if (!value) return {};
          request_weights.emplace_back(*value);
        }
        evaluated[request] = 1;
      }
      firefly::FFInt value(0);
      for (std::size_t term_index = 0; term_index < requests_[request].terms.size();
           ++term_index) {
        const auto target = requests_[request].terms[term_index].target;
        const std::size_t source_position =
            static_cast<std::size_t>(target) * basis_size_ + basis;
        const auto source_output = source_position_to_output_.at(source_position);
        if (source_output == invalid_output) continue;
        const auto found = std::ranges::lower_bound(source_outputs, source_output);
        if (found == source_outputs.end() || *found != source_output)
          throw std::logic_error("requested source output was not evaluated");
        const auto index = static_cast<std::size_t>(found - source_outputs.begin());
        value = value + weights[request][term_index] * source_values[index];
      }
      result.push_back(value);
    }
    return result;
  }

  std::unique_ptr<Native> native;

private:
  static std::optional<ulong> evaluate_coefficient(const FlintRational& coefficient,
                                                   std::span<const ulong> coordinates,
                                                   ulong prime)
  {
    if (coordinates.size() != coefficient.context()->variable_names().size())
      throw std::logic_error("combination coefficient coordinate mismatch");
    nmod_t modulus;
    nmod_init(&modulus, prime);
    const auto numerator = fmpz_mpoly_evaluate_all_nmod(
        fmpz_mpoly_q_numref(coefficient.raw()), coordinates.data(),
        coefficient.context()->raw(), modulus);
    const auto denominator = fmpz_mpoly_evaluate_all_nmod(
        fmpz_mpoly_q_denref(coefficient.raw()), coordinates.data(),
        coefficient.context()->raw(), modulus);
    if (denominator == 0) return std::nullopt;
    return nmod_div(numerator, denominator, modulus);
  }

  static constexpr std::uint32_t invalid_output =
      std::numeric_limits<std::uint32_t>::max();
  std::size_t basis_size_ = 0;
  std::vector<ReductionRequest> requests_;
  std::shared_ptr<const FlintRationalContext> coefficient_context_;
  std::vector<std::vector<FlintRational>> coefficients_;
  std::vector<std::uint32_t> source_position_to_output_;
  std::vector<std::uint32_t> reconstructed_outputs_;
};

} // namespace reduction::detail
