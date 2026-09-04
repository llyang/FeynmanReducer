#pragma once

#include "core/FlintRational.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace reduction::detail {

using ConstantFiniteFieldEvaluator =
    std::function<std::vector<std::uint64_t>(std::uint64_t prime)>;

[[nodiscard]] std::vector<FlintRational> reconstruct_constant_rationals(
    std::size_t output_count,
    const std::shared_ptr<const FlintRationalContext>& context,
    const std::vector<std::uint64_t>& primes,
    const ConstantFiniteFieldEvaluator& evaluate);

} // namespace reduction::detail
