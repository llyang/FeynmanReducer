#pragma once

#include "core/FlintRational.hpp"

#include <memory>

namespace firefly {
class RationalFunction;
}

namespace reduction_detail {

/// Converts FireFly's reconstructed representation, including scanned factors
/// and variable-order metadata, into the original FLINT parameter context.
[[nodiscard]] FlintRational
import_firefly_rational(const firefly::RationalFunction& source,
                        const std::shared_ptr<const FlintRationalContext>& context);

} // namespace reduction_detail
