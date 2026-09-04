#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace reduction::detail {

// Computes all reciprocals with one field division.  The caller supplies the
// prefix scratch so repeated finite-field probes can reuse their allocation.
// No output is modified when an input is zero.
template <typename Field>
[[nodiscard]] std::optional<std::size_t>
batch_inverse(std::span<const Field> values, std::span<Field> inverses,
              std::vector<Field>& prefix)
{
  if (inverses.size() != values.size()) {
    throw std::invalid_argument("batch inverse input/output sizes differ");
  }
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (values[index] == Field(0)) return index;
  }
  if (values.empty()) return std::nullopt;

  prefix.resize(values.size());
  Field product(1);
  for (std::size_t index = 0; index < values.size(); ++index) {
    prefix[index] = product;
    product = product * values[index];
  }

  Field suffix = Field(1) / product;
  for (std::size_t index = values.size(); index-- > 0;) {
    inverses[index] = suffix * prefix[index];
    suffix = suffix * values[index];
  }
  return std::nullopt;
}

} // namespace reduction::detail
