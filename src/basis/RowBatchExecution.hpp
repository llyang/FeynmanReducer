#pragma once
#include "NativeFixedBasisOracle.hpp"
#include <stdexcept>

namespace basis::detail {
struct RowBatchResult {
  std::vector<std::optional<FieldVector>> rows;
  std::size_t attempts = 0, splits = 0, attempted_rows = 0;
  std::size_t replay_calls = 0, selected_outputs = 0;
};
// A union can have extra singular dependencies. Only a failed numeric evaluation
// is retried; exceptions and malformed results propagate. No zero placeholders.
template<class Evaluate>
RowBatchResult evaluate_row_batch(std::span<const std::size_t> rows, Evaluate&& evaluate)
{
  RowBatchResult result;
  result.rows.resize(rows.size());
  const auto attempt = [&](std::span<const std::size_t> requested, std::size_t offset) {
    ++result.attempts;
    result.attempted_rows += requested.size();
    auto got = evaluate(requested);
    if (!got) return false;
    if (got->conventional.size() != requested.size())
      throw std::logic_error("selected native basis probe returned the wrong row count");
    result.replay_calls += got->replay_performed;
    result.selected_outputs += got->selected_reconstructed_outputs;
    for (std::size_t i = 0; i < requested.size(); ++i)
      result.rows[offset+i] = std::move(got->conventional[i]);
    return true;
  };
  if (!rows.empty() && !attempt(rows, 0) && rows.size() > 1) {
    ++result.splits;
    for (std::size_t i = 0; i < rows.size(); ++i)
      attempt(rows.subspan(i, 1), i);
  }
  return result;
}
} // namespace basis::detail
