#pragma once

#include "core/Config.hpp"
#include "core/FlintRational.hpp"
#include "core/ReductionResult.hpp"

#include <firefly/BlackBoxBase.hpp>
#include <firefly/FFInt.hpp>

#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace reduction::detail {

// A view over the original, fully symbolic kernel. Output identifiers are unchanged.
template <typename Native>
class ScaleBlackBox : public firefly::BlackBoxBase<ScaleBlackBox<Native>> {
public:
  ScaleBlackBox(std::unique_ptr<Native> source, std::optional<std::size_t> scale)
      : native(std::move(source)), scale_(scale)
  {}

  void prime_changed()
  {
    native->prime_changed();
  }
  auto reconstructed_outputs() const
  {
    return native->reconstructed_outputs();
  }
  auto total_output_count() const
  {
    return native->total_output_count();
  }

  template <typename T> std::vector<T> operator()(const std::vector<T>& values)
  {
    if constexpr (std::is_same_v<T, firefly::FFInt>) {
      if (!scale_) return (*native)(values);
      return (*native)(expand(values));
    } else {
      throw std::runtime_error("scale reconstruction supports scalar probes only");
    }
  }
  std::vector<firefly::FFInt>
  eval_selected_compact(const std::vector<firefly::FFInt>& values,
                        const std::vector<std::uint32_t>& outputs)
  {
    if (!scale_) return native->eval_selected_compact(values, outputs);
    return native->eval_selected_compact(expand(values), outputs);
  }

  std::unique_ptr<Native> native;

private:
  std::optional<std::size_t> scale_;
  std::vector<firefly::FFInt> expand(const std::vector<firefly::FFInt>& values) const
  {
    if (*scale_ > values.size())
      throw std::logic_error("invalid reconstruction scale index");
    std::vector<firefly::FFInt> full(values.size() + 1);
    for (std::size_t i = 0; i < values.size(); ++i)
      full[i < *scale_ ? i : i + 1] = values[i];
    full[*scale_] = firefly::FFInt(1);
    return full;
  }
};

std::size_t select_reconstruction_scale(const Config& config,
                                        std::span<const std::uint32_t> degrees);

FactorizedRational restore_reconstruction_scale(
    const FactorizedRational& reduced,
    const std::shared_ptr<const FlintRationalContext>& full_context, std::size_t scale,
    const Integral& target, const Integral& master);

void validate_restored_scale(const Config& config, const ReductionResult& result,
                             std::span<const std::uint32_t> outputs,
                             const std::function<void()>& prime_changed,
                             const std::function<std::vector<firefly::FFInt>(
                                 const std::vector<firefly::FFInt>&)>& evaluate);

} // namespace reduction::detail
