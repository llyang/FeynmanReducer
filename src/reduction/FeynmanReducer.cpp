#include "reduction/FeynmanReducer.hpp"

#include "basis/DSeparatingReduction.hpp"
#include "masters/detail/GlobalBasisSelector.hpp"
#include "reduction/BlackBoxFeynman.hpp"
#include "reduction/ConstantRationalReconstruction.hpp"
#include "reduction/FireflyResultImport.hpp"
#include "reduction/ParameterEvaluation.hpp"

#include <firefly/Reconstructor.hpp>

#include <format>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace {

template <typename Function>
std::invoke_result_t<Function>
run_reduction_stage(const ReductionProgressCallback& progress, std::string label,
                    Function&& function)
{
  if (progress) progress(label, ReductionProgressEvent::started);
  try {
    if constexpr (std::is_void_v<std::invoke_result_t<Function>>) {
      std::invoke(std::forward<Function>(function));
      if (progress) progress(label, ReductionProgressEvent::completed);
    } else {
      std::invoke_result_t<Function> result =
          std::invoke(std::forward<Function>(function));
      if (progress) progress(label, ReductionProgressEvent::completed);
      return result;
    }
  } catch (...) {
    if (progress) {
      try {
        progress(label, ReductionProgressEvent::failed);
      } catch (...) {
      }
    }
    throw;
  }
}

} // namespace

namespace {

template <typename BlackBox>
ReductionResult reconstruct_prepared(Config& config,
                                     std::unique_ptr<BlackBox> black_box,
                                     ReductionProgressCallback progress)
{
  if (config.threads == 0) {
    throw std::runtime_error("thread count must be positive");
  }
  constexpr unsigned maximum_bunch_size = 1;
  auto context = std::make_shared<FlintRationalContext>(config.parameters);
  using Reconstructor = firefly::Reconstructor<BlackBox>;
  std::unique_ptr<Reconstructor> reconstructor;
  std::vector<FlintRational> constant_results;
  if (config.parameters.empty()) {
    constant_results =
        run_reduction_stage(progress, "Run constant rational reconstruction", [&] {
          const auto output_count = black_box->reconstructed_outputs().size();
          std::vector<std::uint64_t> primes;
          for (const std::uint64_t prime : firefly::primes()) {
            if (reduction::detail::configuration_is_usable_at_prime(config, prime))
              primes.push_back(prime);
          }
          if (progress && (config.factor_scan || config.shift_scan)) {
            progress("FireFly scans skipped for a fully numeric reconstruction",
                     ReductionProgressEvent::info);
          }
          return reduction::detail::reconstruct_constant_rationals(
              output_count, context, primes, [&](std::uint64_t prime) {
                firefly::FFInt::set_new_prime(prime);
                black_box->prime_changed();
                const auto values = (*black_box)(std::vector<firefly::FFInt>{});
                std::vector<std::uint64_t> result;
                result.reserve(values.size());
                for (const auto& value : values)
                  result.push_back(value.n);
                return result;
              });
        });
  } else {
    reconstructor = run_reduction_stage(
        progress,
        std::format("Run FireFly reconstruction (threads={})", config.threads), [&] {
          if (black_box->reconstructed_outputs().empty()) {
            if (progress) {
              progress(
                  "FireFly reconstruction skipped: all outputs are probabilistic zero",
                  ReductionProgressEvent::info);
            }
            return std::unique_ptr<Reconstructor>{};
          }
          auto instance = std::make_unique<Reconstructor>(
              static_cast<unsigned>(config.parameters.size()), config.threads,
              maximum_bunch_size, *black_box);
          if (config.factor_scan) instance->enable_factor_scan();
          if (config.shift_scan) instance->enable_shift_scan();
          try {
            instance->reconstruct();
          } catch (const std::exception& error) {
            throw std::runtime_error(
                std::format("FireFly reconstruction failed: {}", error.what()));
          }
          return instance;
        });
  }

  return run_reduction_stage(progress, "Materialize reduction result", [&] {
    const auto reconstructed_outputs = black_box->reconstructed_outputs();
    std::vector<FlintRational> reconstructed;
    if (config.parameters.empty()) {
      reconstructed = std::move(constant_results);
    } else if (reconstructor != nullptr) {
      const auto firefly_result = reconstructor->get_result();
      reconstructed.reserve(firefly_result.size());
      for (const auto& value : firefly_result) {
        reconstructed.push_back(
            reduction_detail::import_firefly_rational(value, context));
      }
    }
    if (reconstructed.size() != reconstructed_outputs.size()) {
      throw std::runtime_error(
          std::format("reconstruction result shape mismatch: expected {}, got {}",
                      reconstructed_outputs.size(), reconstructed.size()));
    }

    const std::size_t full_output_count = black_box->total_output_count();
    ReductionResult result{config.integral_header,
                           config.parameters,
                           config.numerics,
                           config.basis,
                           config.targets,
                           context,
                           {}};
    result.coefficients.reserve(full_output_count);
    for (std::size_t output = 0; output < full_output_count; ++output)
      result.coefficients.emplace_back(context);
    for (std::size_t output = 0; output < reconstructed.size(); ++output) {
      const std::size_t full_position = reconstructed_outputs[output];
      if (full_position >= result.coefficients.size())
        throw std::logic_error("reconstructed output position is out of range");
      result.coefficients[full_position] = std::move(reconstructed[output]);
    }
    return result;
  });
}

ReductionResult perform_reduction_impl(Config& config,
                                       const masters::MasterCandidateSet* candidates,
                                       ReductionProgressCallback progress,
                                       ReductionOptions options)
{
  if (config.basis_selection == BasisSelectionPolicy::DSeparating) {
    std::vector<std::uint32_t> relation_source_sectors;
    if (candidates != nullptr) {
      if (!candidates->requires_global_selection()) {
        throw std::invalid_argument(
            "master preselection requires a global-selection candidate set");
      }
      config.basis = masters::detail::select_global_master_basis(
          config, candidates->integrals, candidates->relation_source_sectors);
      relation_source_sectors = candidates->relation_source_sectors;
    }
    if (config.basis.empty())
      throw std::runtime_error(
          "D-separating basis selection requires an initial master basis");
    auto black_box =
        run_reduction_stage(progress, "Prepare shared D-separating reduction", [&] {
          return basis::DSeparatingReduction::prepare(
              config, config.basis, relation_source_sectors, progress, options);
        });
    return reconstruct_prepared(config, std::move(black_box), std::move(progress));
  }

  auto black_box = run_reduction_stage(progress, "Prepare reduction kernel", [&] {
    return candidates == nullptr
               ? BlackBoxFeynman::prepare(config, progress, options)
               : BlackBoxFeynman::prepare(config, *candidates, progress, options);
  });
  return reconstruct_prepared(config, std::move(black_box), std::move(progress));
}

} // namespace

ReductionResult perform_reduction(const Config& config,
                                  ReductionProgressCallback progress,
                                  ReductionOptions options)
{
  Config owned = config;
  return perform_reduction_owned(std::move(owned), std::move(progress), options);
}

ReductionResult perform_reduction_owned(Config config,
                                        ReductionProgressCallback progress,
                                        ReductionOptions options)
{
  return perform_reduction_impl(config, nullptr, std::move(progress), options);
}

ReductionResult perform_reduction(Config config, masters::MasterCandidateSet candidates,
                                  ReductionProgressCallback progress,
                                  ReductionOptions options)
{
  return perform_reduction_impl(config, &candidates, std::move(progress), options);
}
