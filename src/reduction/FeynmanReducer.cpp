#include "reduction/FeynmanReducer.hpp"
#include "reduction/detail/PreparedReconstruction.hpp"
#include "reduction/detail/ReductionStage.hpp"

#include "basis/DSeparatingReduction.hpp"
#include "masters/detail/GlobalBasisSelector.hpp"
#include "reduction/BlackBoxFeynman.hpp"
#include "reduction/ConstantRationalReconstruction.hpp"
#include "reduction/DSeparationCheck.hpp"
#include "reduction/DifferentialEquations.hpp"
#include "reduction/FireflyResultImport.hpp"
#include "reduction/ParameterEvaluation.hpp"
#include "reduction/RequestedOutputBlackBox.hpp"
#include "reduction/ScaleReconstruction.hpp"

#include <firefly/Reconstructor.hpp>

#ifndef FEYNMAN_REDUCER_FIREFLY_OVERLAY
#error "FeynmanReducer requires the repository FireFly Reconstructor overlay"
#endif

#include <algorithm>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using reduction::detail::run_reduction_stage;

template <typename BlackBox>
ReductionResult reconstruct_prepared(Config& config,
                                     std::unique_ptr<BlackBox> native_black_box,
                                     ReductionProgressCallback progress)
{
  if (config.threads == 0) {
    throw std::runtime_error("thread count must be positive");
  }
  using Requested = reduction::detail::RequestedOutputBlackBox<BlackBox>;
  auto requested_black_box =
      std::make_unique<Requested>(config, std::move(native_black_box));
  if (progress && !config.reduction_requests.empty())
    progress(std::format("Requested outputs: source_targets={}, requests={}, "
                         "reconstructed_candidates={}",
                         config.targets.size(), config.reduction_requests.size(),
                         requested_black_box->reconstructed_outputs().size()),
             ReductionProgressEvent::info);
  constexpr unsigned maximum_bunch_size = 1;
  std::optional<std::size_t> scale;
  std::uint64_t scan_probes = 0;
  const bool automatic = config.reconstruction_scale == "auto";
  const bool enabled = config.reconstruction_scale != "off";
  if (enabled && !automatic) {
    if (!config.scale_homogeneous)
      throw std::runtime_error("reconstruction_scale: " +
                               config.scale_homogeneity_reason);
    if (std::ranges::find(config.reconstruction_scale_candidates,
                          config.reconstruction_scale) ==
        config.reconstruction_scale_candidates.end())
      throw std::runtime_error(
          "reconstruction_scale must name a free kinematic parameter in F");
    scale = static_cast<std::size_t>(
        std::ranges::find(config.parameters, config.reconstruction_scale) -
        config.parameters.begin());
  } else if (enabled && config.scale_homogeneous &&
             !config.reconstruction_scale_candidates.empty() &&
             !requested_black_box->reconstructed_outputs().empty()) {
    std::vector<std::uint32_t> degrees(config.parameters.size(), 0);
    if (config.reconstruction_scale_candidates.size() > 1) {
      degrees = run_reduction_stage(progress, "Select reconstruction scale", [&] {
        firefly::RatReconst::reset();
        firefly::Reconstructor<Requested> scanner(
            static_cast<unsigned>(config.parameters.size()), config.threads,
            maximum_bunch_size, *requested_black_box);
        scanner.enable_factor_scan();
        scanner.stop_after_factor_scan();
        scanner.reconstruct();
        scan_probes = scanner.probe_count();
        return scanner.factor_scan_degrees();
      });
      firefly::RatReconst::reset();
      if (progress) {
        std::string scores;
        for (const auto& name : config.reconstruction_scale_candidates) {
          const auto index = static_cast<std::size_t>(
              std::ranges::find(config.parameters, name) - config.parameters.begin());
          if (!scores.empty()) scores += ",";
          scores += std::format("{}:{}", name, degrees.at(index));
        }
        progress(std::format("Scale scan: post_factor_degrees=[{}], probes={}", scores,
                             scan_probes),
                 ReductionProgressEvent::info);
      }
    }
    scale = reduction::detail::select_reconstruction_scale(config, degrees);
  } else if (enabled && progress) {
    progress("Scale normalization skipped: " +
                 (config.scale_homogeneous
                      ? std::string("no scale candidate or no reconstructed outputs")
                      : config.scale_homogeneity_reason),
             ReductionProgressEvent::info);
  }
  auto reconstruction_parameters = config.parameters;
  if (scale) {
    if (progress)
      progress(std::format("Reconstruction scale: {}=1, variables={} -> {}",
                           config.parameters.at(*scale), config.parameters.size(),
                           config.parameters.size() - 1),
               ReductionProgressEvent::info);
    reconstruction_parameters.erase(reconstruction_parameters.begin() +
                                    static_cast<std::ptrdiff_t>(*scale));
  }
  using Adapter = reduction::detail::ScaleBlackBox<Requested>;
  auto black_box = std::make_unique<Adapter>(std::move(requested_black_box), scale);
  auto context = std::make_shared<FlintRationalContext>(config.parameters);
  auto reconstruction_context =
      scale ? std::make_shared<FlintRationalContext>(reconstruction_parameters)
            : context;
  using Reconstructor = firefly::Reconstructor<Adapter>;
  std::unique_ptr<Reconstructor> reconstructor;
  std::vector<FlintRational> constant_results;
  if (reconstruction_parameters.empty()) {
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
              output_count, reconstruction_context, primes, [&](std::uint64_t prime) {
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
          firefly::RatReconst::reset();
          auto instance = std::make_unique<Reconstructor>(
              static_cast<unsigned>(reconstruction_parameters.size()), config.threads,
              maximum_bunch_size, *black_box);
          if (config.factor_scan) instance->enable_factor_scan();
          if (config.shift_scan) instance->enable_shift_scan();
          try {
            instance->reconstruct();
          } catch (const std::exception& error) {
            throw std::runtime_error(
                std::format("FireFly reconstruction failed: {}", error.what()));
          }
          if (progress)
            progress(
                std::format("Reconstruction probes: scale_scan={}, main={}, total={}",
                            scan_probes, instance->probe_count(),
                            scan_probes + instance->probe_count()),
                ReductionProgressEvent::info);
          return instance;
        });
  }

  auto materialized =
      run_reduction_stage(progress, "Materialize reduction result", [&] {
        const auto reconstructed_outputs = black_box->reconstructed_outputs();
        const std::size_t full_output_count = black_box->total_output_count();
        ReductionResult result;
        result.integral_header = config.integral_header;
        result.parameters = config.parameters;
        result.numerics = config.numerics;
        result.basis = config.basis;
        result.targets = config.targets;
        if (config.differential_equations)
          result.differential_parameters = config.kinematic_parameters;
        for (const auto& request : black_box->native->requests())
          result.outputs.push_back(request.output);
        result.context = context;
        result.coefficients.reserve(full_output_count);
        for (std::size_t output = 0; output < full_output_count; ++output)
          result.coefficients.emplace_back(context);
        std::size_t received = 0;
        const auto store = [&](std::size_t output, FactorizedRational value) {
          if (output != received || output >= reconstructed_outputs.size())
            throw std::logic_error("reconstructed output order or count is invalid");
          const std::size_t full_position = reconstructed_outputs[output];
          if (full_position >= result.coefficients.size())
            throw std::logic_error("reconstructed output position is out of range");
          if (scale)
            value = reduction::detail::restore_reconstruction_scale(
                value, context, *scale,
                result.outputs.at(full_position / config.basis.size()).scale_offset,
                config.basis.at(full_position % config.basis.size()));
          result.coefficients[full_position] = std::move(value);
          ++received;
        };
        if (reconstruction_parameters.empty()) {
          for (std::size_t output = 0; output < constant_results.size(); ++output)
            store(output, FactorizedRational(std::move(constant_results[output])));
        } else if (reconstructor != nullptr) {
          reconstructor->consume_results([&](std::uint64_t output, const auto& value) {
            store(output, reduction_detail::import_firefly_rational(
                              value, reconstruction_context));
          });
        }
        if (received != reconstructed_outputs.size()) {
          throw std::runtime_error(
              std::format("reconstruction result shape mismatch: expected {}, got {}",
                          reconstructed_outputs.size(), received));
        }
        // Consumption joins FireFly workers before the prime changes below.
        reconstructor.reset();
        return result;
      });
  if (scale && !black_box->reconstructed_outputs().empty())
    run_reduction_stage(progress, "Validate restored scale", [&] {
      reduction::detail::validate_restored_scale(
          config, materialized, black_box->reconstructed_outputs(),
          [&] { black_box->native->prime_changed(); },
          [&](const std::vector<firefly::FFInt>& values) {
            return (*black_box->native)(values);
          });
    });
  run_reduction_stage(progress, "Check reconstructed D-separation", [&] {
    reduction::detail::check_d_separation(materialized, config.basis_selection,
                                          progress);
  });
  return materialized;
}

ReductionResult perform_reduction_impl(Config& config,
                                       const masters::MasterCandidateSet* candidates,
                                       ReductionProgressCallback progress,
                                       ReductionOptions options)
{
  if (config.basis_selection == BasisSelectionPolicy::DSeparating) {
    if (options.d_separating_kernel == DSeparatingKernelStrategy::Auto) {
      options.d_separating_kernel =
          options.d_separating_shared == DSeparatingSharedOptimization::None
              ? DSeparatingKernelStrategy::ReuseProvisionalReselectFinalBasis
              : DSeparatingKernelStrategy::SharedOracle;
    }
    std::vector<std::uint32_t> relation_source_sectors;
    if (candidates != nullptr) {
      if (!candidates->requires_global_selection()) {
        throw std::invalid_argument(
            "master preselection requires a global-selection candidate set");
      }
      config.basis = masters::detail::select_global_master_basis(
          config, candidates->integrals, candidates->relation_source_sectors);
      relation_source_sectors = candidates->relation_source_sectors;
      options.master_basis_globally_selected = true;
    }
    if (config.basis.empty())
      throw std::runtime_error(
          "D-separating basis selection requires an initial master basis");
    auto black_box =
        run_reduction_stage(progress, "Prepare shared D-separating reduction", [&] {
          return basis::DSeparatingReduction::prepare(
              config, config.basis, relation_source_sectors, progress, options);
        });
    std::unique_ptr<BlackBoxFeynman> final_kernel;
    if (uses_recompact_source(options.d_separating_kernel)) {
      auto source = black_box->take_recompact_source();
      black_box.reset();
      final_kernel = run_reduction_stage(
          progress, "Recompact final D-separating reduction kernel", [&] {
            return BlackBoxFeynman::prepare_from_source(config, std::move(source),
                                                        progress, options);
          });
    } else if (options.d_separating_kernel ==
               DSeparatingKernelStrategy::RebuildFinalBasis) {
      black_box.reset();
      final_kernel = run_reduction_stage(
          progress, "Prepare final D-separating reduction kernel", [&] {
            return BlackBoxFeynman::prepare(config, relation_source_sectors, progress,
                                            options);
          });
    } else {
      return reconstruct_prepared(config, std::move(black_box), std::move(progress));
    }
    if (options.d_separating_replay != TargetReplayPolicy::Exact) {
      const bool enabled =
          final_kernel->configure_target_replay(options.d_separating_replay);
      if (progress)
        progress(
            enabled
                ? (options.d_separating_replay == TargetReplayPolicy::TargetRowsCached
                       ? "Final replay: target-row reuse with cache enabled"
                       : "Final replay: target-row reuse enabled")
                : "Final replay: target-row reuse skipped (master orientation)",
            ReductionProgressEvent::info);
    }
    return reduction::detail::reconstruct_prepared_kernel(
        config, std::move(final_kernel), std::move(progress));
  }

  options.d_separating_kernel = DSeparatingKernelStrategy::SharedOracle;
  auto black_box = run_reduction_stage(progress, "Prepare reduction kernel", [&] {
    if (candidates == nullptr)
      reduction::detail::materialize_differential_equations(config);
    return candidates == nullptr
               ? BlackBoxFeynman::prepare(config, progress, options)
               : BlackBoxFeynman::prepare(config, *candidates, progress, options);
  });
  if (options.default_replay != TargetReplayPolicy::Exact) {
    const bool enabled = black_box->configure_target_replay(options.default_replay);
    if (progress)
      progress(enabled
                   ? "Default replay: target-row reuse enabled"
                   : "Default replay: target-row reuse skipped (master orientation)",
               ReductionProgressEvent::info);
  }
  if (options.default_master_replay != MasterReplayGrouping::Exact) {
    const bool enabled =
        black_box->configure_master_replay(options.default_master_replay);
    if (progress) {
      progress(!enabled ? "Default master replay: skipped (target orientation)"
               : options.default_master_replay == MasterReplayGrouping::TargetRows
                   ? "Default master replay: target-row reuse enabled"
                   : "Default master replay: master-column reuse enabled",
               ReductionProgressEvent::info);
    }
  }
  return reduction::detail::reconstruct_prepared_kernel(config, std::move(black_box),
                                                        std::move(progress));
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

ReductionResult
reduction::detail::reconstruct_prepared_kernel(Config& config,
                                               std::unique_ptr<BlackBoxFeynman> kernel,
                                               ReductionProgressCallback progress)
{
  return reconstruct_prepared(config, std::move(kernel), std::move(progress));
}
