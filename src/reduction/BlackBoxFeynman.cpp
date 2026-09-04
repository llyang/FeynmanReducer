#include "reduction/BlackBoxFeynman.hpp"
#include "core/ProbeValues.hpp"
#include "masters/detail/GlobalBasisSelector.hpp"
#include "reduction/EliminationTape.hpp"
#include "reduction/FiniteFieldArithmetic.hpp"
#include "reduction/ParameterEvaluation.hpp"
#include "topology/IntegralLayout.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <format>
#include <limits>
#include <map>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace {

constexpr std::size_t kProbeAttemptLimit = 128;
constexpr std::size_t kValidationPointsPerPrime = 2;
constexpr std::size_t kResidualProjectionCount = 2;

[[nodiscard]] bool uses_direct_kernel(const Config& config,
                                      const ReductionOptions& options)
{
  if (options.numerator_strategy != NumeratorReductionStrategy::Direct) return false;
  if (options.force_direct_positive_targets) return true;
  return std::ranges::any_of(config.targets, [](const auto& target) {
    return std::ranges::any_of(target.indices,
                               [](const int index) { return index < 0; });
  });
}

[[nodiscard]] std::uint64_t mix_projection_seed(std::uint64_t value)
{
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

[[nodiscard]] std::uint64_t projection_seed(std::span<const firefly::FFInt> values,
                                            std::size_t projection)
{
  std::uint64_t seed = mix_projection_seed(firefly::FFInt::p ^ (projection + 1U));
  for (const auto& value : values)
    seed = mix_projection_seed(seed ^ value.n);
  return seed;
}

[[nodiscard]] firefly::FFInt projection_weight(std::uint64_t seed, std::size_t column)
{
  const std::uint64_t value =
      mix_projection_seed(seed ^ mix_projection_seed(column + 1U));
  return firefly::FFInt(1U + value % (firefly::FFInt::p - 1U));
}

template <typename T>
[[nodiscard]] bool execute_validation_tape(std::vector<T>& matrix, std::vector<T>& rhs,
                                           std::span<const linalg::Instruction> tape)
{
  using enum linalg::OpCode;
  T factor(0);
  for (const auto instruction : tape) {
    switch (instruction.opcode()) {
    case Inv:
      if (matrix[instruction.source()] == T(0)) return false;
      factor = T(1) / matrix[instruction.source()];
      break;
    case MulM:
      matrix[instruction.offset()] = matrix[instruction.offset()] * factor;
      break;
    case MulB:
      rhs[instruction.offset()] = rhs[instruction.offset()] * factor;
      break;
    case LoadF:
      factor = matrix[instruction.source()];
      break;
    case FmaM:
      matrix[instruction.offset()] =
          matrix[instruction.offset()] - factor * matrix[instruction.source()];
      break;
    case FmaB:
      rhs[instruction.offset()] =
          rhs[instruction.offset()] - factor * rhs[instruction.source()];
      break;
    }
  }
  return true;
}

struct ValidationSample {
  std::size_t prime_index = 0;
  std::uint64_t prime = 0;
  std::uint64_t point = 0;
  std::vector<std::uint64_t> values;
  std::vector<std::uint64_t> reference_outputs;
};

std::vector<firefly::FFInt> make_anchor_values(std::size_t parameter_count,
                                               std::uint64_t prime, std::size_t attempt)
{
  std::vector<firefly::FFInt> values;
  values.reserve(parameter_count);
  for (std::size_t parameter = 0; parameter < parameter_count; ++parameter) {
    values.emplace_back(probe_values::planning_field_value(prime, parameter, attempt));
  }
  return values;
}

std::vector<firefly::FFInt> make_validation_values(std::size_t parameter_count,
                                                   std::uint64_t prime,
                                                   std::uint64_t point)
{
  std::vector<firefly::FFInt> values;
  values.reserve(parameter_count);
  for (std::size_t parameter = 0; parameter < parameter_count; ++parameter) {
    values.emplace_back(probe_values::field_value(prime, parameter, point));
  }
  return values;
}

void require_same_results(const std::vector<firefly::FFInt>& reference,
                          const std::vector<firefly::FFInt>& replay,
                          std::size_t prime_index, std::uint64_t prime,
                          std::uint64_t point)
{
  if (reference.size() != replay.size()) {
    throw std::runtime_error(std::format(
        "kernel replay validation failed: kind=shape-mismatch, prime_index={}, "
        "prime={}, point={}, reference_outputs={}, replay_outputs={}",
        prime_index + 1, prime, point, reference.size(), replay.size()));
  }
  for (std::size_t index = 0; index < reference.size(); ++index) {
    if (reference[index] != replay[index]) {
      throw std::runtime_error(std::format(
          "kernel replay validation failed: kind=value-mismatch, prime_index={}, "
          "prime={}, point={}, output={}, reference={}, replay={}",
          prime_index + 1, prime, point, index, reference[index].n, replay[index].n));
    }
  }
}

} // namespace

BlackBoxFeynman::BlackBoxFeynman(const Config& config, ReductionOptions options)
    : cfg(config), numerator_strategy(options.numerator_strategy),
      replay_orientation_preference(options.replay_orientation),
      ansatz_dot_ordering(options.ansatz_dot_ordering),
      direct_pinched_dot_halo(options.direct_pinched_dot_halo),
      direct_group_ordering(options.direct_group_ordering),
      direct_rank_ordering(options.direct_rank_ordering)
{
  for (const auto& master : cfg.basis) {
    if (std::ranges::any_of(master.indices, [](int index) { return index < 0; })) {
      throw std::invalid_argument(
          "negative indices are not supported in the master basis");
    }
  }
  const bool has_negative_targets =
      std::ranges::any_of(cfg.targets, [](const auto& target) {
        return std::ranges::any_of(target.indices, [](int index) { return index < 0; });
      });
  // The direct-jet representation only differs from the projected compact LP
  // representation when a target actually carries a boundary derivative.
  // Falling back here also keeps all downstream polynomial-term selection on
  // the single-layer projected topology for nonnegative target sets.
  if (!has_negative_targets && !options.force_direct_positive_targets)
    numerator_strategy = NumeratorReductionStrategy::Projected;
  if (numerator_strategy == NumeratorReductionStrategy::Projected)
    top_lp_target_plan = compile_top_lp_target_plan(cfg);
  kernel_statistics_.numerator_strategy =
      numerator_strategy == NumeratorReductionStrategy::Direct ? "direct" : "projected";
  std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t> product_ids;
  for (const auto& expression : top_lp_target_plan.expressions) {
    if (top_lp_expression_terms.size() > std::numeric_limits<std::uint32_t>::max())
      throw std::overflow_error("top-LP expression terms exceed 32-bit ids");
    const auto term_begin = static_cast<std::uint32_t>(top_lp_expression_terms.size());
    for (const auto& atom : expression.atoms) {
      std::uint32_t product = 0;
      for (const std::uint32_t factor : atom.polynomial_factors) {
        if (factor >= cfg.extended_lp.polynomial_terms.size())
          throw std::logic_error("top-LP polynomial factor is out of range");
        const auto [found, inserted] =
            product_ids.try_emplace(std::pair{product, factor}, 0);
        if (inserted) {
          if (top_lp_product_nodes.size() >=
              std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("top-LP product nodes exceed 32-bit ids");
          }
          top_lp_product_nodes.push_back({product, factor});
          found->second = static_cast<std::uint32_t>(top_lp_product_nodes.size());
        }
        product = found->second;
      }
      top_lp_maximum_falling_degree =
          std::max(top_lp_maximum_falling_degree, atom.falling_degree);
      top_lp_expression_terms.push_back({atom.weight, product, atom.falling_degree});
    }
    const std::size_t term_count = top_lp_expression_terms.size() - term_begin;
    if (term_count > std::numeric_limits<std::uint32_t>::max())
      throw std::overflow_error("top-LP expression is too long");
    top_lp_expression_programs.push_back(
        {term_begin, static_cast<std::uint32_t>(term_count)});
  }
  kernel_statistics_.maximum_g_shift = top_lp_target_plan.maximum_g_shift;
  kernel_statistics_.top_lp_target_expressions = top_lp_expression_programs.size();
  std::vector<TopLpCoefficientExpression>().swap(top_lp_target_plan.expressions);
  for (const auto& target : cfg.targets) {
    for (const int index : target.indices) {
      if (index < 0) {
        kernel_statistics_.maximum_delta_derivative =
            std::max(kernel_statistics_.maximum_delta_derivative,
                     static_cast<std::size_t>(-static_cast<std::int64_t>(index)));
      }
    }
  }
  const std::size_t coefficient_parameters =
      reduction::detail::coefficient_parameter_count(cfg);
  if (2 * coefficient_parameters > MAX_BILINEAR_BASIS) {
    throw std::runtime_error(std::format(
        "parameter coefficient basis size {} exceeds the supported limit {}",
        2 * coefficient_parameters, MAX_BILINEAR_BASIS));
  }
  const bool uses_extended_lp =
      has_negative_targets || numerator_strategy == NumeratorReductionStrategy::Direct;
  lp_variable_count = uses_extended_lp ? cfg.integral_count : cfg.propagator_count;
  kernel_statistics_.lp_variable_count = lp_variable_count;
  auto compile_lp_programs = [&](const auto& integrals) {
    std::vector<std::size_t> program_ids;
    program_ids.reserve(integrals.size());
    for (const auto& integral : integrals) {
      Integral active_integral;
      if (uses_extended_lp) {
        integral_layout::validate(cfg, integral);
        active_integral = integral;
      } else {
        active_integral = integral_layout::project_active(cfg, integral);
      }
      LpProgram program;
      for (const int index : active_integral.indices) {
        if (__builtin_add_overflow(program.index_sum, static_cast<std::int64_t>(index),
                                   &program.index_sum)) {
          throw std::overflow_error("integral index sum exceeds int64");
        }
      }
      program.delta =
          program.index_sum - static_cast<std::int64_t>(active_integral.indices.size());
      for (const int index : active_integral.indices) {
        if (index > 1) {
          for (std::int64_t factor = 1; factor < index; ++factor)
            program.denominator_factors.push_back(factor);
        }
      }
      const auto found = std::ranges::find(lp_programs, program);
      if (found == lp_programs.end()) {
        program_ids.push_back(lp_programs.size());
        lp_programs.push_back(std::move(program));
      } else {
        program_ids.push_back(
            static_cast<std::size_t>(std::distance(lp_programs.begin(), found)));
      }
    }
    return program_ids;
  };
  target_lp_program_ids = compile_lp_programs(cfg.targets);
  basis_lp_program_ids = compile_lp_programs(cfg.basis);
  for (const auto& program : lp_programs) {
    if (program.delta > 0) {
      maximum_positive_lp_delta =
          std::max(maximum_positive_lp_delta, static_cast<std::size_t>(program.delta));
    } else if (program.delta < 0) {
      if (program.delta == std::numeric_limits<std::int64_t>::min())
        throw std::overflow_error("negative LP delta exceeds size_t");
      maximum_negative_lp_delta =
          std::max(maximum_negative_lp_delta, static_cast<std::size_t>(-program.delta));
    }
  }
  full_lp_reciprocals.reserve(target_lp_program_ids.size() +
                              basis_lp_program_ids.size());
  for (const std::size_t program : target_lp_program_ids) {
    full_lp_reciprocals.push_back(
        {program, CoefficientSelection::LpReciprocalSide::TargetDenominator});
  }
  for (const std::size_t program : basis_lp_program_ids) {
    full_lp_reciprocals.push_back(
        {program, CoefficientSelection::LpReciprocalSide::BasisNumerator});
  }
  std::ranges::sort(full_lp_reciprocals);
  full_lp_reciprocals.erase(
      std::unique(full_lp_reciprocals.begin(), full_lp_reciprocals.end()),
      full_lp_reciprocals.end());
  kernel_statistics_.lp_expressions = lp_programs.size();
}

void BlackBoxFeynman::publish_compact_statistics(
    const reduction::detail::CompactSelection& selection)
{
  kernel_statistics_.provisional_dimension = selection.provisional_dimension;
  kernel_statistics_.provisional_rhs_columns = selection.provisional_rhs_columns;
  kernel_statistics_.provisional_relation_pivots =
      selection.provisional_relation_pivots;
  kernel_statistics_.provisional_score_refresh_columns =
      selection.provisional_score_refresh_columns;
  kernel_statistics_.provisional_incidence_records_scanned =
      selection.provisional_incidence_records_scanned;
  kernel_statistics_.provisional_parallel_refresh_batches =
      selection.provisional_parallel_refresh_batches;
  kernel_statistics_.provisional_parallel_refresh_columns =
      selection.provisional_parallel_refresh_columns;
  kernel_statistics_.provisional_stale_choice_pops =
      selection.provisional_stale_choice_pops;
  kernel_statistics_.provisional_row_eliminations =
      selection.provisional_row_eliminations;
  kernel_statistics_.provisional_parallel_row_batches =
      selection.provisional_parallel_row_batches;
  kernel_statistics_.provisional_parallel_row_eliminations =
      selection.provisional_parallel_row_eliminations;
  kernel_statistics_.provisional_rhs_fallback = selection.provisional_rhs_fallback;
  kernel_statistics_.provisional_relation_elimination_seconds =
      selection.timings.provisional_relation_elimination_ms / 1000.0;
  kernel_statistics_.provisional_back_substitution_seconds =
      selection.timings.provisional_back_substitution_ms / 1000.0;
  kernel_statistics_.provisional_score_refresh_seconds =
      selection.timings.provisional_score_refresh_ms / 1000.0;
  kernel_statistics_.provisional_row_elimination_seconds =
      selection.timings.provisional_row_elimination_ms / 1000.0;
}

void BlackBoxFeynman::publish_ansatz_statistics(
    const reduction::detail::CompactSelection& selection,
    std::span<const reduction::detail::AnsatzColumnMeta> ansatz_metadata,
    std::size_t generated_rows, std::size_t generated_columns,
    std::size_t expansion_rounds, std::size_t expanded_groups,
    std::size_t expanded_points)
{
  kernel_statistics_.generated_rows = generated_rows;
  kernel_statistics_.generated_columns = generated_columns;
  kernel_statistics_.provisional_ansatz_columns = ansatz_metadata.size();
  kernel_statistics_.provisional_ansatz_family_columns.fill(0);
  kernel_statistics_.provisional_ansatz_dot_histogram.clear();
  for (const auto& metadata : ansatz_metadata) {
    ++kernel_statistics_
          .provisional_ansatz_family_columns[static_cast<std::size_t>(metadata.family)];
    if (kernel_statistics_.provisional_ansatz_dot_histogram.size() <=
        metadata.seed_dot_excess) {
      kernel_statistics_.provisional_ansatz_dot_histogram.resize(
          static_cast<std::size_t>(metadata.seed_dot_excess) + 1, 0);
    }
    ++kernel_statistics_.provisional_ansatz_dot_histogram[metadata.seed_dot_excess];
  }
  kernel_statistics_.ansatz_expansion_rounds = expansion_rounds;
  kernel_statistics_.ansatz_expanded_groups = expanded_groups;
  kernel_statistics_.ansatz_expanded_points = expanded_points;
  publish_compact_statistics(selection);
  kernel_statistics_.live_ansatz_columns = selection.ansatz_order.size();
  kernel_statistics_.live_ansatz_dot_histogram.clear();
  for (const std::size_t column : selection.ansatz_order) {
    const auto excess = ansatz_metadata[column].seed_dot_excess;
    if (kernel_statistics_.live_ansatz_dot_histogram.size() <= excess) {
      kernel_statistics_.live_ansatz_dot_histogram.resize(
          static_cast<std::size_t>(excess) + 1, 0);
    }
    ++kernel_statistics_.live_ansatz_dot_histogram[excess];
  }
  kernel_statistics_.compact_dimension = selection.solution_columns.size();
  kernel_statistics_.cross_group_pivots = selection.cross_group_pivots;
}

std::unique_ptr<BlackBoxFeynman>
BlackBoxFeynman::prepare(const Config& config, ReductionProgressCallback progress,
                         ReductionOptions options)
{
  return prepare_impl(config, progress, options);
}

std::unique_ptr<BlackBoxFeynman>
BlackBoxFeynman::prepare(const Config& config,
                         std::span<const std::uint32_t> relation_source_sectors,
                         ReductionProgressCallback progress, ReductionOptions options)
{
  return prepare_impl(config, progress, options, relation_source_sectors);
}

std::unique_ptr<BlackBoxFeynman>
BlackBoxFeynman::prepare(Config& config, const masters::MasterCandidateSet& candidates,
                         ReductionProgressCallback progress, ReductionOptions options)
{
  if (!candidates.requires_global_selection()) {
    throw std::invalid_argument(
        "master preselection requires a global-selection candidate set");
  }
  const bool direct = uses_direct_kernel(config, options);
  const auto original_basis = config.basis;
  const auto selection_start = std::chrono::high_resolution_clock::now();
  try {
    config.basis = masters::detail::select_global_master_basis(
        config, candidates.integrals, candidates.relation_source_sectors);
    const auto selection_elapsed =
        std::chrono::duration<double>(std::chrono::high_resolution_clock::now() -
                                      selection_start)
            .count();
    if (direct && options.direct_group_ordering == DirectGroupOrdering::Auto)
      options.direct_group_ordering = DirectGroupOrdering::ExactJet;
    if (progress) {
      if (direct) {
        const auto group_order = [&] {
          switch (options.direct_group_ordering) {
          case DirectGroupOrdering::Auto:
            return "auto";
          case DirectGroupOrdering::ExactJet:
            return "exact-jet";
          case DirectGroupOrdering::RankSector:
            return "rank-sector";
          case DirectGroupOrdering::SectorRank:
            return "sector-rank";
          }
          return "unknown";
        }();
        progress(std::format("Direct master preselection: input={}, selected={}, "
                             "group_order={}, elapsed_ms={:.2f}",
                             candidates.integrals.size(), config.basis.size(),
                             group_order, selection_elapsed * 1000.0),
                 ReductionProgressEvent::info);
      } else {
        progress(std::format("Projected master preselection: input={}, selected={}, "
                             "elapsed_ms={:.2f}",
                             candidates.integrals.size(), config.basis.size(),
                             selection_elapsed * 1000.0),
                 ReductionProgressEvent::info);
      }
      progress(std::format("Production kernel planning: basis={}, targets={}, "
                           "relation_source_sectors={}",
                           config.basis.size(), config.targets.size(),
                           candidates.relation_source_sectors.size()),
               ReductionProgressEvent::info);
    }
    auto prepared =
        prepare_impl(config, progress, options, candidates.relation_source_sectors);
    prepared->kernel_statistics_.basis_selection_candidates =
        candidates.integrals.size();
    prepared->kernel_statistics_.basis_selection_masters = config.basis.size();
    prepared->kernel_statistics_.basis_selection_seconds = selection_elapsed;
    return prepared;
  } catch (...) {
    config.basis = original_basis;
    throw;
  }
}

std::unique_ptr<BlackBoxFeynman> BlackBoxFeynman::prepare_impl(
    const Config& config, const ReductionProgressCallback& progress,
    ReductionOptions options, std::span<const std::uint32_t> relation_source_sectors)
{
  const auto primes = reduction::detail::usable_firefly_primes(config, 2);

  std::unique_ptr<BlackBoxFeynman> prepared;
  std::size_t accepted_anchor_attempt = 0;
  std::array<std::size_t, 2> accepted_validation_points{};
  std::string last_error;
  for (std::size_t anchor_attempt = 0; anchor_attempt < kProbeAttemptLimit;
       ++anchor_attempt) {
    firefly::FFInt::set_new_prime(primes[0]);
    if (progress) {
      progress(std::format("Kernel planning: anchor_attempt={}", anchor_attempt + 1),
               ReductionProgressEvent::info);
    }
    auto candidate =
        std::unique_ptr<BlackBoxFeynman>(new BlackBoxFeynman(config, options));
    const auto values =
        make_anchor_values(config.parameters.size(), primes[0], anchor_attempt);
    try {
      const auto coeffs = candidate->evaluate_coefficients(values);
      candidate->plan_kernel(coeffs, values, progress, relation_source_sectors);
      std::vector<std::vector<Monomial>>().swap(candidate->top_lp_target_plan.columns);
    } catch (const AnsatzClosureError&) {
      throw;
    } catch (const std::logic_error&) {
      throw;
    } catch (const std::exception& error) {
      last_error = error.what();
      if (last_error.find("symmetry-equivalent") != std::string::npos) {
        throw;
      }
      if (progress) {
        progress(std::format("Kernel planning rejected anchor: {}", last_error),
                 ReductionProgressEvent::info);
      }
      continue;
    }

    bool selection_invalid = false;
    std::vector<std::uint8_t> nonzero_outputs(
        config.targets.size() * config.basis.size(), 0);
    std::vector<ValidationSample> validation_samples;
    validation_samples.reserve(2 * kValidationPointsPerPrime);
    auto evaluate_validation_reference = [&](std::uint64_t point,
                                             std::size_t prime_index) {
      const std::uint64_t prime = primes[prime_index];
      const auto probe = make_validation_values(config.parameters.size(), prime, point);
      try {
        const auto coeffs = candidate->evaluate_coefficients(probe);
        const auto reference = candidate->evaluate_reference(coeffs, probe);
        if (reference.empty()) {
          last_error = std::format(
              "kernel reference validation rejected point: kind=singular-system, "
              "prime_index={}, prime={}, point={}",
              prime_index + 1, prime, point);
          return false;
        }
        if (reference.size() != nonzero_outputs.size()) {
          throw std::logic_error("kernel validation output count is inconsistent");
        }
        for (std::size_t output = 0; output < reference.size(); ++output) {
          if (reference[output] != firefly::FFInt(0)) nonzero_outputs[output] = 1;
        }
        ValidationSample sample;
        sample.prime_index = prime_index;
        sample.prime = prime;
        sample.point = point;
        sample.values.reserve(probe.size());
        for (const auto& value : probe)
          sample.values.push_back(value.n);
        sample.reference_outputs.reserve(reference.size());
        for (const auto& value : reference)
          sample.reference_outputs.push_back(value.n);
        validation_samples.push_back(std::move(sample));
        return true;
      } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        if (message.find("closure validation residual") != std::string::npos) {
          selection_invalid = true;
        }
        last_error = std::format(
            "kernel reference validation rejected point: kind={}, prime_index={}, "
            "prime={}, point={}, reason={}",
            selection_invalid ? "closure-residual" : "evaluation-error",
            prime_index + 1, prime, point, message);
        return false;
      }
    };

    std::array<std::size_t, 2> validation_points{};
    bool validation_failed = false;
    const auto reference_validation_start = std::chrono::steady_clock::now();
    try {
      for (std::size_t prime_index = 0; prime_index < validation_points.size();
           ++prime_index) {
        firefly::FFInt::set_new_prime(primes[prime_index]);
        for (std::size_t attempt = 0;
             !selection_invalid && attempt < kProbeAttemptLimit &&
             validation_points[prime_index] < kValidationPointsPerPrime;
             ++attempt) {
          // Every planning-anchor retry must pass the same validation stream.
          // Otherwise a retry could evade a point that already exposed an
          // invalid numerically inferred Tape structure.
          const std::uint64_t point = probe_values::validation_point_index(attempt);
          if (evaluate_validation_reference(point, prime_index)) {
            ++validation_points[prime_index];
          }
        }
        if (selection_invalid ||
            validation_points[prime_index] != kValidationPointsPerPrime) {
          last_error =
              std::format("prime {} validation found only {} valid point(s): {}",
                          prime_index + 1, validation_points[prime_index], last_error);
          validation_failed = true;
          break;
        }
      }
    } catch (...) {
      firefly::FFInt::set_new_prime(primes[0]);
      throw;
    }
    candidate->kernel_statistics_.reference_validation_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      reference_validation_start)
            .count();
    firefly::FFInt::set_new_prime(primes[0]);
    if (selection_invalid || validation_failed) {
      if (progress) {
        progress(
            std::format("Kernel reference validation rejected anchor: {}", last_error),
            ReductionProgressEvent::info);
      }
      continue;
    }

    candidate->select_reconstructed_outputs(nonzero_outputs);
    const auto replay_finalize_start = std::chrono::steady_clock::now();
    candidate->finalize_replay(progress);
    candidate->kernel_statistics_.replay_finalize_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      replay_finalize_start)
            .count();
    bool replay_validation_failed = false;
    const auto replay_validation_start = std::chrono::steady_clock::now();
    if (!candidate->reconstructed_output_positions.empty()) {
      try {
        for (const auto& sample : validation_samples) {
          firefly::FFInt::set_new_prime(primes[sample.prime_index]);
          candidate->prime_changed();
          std::vector<firefly::FFInt> probe;
          probe.reserve(sample.values.size());
          for (const std::uint64_t value : sample.values)
            probe.emplace_back(value);
          const auto coeffs =
              candidate->evaluate_coefficients(probe, &candidate->replay_coefficients);
          const auto replay = candidate->execute_replay(coeffs, probe);
          std::vector<firefly::FFInt> expected;
          expected.reserve(candidate->reconstructed_output_positions.size());
          for (const std::uint32_t output : candidate->reconstructed_output_positions) {
            expected.emplace_back(sample.reference_outputs[output]);
          }
          if (replay.empty()) {
            throw std::runtime_error(
                std::format("kernel replay validation failed: kind=zero-pivot, "
                            "prime_index={}, prime={}, point={}",
                            sample.prime_index + 1, sample.prime, sample.point));
          }
          require_same_results(expected, replay, sample.prime_index, sample.prime,
                               sample.point);
        }
      } catch (const std::runtime_error& error) {
        last_error = error.what();
        replay_validation_failed = true;
        firefly::FFInt::set_new_prime(primes[0]);
        candidate->prime_changed();
      }
    }
    candidate->kernel_statistics_.replay_validation_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      replay_validation_start)
            .count();
    firefly::FFInt::set_new_prime(primes[0]);
    candidate->prime_changed();
    if (replay_validation_failed) {
      if (progress) {
        progress(
            std::format("Kernel replay validation rejected anchor: {}", last_error),
            ReductionProgressEvent::info);
      }
      continue;
    }
    accepted_anchor_attempt = anchor_attempt;
    accepted_validation_points = validation_points;
    prepared = std::move(candidate);
    break;
  }
  if (prepared == nullptr) {
    throw std::runtime_error(
        std::format("kernel planning and validation failed after {} anchor attempts; "
                    "last_error={}",
                    kProbeAttemptLimit, last_error));
  }

  prepared->reference_evaluation_plan.reset();
  if (progress) {
    progress(std::format("Kernel validation: anchor_attempts={}, prime1_points={}, "
                         "prime2_points={}",
                         accepted_anchor_attempt + 1, accepted_validation_points[0],
                         accepted_validation_points[1]),
             ReductionProgressEvent::info);
    progress(std::format("Output support: total_outputs={}, reconstructed_outputs={}, "
                         "probabilistic_zero_outputs={}",
                         prepared->kernel_statistics_.total_outputs,
                         prepared->kernel_statistics_.reconstructed_outputs,
                         prepared->kernel_statistics_.probabilistic_zero_outputs),
             ReductionProgressEvent::info);
  }
  return prepared;
}

void BlackBoxFeynman::select_reconstructed_outputs(
    std::span<const std::uint8_t> nonzero_outputs)
{
  const std::size_t total = total_output_count();
  if (nonzero_outputs.size() != total)
    throw std::logic_error("reduction output support shape mismatch");
  reconstructed_output_positions.clear();
  reconstructed_output_positions.reserve(total);
  for (std::size_t output = 0; output < total; ++output) {
    if (nonzero_outputs[output] == 0) continue;
    if (output > std::numeric_limits<std::uint32_t>::max())
      throw std::runtime_error("reduction output position exceeds 32-bit ids");
    reconstructed_output_positions.push_back(static_cast<std::uint32_t>(output));
  }
  output_support_initialized = true;
  kernel_statistics_.total_outputs = total;
  kernel_statistics_.reconstructed_outputs = reconstructed_output_positions.size();
  kernel_statistics_.probabilistic_zero_outputs =
      total - reconstructed_output_positions.size();
}

std::vector<firefly::FFInt>
BlackBoxFeynman::evaluate_reference(const EvaluatedCoeffs<firefly::FFInt>& coeffs,
                                    const std::vector<firefly::FFInt>& values) const
{
  using T = firefly::FFInt;
  if (reference_evaluation_plan == nullptr) {
    throw std::runtime_error("reference evaluation plan is unavailable");
  }
  const auto& plan = *reference_evaluation_plan;
  const std::size_t dimension = plan.dimension;
  const std::size_t target_count = cfg.targets.size();
  const std::size_t parameter_count =
      reduction::detail::coefficient_parameter_count(cfg);
  std::vector<T> bilinear_basis(2 * parameter_count);
  bilinear_basis[0] = T(1);
  for (std::size_t parameter = 0; parameter < cfg.kinematic_parameters.size();
       ++parameter) {
    bilinear_basis[parameter + 1] =
        reduction::detail::evaluate_kinematic_parameter(cfg, values, parameter);
  }
  bilinear_basis[parameter_count] = coeffs.minus_half_d;
  for (std::size_t parameter = 1; parameter < parameter_count; ++parameter) {
    bilinear_basis[parameter_count + parameter] =
        coeffs.minus_half_d * bilinear_basis[parameter];
  }

  if (pending_block_replay == nullptr)
    throw std::runtime_error("reference elimination plan is unavailable");
  const auto& pending = *pending_block_replay;
  if (pending.programs.size() != pending.recorded_blocks.size() ||
      pending.programs.size() != pending.operation_tapes.size() ||
      pending.reference_solution_row_by_column.size() != dimension) {
    throw std::logic_error("reference elimination plan shape is inconsistent");
  }

  std::vector<T> global_rhs(dimension * target_count, T(0));
  for (const auto& entry : pending.rhs_skeleton) {
    global_rhs[entry.flat_idx] =
        global_rhs[entry.flat_idx] + T(entry.weight) * bilinear_basis[entry.bb_idx];
  }
  for (const auto& entry : pending.top_lp_rhs_skeleton) {
    if (entry.expression >= coeffs.top_lp_coefficients.size())
      throw std::runtime_error("reference top-LP RHS expression is out of range");
    global_rhs[entry.flat_idx] =
        global_rhs[entry.flat_idx] +
        T(entry.weight) * coeffs.top_lp_coefficients[entry.expression];
  }

  for (std::size_t reverse = pending.programs.size(); reverse > 0; --reverse) {
    const std::size_t block = reverse - 1;
    const auto& program = pending.programs[block];
    const auto& recorded = pending.recorded_blocks[block];
    const std::size_t local_rhs_size =
        static_cast<std::size_t>(program.dimension) * target_count;
    const std::size_t global_rhs_begin =
        static_cast<std::size_t>(program.row_begin) * target_count;
    std::vector<T> local_rhs(local_rhs_size, T(0));
    std::ranges::copy_n(global_rhs.begin() + static_cast<ptrdiff_t>(global_rhs_begin),
                        static_cast<ptrdiff_t>(local_rhs_size), local_rhs.begin());
    if (!pending.operation_tapes[block].empty()) {
      std::vector<T> matrix(recorded.matrix_slot_count, T(0));
      for (const auto& entry : program.skeleton_M) {
        matrix[entry.flat_idx] =
            matrix[entry.flat_idx] + T(entry.weight) * bilinear_basis[entry.bb_idx];
      }
      for (const auto& entry : program.top_lp_skeleton_M) {
        if (entry.expression >= coeffs.top_lp_coefficients.size()) {
          throw std::runtime_error(
              "reference top-LP matrix expression is out of range");
        }
        matrix[entry.flat_idx] =
            matrix[entry.flat_idx] +
            T(entry.weight) * coeffs.top_lp_coefficients[entry.expression];
      }
      if (!execute_validation_tape(matrix, local_rhs, pending.operation_tapes[block]))
        return {};
    }
    std::ranges::copy(local_rhs,
                      global_rhs.begin() + static_cast<ptrdiff_t>(global_rhs_begin));

    std::vector<T> coupling_values(program.couplings.size(), T(0));
    for (const auto& entry : program.skeleton_couplings) {
      coupling_values[entry.flat_idx] = coupling_values[entry.flat_idx] +
                                        T(entry.weight) * bilinear_basis[entry.bb_idx];
    }
    for (const auto& entry : program.top_lp_skeleton_couplings) {
      if (entry.expression >= coeffs.top_lp_coefficients.size()) {
        throw std::runtime_error(
            "reference top-LP coupling expression is out of range");
      }
      coupling_values[entry.flat_idx] =
          coupling_values[entry.flat_idx] +
          T(entry.weight) * coeffs.top_lp_coefficients[entry.expression];
    }
    for (std::size_t coupling = 0; coupling < program.couplings.size(); ++coupling) {
      const auto rows = program.couplings[coupling];
      const std::size_t destination =
          static_cast<std::size_t>(rows.destination_row) * target_count;
      const std::size_t source =
          static_cast<std::size_t>(rows.source_row) * target_count;
      for (std::size_t target = 0; target < target_count; ++target) {
        global_rhs[destination + target] =
            global_rhs[destination + target] -
            coupling_values[coupling] * global_rhs[source + target];
      }
    }
  }

  std::vector<T> solution(dimension * target_count, T(0));
  for (std::size_t column = 0; column < dimension; ++column) {
    const std::uint32_t solution_row =
        column < cfg.basis.size() ? pending.solution_row_by_column[column]
                                  : pending.reference_solution_row_by_column[column];
    const std::size_t source = static_cast<std::size_t>(solution_row) * target_count;
    std::ranges::copy_n(global_rhs.begin() + static_cast<ptrdiff_t>(source),
                        static_cast<ptrdiff_t>(target_count),
                        solution.begin() +
                            static_cast<ptrdiff_t>(column * target_count));
  }

  std::vector<T> evaluated_matrix(plan.matrix_coordinates.size(), T(0));
  for (std::size_t coordinate_index = 0;
       coordinate_index < plan.matrix_coordinates.size(); ++coordinate_index) {
    const auto& coordinate = plan.matrix_coordinates[coordinate_index];
    T value(0);
    for (std::size_t index = coordinate.matrix_term_begin;
         index < coordinate.matrix_term_end; ++index) {
      const auto& term = plan.closure_matrix_terms[index];
      value = value + T(term.weight) * bilinear_basis[term.bb_idx];
    }
    for (std::size_t index = coordinate.top_lp_term_begin;
         index < coordinate.top_lp_term_end; ++index) {
      const auto& term = plan.closure_top_lp_matrix_terms[index];
      if (term.expression >= coeffs.top_lp_coefficients.size())
        throw std::runtime_error("reference top-LP matrix expression is out of range");
      value = value + T(term.weight) * coeffs.top_lp_coefficients[term.expression];
    }
    evaluated_matrix[coordinate_index] = value;
  }

  std::array<std::uint64_t, kResidualProjectionCount> projection_seeds{};
  std::vector<T> target_projection(target_count * kResidualProjectionCount);
  for (std::size_t projection = 0; projection < kResidualProjectionCount;
       ++projection) {
    projection_seeds[projection] = projection_seed(values, projection);
    for (std::size_t target = 0; target < target_count; ++target) {
      target_projection[target * kResidualProjectionCount + projection] =
          projection_weight(projection_seeds[projection], target);
    }
  }
  std::vector<T> projected_solution(dimension * kResidualProjectionCount, T(0));
  for (std::size_t column = 0; column < dimension; ++column) {
    for (std::size_t target = 0; target < target_count; ++target) {
      const T& value = solution[column * target_count + target];
      for (std::size_t projection = 0; projection < kResidualProjectionCount;
           ++projection) {
        projected_solution[column * kResidualProjectionCount + projection] =
            projected_solution[column * kResidualProjectionCount + projection] +
            value * target_projection[target * kResidualProjectionCount + projection];
      }
    }
  }

  std::vector<T> closure_residual(static_cast<std::size_t>(plan.closure_row_count) *
                                      kResidualProjectionCount,
                                  T(0));
  for (const auto& term : plan.closure_rhs_terms) {
    const T value = T(term.weight) * bilinear_basis[term.bb_idx];
    for (std::size_t projection = 0; projection < kResidualProjectionCount;
         ++projection) {
      const std::size_t slot =
          static_cast<std::size_t>(term.row) * kResidualProjectionCount + projection;
      closure_residual[slot] =
          closure_residual[slot] -
          value *
              target_projection[term.target * kResidualProjectionCount + projection];
    }
  }
  for (const auto& term : plan.closure_top_lp_rhs_terms) {
    if (term.expression >= coeffs.top_lp_coefficients.size())
      throw std::runtime_error("reference top-LP RHS expression is out of range");
    const T value = T(term.weight) * coeffs.top_lp_coefficients[term.expression];
    for (std::size_t projection = 0; projection < kResidualProjectionCount;
         ++projection) {
      const std::size_t slot =
          static_cast<std::size_t>(term.row) * kResidualProjectionCount + projection;
      closure_residual[slot] =
          closure_residual[slot] -
          value *
              target_projection[term.target * kResidualProjectionCount + projection];
    }
  }

  for (std::size_t coordinate_index = 0;
       coordinate_index < plan.matrix_coordinates.size(); ++coordinate_index) {
    const T& value = evaluated_matrix[coordinate_index];
    if (value == T(0)) continue;
    const auto& coordinate = plan.matrix_coordinates[coordinate_index];
    for (std::size_t projection = 0; projection < kResidualProjectionCount;
         ++projection) {
      closure_residual[static_cast<std::size_t>(coordinate.row) *
                           kResidualProjectionCount +
                       projection] =
          closure_residual[static_cast<std::size_t>(coordinate.row) *
                               kResidualProjectionCount +
                           projection] +
          value * projected_solution[static_cast<std::size_t>(coordinate.column) *
                                         kResidualProjectionCount +
                                     projection];
    }
  }
  for (std::size_t row = 0; row < plan.closure_row_count; ++row) {
    for (std::size_t projection = 0; projection < kResidualProjectionCount;
         ++projection) {
      if (closure_residual[row * kResidualProjectionCount + projection] != T(0)) {
        throw std::runtime_error(std::format(
            "kernel closure validation residual is nonzero at row={}, projection={}",
            row, projection));
      }
    }
  }

  std::vector<T> result;
  result.reserve(target_count * cfg.basis.size());
  for (std::size_t target = 0; target < target_count; ++target) {
    for (std::size_t basis = 0; basis < cfg.basis.size(); ++basis) {
      T value = solution[basis * target_count + target];
      value = value * coeffs.targets_lp[target];
      value = value * coeffs.basis_lp_inv[basis];
      result.push_back(value);
    }
  }
  return result;
}

void BlackBoxFeynman::prime_changed()
{
  std::lock_guard lock(replay_variant_mutex);
  compile_loaders(firefly::FFInt::p);
  selected_master_structure_cache.clear();
  replay_variant.store(nullptr, std::memory_order_release);
}

std::vector<firefly::FFInt>
BlackBoxFeynman::eval_selected_compact(const std::vector<firefly::FFInt>& values,
                                       const std::vector<std::uint32_t>& active_outputs)
{
  if (!output_support_initialized)
    throw std::logic_error("reduction output support is not initialized");
  if (!std::ranges::is_sorted(active_outputs) ||
      std::ranges::adjacent_find(active_outputs) != active_outputs.end()) {
    throw std::invalid_argument(
        "active reconstruction outputs must be sorted and unique");
  }
  if (!active_outputs.empty() && active_outputs.back() >= replay_outputs.size())
    throw std::out_of_range("active reconstruction output is out of range");
  if (active_outputs.empty()) return {};
  if (active_outputs.size() == replay_outputs.size()) return (*this)(values);

  const auto variant = selected_replay_variant(active_outputs);
  const auto coeffs = evaluate_coefficients(values, &variant->coefficients);
  auto selected = execute_replay(coeffs, values, variant.get());
  if (selected.empty()) return {};
  if (selected.size() != active_outputs.size())
    throw std::logic_error("selected replay result shape is inconsistent");

  return selected;
}

std::vector<firefly::FFInt>
BlackBoxFeynman::eval_selected(const std::vector<firefly::FFInt>& values,
                               const std::vector<std::uint32_t>& active_outputs)
{
  const auto selected = eval_selected_compact(values, active_outputs);

  std::vector<firefly::FFInt> result(replay_outputs.size(), firefly::FFInt(0));
  for (std::size_t index = 0; index < active_outputs.size(); ++index)
    result[active_outputs[index]] = selected[index];
  return result;
}

void BlackBoxFeynman::compile_loaders(std::uint64_t prime)
{
  if (!coefficient_pool_initialized) {
    using ExpressionKey = std::vector<std::pair<std::uint8_t, std::int64_t>>;
    struct ExpressionHash {
      std::size_t operator()(const ExpressionKey& key) const noexcept
      {
        std::size_t seed = key.size();
        for (const auto& [basis, weight] : key) {
          seed ^= std::hash<std::uint8_t>{}(basis) + 0x9e3779b9 + (seed << 6U) +
                  (seed >> 2U);
          seed ^= std::hash<std::int64_t>{}(weight) + 0x9e3779b9 + (seed << 6U) +
                  (seed >> 2U);
        }
        return seed;
      }
    };
    struct ExpressionRecord {
      std::size_t uses = 0;
      std::uint32_t expression = std::numeric_limits<std::uint32_t>::max();
    };
    std::unordered_map<ExpressionKey, ExpressionRecord, ExpressionHash> records;
    auto for_each_group = [](const std::vector<AnsatzEntry>& skeleton, auto&& visit) {
      for (std::size_t begin = 0; begin < skeleton.size();) {
        std::size_t end = begin + 1;
        while (end < skeleton.size() &&
               skeleton[end].flat_idx == skeleton[begin].flat_idx) {
          ++end;
        }
        visit(begin, end);
        begin = end;
      }
    };
    auto make_key = [](const auto& skeleton, std::size_t begin, std::size_t end) {
      ExpressionKey key;
      key.reserve(end - begin);
      for (std::size_t index = begin; index < end; ++index)
        key.emplace_back(skeleton[index].bb_idx, skeleton[index].weight);
      return key;
    };
    auto count_groups = [&](const auto& skeleton) {
      for_each_group(skeleton, [&](std::size_t begin, std::size_t end) {
        ++records[make_key(skeleton, begin, end)].uses;
      });
    };
    count_groups(skeleton_B);
    for (const auto& block : replay_blocks) {
      count_groups(block.skeleton_M);
      count_groups(block.skeleton_couplings);
    }
    auto rewrite_groups = [&](std::vector<AnsatzEntry>& skeleton,
                              std::vector<PooledCoefficientLoad>& pooled) {
      std::vector<AnsatzEntry> unpooled;
      unpooled.reserve(skeleton.size());
      for_each_group(skeleton, [&](std::size_t begin, std::size_t end) {
        auto key = make_key(skeleton, begin, end);
        auto found = records.find(key);
        if (found == records.end())
          throw std::logic_error("coefficient expression count disappeared");
        if (found->second.uses < 2) {
          unpooled.insert(unpooled.end(),
                          skeleton.begin() + static_cast<ptrdiff_t>(begin),
                          skeleton.begin() + static_cast<ptrdiff_t>(end));
          return;
        }
        if (found->second.expression == std::numeric_limits<std::uint32_t>::max()) {
          if (coefficient_expressions.size() >=
              std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("coefficient expression count exceeds 32-bit ids");
          }
          if (key.size() > std::numeric_limits<std::uint8_t>::max())
            throw std::runtime_error("coefficient expression is too long");
          if (coefficient_terms.size() >
              std::numeric_limits<std::uint32_t>::max() - key.size()) {
            throw std::runtime_error("coefficient expression terms exceed 32-bit ids");
          }
          found->second.expression =
              static_cast<std::uint32_t>(coefficient_expressions.size());
          coefficient_expressions.push_back(
              {static_cast<std::uint32_t>(coefficient_terms.size()),
               static_cast<std::uint8_t>(key.size())});
          for (const auto& [basis, weight] : key)
            coefficient_terms.push_back({0, weight, basis});
        }
        pooled.push_back({skeleton[begin].flat_idx, found->second.expression});
      });
      skeleton = std::move(unpooled);
    };
    rewrite_groups(skeleton_B, pooled_loader_B);
    for (auto& block : replay_blocks) {
      rewrite_groups(block.skeleton_M, block.pooled_loader_M);
      rewrite_groups(block.skeleton_couplings, block.pooled_loader_couplings);
    }
    coefficient_pool_initialized = true;
    kernel_statistics_.coefficient_expressions = coefficient_expressions.size();
    kernel_statistics_.pooled_coefficient_loads = pooled_loader_B.size();
    for (const auto& block : replay_blocks) {
      kernel_statistics_.pooled_coefficient_loads +=
          block.pooled_loader_M.size() + block.pooled_loader_couplings.size();
    }
  }

  const finite_field::MontgomeryArithmetic arithmetic(prime);
  const auto field_weight = [prime, &arithmetic](std::int64_t weight) {
    const auto magnitude = (weight < 0 ? static_cast<std::uint64_t>(-(weight + 1)) + 1
                                       : static_cast<std::uint64_t>(weight)) %
                           prime;
    const auto residue = weight >= 0 || magnitude == 0 ? magnitude : prime - magnitude;
    return arithmetic.encode(residue);
  };
  auto build_loader = [&field_weight](const std::vector<AnsatzEntry>& skeleton,
                                      std::vector<LoadInstruction>& loader,
                                      std::vector<std::int64_t>& weights) {
    if (!skeleton.empty()) {
      loader.clear();
      weights.clear();
      loader.reserve(skeleton.size());
      weights.reserve(skeleton.size());
      for (const auto& entry : skeleton) {
        loader.push_back({0, entry.flat_idx, entry.bb_idx, 0});
        weights.push_back(entry.weight);
      }
      for (std::size_t begin = 0; begin < loader.size();) {
        const std::uint32_t slot = loader[begin].flat_idx;
        std::size_t end = begin + 1;
        while (end < loader.size() && loader[end].flat_idx == slot)
          ++end;
        const std::size_t group_size = end - begin;
        if (group_size > MAX_BILINEAR_BASIS) {
          throw std::runtime_error("loader group exceeds the bilinear basis limit");
        }
        loader[begin].group_size = static_cast<std::uint8_t>(group_size);
        begin = end;
      }
    }
    if (loader.size() != weights.size())
      throw std::runtime_error("loader weight shape mismatch");
    for (std::size_t begin = 0; begin < loader.size();) {
      const std::size_t group_size = loader[begin].group_size;
      const std::size_t end = begin + group_size;
      if (group_size == 0 || end > loader.size())
        throw std::runtime_error("loader group metadata is invalid");
      const std::uint32_t slot = loader[begin].flat_idx;
      for (std::size_t index = begin; index < end; ++index) {
        if (loader[index].flat_idx != slot ||
            loader[index].bb_idx >= MAX_BILINEAR_BASIS ||
            (index != begin && loader[index - 1].bb_idx >= loader[index].bb_idx) ||
            (index != begin && loader[index].group_size != 0)) {
          throw std::runtime_error("loader group ordering is invalid");
        }
      }
      if (end < loader.size() && loader[end].flat_idx <= slot)
        throw std::runtime_error("loader slots are not strictly ordered");
      begin = end;
    }
    for (size_t index = 0; index < loader.size(); ++index) {
      loader[index].w_ff = field_weight(weights[index]);
    }
  };
  auto build_top_lp_loader = [&field_weight,
                              this](const std::vector<TopLpRhsEntry>& skeleton,
                                    std::vector<TopLpLoadInstruction>& loader,
                                    std::vector<std::int64_t>& weights) {
    if (!skeleton.empty()) {
      loader.clear();
      weights.clear();
      loader.reserve(skeleton.size());
      weights.reserve(skeleton.size());
      for (const auto& entry : skeleton) {
        loader.push_back({0, entry.flat_idx, entry.expression});
        weights.push_back(entry.weight);
      }
    }
    if (loader.size() != weights.size())
      throw std::runtime_error("top-LP loader weight shape mismatch");
    for (std::size_t index = 0; index < loader.size(); ++index) {
      if (loader[index].expression >= top_lp_expression_programs.size())
        throw std::runtime_error("top-LP loader expression is out of range");
      loader[index].w_ff = field_weight(weights[index]);
    }
  };

  build_loader(skeleton_B, loader_B, loader_B_weights);
  build_top_lp_loader(top_lp_rhs_skeleton, top_lp_loader_B, top_lp_loader_B_weights);
  if (replay_blocks.empty()) {
    throw std::runtime_error("block replay program was not published");
  }
  for (auto& block : replay_blocks) {
    build_loader(block.skeleton_M, block.loader_M, block.loader_M_weights);
    build_top_lp_loader(block.top_lp_skeleton_M, block.top_lp_loader_M,
                        block.top_lp_loader_M_weights);
    build_loader(block.skeleton_couplings, block.loader_couplings,
                 block.loader_coupling_weights);
    build_top_lp_loader(block.top_lp_skeleton_couplings, block.top_lp_loader_couplings,
                        block.top_lp_loader_coupling_weights);
    std::vector<AnsatzEntry>().swap(block.skeleton_M);
    std::vector<TopLpRhsEntry>().swap(block.top_lp_skeleton_M);
    std::vector<AnsatzEntry>().swap(block.skeleton_couplings);
    std::vector<TopLpRhsEntry>().swap(block.top_lp_skeleton_couplings);
  }
  for (auto& term : coefficient_terms)
    term.w_ff = field_weight(term.weight);
  std::vector<AnsatzEntry>().swap(skeleton_B);
  std::vector<TopLpRhsEntry>().swap(top_lp_rhs_skeleton);
}
