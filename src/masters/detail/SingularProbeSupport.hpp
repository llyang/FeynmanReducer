#pragma once

#include "core/Config.hpp"
#include "core/ProbeValues.hpp"
#include "topology/SectorPolynomial.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <span>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace masters::detail {
inline constexpr std::array<std::uint32_t, 2> singular_probe_primes{32003, 31991};

inline std::vector<std::uint32_t> singular_kinematics(std::size_t count,
                                                      std::size_t point)
{
  std::vector<std::uint32_t> values;
  values.reserve(count);
  for (std::size_t j = 0; j < count; ++j)
    values.push_back(probe_values::value(j, point));
  return values;
}

// Integer arithmetic only: Singular workers must never mutate FFInt's prime.
inline std::string singular_sector_polynomial(const TopologyConfig& config,
                                              std::uint32_t sector, std::uint32_t prime,
                                              std::span<const std::uint32_t> kinematics)
{
  std::ostringstream out;
  bool first = true;
  for (const auto& term : config.polynomial_terms) {
    if (term.powers.size() != config.propagator_count ||
        term.weights.size() != kinematics.size() + 1)
      throw std::runtime_error("compiled LP term has an invalid shape");
    if (!polynomial_term_survives_sector(term, sector)) continue;
    const auto residue = [prime](std::int64_t n) {
      const auto p = static_cast<std::int64_t>(prime);
      return static_cast<std::uint64_t>((n % p + p) % p);
    };
    auto coefficient = residue(term.weights[0]);
    for (std::size_t j = 0; j < kinematics.size(); ++j)
      coefficient =
          (coefficient + residue(term.weights[j + 1]) * kinematics[j]) % prime;
    if (coefficient == 0) continue;
    if (!first) out << '+';
    first = false;
    out << coefficient;
    for (std::size_t j = 0; j < term.powers.size(); ++j) {
      if (!term.powers[j]) continue;
      out << "*x" << config.propagator_slots.at(j) + 1;
      if (term.powers[j] != 1) out << '^' << static_cast<unsigned>(term.powers[j]);
    }
  }
  return first ? "0" : out.str();
}

struct SingularProbeTask {
  std::uint32_t sector = 0;
  std::size_t probe = 0;
};

// Stable heavy-first ordering. Each task belongs to exactly one batch.
inline std::vector<std::vector<SingularProbeTask>>
singular_probe_batches(std::vector<std::uint32_t> sectors, unsigned threads,
                       bool regulated)
{
  if (threads == 0) throw std::invalid_argument("threads must be positive");
  std::ranges::sort(sectors, [](auto a, auto b) {
    if (std::popcount(a) != std::popcount(b))
      return std::popcount(a) > std::popcount(b);
    return a < b;
  });
  if (std::ranges::adjacent_find(sectors) != sectors.end())
    throw std::invalid_argument("Singular batch sectors must be unique");
  std::vector<std::vector<SingularProbeTask>> batches;
  std::vector<SingularProbeTask> light;
  for (const auto sector : sectors) {
    for (std::size_t probe = 0; probe < singular_probe_primes.size(); ++probe) {
      const SingularProbeTask task{sector, probe};
      if (regulated && std::popcount(sector) >= 6)
        batches.push_back({task});
      else
        light.push_back(task);
    }
  }
  const auto offset = batches.size();
  const auto count = std::min<std::size_t>(threads, light.size());
  batches.resize(offset + count);
  for (std::size_t i = 0; i < light.size(); ++i)
    batches[offset + i % count].push_back(light[i]);
  return batches;
}
} // namespace masters::detail
