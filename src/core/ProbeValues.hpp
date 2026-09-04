#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace probe_values {

inline constexpr std::array<std::uint32_t, 64> irregular_values{
    631, 137, 887, 269, 719, 193, 947, 431, 571, 109, 823, 353, 997, 229, 683, 461,
    149, 769, 307, 919, 541, 181, 827, 397, 659, 257, 953, 503, 113, 733, 331, 859,
    617, 211, 983, 449, 701, 277, 811, 167, 941, 389, 593, 239, 757, 467, 131, 877,
    313, 647, 199, 929, 557, 283, 797, 421, 103, 727, 367, 839, 607, 251, 971, 479,
};

[[nodiscard]] constexpr std::uint32_t value(std::size_t parameter,
                                            std::size_t attempt = 0)
{
  constexpr std::size_t parameter_stride = 17;
  constexpr std::size_t attempt_stride = 23;
  return irregular_values[(parameter * parameter_stride + attempt * attempt_stride) %
                          irregular_values.size()];
}

[[nodiscard]] constexpr std::uint64_t mix(std::uint64_t input)
{
  input += 0x9e3779b97f4a7c15ULL;
  input = (input ^ (input >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  input = (input ^ (input >> 27U)) * 0x94d049bb133111ebULL;
  return input ^ (input >> 31U);
}

[[nodiscard]] constexpr std::uint64_t
field_value(std::uint64_t prime, std::size_t parameter, std::uint64_t point)
{
  if (prime <= 3) return 1;
  std::uint64_t state = prime;
  state ^= mix(point + 0x6a09e667f3bcc909ULL);
  state ^= mix(static_cast<std::uint64_t>(parameter) + 0xbb67ae8584caa73bULL);
  for (std::uint64_t attempt = 0; attempt < 16; ++attempt) {
    const std::uint64_t sampled = mix(state + attempt * 0x3c6ef372fe94f82bULL) % prime;
    if (sampled != 0 && sampled != 1 && sampled != prime - 1) return sampled;
  }
  return 2 + mix(state) % (prime - 3);
}

[[nodiscard]] constexpr std::uint64_t validation_point_index(std::size_t attempt)
{
  return static_cast<std::uint64_t>(attempt);
}

[[nodiscard]] constexpr std::uint64_t planning_point_index(std::size_t attempt)
{
  constexpr std::uint64_t planning_domain = 0x8000000000000000ULL;
  return planning_domain + static_cast<std::uint64_t>(attempt);
}

/// Return a deterministic generic point for reduction-kernel planning.
///
/// Planning and validation deliberately occupy disjoint point-index domains:
/// validation starts at zero, while planning uses this high domain.  Keeping
/// the planning point in the full finite field makes accidental cancellations
/// overwhelmingly less likely than with the small integer values above.
[[nodiscard]] constexpr std::uint64_t
planning_field_value(std::uint64_t prime, std::size_t parameter, std::size_t attempt)
{
  return field_value(prime, parameter, planning_point_index(attempt));
}

} // namespace probe_values
