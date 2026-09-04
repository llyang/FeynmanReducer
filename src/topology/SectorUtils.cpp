#include "topology/SectorUtils.hpp"

#include "topology/IntegralLayout.hpp"

#include <flint/fmpz.h>
#include <flint/fmpz_mat.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace {

class IntegerMatrix {
public:
  IntegerMatrix(slong rows, slong columns)
  {
    fmpz_mat_init(matrix_, rows, columns);
  }

  ~IntegerMatrix()
  {
    fmpz_mat_clear(matrix_);
  }

  IntegerMatrix(const IntegerMatrix&) = delete;
  IntegerMatrix& operator=(const IntegerMatrix&) = delete;

  fmpz_mat_struct* data()
  {
    return matrix_;
  }

private:
  fmpz_mat_t matrix_;
};

class IntegerMatrixWindow {
public:
  IntegerMatrixWindow(fmpz_mat_struct* matrix, slong row_end, slong column_end)
  {
    fmpz_mat_window_init(window_, matrix, 0, 0, row_end, column_end);
  }

  ~IntegerMatrixWindow()
  {
    fmpz_mat_window_clear(window_);
  }

  IntegerMatrixWindow(const IntegerMatrixWindow&) = delete;
  IntegerMatrixWindow& operator=(const IntegerMatrixWindow&) = delete;

  fmpz_mat_struct* data()
  {
    return window_;
  }

private:
  fmpz_mat_t window_;
};

} // namespace

SectorUtils::SectorUtils(const TopologyConfig& config)
    : SectorUtils(config, integral_layout::default_variable_slots(config))
{}

SectorUtils::SectorUtils(const TopologyConfig& config,
                         std::span<const std::uint32_t> variable_slots)
    : config_(config), variable_slots_(variable_slots.begin(), variable_slots.end()),
      sector_bit_by_slot_(
          std::max<std::size_t>(config.integral_count, config.propagator_count), -1)
{
  if (config_.propagator_count == 0 || config_.propagator_count >= 32) {
    throw std::invalid_argument("propagator count must be in [1, 31]");
  }
  propagator_mask_ = (1U << config_.propagator_count) - 1U;
  if (variable_slots_.empty()) {
    throw std::invalid_argument("LP variable view must not be empty");
  }
  const auto propagator_slots = integral_layout::default_variable_slots(config_);
  for (std::size_t bit = 0; bit < propagator_slots.size(); ++bit) {
    const auto slot = propagator_slots[bit];
    if (slot >= sector_bit_by_slot_.size()) {
      throw std::invalid_argument("active propagator slot is out of range");
    }
    sector_bit_by_slot_[slot] = static_cast<int>(bit);
  }
  for (const auto slot : variable_slots_) {
    if (slot >= sector_bit_by_slot_.size()) {
      throw std::invalid_argument("LP variable slot is out of range");
    }
  }
  term_support_masks_.reserve(config_.polynomial_terms.size());
  for (const auto& term : config_.polynomial_terms) {
    if (term.powers.size() != config_.propagator_count) {
      throw std::invalid_argument("LP term powers must match the propagator count");
    }
    std::uint32_t support = 0;
    for (unsigned index = 0; index < config_.propagator_count; ++index) {
      if (term.powers[index] > 0) support |= 1U << index;
    }
    term_support_masks_.push_back(support);
  }
  valid_sectors_.reserve(std::min<std::size_t>(
      std::size_t{4096}, std::size_t{1} << std::min(config_.propagator_count, 12U)));
}

bool SectorUtils::compute_valid_sector(std::uint32_t sector) const
{
  std::vector<const std::vector<std::uint8_t>*> surviving_powers;
  surviving_powers.reserve(config_.polynomial_terms.size());
  const std::uint32_t absent = propagator_mask_ & ~sector;
  for (std::size_t index = 0; index < config_.polynomial_terms.size(); ++index) {
    if ((term_support_masks_[index] & absent) == 0)
      surviving_powers.push_back(&config_.polynomial_terms[index].powers);
  }
  if (surviving_powers.empty()) {
    return false;
  }

  slong active_variables = 0;
  std::vector<slong> column_mapping(config_.propagator_count, -1);
  for (unsigned index = 0; index < config_.propagator_count; ++index) {
    if (((sector >> index) & 1U) != 0) {
      column_mapping[index] = active_variables++;
    }
  }

  const slong rows = static_cast<slong>(surviving_powers.size());
  const slong columns = active_variables + 1;
  IntegerMatrix augmented(rows, columns);
  for (slong row = 0; row < rows; ++row) {
    for (unsigned index = 0; index < config_.propagator_count; ++index) {
      if (column_mapping[index] != -1) {
        fmpz_set_ui(fmpz_mat_entry(augmented.data(), row, column_mapping[index]),
                    (*surviving_powers[static_cast<std::size_t>(row)])[index]);
      }
    }
    fmpz_one(fmpz_mat_entry(augmented.data(), row, columns - 1));
  }

  IntegerMatrixWindow exponents(augmented.data(), rows, active_variables);
  const slong exponent_rank = fmpz_mat_rank(exponents.data());
  if (exponent_rank == rows) return false;
  return fmpz_mat_rank(augmented.data()) > exponent_rank;
}

std::uint32_t SectorUtils::sector_from_powers(std::span<const int> powers) const
{
  if (powers.size() != variable_slots_.size() + 1) {
    throw std::invalid_argument(
        "power vector must contain one G power and one power per LP variable");
  }
  std::uint32_t sector = 0;
  for (std::size_t index = 1; index < powers.size(); ++index) {
    if (powers[index] >= 0) {
      const int bit = sector_bit_by_slot_[variable_slots_[index - 1]];
      if (bit < 0) {
        throw std::invalid_argument("positive ISP power is outside the top sector");
      }
      sector |= 1U << static_cast<unsigned>(bit);
    }
  }
  return sector;
}

bool SectorUtils::is_valid_sector(std::uint32_t sector) const
{
  if ((sector & ~propagator_mask_) != 0) {
    throw std::invalid_argument("sector contains bits outside the topology");
  }
  const auto found = valid_sectors_.find(sector);
  if (found != valid_sectors_.end()) return found->second;
  const bool valid = compute_valid_sector(sector);
  valid_sectors_.emplace(sector, valid);
  return valid;
}

std::vector<std::uint32_t> SectorUtils::enumerate_nonzero_sectors() const
{
  const std::uint32_t full_sector = propagator_mask_;
  std::vector<std::uint32_t> result;
  // A non-zero sub-sector implies that every super-sector is non-zero:
  // its surviving exponent rows embed with zeroes in the added variables.
  // Therefore a zero sector lets us prune every descendant safely.
  auto visit = [&](auto&& self, std::uint32_t sector,
                   unsigned first_removable) -> void {
    if (sector == 0 || !is_valid_sector(sector)) {
      return;
    }
    result.push_back(sector);
    for (unsigned index = first_removable; index < config_.propagator_count; ++index) {
      const std::uint32_t bit = std::uint32_t{1} << index;
      if ((sector & bit) != 0) {
        self(self, sector & ~bit, index + 1);
      }
    }
  };
  visit(visit, full_sector, 0);
  std::ranges::sort(result);
  return result;
}
