#include "output/GenerateOutput.hpp"

#include "core/AtomicFile.hpp"

#include <filesystem>
#include <stdexcept>
#include <string>

namespace {

void validate_result(const ReductionResult& result)
{
  if (!result.context ||
      result.coefficients.size() != result.targets.size() * result.basis.size()) {
    throw std::runtime_error("invalid reduction result shape");
  }
}

} // namespace

void write_basis_integrals(const std::vector<Integral>& basis,
                           const std::string& header,
                           const std::filesystem::path& output)
{
  core::atomic_file::write(output, "final basis", [&](std::ostream& stream) {
    for (const auto& integral : basis) {
      stream << format_mathematica_integral(header, integral.indices) << '\n';
    }
  });
}

void write_mathematica_result(const ReductionResult& result,
                              const std::filesystem::path& output)
{
  validate_result(result);
  std::vector<std::string> basis_labels;
  basis_labels.reserve(result.basis.size());
  for (const auto& integral : result.basis)
    basis_labels.push_back(
        format_mathematica_integral(result.integral_header, integral.indices));
  core::atomic_file::write(output, "Mathematica result", [&](std::ostream& stream) {
    std::size_t flat = 0;
    stream << "{\n";
    for (std::size_t target_index = 0; target_index < result.targets.size();
         ++target_index) {
      if (target_index != 0) {
        stream << ",\n";
      }
      const auto& target = result.targets[target_index];
      stream << "  "
             << format_mathematica_integral(result.integral_header, target.indices)
             << " -> ";
      bool wrote_term = false;
      for (std::size_t index = 0; index < result.basis.size(); ++index) {
        const FactorizedRational& original = result.coefficients.at(flat++);
        if (original.is_zero()) {
          continue;
        }
        if (wrote_term) {
          stream << " + ";
        }
        wrote_term = true;
        stream << '(' << original.to_string() << ")*" << basis_labels[index];
      }
      if (!wrote_term) {
        stream << '0';
      }
    }
    stream << "\n}\n";
  });
}
