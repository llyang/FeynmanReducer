#include "output/GenerateOutput.hpp"

#include "core/AtomicFile.hpp"

#include <filesystem>
#include <stdexcept>
#include <string>

namespace {

void validate_result(const ReductionResult& result)
{
  const std::size_t output_count =
      result.outputs.empty() ? result.targets.size() : result.outputs.size();
  if (!result.context ||
      result.coefficients.size() != output_count * result.basis.size()) {
    throw std::runtime_error("invalid reduction result shape");
  }
}

std::string output_label(const ReductionResult& result, std::size_t output)
{
  if (result.outputs.empty())
    return format_mathematica_integral(result.integral_header,
                                       result.targets.at(output));
  const auto& descriptor = result.outputs.at(output);
  if (descriptor.differential) {
    return "D[" +
           format_mathematica_integral(result.integral_header, descriptor.integral) +
           "," + descriptor.differential_parameter + "]";
  }
  return descriptor.named
             ? descriptor.name
             : format_mathematica_integral(result.integral_header, descriptor.integral);
}

} // namespace

void write_basis_integrals(const std::vector<Integral>& basis,
                           const std::string& header,
                           const std::filesystem::path& output)
{
  core::atomic_file::write(output, "final basis", [&](std::ostream& stream) {
    for (const auto& integral : basis) {
      stream << format_mathematica_integral(header, integral) << '\n';
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
        format_mathematica_integral(result.integral_header, integral));
  core::atomic_file::write(output, "Mathematica result", [&](std::ostream& stream) {
    stream << "{\n";
    const std::size_t output_count =
        result.outputs.empty() ? result.targets.size() : result.outputs.size();
    bool wrote_output = false;
    for (std::size_t target_index = 0; target_index < output_count; ++target_index) {
      if (!result.outputs.empty() && result.outputs[target_index].differential)
        continue;
      if (wrote_output) {
        stream << ",\n";
      }
      wrote_output = true;
      stream << "  " << output_label(result, target_index) << " -> ";
      bool wrote_term = false;
      for (std::size_t index = 0; index < result.basis.size(); ++index) {
        const FactorizedRational& original =
            result.coefficients.at(target_index * result.basis.size() + index);
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

void write_mathematica_differential_equations(const ReductionResult& result,
                                              const std::filesystem::path& output)
{
  validate_result(result);
  if (result.differential_parameters.empty())
    throw std::runtime_error("result contains no differential equations");
  if (result.outputs.empty())
    throw std::runtime_error("differential-equation output metadata is missing");

  std::vector<std::size_t> rows;
  for (std::size_t index = 0; index < result.outputs.size(); ++index)
    if (result.outputs[index].differential) rows.push_back(index);
  const std::size_t expected =
      result.differential_parameters.size() * result.basis.size();
  if (rows.size() != expected)
    throw std::runtime_error("differential-equation result shape is inconsistent");
  for (std::size_t parameter = 0; parameter < result.differential_parameters.size();
       ++parameter) {
    for (std::size_t row = 0; row < result.basis.size(); ++row) {
      const auto& descriptor =
          result.outputs.at(rows[parameter * result.basis.size() + row]);
      if (descriptor.differential_parameter !=
              result.differential_parameters[parameter] ||
          descriptor.integral != result.basis[row]) {
        throw std::runtime_error(
            "differential-equation row order does not match the final basis");
      }
    }
  }

  core::atomic_file::write(
      output, "Mathematica differential equations", [&](std::ostream& stream) {
        stream << "{\n";
        for (std::size_t parameter = 0;
             parameter < result.differential_parameters.size(); ++parameter) {
          if (parameter != 0) stream << ",\n";
          stream << "  " << result.differential_parameters[parameter] << " -> {\n";
          for (std::size_t row = 0; row < result.basis.size(); ++row) {
            if (row != 0) stream << ",\n";
            stream << "    {";
            const std::size_t output_index =
                rows[parameter * result.basis.size() + row];
            for (std::size_t column = 0; column < result.basis.size(); ++column) {
              if (column != 0) stream << ", ";
              const auto& coefficient =
                  result.coefficients.at(output_index * result.basis.size() + column);
              if (coefficient.is_zero())
                stream << '0';
              else
                stream << '(' << coefficient.to_string() << ')';
            }
            stream << '}';
          }
          stream << "\n  }";
        }
        stream << "\n}\n";
      });
}
