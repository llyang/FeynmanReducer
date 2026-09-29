#include "output/ResearchIrOutput.hpp"

#include "core/AtomicFile.hpp"
#include "core/IntegralFormatting.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <ostream>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

struct ResultRows {
  std::vector<std::size_t> targets;
  std::vector<std::size_t> differential;
};

void write_string(std::ostream& stream, std::string_view value)
{
  stream << '"';
  for (const char character : value) {
    switch (character) {
    case '\\':
      stream << "\\\\";
      break;
    case '"':
      stream << "\\\"";
      break;
    case '\n':
      stream << "\\n";
      break;
    case '\r':
      stream << "\\r";
      break;
    case '\t':
      stream << "\\t";
      break;
    default:
      stream << character;
      break;
    }
  }
  stream << '"';
}

template <typename Range>
void write_numbers(std::ostream& stream, const Range& values, std::uint64_t offset = 0)
{
  stream << '{';
  bool first = true;
  for (const auto value : values) {
    if (!first) stream << ", ";
    first = false;
    stream << static_cast<std::uint64_t>(value) + offset;
  }
  stream << '}';
}

void write_signed_numbers(std::ostream& stream, const std::vector<int>& values)
{
  stream << '{';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) stream << ", ";
    stream << values[index];
  }
  stream << '}';
}

void write_symbols(std::ostream& stream, const std::vector<std::string>& values)
{
  stream << '{';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) stream << ", ";
    stream << values[index];
  }
  stream << '}';
}

std::string exact_constant(const ExactRationalConstant& value)
{
  if (value.denominator == "1") return value.numerator;
  return "(" + value.numerator + ")/(" + value.denominator + ")";
}

void write_coefficient(std::ostream& stream, const FactorizedRational& coefficient)
{
  if (coefficient.is_zero())
    stream << '0';
  else
    stream << '(' << coefficient.to_string() << ')';
}

ResultRows validate_result(const ResearchIrMetadata& metadata,
                           const ReductionResult& result)
{
  if (!result.context) throw std::runtime_error("research IR result has no context");
  if (metadata.loop_order == 0 || metadata.propagator_count == 0 ||
      metadata.integral_slot_count == 0)
    throw std::runtime_error("research IR topology metadata is incomplete");
  if (metadata.top_sector.size() != metadata.integral_slot_count ||
      metadata.propagator_slots.size() != metadata.propagator_count)
    throw std::runtime_error("research IR topology metadata has inconsistent shape");
  if (result.outputs.empty())
    throw std::runtime_error("research IR output descriptors are missing");
  if (result.basis.empty()) throw std::runtime_error("research IR basis is empty");
  if (result.coefficients.size() != result.outputs.size() * result.basis.size())
    throw std::runtime_error("research IR reduction result shape is invalid");
  if (result.differential_parameters.empty())
    throw std::runtime_error("research IR requires differential equations");
  if (std::ranges::find(result.parameters, std::string("d")) == result.parameters.end())
    throw std::runtime_error("research IR requires free dimension symbol d");

  ResultRows rows;
  for (std::size_t index = 0; index < result.outputs.size(); ++index) {
    if (result.outputs[index].differential)
      rows.differential.push_back(index);
    else
      rows.targets.push_back(index);
  }
  const std::size_t expected_differential =
      result.differential_parameters.size() * result.basis.size();
  if (rows.differential.size() != expected_differential)
    throw std::runtime_error("research IR differential-equation shape is invalid");
  for (std::size_t parameter = 0; parameter < result.differential_parameters.size();
       ++parameter) {
    for (std::size_t row = 0; row < result.basis.size(); ++row) {
      const auto& descriptor =
          result.outputs.at(rows.differential[parameter * result.basis.size() + row]);
      if (descriptor.differential_parameter !=
              result.differential_parameters[parameter] ||
          descriptor.integral != result.basis[row])
        throw std::runtime_error(
            "research IR differential rows do not match the basis order");
    }
  }
  for (const auto& integral : result.basis)
    if (integral.dimension_shift != 0)
      throw std::runtime_error("research IR basis contains a dimension shift");
  return rows;
}

std::uint32_t sector_mask(const ResearchIrMetadata& metadata, const Integral& integral)
{
  if (integral.indices.size() != metadata.integral_slot_count)
    throw std::runtime_error("research IR integral has the wrong index count");
  std::uint32_t sector = 0;
  for (std::size_t bit = 0; bit < metadata.propagator_slots.size(); ++bit) {
    const auto slot = metadata.propagator_slots[bit];
    if (slot >= integral.indices.size())
      throw std::runtime_error("research IR propagator slot is out of range");
    if (integral.indices[slot] > 0)
      sector |= std::uint32_t{1} << static_cast<unsigned>(bit);
  }
  return sector;
}

class SectorCatalogue {
public:
  SectorCatalogue(const ResearchIrMetadata& metadata, const ReductionResult& result,
                  const ResultRows& rows)
  {
    for (const auto& sector_class : metadata.sector_classes) {
      auto members = sector_class.members;
      members.push_back(sector_class.representative);
      std::ranges::sort(members);
      members.erase(std::ranges::unique(members).begin(), members.end());
      for (const auto member : members) {
        const auto [iterator, inserted] =
            representatives_.emplace(member, sector_class.representative);
        if (!inserted && iterator->second != sector_class.representative)
          throw std::runtime_error("research IR symmetry classes overlap");
      }
      members_.emplace(sector_class.representative, std::move(members));
    }

    for (const auto& integral : result.basis)
      observe(metadata, integral);
    for (const auto output : rows.targets) {
      const auto& descriptor = result.outputs[output];
      if (!descriptor.named) observe(metadata, descriptor.integral);
    }
  }

  [[nodiscard]] std::uint32_t representative(std::uint32_t sector) const
  {
    const auto found = representatives_.find(sector);
    return found == representatives_.end() ? sector : found->second;
  }

  [[nodiscard]] const std::vector<std::uint32_t>& observed() const
  {
    return observed_;
  }

  [[nodiscard]] std::vector<std::uint32_t> members(std::uint32_t sector) const
  {
    const auto found = members_.find(sector);
    if (found == members_.end()) return {sector};
    return found->second;
  }

  [[nodiscard]] std::vector<std::uint32_t> strict_subsectors(std::uint32_t sector) const
  {
    const auto parents = members(sector);
    std::vector<std::uint32_t> result;
    for (const auto candidate : observed_) {
      if (candidate == sector) continue;
      const auto children = members(candidate);
      bool included = false;
      for (const auto child : children) {
        for (const auto parent : parents) {
          if (child != parent && (child & parent) == child) {
            included = true;
            break;
          }
        }
        if (included) break;
      }
      if (included) result.push_back(candidate);
    }
    return result;
  }

private:
  void observe(const ResearchIrMetadata& metadata, const Integral& integral)
  {
    const auto sector = representative(sector_mask(metadata, integral));
    if (observed_set_.insert(sector).second) {
      observed_.push_back(sector);
      std::ranges::sort(observed_);
    }
  }

  std::unordered_map<std::uint32_t, std::uint32_t> representatives_;
  std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> members_;
  std::unordered_set<std::uint32_t> observed_set_;
  std::vector<std::uint32_t> observed_;
};

void write_integral_fields(std::ostream& stream, const ResearchIrMetadata& metadata,
                           const ReductionResult& result,
                           const SectorCatalogue& sectors, const Integral& integral)
{
  const auto sector = sector_mask(metadata, integral);
  stream << "\"Expression\" -> "
         << format_mathematica_integral(result.integral_header, integral)
         << ", \"Indices\" -> ";
  write_signed_numbers(stream, integral.indices);
  stream << ", \"DimensionShift\" -> " << integral.dimension_shift
         << ", \"SectorMask\" -> " << sector << ", \"CanonicalSectorMask\" -> "
         << sectors.representative(sector);
}

void write_matrix_row(std::ostream& stream, const ReductionResult& result,
                      std::size_t output)
{
  stream << '{';
  for (std::size_t column = 0; column < result.basis.size(); ++column) {
    if (column != 0) stream << ", ";
    write_coefficient(stream,
                      result.coefficients.at(output * result.basis.size() + column));
  }
  stream << '}';
}

} // namespace

ResearchIrMetadata make_research_ir_metadata(const Config& config)
{
  if (!config.differential_equations)
    throw std::runtime_error(
        "experimental research IR requires differential_equations: true");
  if (!config.dimension_parameter_index.has_value() ||
      config.parameters.at(*config.dimension_parameter_index) != "d")
    throw std::runtime_error("experimental research IR requires free dimension d");
  if (config.propagator_count >= 32)
    throw std::runtime_error(
        "experimental research IR supports at most 31 propagators");

  ResearchIrMetadata result;
  result.loop_order = config.loop_count;
  result.propagator_count = config.propagator_count;
  result.integral_slot_count = config.integral_count;
  result.top_sector = config.top_sector;
  result.propagator_slots = config.propagator_slots;
  if (config.symmetry) {
    result.sector_classes.reserve(config.symmetry->sector_classes.size());
    for (const auto& source : config.symmetry->sector_classes) {
      ResearchIrSectorClass target;
      target.representative = source.representative;
      target.members.reserve(source.relations.size());
      for (const auto& relation : source.relations)
        target.members.push_back(relation.target_sector);
      result.sector_classes.push_back(std::move(target));
    }
  }
  return result;
}

void write_mathematica_research_ir(const ResearchIrMetadata& metadata,
                                   const ReductionResult& result,
                                   const std::filesystem::path& output)
{
  const auto rows = validate_result(metadata, result);
  const SectorCatalogue sectors(metadata, result, rows);

  core::atomic_file::write(
      output, "experimental research IR", [&](std::ostream& stream) {
        stream << "<|\n"
               << "  \"Schema\" -> \"FeynmanReducerResearchIR\",\n"
               << "  \"SchemaVersion\" -> "
               << (metadata.parametric_input.empty() ? 1 : 2) << ",\n"
               << "  \"LoopOrder\" -> " << metadata.loop_order << ",\n"
               << "  \"DimensionSymbol\" -> d,\n"
               << "  \"IntegralHeader\" -> " << result.integral_header << ",\n"
               << "  \"Parameters\" -> ";
        write_symbols(stream, result.parameters);
        stream << ",\n  \"FreeKinematicVariables\" -> ";
        write_symbols(stream, result.differential_parameters);
        stream << ",\n  \"Numerics\" -> <|";
        bool first_numeric = true;
        for (const auto& [name, value] : result.numerics) {
          if (!first_numeric) stream << ", ";
          first_numeric = false;
          stream << name << " -> " << exact_constant(value);
        }
        stream << "|>,\n"
               << "  \"Normalization\" -> <|\"Name\" -> "
                  "\"FeynmanReducerConventional\", \"Version\" -> 1, "
                  "\"DimensionShiftConvention\" -> \"AdditiveToD\"|>,\n"
               << "  \"Topology\" -> <|\n"
               << "    \"IntegralSlotCount\" -> " << metadata.integral_slot_count
               << ",\n"
               << "    \"PropagatorCount\" -> " << metadata.propagator_count << ",\n"
               << "    \"PropagatorSlots\" -> ";
        write_numbers(stream, metadata.propagator_slots, 1);
        stream << ",\n    \"TopSector\" -> ";
        write_numbers(stream, metadata.top_sector);
        stream << ",\n    \"SectorClassCoverage\" -> \"ReferencedObjects\",\n"
               << "    \"SectorClasses\" -> {";
        for (std::size_t index = 0; index < sectors.observed().size(); ++index) {
          if (index != 0) stream << ", ";
          const auto sector = sectors.observed()[index];
          stream << "<|\"Id\" -> " << sector << ", \"RepresentativeMask\" -> " << sector
                 << ", \"MemberMasks\" -> ";
          write_numbers(stream, sectors.members(sector));
          stream << ", \"StrictSubsectorClassIds\" -> ";
          write_numbers(stream, sectors.strict_subsectors(sector));
          stream << "|>";
        }
        stream << "}";
        if (!metadata.parametric_input.empty())
          stream << ",\n    \"ParametricInput\" -> " << metadata.parametric_input;
        stream << "\n  |>,\n"
               << "  \"Basis\" -> {\n";
        for (std::size_t index = 0; index < result.basis.size(); ++index) {
          if (index != 0) stream << ",\n";
          stream << "    <|\"Index\" -> " << index + 1 << ", ";
          write_integral_fields(stream, metadata, result, sectors, result.basis[index]);
          stream << "|>";
        }
        stream << "\n  },\n"
               << "  \"Targets\" -> {\n";
        for (std::size_t index = 0; index < rows.targets.size(); ++index) {
          if (index != 0) stream << ",\n";
          const auto& descriptor = result.outputs[rows.targets[index]];
          stream << "    <|\"Index\" -> " << index + 1 << ", \"Kind\" -> ";
          if (descriptor.named)
            write_string(stream, "NamedCombination");
          else if (descriptor.integral.dimension_shift == 0)
            write_string(stream, "OrdinaryIntegral");
          else
            write_string(stream, "DimensionShiftedIntegral");
          stream << ", \"Label\" -> ";
          if (descriptor.named)
            stream << descriptor.name;
          else
            stream << format_mathematica_integral(result.integral_header,
                                                  descriptor.integral);
          if (!descriptor.named) {
            stream << ", ";
            write_integral_fields(stream, metadata, result, sectors,
                                  descriptor.integral);
          }
          stream << ", \"ReductionRow\" -> " << index + 1 << "|>";
        }
        stream << "\n  },\n"
               << "  \"ReductionMatrix\" -> {";
        for (std::size_t index = 0; index < rows.targets.size(); ++index) {
          if (index != 0) stream << ", ";
          write_matrix_row(stream, result, rows.targets[index]);
        }
        stream << "},\n"
               << "  \"DifferentialEquations\" -> <|\n";
        for (std::size_t parameter = 0;
             parameter < result.differential_parameters.size(); ++parameter) {
          if (parameter != 0) stream << ",\n";
          stream << "    " << result.differential_parameters[parameter] << " -> {";
          for (std::size_t row = 0; row < result.basis.size(); ++row) {
            if (row != 0) stream << ", ";
            write_matrix_row(stream, result,
                             rows.differential[parameter * result.basis.size() + row]);
          }
          stream << '}';
        }
        stream << "\n  |>\n"
               << "|>\n";
      });
}
