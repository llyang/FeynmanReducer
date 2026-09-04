#include "symmetry/detail/Graph.hpp"

#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace symmetry::detail {

GraphColorPalette make_lp_color_palette(std::span<const PolynomialTerm> terms)
{
  std::set<std::vector<std::int64_t>> coefficients;
  std::set<std::uint8_t> exponents;
  for (const auto& term : terms) {
    coefficients.insert(term.weights);
    for (const auto exponent : term.powers) {
      if (exponent != 0) {
        exponents.insert(exponent);
      }
    }
  }
  return {
      {coefficients.begin(), coefficients.end()},
      {exponents.begin(), exponents.end()},
  };
}

ColoredGraph make_lp_graph(std::span<const PolynomialTerm> terms,
                           std::size_t variable_count, const GraphColorPalette& palette,
                           std::span<const std::uint8_t> variable_colors)
{
  if (variable_count > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("too many LP variables for symmetry graph");
  }

  if (!variable_colors.empty() && variable_colors.size() != variable_count) {
    throw std::runtime_error("LP variable colors have an unexpected size");
  }
  std::map<std::uint8_t, std::uint32_t> normalized_variable_colors;
  if (variable_colors.empty()) {
    normalized_variable_colors.emplace(0, 0);
  } else {
    for (const auto color : variable_colors)
      normalized_variable_colors.emplace(color, 0);
    std::uint32_t normalized = 0;
    for (auto& [color, value] : normalized_variable_colors) {
      static_cast<void>(color);
      value = normalized++;
    }
  }
  std::uint32_t next_color =
      static_cast<std::uint32_t>(normalized_variable_colors.size());
  std::map<std::vector<std::int64_t>, std::uint32_t> coefficient_colors;
  for (const auto& coefficient : palette.coefficients) {
    coefficient_colors.emplace(coefficient, next_color++);
  }
  std::map<std::uint8_t, std::uint32_t> exponent_colors;
  for (const auto exponent : palette.exponents) {
    exponent_colors.emplace(exponent, next_color++);
  }

  ColoredGraph result;
  result.colors.reserve(variable_count);
  if (variable_colors.empty()) {
    result.colors.assign(variable_count, 0);
  } else {
    for (const auto color : variable_colors)
      result.colors.push_back(normalized_variable_colors.at(color));
  }
  std::vector<std::uint32_t> term_vertices;
  term_vertices.reserve(terms.size());
  for (const auto& term : terms) {
    if (term.powers.size() != variable_count) {
      throw std::runtime_error("LP term has an unexpected variable count");
    }
    if (result.colors.size() >= std::numeric_limits<std::uint32_t>::max()) {
      throw std::runtime_error("LP symmetry graph is too large");
    }
    term_vertices.push_back(static_cast<std::uint32_t>(result.colors.size()));
    const auto color = coefficient_colors.find(term.weights);
    if (color == coefficient_colors.end()) {
      throw std::runtime_error("LP coefficient is missing from graph palette");
    }
    result.colors.push_back(color->second);
  }

  for (std::size_t term_index = 0; term_index < terms.size(); ++term_index) {
    const auto& term = terms[term_index];
    for (std::size_t variable = 0; variable < variable_count; ++variable) {
      const auto exponent = term.powers[variable];
      if (exponent == 0) {
        continue;
      }
      if (result.colors.size() >= std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("LP symmetry graph is too large");
      }
      const auto incidence = static_cast<std::uint32_t>(result.colors.size());
      const auto color = exponent_colors.find(exponent);
      if (color == exponent_colors.end()) {
        throw std::runtime_error("LP exponent is missing from graph palette");
      }
      result.colors.push_back(color->second);
      result.edges.emplace_back(static_cast<std::uint32_t>(variable), incidence);
      result.edges.emplace_back(term_vertices[term_index], incidence);
    }
  }
  return result;
}

} // namespace symmetry::detail
