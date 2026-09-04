#pragma once

#include "compiler/detail/ExactPolynomial.hpp"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace compiler::detail {

struct SymbolTable {
  std::vector<std::string> names;
  std::unordered_map<std::string, std::size_t> indices;

  void add(const std::string& name)
  {
    if (!indices.contains(name)) {
      indices.emplace(name, names.size());
      names.push_back(name);
    }
  }

  [[nodiscard]] std::size_t index(const std::string& name) const
  {
    const auto found = indices.find(name);
    if (found == indices.end()) {
      throw std::runtime_error("unknown symbol: " + name);
    }
    return found->second;
  }
};

void collect_symbols(const std::string& expression, SymbolTable& symbols);
[[nodiscard]] Polynomial parse_poly(const std::string& text,
                                    const SymbolTable& symbols);

} // namespace compiler::detail
