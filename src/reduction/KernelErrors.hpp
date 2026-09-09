#pragma once
#include <stdexcept>
#include <string>
#include <utility>

namespace reduction::detail {
class SymmetryEquivalentBasisError : public std::runtime_error {
public:
  SymmetryEquivalentBasisError()
      : std::runtime_error("basis contains symmetry-equivalent integrals")
  {}
};
class ClosureResidualError : public std::runtime_error {
public:
  explicit ClosureResidualError(std::string message)
      : std::runtime_error(std::move(message))
  {}
};
} // namespace reduction::detail
