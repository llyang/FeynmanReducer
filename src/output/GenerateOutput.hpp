#pragma once

#include "core/IntegralFormatting.hpp"
#include "core/ReductionResult.hpp"

#include <filesystem>
#include <string>
#include <vector>

void write_basis_integrals(const std::vector<Integral>& basis,
                           const std::string& header,
                           const std::filesystem::path& output);

void write_mathematica_result(const ReductionResult& result,
                              const std::filesystem::path& output);
