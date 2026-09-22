#pragma once

#include "core/Config.hpp"

#include <filesystem>
#include <map>

// Compiles the user-facing YAML topology directly into the in-memory
// reduction configuration. Auxiliary paths are resolved relative to the YAML.
[[nodiscard]] TopologyConfig
compile_yaml_topology(const std::filesystem::path& filepath);

// Compiles the topology and the runtime options needed by find_masters without
// reading targets_file.
[[nodiscard]] MasterFinderConfig
compile_yaml_master_finder_config(const std::filesystem::path& filepath);

// Compiles a reduction configuration and its optional targets. A DE-only
// configuration remains target-free until the application has selected the
// master basis and materializes its derivative sources.
[[nodiscard]] Config compile_yaml_config(const std::filesystem::path& filepath);

[[nodiscard]] std::map<std::string, ExactRationalConstant>
compile_yaml_numerics(const std::filesystem::path& filepath);

// Lightweight query used by reduction_validate, which intentionally does not
// compile a topology or support named-combination reference synthesis.
[[nodiscard]] bool yaml_uses_combination_targets(const std::filesystem::path& filepath);

// Lightweight companion query for reduction_validate. Missing/default target
// files are ignored because validation can operate from result files alone.
[[nodiscard]] bool
yaml_uses_dimension_shifted_targets(const std::filesystem::path& filepath);

[[nodiscard]] bool
yaml_generates_differential_equations(const std::filesystem::path& filepath);
