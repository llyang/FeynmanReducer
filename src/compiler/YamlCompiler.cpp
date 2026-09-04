#include "compiler/YamlCompiler.hpp"

#include "compiler/detail/IntegralParser.hpp"
#include "compiler/detail/SymanzikCompiler.hpp"
#include "topology/IntegralLayout.hpp"

#include <yaml-cpp/yaml.h>

#include <array>
#include <filesystem>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

YAML::Node load_yaml_root(const std::filesystem::path& filepath)
{
  const YAML::Node root = YAML::LoadFile(filepath.string());
  if (!root || root.IsNull()) return {};
  if (!root.IsMap()) {
    throw std::runtime_error("top-level YAML value must be a mapping");
  }
  static constexpr std::array<std::string_view, 12> allowed_fields{
      "kinematics",       "propagators",  "top_sector",      "integral_header",
      "symmetry_backend", "targets_file", "threads",         "singular_path",
      "factor_scan",      "shift_scan",   "basis_selection", "numerics"};
  for (const auto& entry : root) {
    if (!entry.first.IsScalar()) {
      throw std::runtime_error("top-level config field names must be strings");
    }
    const std::string field = entry.first.as<std::string>();
    if (std::ranges::find(allowed_fields, field) == allowed_fields.end()) {
      throw std::runtime_error("unknown top-level config field: " + field);
    }
  }
  return root;
}

template <typename RuntimeConfig>
void compile_master_finder_options(const YAML::Node& root, RuntimeConfig& config)
{
  if (root["threads"]) {
    config.threads = root["threads"].as<unsigned>();
    if (config.threads == 0) {
      throw std::runtime_error("threads must be positive");
    }
  }
  if (root["singular_path"]) {
    config.singular_path = root["singular_path"].as<std::string>();
    if (config.singular_path.empty()) {
      throw std::runtime_error("singular_path must be non-empty");
    }
  }
}

} // namespace

TopologyConfig compile_yaml_topology(const std::filesystem::path& filepath)
{
  return compiler::detail::compile_yaml_topology_node(load_yaml_root(filepath));
}

MasterFinderConfig
compile_yaml_master_finder_config(const std::filesystem::path& filepath)
{
  const YAML::Node root = load_yaml_root(filepath);
  MasterFinderConfig config;
  static_cast<TopologyConfig&>(config) =
      compiler::detail::compile_yaml_topology_node(root);
  compile_master_finder_options(root, config);
  return config;
}

Config compile_yaml_config(const std::filesystem::path& filepath)
{
  const YAML::Node root = load_yaml_root(filepath);
  Config config;
  static_cast<TopologyConfig&>(config) =
      compiler::detail::compile_yaml_topology_node(root);
  const std::filesystem::path parent = filepath.parent_path();
  const std::string target_file =
      root["targets_file"] ? root["targets_file"].as<std::string>() : "targets.txt";
  config.targets =
      compiler::detail::parse_integrals(parent / target_file, config.integral_count);
  for (const auto& target : config.targets) {
    try {
      integral_layout::validate(config, target);
    } catch (const std::exception& error) {
      throw std::runtime_error(std::string("invalid target integral: ") + error.what());
    }
  }
  compile_master_finder_options(root, config);
  if (root["basis_selection"]) {
    const std::string selection = root["basis_selection"].as<std::string>();
    if (selection == "default") {
      config.basis_selection = BasisSelectionPolicy::Default;
    } else if (selection == "d-separating") {
      config.basis_selection = BasisSelectionPolicy::DSeparating;
    } else {
      throw std::runtime_error("basis_selection must be 'default' or 'd-separating'");
    }
  }
  if (root["factor_scan"]) config.factor_scan = root["factor_scan"].as<bool>();
  if (root["shift_scan"]) config.shift_scan = root["shift_scan"].as<bool>();
  return config;
}

std::map<std::string, ExactRationalConstant>
compile_yaml_numerics(const std::filesystem::path& filepath)
{
  const YAML::Node root = YAML::LoadFile(filepath.string());
  if (!root || root.IsNull()) return {};
  if (!root.IsMap()) {
    throw std::runtime_error("top-level YAML value must be a mapping");
  }
  return compiler::detail::compile_yaml_numerics_node(root);
}
