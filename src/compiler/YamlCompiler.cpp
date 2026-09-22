#include "compiler/YamlCompiler.hpp"

#include "compiler/detail/CombinationParser.hpp"
#include "compiler/detail/IntegralParser.hpp"
#include "compiler/detail/SymanzikCompiler.hpp"
#include "topology/IntegralLayout.hpp"

#include <yaml-cpp/yaml.h>

#include <array>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <ranges>
#include <regex>
#include <set>
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
  static constexpr std::array<std::string_view, 15> allowed_fields{
      "reconstruction_scale",
      "kinematics",
      "propagators",
      "top_sector",
      "integral_header",
      "symmetry_backend",
      "targets_file",
      "threads",
      "singular_path",
      "factor_scan",
      "shift_scan",
      "basis_selection",
      "numerics",
      "check_master_independence",
      "differential_equations"};
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

std::int64_t integral_scale_offset(const Config& config, const Integral& integral)
{
  std::int64_t sum = 0;
  for (const int index : integral.indices) {
    if (__builtin_add_overflow(sum, static_cast<std::int64_t>(index), &sum) ||
        sum == std::numeric_limits<std::int64_t>::min())
      throw std::overflow_error("integral index sum overflow");
  }
  std::int64_t shift = 0;
  if (__builtin_mul_overflow(static_cast<std::int64_t>(config.loop_count),
                             static_cast<std::int64_t>(integral.dimension_shift / 2),
                             &shift) ||
      __builtin_sub_overflow(shift, sum, &shift))
    throw std::overflow_error("integral scale degree overflow");
  return shift;
}

std::string lower_extension(const std::filesystem::path& path)
{
  std::string extension = path.extension().string();
  std::ranges::transform(extension, extension.begin(), [](char value) {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(value)));
  });
  return extension;
}

void validate_target(const Config& config, const Integral& target)
{
  try {
    integral_layout::validate(config, target);
  } catch (const std::exception& error) {
    throw std::runtime_error(std::string("invalid target integral: ") + error.what());
  }
}

std::uint32_t intern_target(Config& config, const Integral& target)
{
  const auto found = std::ranges::find(config.targets, target);
  if (found != config.targets.end())
    return static_cast<std::uint32_t>(found - config.targets.begin());
  if (config.targets.size() >= std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("too many distinct target integrals");
  config.targets.push_back(target);
  return static_cast<std::uint32_t>(config.targets.size() - 1);
}

void append_integral_requests(
    Config& config, const std::filesystem::path& path,
    std::set<std::pair<int, std::vector<int>>>& integral_outputs)
{
  for (auto& integral :
       compiler::detail::parse_integrals(path, config.integral_count,
                                         config.integral_header)) {
    validate_target(config, integral);
    if (!integral_outputs.insert({integral.dimension_shift, integral.indices}).second)
      throw std::runtime_error("duplicate target integral across target files");
    const auto target = intern_target(config, integral);
    ReductionRequest request;
    request.output.integral = integral;
    request.output.scale_offset = integral_scale_offset(config, integral);
    request.terms.push_back({target, "1"});
    config.reduction_requests.push_back(std::move(request));
  }
}

void append_combination_requests(Config& config, const std::filesystem::path& path,
                                 std::set<std::string>& named_outputs)
{
  for (auto& combination : compiler::detail::parse_combinations(path, config)) {
    if (!named_outputs.insert(combination.output.name).second)
      throw std::runtime_error("duplicate combination name across target files: " +
                               combination.output.name);
    ReductionRequest request;
    request.output = std::move(combination.output);
    for (auto& term : combination.terms) {
      const auto target = intern_target(config, term.integral);
      request.terms.push_back({target, std::move(term.coefficient)});
    }
    config.reduction_requests.push_back(std::move(request));
  }
}

void compile_targets(const YAML::Node& root, const std::filesystem::path& parent,
                     Config& config)
{
  const YAML::Node configured = root["targets_file"];
  if (!configured && config.differential_equations) return;
  if (!configured || configured.IsScalar()) {
    const std::string filename =
        configured ? configured.as<std::string>() : std::string("targets.txt");
    if (filename.empty()) throw std::runtime_error("targets_file must be non-empty");
    const auto path = parent / filename;
    const auto extension = lower_extension(path);
    if (extension == ".yaml" || extension == ".yml") {
      std::set<std::string> names;
      append_combination_requests(config, path, names);
    } else {
      config.targets = compiler::detail::parse_integrals(
          path, config.integral_count, config.integral_header);
      for (const auto& target : config.targets)
        validate_target(config, target);
    }
    return;
  }
  if (!configured.IsSequence() || configured.size() == 0)
    throw std::runtime_error("targets_file must be a path or a non-empty path list");
  std::set<std::pair<int, std::vector<int>>> integral_outputs;
  std::set<std::string> named_outputs;
  for (const auto& item : configured) {
    if (!item.IsScalar())
      throw std::runtime_error("targets_file list entries must be paths");
    const std::string filename = item.as<std::string>();
    if (filename.empty())
      throw std::runtime_error("targets_file list entries must be non-empty");
    const auto path = parent / filename;
    const auto extension = lower_extension(path);
    if (extension == ".txt") {
      append_integral_requests(config, path, integral_outputs);
    } else if (extension == ".yaml" || extension == ".yml") {
      append_combination_requests(config, path, named_outputs);
    } else {
      throw std::runtime_error("target list entries must use .txt, .yaml, or .yml: " +
                               path.string());
    }
  }
  if (config.targets.empty() || config.reduction_requests.empty())
    throw std::runtime_error("target files contain no reduction requests");
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
  if (root["differential_equations"])
    config.differential_equations = root["differential_equations"].as<bool>();
  if (config.differential_equations && config.kinematic_parameters.empty()) {
    throw std::runtime_error(
        "differential_equations requires at least one free kinematic parameter");
  }
  const std::filesystem::path parent = filepath.parent_path();
  compile_targets(root, parent, config);
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
  if (root["check_master_independence"])
    config.check_master_independence = root["check_master_independence"].as<bool>();
  if (root["factor_scan"]) config.factor_scan = root["factor_scan"].as<bool>();
  if (root["shift_scan"]) config.shift_scan = root["shift_scan"].as<bool>();
  if (root["reconstruction_scale"])
    config.reconstruction_scale = root["reconstruction_scale"].as<std::string>();
  if (config.reconstruction_scale != "auto" && config.reconstruction_scale != "off") {
    if (!config.scale_homogeneous)
      throw std::runtime_error("reconstruction_scale: " +
                               config.scale_homogeneity_reason);
    if (std::ranges::find(config.reconstruction_scale_candidates,
                          config.reconstruction_scale) ==
        config.reconstruction_scale_candidates.end())
      throw std::runtime_error(
          "reconstruction_scale must name a free kinematic parameter in F");
  }
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

bool yaml_uses_combination_targets(const std::filesystem::path& filepath)
{
  const YAML::Node root = load_yaml_root(filepath);
  const YAML::Node targets = root["targets_file"];
  const auto is_combination = [](const YAML::Node& node) {
    if (!node.IsScalar())
      throw std::runtime_error("targets_file entries must be paths");
    const auto extension = lower_extension(node.as<std::string>());
    return extension == ".yaml" || extension == ".yml";
  };
  if (!targets) return false;
  if (targets.IsScalar()) return is_combination(targets);
  if (!targets.IsSequence())
    throw std::runtime_error("targets_file must be a path or a path list");
  return std::ranges::any_of(targets, is_combination);
}

bool yaml_uses_dimension_shifted_targets(const std::filesystem::path& filepath)
{
  const YAML::Node root = load_yaml_root(filepath);
  const YAML::Node targets = root["targets_file"];
  std::vector<std::filesystem::path> paths;
  if (!targets) {
    paths.emplace_back("targets.txt");
  } else if (targets.IsScalar()) {
    paths.push_back(targets.as<std::string>());
  } else if (targets.IsSequence()) {
    for (const auto& target : targets) {
      if (!target.IsScalar())
        throw std::runtime_error("targets_file entries must be paths");
      paths.push_back(target.as<std::string>());
    }
  } else {
    throw std::runtime_error("targets_file must be a path or a path list");
  }

  static const std::regex shifted(
      R"(^\s*[A-Za-z$][A-Za-z0-9$]*\s*\[\s*([+-]?[0-9]+)\s*,\s*\{)");
  const auto parent = filepath.parent_path();
  for (const auto& configured : paths) {
    const auto extension = lower_extension(configured);
    if (extension == ".yaml" || extension == ".yml") continue;
    const auto path = configured.is_absolute() ? configured : parent / configured;
    std::ifstream input(path);
    if (!input) continue;
    std::string line;
    std::smatch match;
    while (std::getline(input, line)) {
      if (std::regex_search(line, match, shifted) && std::stoi(match[1].str()) != 0)
        return true;
    }
  }
  return false;
}

bool yaml_generates_differential_equations(const std::filesystem::path& filepath)
{
  const YAML::Node root = load_yaml_root(filepath);
  const YAML::Node enabled = root["differential_equations"];
  return enabled && enabled.as<bool>();
}
