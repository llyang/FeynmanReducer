#pragma once

#include "core/Config.hpp"

#include <yaml-cpp/node/node.h>

#include <map>

namespace compiler::detail {

[[nodiscard]] TopologyConfig compile_yaml_topology_node(const YAML::Node& root);

[[nodiscard]] std::map<std::string, ExactRationalConstant>
compile_yaml_numerics_node(const YAML::Node& root);

} // namespace compiler::detail
