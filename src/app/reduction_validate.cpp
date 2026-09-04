#include "compiler/YamlCompiler.hpp"
#include "validation/ReductionValidator.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Options {
  std::filesystem::path config = "config.yaml";
  std::filesystem::path result;
  std::filesystem::path basis;
  std::filesystem::path kira_result;
  std::filesystem::path kira_basis;
  bool help = false;
};

Options parse_arguments(int argc, char** argv)
{
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto value = [&]() -> std::filesystem::path {
      if (index + 1 >= argc) {
        throw std::runtime_error(argument + " requires a path");
      }
      return argv[++index];
    };
    if (argument == "-i" || argument == "--input") {
      options.config = value();
    } else if (argument == "--result") {
      options.result = value();
    } else if (argument == "--basis") {
      options.basis = value();
    } else if (argument == "--kira-result") {
      options.kira_result = value();
    } else if (argument == "--kira-basis") {
      options.kira_basis = value();
    } else if (argument == "-h" || argument == "--help") {
      options.help = true;
    } else {
      throw std::runtime_error("unknown argument: " + argument);
    }
  }
  return options;
}

std::filesystem::path resolve(const std::filesystem::path& parent,
                              const std::filesystem::path& configured,
                              const std::filesystem::path& fallback)
{
  const auto path = configured.empty() ? fallback : configured;
  return path.is_absolute() ? path : parent / path;
}

} // namespace

int main(int argc, char** argv)
{
  try {
    const Options options = parse_arguments(argc, argv);
    if (options.help) {
      std::cout
          << "Usage: reduction_validate [-i config.yaml]\n"
             "                          [--result outputs/results.m]\n"
             "                          [--basis outputs/final_basis.txt]\n"
             "                          [--kira-result validation/kira_integrals.m]\n"
             "                          [--kira-basis validation/masters]\n";
      return 0;
    }
    if (!std::filesystem::is_regular_file(options.config)) {
      throw std::runtime_error("config file does not exist: " +
                               options.config.string());
    }
    const auto parent = options.config.parent_path();
    const auto numerics = compile_yaml_numerics(options.config);
    const validation::ValidationFiles files{
        resolve(parent, options.result, "outputs/results.m"),
        resolve(parent, options.basis, "outputs/final_basis.txt"),
        resolve(parent, options.kira_result, "validation/kira_integrals.m"),
        resolve(parent, options.kira_basis, "validation/masters"),
        numerics,
    };
    const auto report = validation::validate_reduction(files);
    if (!report.passed()) {
      std::cerr << "[ERROR] reduction_validate: validation failed: "
                << report.difference_count << " difference(s)\n";
      for (const auto& diagnostic : report.diagnostics) {
        std::cerr << "- " << diagnostic << '\n';
      }
      if (report.diagnostics.size() < report.difference_count) {
        std::cerr << "- ... " << report.difference_count - report.diagnostics.size()
                  << " additional difference(s) omitted\n";
      }
      return 1;
    }
    std::cout << "reduction validation passed\n"
              << "targets: " << report.target_count << '\n'
              << "masters: " << report.master_count << '\n'
              << "coefficients: " << report.coefficient_count << '\n';
    if (report.skipped_identity_count != 0) {
      std::cout << "skipped identities: " << report.skipped_identity_count << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[ERROR] reduction_validate: " << error.what() << '\n';
    return 2;
  }
}
