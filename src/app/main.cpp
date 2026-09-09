#include "app/FireflyLogArchive.hpp"
#include "app/RunLog.hpp"
#include "app/RunTiming.hpp"
#include "compiler/YamlCompiler.hpp"
#include "masters/MasterFinder.hpp"
#include "output/GenerateOutput.hpp"
#include "reduction/FeynmanReducer.hpp"
#include "symmetry/Symmetry.hpp"

#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

struct Options {
  std::filesystem::path input = "config.yaml";
  std::filesystem::path output_directory;
  std::filesystem::path mathematica_output;
  std::filesystem::path basis_output;
  std::filesystem::path firefly_log_output;
  std::optional<unsigned> threads;
  NumeratorReductionStrategy numerator_strategy = NumeratorReductionStrategy::Projected;
};

unsigned parse_threads(const std::string& text)
{
  unsigned value = 0;
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size() || value == 0) {
    throw std::runtime_error("thread count must be a positive integer");
  }
  return value;
}

Options parse_arguments(int argc, char** argv)
{
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto value = [&]() -> std::string {
      if (index + 1 >= argc) {
        throw std::runtime_error(argument + " requires a value");
      }
      return argv[++index];
    };
    if (argument == "-i" || argument == "--input") {
      options.input = value();
    } else if (argument == "-j" || argument == "--threads") {
      options.threads = parse_threads(value());
    } else if (argument == "--numerator-strategy") {
      const std::string strategy = value();
      if (strategy == "projected") {
        options.numerator_strategy = NumeratorReductionStrategy::Projected;
      } else if (strategy == "direct") {
        options.numerator_strategy = NumeratorReductionStrategy::Direct;
      } else {
        throw std::runtime_error("numerator strategy must be 'projected' or 'direct'");
      }
    } else if (argument == "-h" || argument == "--help") {
      std::cout << "Usage: FeynmanReducer [-i config.yaml] [-j N]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + argument);
    }
  }
  const std::filesystem::path parent = options.input.parent_path();
  options.output_directory = parent / "outputs";
  options.mathematica_output = options.output_directory / "results.m";
  options.basis_output = options.output_directory / "final_basis.txt";
  options.firefly_log_output = options.output_directory / "firefly.log";
  return options;
}

} // namespace

int main(int argc, char** argv)
{
  RunTiming timing(std::cout, std::cerr);
  std::optional<RunLog> run_log;
  try {
    const Options options = parse_arguments(argc, argv);
    run_log.emplace(options.output_directory, timing.started_at(), std::cout,
                    std::cerr);
    timing.set_info_output(run_log->detail());
    timing.summary("FeynmanReducer started");
    timing.summary(std::format("Reduction log: path={}", run_log->path().string()));
    Config config = timing.run_stage("Compile configuration", [&] {
      Config compiled = compile_yaml_config(options.input);
      if (options.threads.has_value()) {
        compiled.threads = *options.threads;
      }
      return compiled;
    });
    timing.summary(std::format(
        "Configuration: input={}, active_propagators={}, integral_slots={}, "
        "targets={}, parameters={}",
        options.input.string(), config.propagator_count, config.integral_count,
        config.targets.size(), config.parameters.size()));
    if (config.symmetry) {
      std::size_t subsector_generator_count = 0;
      for (const auto& sector_class : config.symmetry->sector_classes) {
        subsector_generator_count += sector_class.generators.size();
      }
      timing.info(std::format(
          "Symmetry: backend={}, generators={}, nonzero_sectors={}, "
          "sector_classes={}, representative_generators={}",
          symmetry::backend_name(config.symmetry->backend),
          config.symmetry->generators.size(), config.symmetry->nonzero_sector_count,
          config.symmetry->sector_classes.size(), subsector_generator_count));
    }
    auto master_candidates =
        timing.run_stage(std::format("Find master basis (workers={})", config.threads),
                         [&] { return masters::find_master_candidates(config); });
    if (!master_candidates.requires_global_selection()) {
      config.basis = std::move(master_candidates.integrals);
      timing.summary(std::format("Master basis: integrals={}, source=isolated",
                                 config.basis.size()));
    } else {
      timing.summary(std::format("Master candidates: integrals={}, source_sectors={}",
                                 master_candidates.integrals.size(),
                                 master_candidates.relation_source_sectors.size()));
    }
    FireflyLogArchive firefly_log("firefly.log", options.firefly_log_output);
    auto reduction_progress = [&](std::string_view label,
                                  ReductionProgressEvent event) {
      switch (event) {
      case ReductionProgressEvent::started:
        timing.stage_started(std::string(label));
        break;
      case ReductionProgressEvent::completed:
        timing.stage_completed(label);
        break;
      case ReductionProgressEvent::failed:
        timing.stage_failed(label);
        break;
      case ReductionProgressEvent::warning:
        timing.warning(label);
        break;
      case ReductionProgressEvent::info:
        timing.info(label);
        break;
      }
    };
    ReductionResult result =
        master_candidates.requires_global_selection()
            ? perform_reduction(std::move(config), std::move(master_candidates),
                                reduction_progress, {options.numerator_strategy})
            : perform_reduction_owned(std::move(config), reduction_progress,
                                      {options.numerator_strategy});
    bool archived_firefly_log = false;
    timing.run_stage("Write output files", [&] {
      archived_firefly_log = firefly_log.finish();
      write_mathematica_result(result, options.mathematica_output);
      write_basis_integrals(result.basis, result.integral_header, options.basis_output);
    });
    timing.summary(std::format("Output: type=mathematica, path={}",
                               options.mathematica_output.string()));
    timing.summary(
        std::format("Output: type=basis, path={}", options.basis_output.string()));
    if (archived_firefly_log) {
      timing.summary(std::format("Output: type=firefly_log, path={}",
                                 options.firefly_log_output.string()));
    }
    timing.complete("FeynmanReducer");
    return 0;
  } catch (const std::exception& error) {
    timing.error(std::format("FeynmanReducer: {}", error.what()));
    return 2;
  }
}
