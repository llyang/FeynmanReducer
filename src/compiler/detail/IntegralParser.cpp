#include "compiler/detail/IntegralParser.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <ranges>
#include <stdexcept>
#include <string>

namespace compiler::detail {

std::vector<Integral> parse_integrals(const std::filesystem::path& path,
                                      unsigned expected_size)
{
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open integral file: " + path.string());
  }
  std::string content{std::istreambuf_iterator<char>(input),
                      std::istreambuf_iterator<char>()};
  for (char& character : content) {
    if (character == '{') {
      character = '[';
    } else if (character == '}') {
      character = ']';
    }
  }
  std::vector<Integral> result;
  std::vector<int> current;
  std::string number;
  bool inside = false;
  auto flush_number = [&] {
    if (!number.empty()) {
      current.push_back(std::stoi(number));
      number.clear();
    }
  };
  for (const char character : content) {
    if (character == '[') {
      if (!inside) {
        current.clear();
        inside = true;
      }
    } else if (character == ']') {
      if (inside) {
        flush_number();
        if (!current.empty()) {
          if (current.size() != expected_size) {
            throw std::runtime_error("integral length mismatch in " + path.string());
          }
          result.push_back({current});
        }
        inside = false;
      }
    } else if (inside && (std::isdigit(static_cast<unsigned char>(character)) ||
                          (character == '-' && number.empty()))) {
      number.push_back(character);
    } else if (inside && (character == ',' ||
                          std::isspace(static_cast<unsigned char>(character)))) {
      flush_number();
    }
  }
  if (inside) {
    throw std::runtime_error("unclosed integral list in " + path.string());
  }
  if (result.empty()) {
    throw std::runtime_error("no integrals found in " + path.string());
  }
  return result;
}

} // namespace compiler::detail
