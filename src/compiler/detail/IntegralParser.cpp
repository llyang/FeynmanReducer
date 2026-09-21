#include "compiler/detail/IntegralParser.hpp"

#include <cctype>
#include <charconv>
#include <fstream>
#include <stdexcept>
#include <string>

namespace compiler::detail {
namespace {

class IntegralCursor {
public:
  explicit IntegralCursor(std::string_view input) : input_(input) {}

  void skip_space()
  {
    while (position_ < input_.size() &&
           std::isspace(static_cast<unsigned char>(input_[position_])) != 0)
      ++position_;
  }

  [[nodiscard]] bool done()
  {
    skip_space();
    return position_ == input_.size();
  }

  [[nodiscard]] char peek()
  {
    skip_space();
    return position_ == input_.size() ? '\0' : input_[position_];
  }

  bool consume(char expected)
  {
    skip_space();
    if (position_ == input_.size() || input_[position_] != expected) return false;
    ++position_;
    return true;
  }

  void require(char expected, std::string_view message)
  {
    if (!consume(expected)) throw std::runtime_error(std::string(message));
  }

  [[nodiscard]] std::string identifier()
  {
    skip_space();
    if (position_ == input_.size() ||
        !(std::isalpha(static_cast<unsigned char>(input_[position_])) != 0 ||
          input_[position_] == '$'))
      throw std::runtime_error("integral head must be a Mathematica symbol");
    const std::size_t begin = position_++;
    while (position_ < input_.size() &&
           (std::isalnum(static_cast<unsigned char>(input_[position_])) != 0 ||
            input_[position_] == '$'))
      ++position_;
    return std::string(input_.substr(begin, position_ - begin));
  }

  [[nodiscard]] int integer()
  {
    skip_space();
    const std::size_t begin = position_;
    if (position_ < input_.size() &&
        (input_[position_] == '-' || input_[position_] == '+'))
      ++position_;
    const std::size_t digits = position_;
    while (position_ < input_.size() &&
           std::isdigit(static_cast<unsigned char>(input_[position_])) != 0)
      ++position_;
    if (digits == position_)
      throw std::runtime_error("integral index must be an integer");
    int result = 0;
    const auto text = input_.substr(begin, position_ - begin);
    const char* first = text.data();
    if (*first == '+') ++first;
    const auto [end, error] = std::from_chars(first, text.data() + text.size(), result);
    if (error != std::errc{} || end != text.data() + text.size())
      throw std::runtime_error("integral integer is out of range");
    return result;
  }

private:
  std::string_view input_;
  std::size_t position_ = 0;
};

std::vector<int> parse_list(IntegralCursor& cursor, char open, char close)
{
  cursor.require(open, "missing integral index list");
  std::vector<int> result;
  if (cursor.consume(close)) return result;
  for (;;) {
    result.push_back(cursor.integer());
    if (cursor.consume(close)) return result;
    cursor.require(',', "integral indices must be comma-separated");
  }
}

Integral parse_one(IntegralCursor& cursor, unsigned expected_size)
{
  Integral result;
  const std::string head = cursor.identifier();
  static_cast<void>(head);
  cursor.require('[', "missing '[' after integral head");
  const int leading = cursor.integer();
  if (cursor.consume(']')) {
    result.indices.push_back(leading);
    if (result.indices.size() != expected_size)
      throw std::runtime_error("integral length mismatch");
    return result;
  }
  cursor.require(',', "integral requires at least two indices or a shifted index list");
  if (cursor.peek() == '{') {
    result.dimension_shift = leading;
    result.indices = parse_list(cursor, '{', '}');
    cursor.require(']', "missing ']' after shifted integral");
  } else {
    result.indices.push_back(leading);
    for (;;) {
      result.indices.push_back(cursor.integer());
      if (cursor.consume(']')) break;
      cursor.require(',', "integral indices must be comma-separated");
    }
  }
  if (result.indices.size() != expected_size)
    throw std::runtime_error("integral length mismatch");
  if (result.dimension_shift % 2 != 0)
    throw std::runtime_error("integral dimension shift must be even");
  return result;
}

} // namespace

Integral parse_integral_expression(std::string_view expression, unsigned expected_size)
{
  IntegralCursor cursor(expression);
  Integral result = parse_one(cursor, expected_size);
  if (!cursor.done()) throw std::runtime_error("unexpected text after integral");
  return result;
}

std::vector<Integral> parse_integrals(const std::filesystem::path& path,
                                      unsigned expected_size)
{
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open integral file: " + path.string());
  std::vector<Integral> result;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    const auto comment = line.find('#');
    const std::string_view expression(
        line.data(), comment == std::string::npos ? line.size() : comment);
    IntegralCursor empty(expression);
    if (empty.done()) continue;
    try {
      result.push_back(parse_integral_expression(expression, expected_size));
    } catch (const std::exception& error) {
      throw std::runtime_error(path.string() + ":" + std::to_string(line_number) +
                               ": " + error.what());
    }
  }
  if (result.empty())
    throw std::runtime_error("no integrals found in " + path.string());
  return result;
}

} // namespace compiler::detail
