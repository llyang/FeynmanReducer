#include "compiler/detail/ExpressionParser.hpp"

#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace compiler::detail {

struct Token {
  enum class Kind : std::uint8_t {
    End,
    Integer,
    Symbol,
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    Left,
    Right,
  };
  Kind kind = Kind::End;
  std::string text;
};

class Lexer {
public:
  explicit Lexer(std::string input) : input_(std::move(input)) {}

  Token next()
  {
    while (position_ < input_.size() &&
           std::isspace(static_cast<unsigned char>(input_[position_]))) {
      ++position_;
    }
    if (position_ == input_.size()) {
      return {};
    }
    const char current = input_[position_];
    if (std::isdigit(static_cast<unsigned char>(current))) {
      const std::size_t start = position_;
      while (position_ < input_.size() &&
             std::isdigit(static_cast<unsigned char>(input_[position_]))) {
        ++position_;
      }
      return {Token::Kind::Integer, input_.substr(start, position_ - start)};
    }
    if (std::isalpha(static_cast<unsigned char>(current)) || current == '_') {
      const std::size_t start = position_;
      while (position_ < input_.size() &&
             (std::isalnum(static_cast<unsigned char>(input_[position_])) ||
              input_[position_] == '_')) {
        ++position_;
      }
      return {Token::Kind::Symbol, input_.substr(start, position_ - start)};
    }
    ++position_;
    switch (current) {
    case '+':
      return {Token::Kind::Plus, "+"};
    case '-':
      return {Token::Kind::Minus, "-"};
    case '*':
      return {Token::Kind::Star, "*"};
    case '/':
      return {Token::Kind::Slash, "/"};
    case '^':
      return {Token::Kind::Caret, "^"};
    case '(':
      return {Token::Kind::Left, "("};
    case ')':
      return {Token::Kind::Right, ")"};
    default:
      throw std::runtime_error(std::string("unsupported expression character: ") +
                               current);
    }
  }

private:
  std::string input_;
  std::size_t position_ = 0;
};

void collect_symbols(const std::string& expression, SymbolTable& symbols)
{
  Lexer lexer(expression);
  for (Token token = lexer.next(); token.kind != Token::Kind::End;
       token = lexer.next()) {
    if (token.kind == Token::Kind::Symbol) {
      symbols.add(token.text);
    }
  }
}

class Parser {
public:
  Parser(std::string input, const SymbolTable& symbols)
      : lexer_(std::move(input)), symbols_(symbols), current_(lexer_.next())
  {}

  Polynomial parse()
  {
    Polynomial result = parse_expression();
    if (current_.kind != Token::Kind::End) {
      throw std::runtime_error("unexpected trailing token: " + current_.text);
    }
    return result;
  }

private:
  void consume(Token::Kind expected)
  {
    if (current_.kind != expected) {
      throw std::runtime_error("unexpected token: " + current_.text);
    }
    current_ = lexer_.next();
  }

  Polynomial parse_expression()
  {
    Polynomial result = parse_term();
    while (current_.kind == Token::Kind::Plus || current_.kind == Token::Kind::Minus) {
      const Token::Kind operation = current_.kind;
      consume(operation);
      Polynomial rhs = parse_term();
      result = operation == Token::Kind::Plus ? result + rhs : result - rhs;
    }
    return result;
  }

  Polynomial parse_term()
  {
    Polynomial result = parse_unary();
    while (current_.kind == Token::Kind::Star || current_.kind == Token::Kind::Slash) {
      const Token::Kind operation = current_.kind;
      consume(operation);
      Polynomial rhs = parse_unary();
      result = operation == Token::Kind::Star
                   ? result * rhs
                   : scale(result, Rational(1) / constant_value(rhs));
    }
    return result;
  }

  Polynomial parse_unary()
  {
    if (current_.kind == Token::Kind::Plus) {
      consume(Token::Kind::Plus);
      return parse_unary();
    }
    if (current_.kind == Token::Kind::Minus) {
      consume(Token::Kind::Minus);
      return -parse_unary();
    }
    return parse_power();
  }

  Polynomial parse_power()
  {
    Polynomial result = parse_primary();
    if (current_.kind == Token::Kind::Caret) {
      consume(Token::Kind::Caret);
      if (current_.kind != Token::Kind::Integer) {
        throw std::runtime_error("power exponent must be a non-negative integer");
      }
      const int exponent = std::stoi(current_.text);
      consume(Token::Kind::Integer);
      result = power(std::move(result), exponent);
    }
    return result;
  }

  Polynomial parse_primary()
  {
    if (current_.kind == Token::Kind::Integer) {
      Polynomial result = constant_poly(symbols_.names.size(), Rational(current_.text));
      consume(Token::Kind::Integer);
      return result;
    }
    if (current_.kind == Token::Kind::Symbol) {
      const std::string symbol = current_.text;
      consume(Token::Kind::Symbol);
      return symbol_poly(symbols_.names.size(), symbols_.index(symbol));
    }
    if (current_.kind == Token::Kind::Left) {
      consume(Token::Kind::Left);
      Polynomial result = parse_expression();
      consume(Token::Kind::Right);
      return result;
    }
    throw std::runtime_error("unexpected token: " + current_.text);
  }

  Lexer lexer_;
  const SymbolTable& symbols_;
  Token current_;
};

Polynomial parse_poly(const std::string& text, const SymbolTable& symbols)
{
  return Parser(text, symbols).parse();
}

} // namespace compiler::detail
