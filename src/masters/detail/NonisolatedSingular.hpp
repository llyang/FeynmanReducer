#pragma once

#include <string_view>

namespace masters::detail {

inline constexpr std::string_view kNonisolatedSingularProcedures = R"SINGULAR(
proc fr_count_nonisolated(ideal input_ideal, int depth)
{
  ideal current = std(input_ideal);
  int current_vdim = vdim(current);
  list terminal_ideals;
  if (current_vdim >= 0)
  {
    if (current_vdim > 0)
    {
      terminal_ideals[1] = current;
    }
    return(list(current_vdim, 0, terminal_ideals));
  }
  if (depth > nvars(basering))
  {
    return(list(0, 1, terminal_ideals));
  }

  list decomposition = primdecGTZ(current);
  int total = 0;
  int component_index;
  for (component_index = 1; component_index <= size(decomposition);
       component_index++)
  {
    ideal component = std(decomposition[component_index][1]);
    int component_vdim = vdim(component);
    if (component_vdim >= 0)
    {
      total = total + component_vdim;
      if (component_vdim > 0)
      {
        terminal_ideals[size(terminal_ideals) + 1] = component;
      }
    }
    else
    {
      ideal associated_prime = std(decomposition[component_index][2]);
      ideal component_mod_prime = reduce(component, associated_prime);
      ideal prime_mod_component = reduce(associated_prime, component);
      int component_is_radical = 1;
      int equality_index;
      for (equality_index = 1; equality_index <= size(component_mod_prime);
           equality_index++)
      {
        if (component_mod_prime[equality_index] != 0)
        {
          component_is_radical = 0;
        }
      }
      for (equality_index = 1; equality_index <= size(prime_mod_component);
           equality_index++)
      {
        if (prime_mod_component[equality_index] != 0)
        {
          component_is_radical = 0;
        }
      }
      if (component_is_radical == 0)
      {
        return(list(total, 5, terminal_ideals));
      }
      ideal reduced_component = interred(component);
      ideal linear_part;
      poly nonlinear_part = 0;
      int nonlinear_count = 0;
      int generator_index;
      for (generator_index = 1; generator_index <= size(reduced_component);
           generator_index++)
      {
        if (reduced_component[generator_index] != 0)
        {
          if (deg(reduced_component[generator_index]) <= 1)
          {
            linear_part[size(linear_part) + 1]
                = reduced_component[generator_index];
          }
          else
          {
            nonlinear_count++;
            nonlinear_part = reduced_component[generator_index];
          }
        }
      }
      if (nonlinear_count != 0)
      {
        if (nonlinear_count != 1)
        {
          return(list(total, 2, terminal_ideals));
        }
        linear_part = interred(std(linear_part));
        nonlinear_part = reduce(nonlinear_part, linear_part);
        if (nonlinear_part == 0 || deg(nonlinear_part) <= 1)
        {
          return(list(total, 3, terminal_ideals));
        }

        intvec pivot_variables;
        pivot_variables[nvars(basering)] = 0;
        int linear_index;
        int variable_index;
        intvec leading_exponent;
        for (linear_index = 1; linear_index <= size(linear_part); linear_index++)
        {
          if (linear_part[linear_index] != 0)
          {
            leading_exponent = leadexp(linear_part[linear_index]);
            for (variable_index = 1; variable_index <= nvars(basering);
                 variable_index++)
            {
              if (leading_exponent[variable_index] != 0)
              {
                pivot_variables[variable_index] = 1;
                break;
              }
            }
          }
        }

        ideal child = linear_part;
        for (variable_index = 1; variable_index <= nvars(basering);
             variable_index++)
        {
          if (pivot_variables[variable_index] == 0)
          {
            child[size(child) + 1]
                = diff(nonlinear_part, var(variable_index));
          }
        }
        list saturation = sat(child, ideal(nonlinear_part));
        child = std(saturation[1]);
        ideal component_standard = std(component);
        if (dim(child) >= dim(component_standard))
        {
          return(list(total, 4, terminal_ideals));
        }
        list child_result = fr_count_nonisolated(child, depth + 1);
        if (child_result[2] != 0)
        {
          return(list(total, child_result[2], terminal_ideals));
        }
        total = total + child_result[1];
        list child_terminals = child_result[3];
        int child_index;
        for (child_index = 1; child_index <= size(child_terminals); child_index++)
        {
          terminal_ideals[size(terminal_ideals) + 1]
              = child_terminals[child_index];
        }
      }
    }
  }
  return(list(total, 0, terminal_ideals));
}

proc fr_nonisolated_max_exp(poly candidate)
{
  intvec exponents = leadexp(candidate);
  int maximum = 0;
  int variable_index;
  for (variable_index = 1; variable_index <= size(exponents); variable_index++)
  {
    if (exponents[variable_index] > maximum)
    {
      maximum = exponents[variable_index];
    }
  }
  return(maximum);
}

proc fr_nonisolated_preferred(poly lhs, poly rhs)
{
  int lhs_maximum = fr_nonisolated_max_exp(lhs);
  int rhs_maximum = fr_nonisolated_max_exp(rhs);
  if (lhs_maximum < rhs_maximum) { return(1); }
  if (lhs_maximum > rhs_maximum) { return(0); }
  intvec lhs_exponents = leadexp(lhs);
  intvec rhs_exponents = leadexp(rhs);
  int variable_index;
  for (variable_index = 1; variable_index <= size(lhs_exponents);
       variable_index++)
  {
    if (lhs_exponents[variable_index] > rhs_exponents[variable_index])
    {
      return(1);
    }
    if (lhs_exponents[variable_index] < rhs_exponents[variable_index])
    {
      return(0);
    }
  }
  return(0);
}

proc fr_select_nonisolated_basis(list terminal_ideals, int expected_dimension)
{
  ideal selected_basis;
  if (expected_dimension == 0)
  {
    if (size(terminal_ideals) != 0)
    {
      return(list(selected_basis, 6));
    }
    return(list(selected_basis, 0));
  }

  list standard_ideals;
  list quotient_bases;
  int coordinate_count = 0;
  int terminal_index;
  for (terminal_index = 1; terminal_index <= size(terminal_ideals);
       terminal_index++)
  {
    ideal terminal = std(terminal_ideals[terminal_index]);
    int terminal_dimension = vdim(terminal);
    if (terminal_dimension <= 0)
    {
      return(list(selected_basis, 6));
    }
    ideal quotient_basis = kbase(terminal);
    if (size(quotient_basis) != terminal_dimension)
    {
      return(list(selected_basis, 6));
    }
    coordinate_count = coordinate_count + terminal_dimension;
    standard_ideals[size(standard_ideals) + 1] = terminal;
    quotient_bases[size(quotient_bases) + 1] = quotient_basis;
  }
  if (coordinate_count != expected_dimension)
  {
    return(list(selected_basis, 6));
  }

  matrix coordinates[expected_dimension][expected_dimension];
  ideal degree_candidates;
  int selected = 0;
  int current_degree = 0;
  int selection_error = 0;
  int candidate_index;
  int ordering_index;
  int best_index;
  int basis_index;
  int row_offset;
  int found;
  intvec used;
  intvec normal_exponents;
  number normal_coefficient;
  poly candidate;
  poly normal_form;
  while (selected < expected_dimension && current_degree < expected_dimension
         && selection_error == 0)
  {
    if (current_degree == 0)
    {
      degree_candidates = 1;
    }
    else
    {
      degree_candidates = maxideal(current_degree);
    }
    used = 0;
    used[size(degree_candidates)] = 0;
    for (ordering_index = 1; ordering_index <= size(degree_candidates);
         ordering_index++)
    {
      best_index = 0;
      for (candidate_index = 1; candidate_index <= size(degree_candidates);
           candidate_index++)
      {
        if (used[candidate_index] == 0)
        {
          if (best_index == 0)
          {
            best_index = candidate_index;
          }
          else
          {
            if (fr_nonisolated_preferred(degree_candidates[candidate_index],
                                         degree_candidates[best_index]))
            {
              best_index = candidate_index;
            }
          }
        }
      }
      used[best_index] = 1;
      candidate = degree_candidates[best_index];
      for (basis_index = 1; basis_index <= expected_dimension; basis_index++)
      {
        coordinates[basis_index, selected + 1] = 0;
      }
      row_offset = 0;
      for (terminal_index = 1; terminal_index <= size(standard_ideals);
           terminal_index++)
      {
        ideal terminal = std(standard_ideals[terminal_index]);
        ideal quotient_basis = quotient_bases[terminal_index];
        normal_form = reduce(candidate, terminal);
        while (normal_form != 0 && selection_error == 0)
        {
          normal_exponents = leadexp(normal_form);
          normal_coefficient = leadcoef(normal_form);
          found = 0;
          for (basis_index = 1; basis_index <= size(quotient_basis);
               basis_index++)
          {
            if (normal_exponents == leadexp(quotient_basis[basis_index]))
            {
              coordinates[row_offset + basis_index, selected + 1]
                  = normal_coefficient;
              found = 1;
              break;
            }
          }
          if (found == 0)
          {
            selection_error = 7;
          }
          normal_form = normal_form - lead(normal_form);
        }
        row_offset = row_offset + size(quotient_basis);
      }
      if (selection_error == 0 && rank(coordinates) > selected)
      {
        selected++;
        selected_basis[selected] = candidate;
      }
      if (selected == expected_dimension || selection_error != 0)
      {
        break;
      }
    }
    current_degree++;
  }
  if (selection_error != 0)
  {
    return(list(selected_basis, selection_error));
  }
  if (selected != expected_dimension)
  {
    return(list(selected_basis, 8));
  }
  return(list(selected_basis, 0));
}
)SINGULAR";

} // namespace masters::detail
