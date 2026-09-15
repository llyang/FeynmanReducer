#include "FactorSaturation.hpp"
#include <algorithm>
#include <chrono>
#include <flint/nmod_poly.h>
#include <flint/nmod_poly_factor.h>
#include <set>
#include <stdexcept>
#include <tuple>

namespace basis::factor_saturation {
namespace {
struct Poly {
  nmod_poly_t p;
  explicit Poly(const PrimeField& f)
  {
    nmod_poly_init(p, f.prime());
  }
  Poly(const PrimeField& f, const FieldVector& v) : Poly(f)
  {
    for (std::size_t i = 0; i < v.size(); ++i)
      nmod_poly_set_coeff_ui(p, static_cast<slong>(i), v[i]);
  }
  ~Poly()
  {
    nmod_poly_clear(p);
  }
  Poly(const Poly&) = delete;
  FieldVector vector() const
  {
    FieldVector v(static_cast<std::size_t>(std::max<slong>(1, nmod_poly_length(p))));
    for (std::size_t i = 0; i < v.size(); ++i)
      v[i] = nmod_poly_get_coeff_ui(p, static_cast<slong>(i));
    return v;
  }
};
bool regular(const PrimeField& f, const RationalRow& row, const FieldVector& h)
{
  return std::ranges::all_of(row,
                             [&](const auto& x) { return valuation(f, x, h) >= 0; });
}
} // namespace
RationalFunction normalize(const PrimeField& f, RationalFunction x)
{
  Poly n(f, x.numerator), d(f, x.denominator), g(f), q(f);
  if (nmod_poly_is_zero(d.p)) throw std::domain_error("zero rational denominator");
  if (nmod_poly_is_zero(n.p)) return {{0}, {1}};
  nmod_poly_gcd(g.p, n.p, d.p);
  nmod_poly_divexact(q.p, n.p, g.p);
  nmod_poly_swap(q.p, n.p);
  nmod_poly_divexact(q.p, d.p, g.p);
  nmod_poly_swap(q.p, d.p);
  const auto scale = f.inverse(nmod_poly_get_coeff_ui(d.p, nmod_poly_degree(d.p)));
  nmod_poly_scalar_mul_nmod(n.p, n.p, scale);
  nmod_poly_scalar_mul_nmod(d.p, d.p, scale);
  return {n.vector(), d.vector()};
}
RationalFunction multiply(const PrimeField& f, const RationalFunction& a,
                          const RationalFunction& b)
{
  Poly an(f, a.numerator), ad(f, a.denominator), bn(f, b.numerator),
      bd(f, b.denominator), n(f), d(f);
  nmod_poly_mul(n.p, an.p, bn.p);
  nmod_poly_mul(d.p, ad.p, bd.p);
  return normalize(f, {n.vector(), d.vector()});
}
RationalFunction divide(const PrimeField& f, const RationalFunction& a,
                        const RationalFunction& b)
{
  return multiply(f, a, {b.denominator, b.numerator});
}
RationalFunction subtract(const PrimeField& f, const RationalFunction& a,
                          const RationalFunction& b)
{
  Poly an(f, a.numerator), ad(f, a.denominator), bn(f, b.numerator),
      bd(f, b.denominator), n(f), t(f), d(f);
  nmod_poly_mul(n.p, an.p, bd.p);
  nmod_poly_mul(t.p, bn.p, ad.p);
  nmod_poly_sub(n.p, n.p, t.p);
  nmod_poly_mul(d.p, ad.p, bd.p);
  return normalize(f, {n.vector(), d.vector()});
}
std::vector<FieldVector> factors(const PrimeField& f, const FieldVector& x)
{
  Poly p(f, x);
  std::vector<FieldVector> result;
  if (nmod_poly_degree(p.p) <= 0) return result;
  nmod_poly_factor_t fs;
  nmod_poly_factor_init(fs);
  nmod_poly_factor(fs, p.p);
  for (slong i = 0; i < fs->num; ++i) {
    Poly q(f);
    nmod_poly_make_monic(q.p, fs->p + i);
    result.push_back(q.vector());
  }
  nmod_poly_factor_clear(fs);
  std::ranges::sort(result, [](const auto& a, const auto& b) {
    return std::tuple{a.size(), a} < std::tuple{b.size(), b};
  });
  return result;
}
FieldVector noncommon_part(const PrimeField& f, std::span<const FieldVector> xs,
                           std::size_t point)
{
  Poly g(f), p(f), q(f), d(f, xs[point]);
  bool first = true;
  for (const auto& x : xs) {
    if (x.size() <= 1) continue;
    Poly v(f, x);
    if (first) {
      nmod_poly_make_monic(g.p, v.p);
      first = false;
    } else {
      nmod_poly_gcd(p.p, g.p, v.p);
      nmod_poly_swap(g.p, p.p);
    }
  }
  if (first || nmod_poly_degree(d.p) <= 0) return {1};
  nmod_poly_divexact(q.p, d.p, g.p);
  nmod_poly_make_monic(q.p, q.p);
  return q.vector();
}
int valuation(const PrimeField& f, const RationalFunction& x, const FieldVector& h)
{
  Poly n(f, x.numerator), d(f, x.denominator), p(f, h), q(f), r(f);
  if (nmod_poly_degree(p.p) <= 0)
    throw std::invalid_argument("valuation requires nonconstant factor");
  if (nmod_poly_is_zero(d.p)) throw std::domain_error("zero denominator in valuation");
  if (nmod_poly_is_zero(n.p)) return infinite_valuation;
  const auto multiplicity = [&](Poly& a) {
    int count = 0;
    while (nmod_poly_degree(a.p) >= nmod_poly_degree(p.p)) {
      nmod_poly_divrem(q.p, r.p, a.p, p.p);
      if (!nmod_poly_is_zero(r.p)) break;
      ++count;
      nmod_poly_swap(a.p, q.p);
    }
    return count;
  };
  return multiplicity(n) - multiplicity(d);
}
RationalRow rebase(const PrimeField& f, const RationalRow& a, const RationalRow& b,
                   std::size_t j)
{
  if (a.size() != b.size() || j >= a.size())
    throw std::invalid_argument("rational rebase shape mismatch");
  RationalRow out(a.size());
  const auto ratio = divide(f, a[j], b[j]);
  for (std::size_t i = 0; i < a.size(); ++i)
    out[i] = i == j ? ratio : subtract(f, a[i], multiply(f, ratio, b[i]));
  return out;
}
RationalRow apply_steps(const PrimeField& f, RationalRow row,
                        std::span<const Step> steps, const RowProvider& provider)
{
  std::vector<RationalRow> pivots;
  for (const auto& step : steps) {
    auto pivot = provider(step.candidate);
    if (!pivot) throw std::runtime_error("missing rational pivot coordinates");
    for (std::size_t k = 0; k < pivots.size(); ++k)
      *pivot = rebase(f, *pivot, pivots[k], steps[k].slot);
    row = rebase(f, row, *pivot, step.slot);
    pivots.push_back(std::move(*pivot));
  }
  return row;
}
namespace {
SaturationResult saturate_ranked(
    const PrimeField& f, const FieldVector& h, std::span<const std::size_t> initial,
    std::span<const std::uint32_t> sectors, std::size_t target_count,
    const RowProvider& provider, std::span<const FactorObservation> known,
    const std::set<std::vector<std::size_t>>& committed_seen, const PairScorer& score,
    std::size_t maximum_steps, const RowPrefetch& prefetch)
{
  using Clock = std::chrono::steady_clock;
  const auto elapsed = [](auto start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
  };
  const auto key = [](std::vector<std::size_t> ids) {
    std::ranges::sort(ids);
    return ids;
  };
  SaturationResult result;
  result.selected.assign(initial.begin(), initial.end());
  if (initial.empty() || initial.size() > sectors.size())
    throw std::invalid_argument("invalid initial basis");
  std::vector<RationalRow> rows;
  for (std::size_t c = 0; c < sectors.size() + target_count; ++c) {
    if (prefetch && c % 32 == 0)
      prefetch(c, std::min<std::size_t>(32, sectors.size() + target_count - c));
    auto row = provider(c);
    if (!row) return result;
    if (row->size() != initial.size())
      throw std::invalid_argument("ranked candidate shape mismatch");
    rows.push_back(std::move(*row));
    if (c < sectors.size()) {
      ++result.covered;
      result.regular += regular(f, rows.back(), h);
    }
  }
  auto visited = committed_seen;
  visited.insert(key(result.selected));
  for (;;) {
    PairDecision decision;
    auto started = Clock::now();
    for (const auto& o : known)
      if (o.factor != h && std::ranges::all_of(rows, [&](const auto& row) {
            return regular(f, row, o.factor);
          }))
        decision.protected_factors.push_back(o.factor);
    std::vector<PivotPair> pairs;
    result.blocked.clear();
    for (std::size_t c = 0; c < sectors.size(); ++c) {
      if (regular(f, rows[c], h)) continue;
      std::vector<int> vs;
      for (const auto& x : rows[c])
        vs.push_back(valuation(f, x, h));
      const auto minimum = *std::ranges::min_element(vs);
      std::vector<std::size_t> all;
      bool eligible = false;
      for (std::size_t j = 0; j < vs.size(); ++j) {
        if (vs[j] != minimum) continue;
        all.push_back(j);
        if (sectors[c] != sectors[result.selected[j]]) continue;
        eligible = true;
        ++decision.legal_pairs;
        auto child = result.selected;
        child[j] = c;
        if (visited.contains(key(std::move(child)))) {
          ++decision.cycle_excluded;
          continue;
        }
        PivotPair pair{c, j, minimum};
        for (const auto& q : decision.protected_factors) {
          const auto v = valuation(f, rows[c][j], q);
          if (v < 0) throw std::logic_error("protected factor is not regular");
          if (v > 0) {
            ++pair.reintroduced;
            pair.reintroduced_degree += q.size() - 1;
          }
        }
        ++decision.risk_groups[{pair.reintroduced, pair.reintroduced_degree}];
        pairs.push_back(pair);
      }
      if (!eligible) result.blocked.push_back({c, minimum, std::move(all)});
    }
    if (pairs.empty()) {
      decision.scoring_seconds = elapsed(started);
      result.status = result.regular == sectors.size() ? "candidate_pool_regular"
                      : decision.cycle_excluded        ? "cycle_blocked"
                                                       : "eligibility_blocked";
      result.pair_decisions.push_back(std::move(decision));
      return result;
    }
    if (result.steps.size() == maximum_steps) {
      decision.scoring_seconds = elapsed(started);
      result.pair_decisions.push_back(std::move(decision));
      result.status = "budget_exhausted";
      return result;
    }
    const auto best_risk = decision.risk_groups.begin()->first;
    std::erase_if(pairs, [&](const auto& p) {
      return std::pair{p.reintroduced, p.reintroduced_degree} != best_risk;
    });
    const auto scored = score(pairs, result.steps);
    const auto& scores = scored.values;
    if (scores.size() != pairs.size())
      throw std::logic_error("pair score count mismatch");
    const auto rank = [&](std::size_t i) {
      const auto& s = scores[i];
      return std::tuple{s.classification, s.moving_degree, s.degree, pairs[i].candidate,
                        pairs[i].slot};
    };
    std::size_t best = 0;
    for (std::size_t i = 0; i < pairs.size(); ++i) {
      if (rank(i) < rank(best)) best = i;
      decision.incomplete_scores += scores[i].classification == 2;
    }
    decision.scored_pairs = pairs.size();
    decision.selected = pairs[best];
    decision.numerator = scores[best];
    decision.data_seconds = scored.data_seconds;
    decision.coordinate_seconds = scored.coordinate_seconds;
    decision.scoring_seconds = std::max(0.0, elapsed(started) - scored.data_seconds -
                                                 scored.coordinate_seconds);
    started = Clock::now();
    const auto pair = pairs[best];
    const auto pivot = rows[pair.candidate];
    const auto before = result.regular;
    result.regular = 0;
    for (std::size_t c = 0; c < rows.size(); ++c) {
      const auto was_regular = regular(f, rows[c], h);
      rows[c] = rebase(f, rows[c], pivot, pair.slot);
      const auto now_regular = regular(f, rows[c], h);
      if (was_regular && !now_regular)
        throw std::logic_error("ranked single-factor monotonicity violated");
      if (c < sectors.size()) result.regular += now_regular;
    }
    if (result.regular <= before)
      throw std::logic_error("ranked single-factor step made no progress");
    // The old masters belong to C: a positive pivot valuation necessarily
    // reintroduces q through 1/b_j, whereas a unit preserves all regular rows.
    for (const auto& q : decision.protected_factors) {
      const bool predicted = valuation(f, pivot[pair.slot], q) > 0;
      const bool actual = std::ranges::any_of(
          rows, [&](const auto& row) { return !regular(f, row, q); });
      if (predicted != actual)
        throw std::logic_error("protected-factor pivot prediction disagrees");
    }
    result.selected[pair.slot] = pair.candidate;
    result.steps.push_back({pair.candidate, pair.slot, pair.valuation, result.regular});
    if (result.steps.size() > sectors.size() - initial.size())
      throw std::logic_error("ranked single-factor step bound exceeded");
    visited.insert(key(result.selected));
    decision.coordinate_seconds += elapsed(started);
    result.pair_decisions.push_back(std::move(decision));
  }
}
} // namespace

SequenceResult sequential(const PrimeField& field, std::span<const std::size_t> initial,
                          std::span<const std::uint32_t> sectors,
                          std::size_t target_count, const RowProvider& original,
                          const FactorFinder& find, const SequenceValidator& validate,
                          std::size_t maximum_states, const PairScorer& score_pairs,
                          const RowPrefetch& prefetch)
{
  if (!score_pairs)
    throw std::invalid_argument("factor saturation requires a pair scorer");
  struct CachedRow {
    std::optional<RationalRow> row;
    std::size_t version = 0;
  };
  std::map<std::size_t, CachedRow> cache;
  std::vector<RationalRow> pivots;
  SequenceResult result;
  result.selected.assign(initial.begin(), initial.end());
  const auto key = [](std::span<const std::size_t> selected) {
    std::vector<std::size_t> sorted(selected.begin(), selected.end());
    std::ranges::sort(sorted);
    return sorted;
  };
  std::set<std::vector<std::size_t>> seen{key(initial)};
  const RowProvider current = [&](std::size_t row) -> std::optional<RationalRow> {
    auto [it, inserted] = cache.try_emplace(row);
    auto& entry = it->second;
    if (inserted) entry.row = original(row);
    if (!entry.row) return std::nullopt;
    while (entry.version < pivots.size()) {
      *entry.row = rebase(field, *entry.row, pivots[entry.version],
                          result.steps[entry.version].slot);
      ++entry.version;
    }
    return entry.row;
  };
  std::vector<FactorObservation> observations;
  const auto publish = [&](SequenceRound round) {
    round.cumulative_steps = result.steps;
    result.rounds.push_back(std::move(round));
  };
  for (;;) {
    auto scan = find(result.steps);
    if (scan.status != "complete") {
      result.status = scan.status;
      return result;
    }
    std::ranges::sort(scan.factors, [](const auto& a, const auto& b) {
      return std::tuple{a.size(), a} < std::tuple{b.size(), b};
    });
    scan.factors.erase(std::unique(scan.factors.begin(), scan.factors.end()),
                       scan.factors.end());
    if (scan.factors.empty()) {
      result.status = validate(result.selected, result.steps);
      if (result.status == "moving_pole") result.status = "anchor_inconclusive";
      return result;
    }
    for (const auto& h : scan.factors)
      if (std::ranges::none_of(observations,
                               [&](const auto& o) { return o.factor == h; }))
        observations.push_back({h, 0, 0, false, result.rounds.size()});
    bool committed = false;
    for (const auto& h : scan.factors) {
      SequenceRound round;
      round.factor = h;
      if (result.states_used == maximum_states) {
        result.status = "budget_exhausted";
        return result;
      }
      const PairScorer full_score = [&](std::span<const PivotPair> pairs,
                                        std::span<const Step> suffix) {
        auto path = result.steps;
        path.insert(path.end(), suffix.begin(), suffix.end());
        return score_pairs(pairs, path);
      };
      round.saturation = saturate_ranked(
          field, h, result.selected, sectors, target_count, current, observations, seen,
          full_score, maximum_states - result.states_used, prefetch);
      result.states_used += round.saturation.steps.size();
      round.status = round.saturation.status;
      if (round.status == "budget_exhausted") {
        publish(std::move(round));
        result.status = "budget_exhausted";
        return result;
      }
      if (round.status != "candidate_pool_regular") {
        publish(std::move(round));
        continue;
      }
      bool incomplete = false, target_pole = false;
      for (std::size_t row = sectors.size(); row < sectors.size() + target_count;
           ++row) {
        auto coordinates = current(row);
        if (!coordinates) {
          incomplete = true;
          break;
        }
        *coordinates = apply_steps(field, std::move(*coordinates),
                                   round.saturation.steps, current);
        if (!regular(field, *coordinates, h)) target_pole = true;
      }
      if (incomplete) {
        round.status = "interpolation_incomplete";
        publish(std::move(round));
        continue;
      }
      if (target_pole) {
        round.status = "sampled_pool_insufficient";
        publish(std::move(round));
        result.status = "sampled_pool_insufficient";
        return result;
      }
      if (round.saturation.steps.empty()) {
        round.status = "no_progress";
        publish(std::move(round));
        continue;
      }
      // Prepare the entire suffix before publishing it. Rows in cache remain at
      // their old versions and advance on demand after this transaction commits.
      std::vector<RationalRow> suffix_pivots;
      for (const auto& step : round.saturation.steps) {
        auto pivot = current(step.candidate);
        if (!pivot) throw std::logic_error("committing an unavailable pivot");
        for (std::size_t k = 0; k < suffix_pivots.size(); ++k)
          *pivot =
              rebase(field, *pivot, suffix_pivots[k], round.saturation.steps[k].slot);
        suffix_pivots.push_back(std::move(*pivot));
      }
      result.steps.insert(result.steps.end(), round.saturation.steps.begin(),
                          round.saturation.steps.end());
      pivots.insert(pivots.end(), std::make_move_iterator(suffix_pivots.begin()),
                    std::make_move_iterator(suffix_pivots.end()));
      {
        auto intermediate = result.selected;
        for (std::size_t i = 0; i + 1 < round.saturation.steps.size(); ++i) {
          const auto& step = round.saturation.steps[i];
          intermediate[step.slot] = step.candidate;
          seen.insert(key(intermediate));
        }
      }
      result.selected = round.saturation.selected;
      for (auto& o : observations) {
        const bool previously_cleared =
            std::ranges::any_of(result.rounds, [&](const auto& r) {
              return r.factor == o.factor && r.status == "committed";
            });
        const auto previous_candidates = o.candidate_poles,
                   previous_targets = o.target_poles;
        o.candidate_poles = 0;
        o.target_poles = 0;
        o.reintroduced = false;
        for (std::size_t row = 0; row < sectors.size() + target_count; ++row) {
          auto coordinates = current(row);
          if (!coordinates)
            throw std::logic_error("committed saturation has incomplete coverage");
          if (!regular(field, *coordinates, o.factor)) {
            if (row < sectors.size())
              ++o.candidate_poles;
            else
              ++o.target_poles;
          }
        }
        o.reintroduced = previously_cleared && previous_candidates == 0 &&
                         previous_targets == 0 &&
                         (o.candidate_poles != 0 || o.target_poles != 0);
      }
      round.observations = observations;
      round.status = "committed";
      const bool fresh = seen.insert(key(result.selected)).second;
      if (!fresh) round.status = "cycle";
      publish(std::move(round));
      if (!fresh) {
        result.status = "cycle";
        return result;
      }
      committed = true;
      break;
    }
    if (!committed) {
      result.status = "all_factors_blocked_or_incomplete";
      return result;
    }
  }
}

} // namespace basis::factor_saturation
