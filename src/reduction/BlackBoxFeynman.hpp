#pragma once

#include <array>
#include <atomic>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <firefly/BlackBoxBase.hpp>
#include <firefly/FFInt.hpp>

#include "core/AtomicSharedPtr.hpp"
#include "core/Config.hpp"
#include "core/MasterCandidates.hpp"
#include "reduction/CompiledReplay.hpp"
#include "reduction/KernelPlanning.hpp"
#include "reduction/Monomial.hpp"
#include "reduction/ReductionOptions.hpp"
#include "reduction/ReductionProgress.hpp"
#include "reduction/TopLpTargets.hpp"

namespace reduction::detail {
struct KernelPublicationInput;
}

struct ReductionKernelStatistics {
  std::string compact_policy = "sector-markowitz";
  std::string_view block_layout = "seed-sector-scc";
  std::string_view numerator_strategy = "projected";
  std::size_t lp_variable_count = 0;

  std::size_t maximum_g_shift = 0;
  std::size_t maximum_delta_derivative = 0;
  std::size_t top_lp_target_expressions = 0;
  std::size_t top_lp_replay_expressions = 0;
  std::size_t top_lp_replay_product_nodes = 0;
  std::size_t top_lp_replay_polynomial_factors = 0;
  std::size_t jet_state_count = 0;
  std::size_t jet_row_count = 0;
  std::size_t jet_relation_count = 0;
  std::size_t jet_group_count = 0;
  std::size_t ansatz_expansion_rounds = 0;
  std::size_t ansatz_expanded_groups = 0;
  std::size_t ansatz_expanded_points = 0;
  std::size_t basis_selection_candidates = 0;
  std::size_t basis_selection_masters = 0;
  std::size_t generated_rows = 0;
  std::size_t generated_columns = 0;
  std::size_t provisional_ansatz_columns = 0;
  std::array<std::size_t, ansatz_family_count> provisional_ansatz_family_columns{};
  std::vector<std::size_t> provisional_ansatz_dot_histogram;
  std::size_t provisional_dimension = 0;
  std::size_t provisional_rhs_columns = 0;
  double master_rank_completion_seconds = 0.0;
  double master_rank_check_seconds = 0.0;
  std::size_t provisional_relation_pivots = 0;
  std::size_t provisional_score_refresh_columns = 0;
  std::size_t provisional_incidence_records_scanned = 0;
  std::size_t provisional_parallel_refresh_batches = 0;
  std::size_t provisional_parallel_refresh_columns = 0;
  std::size_t provisional_stale_choice_pops = 0;
  std::size_t provisional_choice_queue_compactions = 0;
  std::size_t provisional_compacted_choice_entries = 0;
  std::size_t provisional_maximum_choice_queue_size = 0;
  std::size_t provisional_row_eliminations = 0;
  std::size_t provisional_parallel_row_batches = 0;
  std::size_t provisional_parallel_row_eliminations = 0;
  bool provisional_rhs_fallback = false;
  std::size_t live_ansatz_columns = 0;
  std::vector<std::size_t> live_ansatz_dot_histogram;
  std::size_t compact_dimension = 0;
  std::size_t cross_group_pivots = 0;
  std::size_t block_count = 0;
  std::size_t maximum_block_dimension = 0;
  std::size_t physical_target_rhs = 0;
  std::size_t target_sector_batches = 0;
  std::size_t maximum_block_targets = 0;
  std::string_view replay_orientation = "target";
  std::size_t solver_rhs_columns = 0;
  std::size_t master_factor_slots = 0;
  std::size_t master_forward_slots = 0;
  std::size_t master_live_slots = 0;
  std::size_t master_forward_operations = 0;
  std::size_t master_live_operations = 0;
  std::size_t contraction_fmas = 0;
  std::size_t estimated_target_operations = 0;
  std::size_t estimated_master_operations = 0;
  std::size_t tape_instructions = 0;
  std::size_t matrix_slots = 0;
  std::size_t rhs_slots = 0;
  std::size_t target_rhs_slots = 0;
  std::size_t coupling_fmas = 0;
  std::size_t rhs_ranges = 0;
  std::size_t rhs_ranged_operations = 0;
  std::size_t coefficient_expressions = 0;
  std::size_t pooled_coefficient_loads = 0;
  std::size_t lp_expressions = 0;
  std::size_t total_outputs = 0;
  std::size_t reconstructed_outputs = 0;
  std::size_t probabilistic_zero_outputs = 0;
  std::array<std::size_t, 6> tape_opcode_counts{};

  double reference_plan_seconds = 0.0;
  double basis_selection_seconds = 0.0;
  double block_recording_seconds = 0.0;
  double reference_validation_seconds = 0.0;
  double replay_finalize_seconds = 0.0;
  double replay_validation_seconds = 0.0;
  double provisional_relation_elimination_seconds = 0.0;
  double provisional_back_substitution_seconds = 0.0;
  double provisional_score_refresh_seconds = 0.0;
  double provisional_row_elimination_seconds = 0.0;
  std::size_t compiled_groups = 0;
  std::size_t compiled_matrix_operations = 0;
  std::size_t compiled_rhs_operations = 0;
  std::size_t compiled_rhs_ranges = 0;
  std::size_t compiled_tape_bytes = 0;
  std::size_t matrix_contiguous_operations = 0;
  std::size_t matrix_total_operations = 0;
};

/// =========================================================================
/// 费曼积分约化的 FireFly 黑盒接口
///
/// 设计采用两阶段策略：
///   阶段一（准备）: FireFly 启动前扫描拟设边界，构建并验证 Tape。
///   阶段二（回放）: FireFly 的所有求值仅装载矩阵元并回放 Tape，
///                   避免在 worker 回调中改变 kernel 结构。
///
/// FireFly 框架通过 CRTP 模式调用 operator()，每次传入一组
/// 有限域上的参数值，返回约化系数的有限域值。
/// =========================================================================
class BlackBoxFeynman : public firefly::BlackBoxBase<BlackBoxFeynman> {

  // Low-level preparation borrows an immutable Config. perform_reduction keeps
  // its owned Config alive for the complete FireFly reconstruction; direct
  // users of this class must provide the same lifetime and immutability.
  const Config& cfg; ///< 配置信息（多项式、基底、目标积分等）
  static constexpr size_t MAX_BILINEAR_BASIS = 255;

  /// 稀疏符号装载项：weight * bilinear_basis[bb_idx]
  struct AnsatzEntry {
    std::uint32_t flat_idx;
    std::uint8_t bb_idx;
    std::int64_t weight;

    bool operator==(const AnsatzEntry&) const = default;
  };
  static_assert(sizeof(AnsatzEntry) == 16);
  static_assert(sizeof(linalg::SparseEntry<firefly::FFInt>) == 16);

  struct ReferenceMatrixTerm {
    std::uint32_t row;
    std::uint32_t column;
    std::uint8_t bb_idx;
    std::int64_t weight;
  };

  struct ReferenceTopLpMatrixTerm {
    std::uint32_t row;
    std::uint32_t column;
    std::uint32_t expression;
    std::int64_t weight;
  };

  struct ReferenceRhsTerm {
    std::uint32_t row;
    std::uint32_t target;
    std::uint8_t bb_idx;
    std::int64_t weight;
  };

  struct ReferenceTopLpRhsTerm {
    std::uint32_t row;
    std::uint32_t target;
    std::uint32_t expression;
    std::int64_t weight;
  };

  struct ReferenceMatrixCoordinate {
    std::uint32_t row = 0;
    std::uint32_t column = 0;
    std::uint32_t matrix_term_begin = 0;
    std::uint32_t matrix_term_end = 0;
    std::uint32_t top_lp_term_begin = 0;
    std::uint32_t top_lp_term_end = 0;
  };

  struct ReferenceEvaluationPlan {
    std::uint32_t dimension = 0;
    std::uint32_t closure_row_count = 0;
    std::vector<std::uint32_t> reference_row_by_closure_row;
    std::vector<ReferenceMatrixTerm> closure_matrix_terms;
    std::vector<ReferenceTopLpMatrixTerm> closure_top_lp_matrix_terms;
    std::vector<ReferenceMatrixCoordinate> matrix_coordinates;
    std::vector<ReferenceRhsTerm> closure_rhs_terms;
    std::vector<ReferenceTopLpRhsTerm> closure_top_lp_rhs_terms;
  };

  // ─── 骨架数据（探测阶段生成，回放阶段使用）───
  std::vector<AnsatzEntry> skeleton_B; ///< B 矩阵的非零元

  /// 预编译的矩阵装载指令
  struct LoadInstruction {
    uint64_t w_ff;           ///< 当前有限域下预编码为 Montgomery 形式的权重
    std::uint32_t flat_idx;  ///< 紧凑工作区槽位
    std::uint8_t bb_idx;     ///< 双线性基底索引
    std::uint8_t group_size; ///< 仅组首保存同一槽位的连续指令数
  };
  static_assert(sizeof(LoadInstruction) == 16);

  struct PooledCoefficientLoad {
    std::uint32_t flat_idx;
    std::uint32_t expression;
  };

  struct TopLpRhsEntry {
    std::uint32_t flat_idx;
    std::uint32_t expression;
    std::int64_t weight;
  };

  struct TopLpLoadInstruction {
    std::uint64_t w_ff;
    std::uint32_t flat_idx;
    std::uint32_t expression;
  };
  static_assert(sizeof(TopLpLoadInstruction) == 16);

  struct CoefficientExpression {
    std::uint32_t term_begin;
    std::uint8_t term_count;
  };

  struct CoefficientTerm {
    std::uint64_t w_ff = 0;
    std::int64_t weight = 0;
    std::uint8_t bb_idx = 0;
  };

  std::vector<LoadInstruction> loader_B;      ///< B 矩阵的快速装载指令流
  std::vector<std::int64_t> loader_B_weights; ///< 质数切换用冷数据
  std::vector<PooledCoefficientLoad> pooled_loader_B;
  std::vector<TopLpRhsEntry> top_lp_rhs_skeleton;
  std::vector<TopLpLoadInstruction> top_lp_loader_B;
  std::vector<std::int64_t> top_lp_loader_B_weights;

  struct BlockCoupling {
    std::uint32_t destination_row;
    std::uint32_t source_row;
    std::uint32_t target_begin = 0;
    std::uint32_t target_end = 0;
    std::uint64_t master_mask = 0;
  };

  struct CouplingOperation {
    std::uint32_t destination;
    std::uint32_t source;

    bool operator==(const CouplingOperation&) const = default;
  };

  struct PendingMasterOperation {
    std::uint32_t destination_row;
    std::uint32_t source_row;
    std::uint32_t factor_group;
    std::uint32_t range_begin = 0;
    std::uint32_t range_end = 0;
    std::uint64_t mask = 0;
    bool scale;
  };

  enum class MasterOperationKind : std::uint32_t { Scale, Eliminate };

  struct CompiledMasterOperation {
    std::uint32_t factor_group;
    std::uint32_t range_begin;
    std::uint32_t range_end;
    MasterOperationKind kind;
  };
  static_assert(sizeof(CompiledMasterOperation) == 16);

  struct CompiledMasterCoupling {
    std::uint32_t range_begin;
    std::uint32_t range_end;
  };
  static_assert(sizeof(CompiledMasterCoupling) == 8);

  struct MasterColumnRange {
    std::uint32_t begin;
    std::uint32_t end;
  };

  struct MasterSlotRange {
    std::uint32_t destination;
    std::uint32_t source;
    std::uint32_t column_begin;
    std::uint32_t count;
  };

  struct ContiguousSlotRun {
    std::uint32_t begin;
    std::uint32_t count;
  };

  struct MasterReplayProgram {
    std::vector<std::uint32_t> factor_capture_slots;
    std::uint32_t factor_count = 0;
    std::vector<PendingMasterOperation> pending_operations;
    std::vector<CompiledMasterOperation> operations;
    std::vector<CompiledMasterCoupling> couplings;
    std::vector<MasterColumnRange> column_ranges;
    std::vector<MasterSlotRange> slot_ranges;
    bool uses_masks = false;
  };

  struct BlockReplayProgram {
    std::uint32_t row_begin = 0;
    std::uint32_t dimension = 0;
    std::uint32_t matrix_workspace_size = 0;
    std::vector<linalg::Instruction> tape;
    linalg::CompiledTape compiled_tape;
    std::unique_ptr<MasterReplayProgram> master;

    std::vector<AnsatzEntry> skeleton_M;
    std::vector<LoadInstruction> loader_M;
    std::vector<std::int64_t> loader_M_weights;
    std::vector<PooledCoefficientLoad> pooled_loader_M;
    std::vector<TopLpRhsEntry> top_lp_skeleton_M;
    std::vector<TopLpLoadInstruction> top_lp_loader_M;
    std::vector<std::int64_t> top_lp_loader_M_weights;

    std::vector<BlockCoupling> couplings;
    std::vector<std::uint32_t> coupling_targets;
    std::vector<CouplingOperation> coupling_operations;
    std::vector<AnsatzEntry> skeleton_couplings;
    std::vector<LoadInstruction> loader_couplings;
    std::vector<std::int64_t> loader_coupling_weights;
    std::vector<PooledCoefficientLoad> pooled_loader_couplings;
    std::vector<TopLpRhsEntry> top_lp_skeleton_couplings;
    std::vector<TopLpLoadInstruction> top_lp_loader_couplings;
    std::vector<std::int64_t> top_lp_loader_coupling_weights;
  };

  struct PendingBlockReplay {
    std::vector<BlockReplayProgram> programs;
    std::vector<linalg::TapeResult> recorded_blocks;
    std::vector<std::vector<linalg::Instruction>> operation_tapes;
    std::vector<std::uint32_t> solution_row_by_column;
    std::vector<std::uint32_t> reference_solution_row_by_column;
    std::vector<std::size_t> block_of_position;
    std::vector<AnsatzEntry> rhs_skeleton;
    std::vector<TopLpRhsEntry> top_lp_rhs_skeleton;
  };

  // Published replay state is always block triangular. A matrix without useful
  // internal boundaries is represented by one block, not a separate solver.
  std::vector<BlockReplayProgram> replay_blocks;
  std::vector<CoefficientExpression> coefficient_expressions;
  std::vector<CoefficientTerm> coefficient_terms;
  bool coefficient_pool_initialized = false;
  std::unique_ptr<ReferenceEvaluationPlan> reference_evaluation_plan;

  std::unique_ptr<PendingBlockReplay> pending_block_replay;
  std::vector<std::uint32_t> reconstructed_output_positions;
  bool output_support_initialized = false;

  enum class ReplayOrientation { Target, Master };
  ReplayOrientation replay_orientation = ReplayOrientation::Target;

  size_t square_dim = 0; ///< 压缩方阵的维数（= 满秩列数）
  size_t matrix_workspace_size = 0;
  size_t rhs_workspace_size = 0;
  size_t target_rhs_workspace_size = 0;
  ReductionKernelStatistics kernel_statistics_;
  NumeratorReductionStrategy numerator_strategy = NumeratorReductionStrategy::Projected;
  ReplayOrientationPreference replay_orientation_preference =
      ReplayOrientationPreference::Auto;
  AnsatzDotOrdering ansatz_dot_ordering = AnsatzDotOrdering::Auto;
  unsigned direct_pinched_dot_halo = 1;
  DirectGroupOrdering direct_group_ordering = DirectGroupOrdering::Auto;
  DirectRankOrdering direct_rank_ordering = DirectRankOrdering::LowFirst;
  bool check_master_independence = false;
  TopLpTargetPlan top_lp_target_plan;

  struct ReplayOutput {
    std::uint32_t rhs_slot;
    std::uint32_t target;
    std::uint32_t basis;
    std::uint32_t contraction_begin = 0;
    std::uint32_t contraction_end = 0;
  };
  std::vector<ReplayOutput> replay_outputs;

  struct MasterContraction {
    std::uint32_t target_slot;
    std::uint32_t master_slot;
  };
  std::vector<MasterContraction> master_contractions;
  std::vector<std::uint32_t> master_rhs_columns;
  std::vector<std::uint32_t> master_seed_slots;
  std::size_t master_factor_workspace_size = 0;

  struct CoefficientSelection {
    enum class LpReciprocalSide : std::uint8_t {
      TargetDenominator,
      BasisNumerator,
    };
    struct LpReciprocalRequest {
      std::size_t program = 0;
      LpReciprocalSide side = LpReciprocalSide::TargetDenominator;

      auto operator<=>(const LpReciprocalRequest&) const = default;
    };

    std::vector<std::uint32_t> top_lp_expressions;
    std::vector<std::uint32_t> top_lp_product_nodes;
    std::vector<std::uint32_t> extended_polynomial_factors;
    std::vector<std::uint32_t> pooled_expressions;
    std::vector<std::size_t> lp_programs;
    std::vector<LpReciprocalRequest> lp_reciprocals;
    std::vector<std::uint32_t> targets;
    std::vector<std::uint32_t> basis;
    std::uint16_t maximum_falling_degree = 0;
    std::size_t maximum_positive_lp_delta = 0;
    std::size_t maximum_negative_lp_delta = 0;
  };
  CoefficientSelection replay_coefficients;

  struct SelectedMasterBlock {
    std::vector<CompiledMasterOperation> operations;
    std::vector<CompiledMasterCoupling> couplings;
    std::vector<MasterSlotRange> slot_ranges;
    std::vector<LoadInstruction> loader_couplings;
    std::vector<PooledCoefficientLoad> pooled_loader_couplings;
    std::vector<TopLpLoadInstruction> top_lp_loader_couplings;
  };

  struct SelectedMasterStructure {
    std::vector<std::uint32_t> master_columns;
    std::vector<SelectedMasterBlock> blocks;
    std::vector<ContiguousSlotRun> rhs_clear_runs;
    CoefficientSelection coefficient_seed;
  };

  struct SelectedMasterReplay {
    std::shared_ptr<const SelectedMasterStructure> structure;
    std::vector<ContiguousSlotRun> rhs_clear_runs;
    std::vector<ContiguousSlotRun> target_clear_runs;
  };

  struct ReplayVariant {
    std::vector<std::uint32_t> active_outputs;
    CoefficientSelection coefficients;
    std::vector<BlockReplayProgram> blocks;
    std::vector<LoadInstruction> loader_B;
    std::vector<PooledCoefficientLoad> pooled_loader_B;
    std::vector<TopLpLoadInstruction> top_lp_loader_B;
    std::vector<ReplayOutput> outputs;
    std::vector<std::uint32_t> master_columns;
    std::optional<SelectedMasterReplay> selected_master;
    std::size_t matrix_workspace_size = 0;
    std::size_t rhs_workspace_size = 0;
  };
  mutable std::mutex replay_variant_mutex;
  mutable core::AtomicSharedPtr<const ReplayVariant> replay_variant;
  // Share heavy master programs along the current/in-flight inclusion chain,
  // without extending their lifetime beyond the replay variants that use them.
  // A non-comparable public request starts a new chain.
  mutable std::vector<std::weak_ptr<const SelectedMasterStructure>>
      selected_master_structure_cache;

  struct TopLpProductNode {
    std::uint32_t parent;
    std::uint32_t polynomial_factor;
  };
  struct TopLpExpressionTerm {
    std::int64_t weight;
    std::uint32_t product;
    std::uint16_t falling_degree;
  };
  struct TopLpExpressionProgram {
    std::uint32_t term_begin;
    std::uint32_t term_count;
  };
  std::vector<TopLpProductNode> top_lp_product_nodes;
  std::vector<TopLpExpressionTerm> top_lp_expression_terms;
  std::vector<TopLpExpressionProgram> top_lp_expression_programs;
  std::uint16_t top_lp_maximum_falling_degree = 0;

  struct LpProgram {
    std::int64_t index_sum = 0;
    std::int64_t delta = 0;
    std::vector<std::int64_t> denominator_factors;

    bool operator==(const LpProgram&) const = default;
  };
  std::vector<LpProgram> lp_programs;
  std::vector<std::size_t> target_lp_program_ids;
  std::vector<std::size_t> basis_lp_program_ids;
  std::vector<CoefficientSelection::LpReciprocalRequest> full_lp_reciprocals;
  std::size_t maximum_positive_lp_delta = 0;
  std::size_t maximum_negative_lp_delta = 0;
  std::size_t lp_variable_count = 0;

  template <typename T> struct LpFraction {
    T numerator;
    T denominator;
  };

  /// Evaluate an LP normalization without performing a field division.
  template <typename T>
  LpFraction<T> evaluate_lp_fraction(const LpProgram& program,
                                     std::span<const T> positive_delta_products,
                                     std::span<const T> negative_delta_products) const
  {
    T num(1), den(1);
    if (program.index_sum % 2 != 0) num = T(0) - num; // (-1)^A
    if (program.delta > 0) {
      const auto degree = static_cast<std::size_t>(program.delta);
      if (degree >= positive_delta_products.size())
        throw std::logic_error("positive LP delta product is unavailable");
      num = num * positive_delta_products[degree];
    } else if (program.delta < 0) {
      if (program.delta == std::numeric_limits<std::int64_t>::min())
        throw std::overflow_error("negative LP delta exceeds size_t");
      const auto degree = static_cast<std::size_t>(-program.delta);
      if (degree >= negative_delta_products.size())
        throw std::logic_error("negative LP delta product is unavailable");
      den = den * negative_delta_products[degree];
    }
    for (const std::int64_t factor : program.denominator_factors)
      den = den * T(factor);
    return {num, den};
  }

  /// Evaluate all coefficient programs in the current finite field.
  [[nodiscard]] EvaluatedCoeffs<firefly::FFInt>
  evaluate_coefficients(const std::vector<firefly::FFInt>& values,
                        const CoefficientSelection* selection = nullptr) const;

  /// 探测能解出所有目标积分的拟设边界，并生成压缩方阵的骨架结构
  ///
  /// Projected 和 direct 都从各自的初始 domain 开始，按 residual group 执行局部
  /// expansion。无法闭合时抛出结构性错误；成功后录制消元 Tape。
  void plan_kernel(const EvaluatedCoeffs<firefly::FFInt>& coeffs,
                   const std::vector<firefly::FFInt>& values,
                   const ReductionProgressCallback& progress,
                   std::span<const std::uint32_t> relation_source_sectors);
  void plan_kernel_impl(const EvaluatedCoeffs<firefly::FFInt>& coeffs,
                        const std::vector<firefly::FFInt>& values,
                        const ReductionProgressCallback& progress,
                        std::span<const std::uint32_t> relation_source_sectors);
  void
  plan_direct_numerator_kernel(const EvaluatedCoeffs<firefly::FFInt>& coeffs,
                               const std::vector<firefly::FFInt>& values,
                               const ReductionProgressCallback& progress,
                               std::span<const std::uint32_t> relation_source_sectors);
  void publish_kernel_plan(reduction::detail::KernelPublicationInput&& input,
                           const EvaluatedCoeffs<firefly::FFInt>& coeffs,
                           const std::vector<firefly::FFInt>& values);
  void publish_compact_statistics(const reduction::detail::CompactSelection& selection);
  void publish_ansatz_statistics(
      const reduction::detail::CompactSelection& selection,
      std::span<const reduction::detail::AnsatzColumnMeta> ansatz_metadata,
      std::size_t generated_rows, std::size_t generated_columns,
      std::size_t expansion_rounds, std::size_t expanded_groups,
      std::size_t expanded_points);
  void build_reference_evaluation_plan(
      const reduction::detail::KernelPublicationInput& input);
  void
  build_pending_block_replay(const reduction::detail::KernelPublicationInput& input,
                             const EvaluatedCoeffs<firefly::FFInt>& coeffs,
                             const std::vector<firefly::FFInt>& values);

  [[nodiscard]] std::vector<firefly::FFInt>
  evaluate_reference(const EvaluatedCoeffs<firefly::FFInt>& coeffs,
                     const std::vector<firefly::FFInt>& values) const;
  /// 针对当前的有限域特征 p，预编译稀疏装载指令流
  void compile_loaders(std::uint64_t prime);

  /// Hot replay is deliberately specialized to scalar FFInt below.
  /// FireFly bunch types are rejected by operator() before initialization.

  [[nodiscard]] std::vector<firefly::FFInt>
  execute_replay(const EvaluatedCoeffs<firefly::FFInt>& coeffs,
                 const std::vector<firefly::FFInt>& values,
                 const ReplayVariant* variant = nullptr) const;

  [[nodiscard]] std::shared_ptr<const ReplayVariant>
  selected_replay_variant(std::span<const std::uint32_t> active_outputs) const;
  [[nodiscard]] std::shared_ptr<ReplayVariant>
  build_replay_variant(std::span<const std::uint32_t> active_outputs) const;

  void select_reconstructed_outputs(std::span<const std::uint8_t> nonzero_outputs);
  void finalize_replay(const ReductionProgressCallback& progress);
  void complete_coefficient_selection(CoefficientSelection& selection) const;

  explicit BlackBoxFeynman(const Config& config, ReductionOptions options);

  [[nodiscard]] static std::unique_ptr<BlackBoxFeynman>
  prepare_impl(const Config& config, const ReductionProgressCallback& progress,
               ReductionOptions options,
               std::span<const std::uint32_t> relation_source_sectors = {});

public:
  // These low-level entry points deliberately borrow config to avoid copying
  // the compiled polynomial data. Temporaries are rejected explicitly.
  [[nodiscard]] static std::unique_ptr<BlackBoxFeynman>
  prepare(const Config& config, ReductionProgressCallback progress = {},
          ReductionOptions options = {});
  static std::unique_ptr<BlackBoxFeynman>
  prepare(Config&&, ReductionProgressCallback = {}, ReductionOptions = {}) = delete;

  // Fixed-basis preparation with relation-source sectors retained from an
  // earlier global master selection. This is used by opt-in workflows that
  // must prepare a target superset once and replay it through a later output
  // coordinate transform.
  [[nodiscard]] static std::unique_ptr<BlackBoxFeynman>
  prepare(const Config& config, std::span<const std::uint32_t> relation_source_sectors,
          ReductionProgressCallback progress, ReductionOptions options = {});
  static std::unique_ptr<BlackBoxFeynman> prepare(Config&&,
                                                  std::span<const std::uint32_t>,
                                                  ReductionProgressCallback,
                                                  ReductionOptions = {}) = delete;

  [[nodiscard]] static std::unique_ptr<BlackBoxFeynman>
  prepare(Config& config, const masters::MasterCandidateSet& candidates,
          ReductionProgressCallback progress = {}, ReductionOptions options = {});

  [[nodiscard]] const ReductionKernelStatistics& kernel_statistics() const noexcept
  {
    return kernel_statistics_;
  }

  [[nodiscard]] std::span<const std::uint32_t> reconstructed_outputs() const noexcept
  {
    return reconstructed_output_positions;
  }

  [[nodiscard]] std::size_t total_output_count() const noexcept
  {
    return cfg.targets.size() * cfg.basis.size();
  }

  /// FireFly 回调: 质数切换时调用
  void prime_changed();

  /// FireFly overlay callback: evaluate only outputs that remain active in the
  /// current prime field. The returned vector keeps the projected output shape.
  [[nodiscard]] std::vector<firefly::FFInt>
  eval_selected(const std::vector<firefly::FFInt>& values,
                const std::vector<std::uint32_t>& active_outputs);

  /// FireFly overlay callback: evaluate active outputs and return only their
  /// values, in the exact order supplied by active_outputs.
  [[nodiscard]] std::vector<firefly::FFInt>
  eval_selected_compact(const std::vector<firefly::FFInt>& values,
                        const std::vector<std::uint32_t>& active_outputs);

  /// FireFly 黑盒核心回调: 给定有限域参数值，返回需要重构的约化系数
  ///
  /// Kernel 在构造 FireFly reconstructor 前完成准备；所有调用只回放 Tape。
  /// 返回顺序由 reconstructed_outputs() 给出；完整布局在物化阶段恢复。
  template <typename T> std::vector<T> operator()(const std::vector<T>& values)
  {
    if constexpr (!std::is_same_v<T, firefly::FFInt>) {
      throw std::runtime_error("only bunch_size = 1 (FFInt) is supported");
    } else {
      if (output_support_initialized && reconstructed_output_positions.empty())
        return {};
      auto coeffs = evaluate_coefficients(values, &replay_coefficients);
      if (!output_support_initialized)
        throw std::logic_error("reduction output support is not initialized");
      return execute_replay(coeffs, values);
    }
  }
};
