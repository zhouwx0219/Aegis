#pragma once
// 成本模型与统计：CAST 的优化目标是"最小化为提交一个合格 winner 所浪费的算力"。
// 与传统并发控制(最大化吞吐/最小化延迟)的目标不同。
#include <cstddef>

namespace cast::cost {

// 单位成本。c_gen 为一次候选生成(LLM+工具)的成本；c_merge 为一次语义 rebase(仅 KV 操作)的成本。
// 关键假设：c_merge << c_gen（agent 场景下二者差 4~5 个数量级）。
struct CostModel {
  double c_gen = 1.0;
  double c_merge = 0.01;
};

struct CostStats {
  std::size_t n_tasks = 0;               // 已处理任务数（每次 CommitTask +1）
  std::size_t candidates_generated = 0;  // 累计生成的候选数
  std::size_t n_merge = 0;               // 语义合并次数（省下昂贵重跑）
  std::size_t n_reselect = 0;            // 重选已有候选次数（零生成成本）
  std::size_t n_regen = 0;               // 重新生成次数（昂贵）

  // 浪费算力 = loser 候选 + 重生成 + 合并开销。
  // 理想下界 = 生成 1 个候选直接提交（不计入浪费）。
  double WastedCompute(const CostModel& m) const {
    const std::size_t losers = candidates_generated > n_tasks ? candidates_generated - n_tasks : 0;
    return static_cast<double>(losers) * m.c_gen +
           static_cast<double>(n_regen) * m.c_gen +
           static_cast<double>(n_merge) * m.c_merge;
  }

  // 总算力 = 所有生成 + 所有重生成 + 所有合并。
  double TotalCompute(const CostModel& m) const {
    return static_cast<double>(candidates_generated + n_regen) * m.c_gen +
           static_cast<double>(n_merge) * m.c_merge;
  }
};

}  // namespace cast::cost
