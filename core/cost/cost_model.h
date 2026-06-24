#pragma once

#include <cstddef>

namespace cast::cost {

// Unit costs for agent planning. c_gen models one candidate generation
// (for example, an LLM/tool call); c_merge models one semantic rebase.
struct CostModel {
  double c_gen = 1.0;
  double c_merge = 0.01;
};

struct CostStats {
  std::size_t n_tasks = 0;
  std::size_t candidates_generated = 0;
  std::size_t n_merge = 0;
  std::size_t n_reselect = 0;
  std::size_t n_regen = 0;

  double WastedCompute(const CostModel& m) const {
    const std::size_t losers =
        candidates_generated > n_tasks ? candidates_generated - n_tasks : 0;
    return static_cast<double>(losers) * m.c_gen +
           static_cast<double>(n_regen) * m.c_gen +
           static_cast<double>(n_merge) * m.c_merge;
  }

  double TotalCompute(const CostModel& m) const {
    return static_cast<double>(candidates_generated + n_regen) * m.c_gen +
           static_cast<double>(n_merge) * m.c_merge;
  }
};

}  // namespace cast::cost
