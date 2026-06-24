#pragma once

#include <string>
#include <vector>

#include "core/branch/speculative_branch.h"
#include "core/concurrency/concurrency_control.h"
#include "core/cost/cost_model.h"

namespace cast::txn {

struct CommitOutcome {
  bool committed = false;
  bool rejected = false;
  bool needs_regeneration = false;
  std::string winner_branch_id;
  std::string action;
  std::string reason;
  std::vector<std::string> conflict_object_ids;
};

class CommitProtocol {
 public:
  virtual ~CommitProtocol() = default;
  virtual const char* Name() const = 0;
  virtual const char* Family() const { return "commit_protocol"; }
  virtual const char* Description() const { return ""; }
  virtual CommitOutcome CommitTask(
      std::vector<branch::SpeculativeBranch>& candidates,
      const concurrency::ConcurrencyControl& cc,
      cost::CostStats* stats) = 0;
};

}  // namespace cast::txn
