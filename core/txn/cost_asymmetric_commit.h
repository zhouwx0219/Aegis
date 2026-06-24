#pragma once

#include <algorithm>
#include <numeric>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/branch/speculative_branch.h"
#include "core/cost/cost_model.h"
#include "core/storage/versioned_kv.h"
#include "core/txn/commit_protocol.h"

namespace cast::txn {

class CostAsymmetricCommit final : public CommitProtocol {
 public:
  CostAsymmetricCommit(storage::VersionedKVStore& store, cost::CostModel model)
      : store_(store) {
    (void)model;
  }

  const char* Name() const override { return "cost-asymmetric"; }
  const char* Family() const override { return "commit_protocol"; }
  const char* Description() const override {
    return "Cost-aware multi-branch commit with semantic merge, reselect, and regeneration";
  }

  CommitOutcome CommitTask(
      std::vector<branch::SpeculativeBranch>& candidates,
      const concurrency::ConcurrencyControl& cc,
      cost::CostStats* stats) override {
    cost::CostStats local_stats;
    if (stats == nullptr) stats = &local_stats;
    CommitOutcome outcome;
    if (candidates.empty()) {
      outcome.action = "abort";
      outcome.reason = "no candidates";
      return outcome;
    }
    stats->candidates_generated += candidates.size();
    stats->n_tasks += 1;

    std::vector<std::size_t> order(candidates.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t left,
                                                     std::size_t right) {
      return candidates[left].quality > candidates[right].quality;
    });

    bool has_conflict = false;
    std::size_t conflict_index = order.front();
    std::string rejection_reason;
    for (std::size_t position = 0; position < order.size(); ++position) {
      const std::size_t index = order[position];
      TryResult result = TryCommit(candidates[index], cc);
      if (result.committed) {
        stats->n_merge += result.merges;
        if (position > 0) ++stats->n_reselect;
        outcome.committed = true;
        outcome.winner_branch_id = candidates[index].branch_id;
        outcome.action = position > 0 ? "reselect" : result.action;
        return outcome;
      }
      if (result.failure == FailureKind::kConflict && !has_conflict) {
        has_conflict = true;
        conflict_index = index;
        outcome.conflict_object_ids = result.conflict_object_ids;
      }
      if (result.failure == FailureKind::kSemanticReject &&
          rejection_reason.empty()) {
        rejection_reason = result.reason;
      }
    }

    if (!has_conflict) {
      outcome.rejected = true;
      outcome.action = "reject";
      outcome.reason = rejection_reason.empty() ? "semantic condition rejected"
                                                 : rejection_reason;
      return outcome;
    }

    outcome.needs_regeneration = true;
    outcome.winner_branch_id = candidates[conflict_index].branch_id;
    outcome.action = "regenerate_required";
    outcome.reason = std::string(cc.Name()) +
                     ": conflict requires a newly generated candidate";
    return outcome;
  }

 private:
  enum class FailureKind { kNone, kConflict, kSemanticReject };

  struct TryResult {
    bool committed = false;
    std::size_t merges = 0;
    std::string action;
    FailureKind failure = FailureKind::kNone;
    std::string reason;
    std::vector<std::string> conflict_object_ids;
  };

  TryResult TryCommit(const branch::SpeculativeBranch& candidate,
                      const concurrency::ConcurrencyControl& cc) {
    std::vector<storage::VersionCheck> checks;
    std::vector<storage::WriteOp> writes;
    std::unordered_set<std::string> read_targets;
    std::unordered_set<std::string> write_targets;
    std::size_t merges = 0;
    bool rebased = false;

    for (const auto& read : candidate.read_set) {
      if (read.object_id.empty()) {
        return {false, 0, "", FailureKind::kSemanticReject,
                "candidate contains empty read target", {}};
      }
      if (!read_targets.insert(read.object_id).second) continue;
      checks.push_back({read.object_id, read.version});
    }

    for (const auto& write : candidate.writes) {
      if (!write_targets.insert(write.object_id).second) {
        return {false, 0, "", FailureKind::kSemanticReject,
                "candidate contains duplicate write target: " + write.object_id,
                {}};
      }
      const auto current = store_.Get(write.object_id);
      const auto resolved = cc.Resolve(write, current);
      switch (resolved.decision) {
        case concurrency::CCDecision::kSkip:
          continue;
        case concurrency::CCDecision::kConflict:
          return {false, 0, "", FailureKind::kConflict, resolved.reason,
                  {write.object_id}};
        case concurrency::CCDecision::kReject:
          return {false, 0, "", FailureKind::kSemanticReject, resolved.reason,
                  {}};
        case concurrency::CCDecision::kApply:
          checks.push_back({write.object_id, current.version});
          writes.push_back({write.object_id, resolved.value});
          rebased = rebased || resolved.rebased;
          if (resolved.merged) ++merges;
          break;
      }
    }

    if (store_.BatchPutIfVersion(checks, writes)) {
      return {true, merges, rebased ? "merge" : "direct", FailureKind::kNone,
              "", {}};
    }
    return {false, 0, "", FailureKind::kConflict,
            "atomic version check failed", ConflictTargets(checks)};
  }

  std::vector<std::string> ConflictTargets(
      const std::vector<storage::VersionCheck>& checks) const {
    std::vector<std::string> targets;
    std::unordered_set<std::string> seen;
    for (const auto& check : checks) {
      if (store_.GetVersion(check.key) == check.expected_version) continue;
      if (seen.insert(check.key).second) targets.push_back(check.key);
    }
    return targets;
  }

  storage::VersionedKVStore& store_;
};

}  // namespace cast::txn
