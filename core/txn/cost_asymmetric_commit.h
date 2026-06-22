#pragma once
// Cost-asymmetric commit protocol.
// Resolution order: direct -> semantic rebase -> reselect -> regenerate.
// Semantic rejection (for example exhausted escrow or failed CAS) is not a
// version conflict and therefore must not trigger an expensive regeneration.
#include <algorithm>
#include <string>
#include <vector>

#include "core/branch/speculative_branch.h"
#include "core/cost/cost_model.h"
#include "core/intent/policy_dispatcher.h"
#include "core/storage/versioned_object_store.h"

namespace cast::txn {

enum class CommitStrategy {
  kStrictOCC,
  kCAST,
};

struct CommitOutcome {
  bool committed = false;
  bool rejected = false;
  std::string winner_branch_id;
  std::string action;  // direct | merge | reselect | regenerate | reject | abort
  std::string reason;
};

class CostAsymmetricCommit {
 public:
  CostAsymmetricCommit(storage::VersionedObjectStore& store, cost::CostModel model)
      : store_(store), model_(model) {}

  CommitOutcome CommitTask(std::vector<branch::SpeculativeBranch>& candidates,
                           CommitStrategy strategy, cost::CostStats* stats) {
    CommitOutcome out;
    if (candidates.empty()) {
      out.action = "abort";
      out.reason = "no candidates";
      return out;
    }
    stats->candidates_generated += candidates.size();
    stats->n_tasks += 1;

    std::size_t winner_idx = 0;
    for (std::size_t i = 1; i < candidates.size(); ++i) {
      if (candidates[i].quality > candidates[winner_idx].quality) winner_idx = i;
    }

    TryResult first = TryCommit(candidates[winner_idx], strategy);
    if (first.committed) {
      stats->n_merge += first.merges;
      out.committed = true;
      out.winner_branch_id = candidates[winner_idx].branch_id;
      out.action = first.action;
      return out;
    }

    bool has_conflict = first.failure == FailureKind::kConflict;
    std::size_t retry_idx = winner_idx;
    std::string reject_reason =
        first.failure == FailureKind::kSemanticReject ? first.reason : "";

    if (strategy == CommitStrategy::kCAST) {
      for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (i == winner_idx) continue;
        TryResult tr = TryCommit(candidates[i], strategy);
        if (tr.committed) {
          stats->n_merge += tr.merges;
          stats->n_reselect += 1;
          out.committed = true;
          out.winner_branch_id = candidates[i].branch_id;
          out.action = "reselect";
          return out;
        }
        if (!has_conflict && tr.failure == FailureKind::kConflict) {
          has_conflict = true;
          retry_idx = i;
        }
        if (tr.failure == FailureKind::kSemanticReject && reject_reason.empty()) {
          reject_reason = tr.reason;
        }
      }
    }

    // All alternatives failed a business condition. Regeneration cannot create
    // inventory or make a false CAS condition true, so reject without paying c_gen.
    if (!has_conflict) {
      out.rejected = true;
      out.action = "reject";
      out.reason = reject_reason.empty() ? "semantic condition rejected" : reject_reason;
      return out;
    }

    stats->n_regen += 1;
    std::string refresh_reason;
    if (!RefreshBaseline(candidates[retry_idx], &refresh_reason)) {
      out.rejected = true;
      out.action = "reject";
      out.reason = refresh_reason;
      return out;
    }

    TryResult retried = TryCommit(candidates[retry_idx], strategy);
    stats->n_merge += retried.merges;
    out.committed = retried.committed;
    out.rejected = retried.failure == FailureKind::kSemanticReject;
    out.winner_branch_id = candidates[retry_idx].branch_id;
    out.action = out.rejected ? "reject" : "regenerate";
    if (!retried.committed) {
      out.reason = retried.reason.empty() ? "regenerate still failed" : retried.reason;
    }
    return out;
  }

 private:
  enum class FailureKind { kNone, kConflict, kSemanticReject };

  struct TryResult {
    bool committed = false;
    std::size_t merges = 0;
    std::string action;
    FailureKind failure = FailureKind::kNone;
    std::string reason;
  };

  TryResult TryCommit(const branch::SpeculativeBranch& b, CommitStrategy strategy) {
    std::vector<storage::VersionCheck> checks;
    std::vector<storage::WriteOp> writes;
    std::size_t merges = 0;
    bool used_rebase = false;

    for (const auto& w : b.writes) {
      const auto cur = store_.Get(w.object_id);
      const auto cls = intent::PolicyDispatcher::Classify(w.intent);

      if (cls == intent::PolicyDispatcher::ConcurrencyClass::kReadOnly) continue;

      if (cur.version == w.base_version) {
        // CAS and constrained DELTA are business predicates, not merely version
        // checks. Re-evaluate them even on the fast/direct path.
        if (cls == intent::PolicyDispatcher::ConcurrencyClass::kConditionalRebase ||
            cls == intent::PolicyDispatcher::ConcurrencyClass::kConstrainedCommutative) {
          const auto rr = intent::PolicyDispatcher::ResolveWrite(
              w.base_value, w.branch_value, w.intent, cur.value);
          if (!rr.success) {
            return TryResult{false, 0, "", FailureKind::kSemanticReject, rr.reason};
          }
          if (rr.should_write) {
            checks.push_back({w.object_id, cur.version});
            writes.push_back({w.object_id, rr.value});
          }
        } else {
          checks.push_back({w.object_id, w.base_version});
          writes.push_back({w.object_id, w.branch_value});
        }
        continue;
      }

      const bool rebindable =
          strategy == CommitStrategy::kCAST &&
          (cls == intent::PolicyDispatcher::ConcurrencyClass::kCommutativeRebase ||
           cls == intent::PolicyDispatcher::ConcurrencyClass::kConstrainedCommutative ||
           cls == intent::PolicyDispatcher::ConcurrencyClass::kConditionalRebase);
      if (!rebindable) {
        return TryResult{false, 0, "", FailureKind::kConflict, "version conflict"};
      }

      const auto rr = intent::PolicyDispatcher::ResolveWrite(
          w.base_value, w.branch_value, w.intent, cur.value);
      if (!rr.success) {
        return TryResult{false, 0, "", FailureKind::kSemanticReject, rr.reason};
      }
      if (rr.should_write) {
        checks.push_back({w.object_id, cur.version});
        writes.push_back({w.object_id, rr.value});
      }
      used_rebase = true;
      if (cls == intent::PolicyDispatcher::ConcurrencyClass::kCommutativeRebase ||
          cls == intent::PolicyDispatcher::ConcurrencyClass::kConstrainedCommutative) {
        ++merges;
      }
    }

    if (store_.BatchPutIfVersion(checks, writes)) {
      return TryResult{true, merges, used_rebase ? "merge" : "direct",
                       FailureKind::kNone, ""};
    }
    return TryResult{false, 0, "", FailureKind::kConflict,
                     "atomic version check failed"};
  }

  bool RefreshBaseline(branch::SpeculativeBranch& b, std::string* reason) {
    for (auto& w : b.writes) {
      const auto cur = store_.Get(w.object_id);
      const auto cls = intent::PolicyDispatcher::Classify(w.intent);

      // Regenerating an ordered APPEND means replaying its declared payload on
      // the newest value. OVERWRITE keeps its generated target value.
      const bool recompute =
          w.intent.intent_type == intent::IntentType::kAppend ||
          cls == intent::PolicyDispatcher::ConcurrencyClass::kCommutativeRebase ||
          cls == intent::PolicyDispatcher::ConcurrencyClass::kConstrainedCommutative ||
          cls == intent::PolicyDispatcher::ConcurrencyClass::kConditionalRebase;
      if (recompute) {
        const auto rr = intent::PolicyDispatcher::ResolveWrite(
            w.base_value, w.branch_value, w.intent, cur.value);
        if (!rr.success) {
          if (reason) *reason = rr.reason;
          return false;
        }
        if (rr.should_write) w.branch_value = rr.value;
      }
      w.base_value = cur.value;
      w.base_version = cur.version;
    }
    return true;
  }

  storage::VersionedObjectStore& store_;
  cost::CostModel model_;
};

}  // namespace cast::txn
