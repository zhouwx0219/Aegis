#pragma once
// 成本不对称提交协议（CAST 的核心）。
// 与 strict OCC 的根本区别：遇到提交冲突时不默认 abort→重跑(昂贵)，
// 而是优先用语义可合并性 rebase 合并(merge)或改提交已有候选(reselect)，
// 把昂贵的 regenerate 压到最后手段。目标：最小化浪费的算力。
#include <algorithm>
#include <string>
#include <vector>

#include "core/branch/speculative_branch.h"
#include "core/cost/cost_model.h"
#include "core/intent/policy_dispatcher.h"
#include "core/storage/versioned_object_store.h"

namespace cast::txn {

enum class CommitStrategy {
  kStrictOCC,  // baseline：任何版本冲突 → abort → regenerate
  kCAST,       // ours：commutative/conditional 冲突先 rebase 合并；再 reselect；最后才 regenerate
};

struct CommitOutcome {
  bool committed = false;
  std::string winner_branch_id;
  std::string action;  // "direct" | "merge" | "reselect" | "regenerate" | "abort"
  std::string reason;
};

class CostAsymmetricCommit {
 public:
  CostAsymmetricCommit(storage::VersionedObjectStore& store, cost::CostModel model)
      : store_(store), model_(model) {}

  // 提交一个任务：从若干候选中选 winner 提交；按 strategy 处理冲突；统计成本。
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

    // winner = 质量最高
    std::size_t winner_idx = 0;
    for (std::size_t i = 1; i < candidates.size(); ++i) {
      if (candidates[i].quality > candidates[winner_idx].quality) winner_idx = i;
    }

    // 第一轮尝试提交 winner
    TryResult tr = TryCommit(candidates[winner_idx], strategy);
    if (tr.committed) {
      stats->n_merge += tr.merges;
      out.committed = true;
      out.winner_branch_id = candidates[winner_idx].branch_id;
      out.action = tr.action;
      return out;
    }

    // CAST：先试 reselect 其他已生成候选（零额外生成成本）
    if (strategy == CommitStrategy::kCAST) {
      for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (i == winner_idx) continue;
        TryResult tr2 = TryCommit(candidates[i], strategy);
        if (tr2.committed) {
          stats->n_merge += tr2.merges;
          stats->n_reselect += 1;
          out.committed = true;
          out.winner_branch_id = candidates[i].branch_id;
          out.action = "reselect";
          return out;
        }
      }
    }

    // 最后手段：regenerate（重读最新基线后重试，必成功）——这是唯一花 c_gen 的路径
    stats->n_regen += 1;
    RefreshBaseline(candidates[winner_idx]);
    TryResult tr3 = TryCommit(candidates[winner_idx], strategy);
    stats->n_merge += tr3.merges;
    out.committed = tr3.committed;
    out.winner_branch_id = candidates[winner_idx].branch_id;
    out.action = "regenerate";
    if (!tr3.committed) out.reason = "regenerate still failed";
    return out;
  }

 private:
  struct TryResult {
    bool committed = false;
    std::size_t merges = 0;
    std::string action;  // "direct" | "merge"
  };

  // 尝试把一个候选原子提交。CAST 对 commutative/conditional 冲突做 rebase；
  // strict 冲突或 OCC 策略下任何版本变化都视作冲突。
  TryResult TryCommit(const branch::SpeculativeBranch& b, CommitStrategy strategy) {
    std::vector<storage::VersionCheck> checks;
    std::vector<storage::WriteOp> writes;
    std::size_t merges = 0;
    bool used_rebase = false;

    for (const auto& w : b.writes) {
      const auto cur = store_.Get(w.object_id);
      if (cur.version == w.base_version) {
        // 版本未变：CAS 仍需校验条件（避免 regenerate 强行对齐版本后绕过条件 -> 重复占用）
        if (w.intent.intent_type == intent::IntentType::kCas &&
            w.intent.condition.type == intent::ConditionType::kValueEquals &&
            cur.value != w.intent.condition.expected_value) {
          return TryResult{false, 0, ""};
        }
        // 否则直接用分支缓冲值
        checks.push_back({w.object_id, w.base_version});
        writes.push_back({w.object_id, w.branch_value});
        continue;
      }
      // 版本已变：是否可语义重绑定？
      const auto cls = intent::PolicyDispatcher::Classify(w.intent.intent_type);
      const bool rebindable =
          strategy == CommitStrategy::kCAST &&
          (cls == intent::PolicyDispatcher::ConcurrencyClass::kCommutativeRebase ||
           cls == intent::PolicyDispatcher::ConcurrencyClass::kConditionalRebase);
      if (!rebindable) {
        return TryResult{false, 0, ""};  // strict 冲突或 OCC：整体冲突
      }
      const auto rr = intent::PolicyDispatcher::ResolveWrite(w.base_value, w.branch_value,
                                                             w.intent, cur.value);
      if (!rr.success) {
        return TryResult{false, 0, ""};  // 如 CAS 条件不再成立
      }
      checks.push_back({w.object_id, cur.version});
      writes.push_back({w.object_id, rr.value});
      used_rebase = true;
      if (cls == intent::PolicyDispatcher::ConcurrencyClass::kCommutativeRebase) merges += 1;
    }

    if (store_.BatchPutIfVersion(checks, writes)) {
      return TryResult{true, merges, used_rebase ? "merge" : "direct"};
    }
    return TryResult{false, 0, ""};
  }

  // 模拟"基于最新状态重新生成候选"：把基线对齐到当前 store，并把目标值重绑定到最新值。
  void RefreshBaseline(branch::SpeculativeBranch& b) {
    for (auto& w : b.writes) {
      const auto cur = store_.Get(w.object_id);
      const auto rr = intent::PolicyDispatcher::ResolveWrite(w.base_value, w.branch_value,
                                                             w.intent, cur.value);
      w.base_value = cur.value;
      w.base_version = cur.version;
      if (rr.success) w.branch_value = rr.value;
    }
  }

  storage::VersionedObjectStore& store_;
  cost::CostModel model_;
};

}  // namespace cast::txn
