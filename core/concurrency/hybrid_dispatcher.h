#pragma once
// 混合并发控制内核：per-object / per-intent 自适应 CC 分发器（Paper B 支柱二，现下沉至 C++）。
//
// 对一个异构对象群(strict 行 / 无约束可交换计数器 / 带下界约束库存 / 只读)按对象意图类路由：
//   READ        -> 快照放行(SI，不计冲突)
//   COMM_FREE   -> 语义合并(CRDT rebase，永不 abort)
//   COMM_CONSTR -> escrow 额度预留(可交换但守下界 stock>=0)
//   STRICT      -> OCC 版本校验(strict-strict 才判真冲突)
// 验证层(would_abort)按策略放宽冲突判定；提交层(apply)按类做无成本解析。
// 与 agent/experiments/hybrid_cc_adaptive.py 同口径，原 Python 的 apply_write/aborts/commit_one
// 逻辑整体移植到此内核；Python 仅保留 agent 运行时(线程/生成/sleep 代表 c_gen/重试编排)。
//
// 线程安全：每个提交方法持内部 mutex（提交临界区，对应原 Python 的 store_lock）。
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace cast::concurrency {

enum class CCClass { kRead, kCommFree, kCommConstr, kStrict };

// 统一单协议 vs 自适应混合，作为同框架公平对比的策略集合。
enum class HybridPolicy { kOCC, kMVCC, k2PL, kMergeAll, kHybrid };

// 一个候选：读集 / 写集(对象 id，按 catalog 决定其并发类) + 生成时拍下的版本基线。
struct AdaptiveCandidate {
  std::vector<std::string> reads;
  std::vector<std::string> writes;
  std::unordered_map<std::string, std::uint64_t> base;  // 生成时版本快照
};

struct DispatchStats {
  long long committed = 0;
  long long reselect = 0;       // 提交了非首选候选(零生成成本)
  long long regen = 0;          // 重新生成(昂贵，由 agent 运行时驱动 commit_regen)
  long long merge = 0;          // 可交换合并(commfree 累加 / merge-all 盲并)
  long long escrow_grant = 0;   // escrow 授予(HYBRID 带约束扣减成功)
  long long escrow_reject = 0;  // escrow 拒绝(缺货，正确拒绝：非超卖、非浪费)
  long long oversell = 0;       // 破下界次数(stock<0)：仅 merge-all 盲并会发生
};

enum class TaskOutcome { kDirect, kReselect, kAllAborted };

class HybridDispatcher {
 public:
  explicit HybridDispatcher(HybridPolicy policy) : policy_(policy) {}

  // —— catalog / 初始化 ——（建池阶段单线程调用）
  void init_strict(const std::string& o) { put(o, CCClass::kStrict, 0); }
  void init_counter(const std::string& o) { put(o, CCClass::kCommFree, 0); }
  void init_stock(const std::string& o, long long s0) { put(o, CCClass::kCommConstr, s0); }
  void init_read(const std::string& o) { put(o, CCClass::kRead, 0); }

  // 取若干对象的当前版本（候选生成时拍快照用）。
  std::unordered_map<std::string, std::uint64_t> snapshot(const std::vector<std::string>& objs) {
    std::lock_guard<std::mutex> lk(mu_);
    std::unordered_map<std::string, std::uint64_t> m;
    m.reserve(objs.size());
    for (const auto& o : objs) m[o] = ver_[o];
    return m;
  }

  // 乐观策略的整任务提交：按序尝试候选，首个不冲突者提交(winner/reselect)；全冲突返回 kAllAborted。
  // 整个候选循环在一把锁内原子完成（对应原 Python 的单个 store_lock 临界区）。
  TaskOutcome commit_task(const std::vector<AdaptiveCandidate>& cands) {
    std::lock_guard<std::mutex> lk(mu_);
    for (std::size_t i = 0; i < cands.size(); ++i) {
      if (!would_abort(cands[i])) {
        apply(cands[i]);
        ++stats_.committed;
        if (i > 0) ++stats_.reselect;
        return i == 0 ? TaskOutcome::kDirect : TaskOutcome::kReselect;
      }
    }
    return TaskOutcome::kAllAborted;
  }

  // 重生成提交：agent 运行时已重读最新基线(故不冲突)，强制提交首选候选。
  void commit_regen(const AdaptiveCandidate& c) {
    std::lock_guard<std::mutex> lk(mu_);
    apply(c);
    ++stats_.committed;
    ++stats_.regen;
  }

  // 2PL 提交：调用方已持对象锁(故无并发冲突)，无 abort 直接提交。
  void commit_2pl(const AdaptiveCandidate& c) {
    std::lock_guard<std::mutex> lk(mu_);
    apply(c);
    ++stats_.committed;
  }

  DispatchStats stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    return stats_;
  }
  long long value_of(const std::string& o) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = ival_.find(o);
    return it == ival_.end() ? 0 : it->second;
  }

 private:
  void put(const std::string& o, CCClass c, long long v) {
    cls_[o] = c;
    ival_[o] = v;
    ver_[o] = 0;
  }

  bool changed(const AdaptiveCandidate& c, const std::string& o) const {
    auto it = c.base.find(o);
    if (it == c.base.end()) return false;  // 基线覆盖候选所有读写对象，缺失视作未变
    auto vit = ver_.find(o);
    return vit != ver_.end() && vit->second != it->second;
  }

  // 验证层：按策略判该候选是否冲突(需 reselect/regen)。
  bool would_abort(const AdaptiveCandidate& c) const {
    switch (policy_) {
      case HybridPolicy::kOCC:  // 读集+写集全严格
        for (const auto& o : c.reads) if (changed(c, o)) return true;
        for (const auto& o : c.writes) if (changed(c, o)) return true;
        return false;
      case HybridPolicy::kMVCC:  // 读放行(快照)，写写仍冲突
        for (const auto& o : c.writes) if (changed(c, o)) return true;
        return false;
      case HybridPolicy::kMergeAll:
      case HybridPolicy::kHybrid:  // 读放行 + 可交换写放行，仅 strict-strict
        for (const auto& o : c.writes)
          if (clsOf(o) == CCClass::kStrict && changed(c, o)) return true;
        return false;
      case HybridPolicy::k2PL:  // 锁内提交，无 abort（实际走 commit_2pl）
        return false;
    }
    return false;
  }

  CCClass clsOf(const std::string& o) const {
    auto it = cls_.find(o);
    return it == cls_.end() ? CCClass::kStrict : it->second;
  }

  // 提交层：按对象并发类做无成本解析并落库；记 merge/escrow/oversell。
  void apply(const AdaptiveCandidate& c) {
    for (const auto& o : c.writes) {
      switch (clsOf(o)) {
        case CCClass::kStrict:  // 覆盖写：值无关紧要，仅升版本
          ++ival_[o];
          ++ver_[o];
          break;
        case CCClass::kCommFree:  // 无约束可交换：累加合并，永远安全
          ++ival_[o];
          ++ver_[o];
          ++stats_.merge;
          break;
        case CCClass::kCommConstr:  // 带下界约束的扣减
          if (policy_ == HybridPolicy::kMergeAll) {
            long long nv = ival_[o] - 1;  // 盲并：不查下界 -> 可超卖
            if (nv < 0) ++stats_.oversell;
            ival_[o] = nv;
            ++ver_[o];
            ++stats_.merge;
          } else {
            if (ival_[o] > 0) {  // escrow 守界：剩余>0 才扣
              --ival_[o];
              ++ver_[o];
              if (policy_ == HybridPolicy::kHybrid) ++stats_.escrow_grant;
            } else {  // 缺货：正确拒绝(不写、不超卖、不重跑)
              if (policy_ == HybridPolicy::kHybrid) ++stats_.escrow_reject;
            }
          }
          break;
        case CCClass::kRead:  // 只读：不落库
          break;
      }
    }
  }

  HybridPolicy policy_;
  mutable std::mutex mu_;
  std::unordered_map<std::string, CCClass> cls_;
  std::unordered_map<std::string, long long> ival_;
  std::unordered_map<std::string, std::uint64_t> ver_;
  DispatchStats stats_;
};

}  // namespace cast::concurrency
