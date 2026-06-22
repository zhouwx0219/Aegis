#pragma once
// Escrow 额度账户（O'Neil 1986）：把"带下界约束的扣减"表达为额度预留，
// 从而把"超卖边界"转成"可并发收益"。
//
// 语义（与 agent/experiments/escrow_experiment.py 的 run_escrow 对齐，现已下沉至此 C++ 内核）：
//   - 预留可交换、互不阻塞：只要 剩余 - q >= 下界，就授予并扣减，永不 abort/重跑；
//   - 任何使 剩余 < 下界 的预留被拒绝（不改状态）⟹ 不超卖（约束保持）。
// 这是 CAST/HYBRID 的"约束可交换"并发类（kConstrainedCommutative）的状态载体。
#include <cstdint>

namespace cast::concurrency {

class EscrowAccount {
 public:
  EscrowAccount() = default;
  explicit EscrowAccount(long long capacity, long long lower_bound = 0)
      : remaining_(capacity), lower_bound_(lower_bound) {}

  // 预留 q（q>0 表示扣减）：剩余 - q >= 下界 则授予(扣减)返回 true；否则拒绝(状态不变)返回 false。
  bool Reserve(long long q) {
    if (remaining_ - q >= lower_bound_) {
      remaining_ -= q;
      ++granted_;
      return true;
    }
    ++rejected_;
    return false;
  }

  // 释放（补偿/退预留）：无条件加回，可交换。
  void Release(long long q) { remaining_ += q; }

  long long remaining() const { return remaining_; }
  long long lower_bound() const { return lower_bound_; }
  long long granted() const { return granted_; }
  long long rejected() const { return rejected_; }
  // 该账户是否破过下界（正确性自检：escrow 下应恒为 false）。
  bool oversold() const { return remaining_ < lower_bound_; }

 private:
  long long remaining_ = 0;
  long long lower_bound_ = 0;
  long long granted_ = 0;
  long long rejected_ = 0;
};

}  // namespace cast::concurrency
