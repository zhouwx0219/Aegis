#pragma once

#include <mutex>
#include <stdexcept>

namespace cast::concurrency {

class EscrowAccount {
 public:
  EscrowAccount() = default;
  explicit EscrowAccount(long long capacity, long long lower_bound = 0)
      : remaining_(capacity), lower_bound_(lower_bound) {
    if (capacity < lower_bound) {
      throw std::invalid_argument("capacity must be >= lower_bound");
    }
  }

  bool Reserve(long long quantity) {
    if (quantity <= 0) {
      throw std::invalid_argument("reservation quantity must be positive");
    }
    std::lock_guard<std::mutex> lock(mu_);
    if (remaining_ - quantity >= lower_bound_) {
      remaining_ -= quantity;
      ++granted_;
      return true;
    }
    ++rejected_;
    return false;
  }

  void Release(long long quantity) {
    if (quantity <= 0) {
      throw std::invalid_argument("release quantity must be positive");
    }
    std::lock_guard<std::mutex> lock(mu_);
    remaining_ += quantity;
  }

  long long remaining() const {
    std::lock_guard<std::mutex> lock(mu_);
    return remaining_;
  }
  long long lower_bound() const { return lower_bound_; }
  long long granted() const {
    std::lock_guard<std::mutex> lock(mu_);
    return granted_;
  }
  long long rejected() const {
    std::lock_guard<std::mutex> lock(mu_);
    return rejected_;
  }
  bool oversold() const {
    std::lock_guard<std::mutex> lock(mu_);
    return remaining_ < lower_bound_;
  }

 private:
  mutable std::mutex mu_;
  long long remaining_ = 0;
  long long lower_bound_ = 0;
  long long granted_ = 0;
  long long rejected_ = 0;
};

}  // namespace cast::concurrency
