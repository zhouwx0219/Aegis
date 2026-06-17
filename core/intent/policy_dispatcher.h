#pragma once
// 并发策略分发器：把写意图分类成并发类，并在"最新存储值"上做语义重绑定(rebase)。
// 移植自现有 data_agent_system/intent/policy_dispatcher.h，是混合并发与成本不对称提交的核心复用件。
#include <charconv>
#include <optional>
#include <string>

#include "core/intent/intent.h"

namespace cast::intent {

class PolicyDispatcher {
 public:
  enum class ConcurrencyClass {
    kReadOnly,
    kStrict,             // OVERWRITE：必须版本未变才提交
    kCommutativeRebase,  // APPEND/DELTA：可在最新值上交换式重绑定
    kConditionalRebase,  // CAS：条件仍成立才提交
  };

  struct ResolveResult {
    bool success = true;
    bool should_write = false;
    std::string value;
    std::string reason;
  };

  static ConcurrencyClass Classify(IntentType t) {
    switch (t) {
      case IntentType::kRead: return ConcurrencyClass::kReadOnly;
      case IntentType::kOverwrite: return ConcurrencyClass::kStrict;
      case IntentType::kAppend:
      case IntentType::kDelta: return ConcurrencyClass::kCommutativeRebase;
      case IntentType::kCas: return ConcurrencyClass::kConditionalRebase;
    }
    return ConcurrencyClass::kStrict;
  }

  static const char* ConcurrencyClassName(ConcurrencyClass c) {
    switch (c) {
      case ConcurrencyClass::kReadOnly: return "read_only";
      case ConcurrencyClass::kStrict: return "strict";
      case ConcurrencyClass::kCommutativeRebase: return "commutative_rebase";
      case ConcurrencyClass::kConditionalRebase: return "conditional_rebase";
    }
    return "strict";
  }

  // 在最新存储值 current_store_value 上重绑定一次写：
  //   base_value   = 分支读到的基线值
  //   branch_value = 分支缓冲的新值
  // 成功则给出可直接写入 store 的最终 value。
  static ResolveResult ResolveWrite(const std::string& base_value,
                                    const std::string& branch_value,
                                    const WriteIntent& intent,
                                    const std::string& current_store_value) {
    ResolveResult r;
    switch (intent.intent_type) {
      case IntentType::kRead:
        r.should_write = false;
        return r;
      case IntentType::kOverwrite:
        r.should_write = true;
        r.value = branch_value;
        return r;
      case IntentType::kAppend: {
        r.should_write = true;
        // 把分支相对基线追加的增量拼接到最新值上。
        if (branch_value.size() >= base_value.size() &&
            branch_value.compare(0, base_value.size(), base_value) == 0) {
          r.value = current_store_value + branch_value.substr(base_value.size());
        } else {
          r.value = current_store_value + intent.payload;
        }
        return r;
      }
      case IntentType::kDelta: {
        r.should_write = true;
        const auto cur = ParseInt(current_store_value);
        const auto base = ParseInt(base_value);
        const auto bv = ParseInt(branch_value);
        if (cur && base && bv) {
          r.value = std::to_string(*cur + (*bv - *base));  // 在最新值上重算增量
          return r;
        }
        const auto delta = ParseInt(intent.payload);
        if (!cur || !delta) {
          r.success = false;
          r.reason = "delta rebase requires integer values";
          return r;
        }
        r.value = std::to_string(*cur + *delta);
        return r;
      }
      case IntentType::kCas:
        if (intent.condition.type == ConditionType::kValueEquals &&
            current_store_value != intent.condition.expected_value) {
          r.success = false;
          r.reason = "cas condition no longer holds";
          return r;
        }
        r.should_write = true;
        r.value = branch_value;
        return r;
    }
    r.success = false;
    r.reason = "unknown intent type";
    return r;
  }

 private:
  static std::optional<long long> ParseInt(const std::string& s) {
    long long v = 0;
    const auto* b = s.data();
    const auto* e = s.data() + s.size();
    const auto res = std::from_chars(b, e, v);
    if (res.ec != std::errc() || res.ptr != e) return std::nullopt;
    return v;
  }
};

}  // namespace cast::intent
