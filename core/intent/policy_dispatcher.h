#pragma once

#include <charconv>
#include <optional>
#include <string>

#include "core/intent/intent.h"

namespace cast::intent {

class PolicyDispatcher {
 public:
  enum class ConcurrencyClass {
    kReadOnly,
    kStrict,
    kCommutativeRebase,
    kConstrainedCommutative,
    kConditionalRebase,
  };

  struct ResolveResult {
    bool success = true;
    bool should_write = false;
    std::string value;
    std::string reason;
  };

  static ConcurrencyClass Classify(IntentType type) {
    switch (type) {
      case IntentType::kRead:
        return ConcurrencyClass::kReadOnly;
      case IntentType::kOverwrite:
        return ConcurrencyClass::kStrict;
      case IntentType::kAppend:
        return ConcurrencyClass::kStrict;
      case IntentType::kDelta:
        return ConcurrencyClass::kCommutativeRebase;
      case IntentType::kCas:
        return ConcurrencyClass::kConditionalRebase;
    }
    return ConcurrencyClass::kStrict;
  }

  static ConcurrencyClass Classify(const WriteIntent& intent) {
    if (intent.intent_type == IntentType::kAppend && intent.commutative) {
      return ConcurrencyClass::kCommutativeRebase;
    }
    if (intent.intent_type == IntentType::kDelta && intent.constrained) {
      return ConcurrencyClass::kConstrainedCommutative;
    }
    return Classify(intent.intent_type);
  }

  static const char* ConcurrencyClassName(ConcurrencyClass concurrency_class) {
    switch (concurrency_class) {
      case ConcurrencyClass::kReadOnly:
        return "read_only";
      case ConcurrencyClass::kStrict:
        return "strict";
      case ConcurrencyClass::kCommutativeRebase:
        return "commutative_rebase";
      case ConcurrencyClass::kConstrainedCommutative:
        return "constrained_commutative_escrow";
      case ConcurrencyClass::kConditionalRebase:
        return "conditional_rebase";
    }
    return "strict";
  }

  static ResolveResult ResolveWrite(const std::string& base_value,
                                    const std::string& branch_value,
                                    const WriteIntent& intent,
                                    const std::string& current_store_value) {
    ResolveResult result;
    switch (intent.intent_type) {
      case IntentType::kRead:
        result.should_write = false;
        return result;
      case IntentType::kOverwrite:
        result.should_write = true;
        result.value = branch_value;
        return result;
      case IntentType::kAppend: {
        result.should_write = true;
        if (branch_value.size() >= base_value.size() &&
            branch_value.compare(0, base_value.size(), base_value) == 0) {
          result.value = current_store_value + branch_value.substr(base_value.size());
        } else {
          result.value = current_store_value + intent.payload;
        }
        return result;
      }
      case IntentType::kDelta: {
        const auto current = ParseInt(current_store_value);
        const auto base = ParseInt(base_value);
        const auto branch = ParseInt(branch_value);
        long long new_value = 0;
        if (current && base && branch) {
          new_value = *current + (*branch - *base);
        } else {
          const auto delta = ParseInt(intent.payload);
          if (!current || !delta) {
            result.success = false;
            result.reason = "delta rebase requires integer values";
            return result;
          }
          new_value = *current + *delta;
        }
        if (intent.constrained && new_value < intent.lower_bound) {
          result.success = false;
          result.reason = "escrow: constrained delta would breach lower bound";
          return result;
        }
        result.should_write = true;
        result.value = std::to_string(new_value);
        return result;
      }
      case IntentType::kCas:
        if (intent.condition.type == ConditionType::kValueEquals &&
            current_store_value != intent.condition.expected_value) {
          result.success = false;
          result.reason = "cas condition no longer holds";
          return result;
        }
        result.should_write = true;
        result.value = branch_value;
        return result;
    }
    result.success = false;
    result.reason = "unknown intent type";
    return result;
  }

 private:
  static std::optional<long long> ParseInt(const std::string& s) {
    long long value = 0;
    const auto* begin = s.data();
    const auto* end = s.data() + s.size();
    const auto parse = std::from_chars(begin, end, value);
    if (parse.ec != std::errc() || parse.ptr != end) return std::nullopt;
    return value;
  }
};

}  // namespace cast::intent
