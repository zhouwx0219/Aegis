#pragma once

#include <cstdint>
#include <string>

namespace cast::intent {

enum class IntentType {
  kRead,
  kOverwrite,
  kAppend,
  kDelta,
  kCas,
};

enum class ConditionType {
  kNone,
  kValueEquals,
};

struct Condition {
  ConditionType type = ConditionType::kNone;
  std::string expected_value;
};

// Carries the semantic meaning of a write in addition to its materialized value.
struct WriteIntent {
  std::string object_id;
  IntentType intent_type = IntentType::kRead;
  std::string payload;
  Condition condition;

  // Ordered append is strict by default. Enable this only when the append
  // operator is associative and commutative.
  bool commutative = false;

  // A constrained delta must not move the resolved value below lower_bound.
  bool constrained = false;
  long long lower_bound = 0;
};

inline const char* IntentTypeName(IntentType type) {
  switch (type) {
    case IntentType::kRead:
      return "READ";
    case IntentType::kOverwrite:
      return "OVERWRITE";
    case IntentType::kAppend:
      return "APPEND";
    case IntentType::kDelta:
      return "DELTA";
    case IntentType::kCas:
      return "CAS";
  }
  return "READ";
}

}  // namespace cast::intent
