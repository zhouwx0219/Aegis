#pragma once

#include <cstdint>
#include <string>

namespace cast::intent {

enum class IntentType {
  kRead,
  kWrite,
};

enum class ConditionType {
  kNone,
  kValueEquals,
};

struct Condition {
  ConditionType type = ConditionType::kNone;
  std::string expected_value;
};

// Describes the logical operation behind a materialized write.
struct WriteIntent {
  std::string object_id;
  IntentType intent_type = IntentType::kRead;
  std::string payload;
  Condition condition;
};

inline const char* IntentTypeName(IntentType type) {
  switch (type) {
    case IntentType::kRead:
      return "READ";
    case IntentType::kWrite:
      return "WRITE";
  }
  return "READ";
}

}  // namespace cast::intent
