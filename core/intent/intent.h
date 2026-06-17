#pragma once
// 写意图：记录"这次写的语义是什么"，而不仅是"写成什么值"。
// 沿用现有 Data-Agent-System 的五类意图。
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

struct WriteIntent {
  std::string object_id;
  IntentType intent_type = IntentType::kRead;
  std::string payload;  // append 片段 / delta 增量 / 其他语义负载
  Condition condition;  // 供 CAS 使用
};

inline const char* IntentTypeName(IntentType t) {
  switch (t) {
    case IntentType::kRead: return "READ";
    case IntentType::kOverwrite: return "OVERWRITE";
    case IntentType::kAppend: return "APPEND";
    case IntentType::kDelta: return "DELTA";
    case IntentType::kCas: return "CAS";
  }
  return "READ";
}

}  // namespace cast::intent
