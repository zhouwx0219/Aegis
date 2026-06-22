#pragma once
// 写意图：记录"这次写的语义是什么"，而不仅是"写成什么值"。
// 沿用现有 Data-Agent-System 的五类意图。
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

struct WriteIntent {
  std::string object_id;
  IntentType intent_type = IntentType::kRead;
  std::string payload;  // append 片段 / delta 增量 / 其他语义负载
  Condition condition;  // 供 CAS 使用

  // —— 约束可交换（escrow）扩展 ——
  // 仅对 DELTA 有意义：constrained=true 表示该扣减带下界约束（如库存 stock>=lower_bound）。
  // 默认 false ⟹ 行为与历史完全一致（既有 CostAsymmetricCommit 调用方不设置此字段）。
  // 约束扣减的状态化预留由 cast::concurrency::EscrowAccount / HybridDispatcher 负责，
  // 此处只承载"这是一笔带下界的可交换写"这一意图标注。
  // Ordered APPEND is strict by default. Set true only for a commutative
  // collection-style append whose operator satisfies associativity/commutativity.
  bool commutative = false;

  bool constrained = false;
  long long lower_bound = 0;
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
