#pragma once
// 统一对象抽象（事务语义扩大化的落点）。
// 第一版聚焦三类对象：结构化行(row)、文本(text)、数值计数器(counter)。
// kGeneric/kCandidateResult 保留以兼容现有 Data-Agent-System 的 ObjectType 语义。
#include <cstdint>
#include <string>

namespace cast::object {

enum class ObjectType {
  kGeneric,
  kRow,
  kText,
  kCounter,
  kCandidateResult,
};

// 底层存储读出的版本化值。version==0 且 exists==false 表示对象不存在。
struct VersionedValue {
  std::string value;
  std::uint64_t version = 0;
  bool exists = false;
};

inline const char* ObjectTypeName(ObjectType t) {
  switch (t) {
    case ObjectType::kGeneric: return "generic";
    case ObjectType::kRow: return "row";
    case ObjectType::kText: return "text";
    case ObjectType::kCounter: return "counter";
    case ObjectType::kCandidateResult: return "candidate_result";
  }
  return "generic";
}

}  // namespace cast::object
