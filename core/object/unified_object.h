#pragma once
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
