#pragma once

#include <cstdint>
#include <string>

namespace cast::object {

enum class ObjectType {
  kGeneric,
  kRow,
  kText,
  kCounter,
};

// Versioned value read from storage. version==0 and exists==false means missing.
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
  }
  return "generic";
}

}  // namespace cast::object
