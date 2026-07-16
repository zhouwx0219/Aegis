#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/object/unified_object.h"

namespace cast::storage {

using cast::object::VersionedValue;

struct VersionCheck {
  std::string key;
  std::uint64_t expected_version = 0;
};

struct WriteOp {
  std::string key;
  std::string value;
};

class VersionedKVStore {
 public:
  virtual ~VersionedKVStore() = default;
  virtual VersionedValue Get(const std::string& key) const = 0;
  virtual std::uint64_t GetVersion(const std::string& key) const = 0;
  virtual void Put(const std::string& key, const std::string& value) = 0;
  virtual bool PutIfVersion(const std::string& key, std::uint64_t expected,
                            const std::string& value) = 0;
  virtual bool DeleteIfVersion(const std::string& key,
                               std::uint64_t expected) = 0;
  virtual bool ValidateVersions(
      const std::vector<VersionCheck>& checks) const = 0;
  virtual bool BatchPutIfVersion(const std::vector<VersionCheck>& checks,
                                 const std::vector<WriteOp>& writes) = 0;
  virtual const char* BackendName() const = 0;
};

}  // namespace cast::storage
