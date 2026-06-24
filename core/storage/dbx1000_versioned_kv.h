#pragma once

#include <cstddef>
#include <memory>

#include "core/storage/versioned_kv.h"

namespace cast::storage {

class Dbx1000VersionedKVStore final : public VersionedKVStore {
 public:
  explicit Dbx1000VersionedKVStore(std::size_t max_key_bytes = 512,
                                   std::size_t max_value_bytes = 8192,
                                   std::size_t bucket_count = 4096);
  ~Dbx1000VersionedKVStore() override;

  Dbx1000VersionedKVStore(const Dbx1000VersionedKVStore&) = delete;
  Dbx1000VersionedKVStore& operator=(const Dbx1000VersionedKVStore&) = delete;

  VersionedValue Get(const std::string& key) const override;
  std::uint64_t GetVersion(const std::string& key) const override;
  void Put(const std::string& key, const std::string& value) override;
  bool PutIfVersion(const std::string& key, std::uint64_t expected,
                    const std::string& value) override;
  bool DeleteIfVersion(const std::string& key,
                       std::uint64_t expected) override;
  bool BatchPutIfVersion(const std::vector<VersionCheck>& checks,
                         const std::vector<WriteOp>& writes) override;
  const char* BackendName() const override { return "dbx1000"; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

using VersionedObjectStore = Dbx1000VersionedKVStore;

}  // namespace cast::storage
