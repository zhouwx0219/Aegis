#include "core/storage/dbx1000_versioned_kv.h"

#include <mm_malloc.h>

#include <cstring>
#include <array>
#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "catalog.h"
#include "helper.h"
#include "index_hash.h"
#include "row.h"
#include "table.h"

namespace cast::storage {
namespace {

constexpr int kVersionColumn = 0;
constexpr int kExistsColumn = 1;
constexpr int kKeySizeColumn = 2;
constexpr int kValueSizeColumn = 3;
constexpr int kKeyColumn = 4;
constexpr int kValueColumn = 5;
constexpr char kTableName[] = "CAST_DAS_VERSIONED_KV";
constexpr std::size_t kLockShardCount = 256;

std::uint64_t HashKey(const std::string& key) {
  std::uint64_t hash = 14695981039346656037ULL;
  for (unsigned char byte : key) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  return hash;
}

void AddColumn(Catalog& schema, const char* name, std::size_t size,
               const char* type) {
  schema.add_col(const_cast<char*>(name), size, const_cast<char*>(type));
}

}  // namespace

struct Dbx1000VersionedKVStore::Impl {
  Impl(std::size_t key_bytes, std::size_t value_bytes, std::size_t buckets)
      : max_key_bytes(key_bytes), max_value_bytes(value_bytes) {
    if (max_key_bytes == 0 || max_value_bytes == 0 || buckets == 0) {
      throw std::invalid_argument("DBx1000 KV dimensions must be positive");
    }
    schema.init(kTableName, 6);
    AddColumn(schema, "VERSION", sizeof(std::uint64_t), "UINT64");
    AddColumn(schema, "EXISTS", sizeof(std::uint64_t), "UINT64");
    AddColumn(schema, "KEY_SIZE", sizeof(std::uint64_t), "UINT64");
    AddColumn(schema, "VALUE_SIZE", sizeof(std::uint64_t), "UINT64");
    AddColumn(schema, "KEY", max_key_bytes, "CHAR");
    AddColumn(schema, "VALUE", max_value_bytes, "CHAR");
    table.init(&schema);
    index.init(1, &table, buckets);
  }

  ~Impl() {
    for (row_t* row : rows) {
      row->free_row();
      _mm_free(row);
    }
    for (itemid_t* item : items) delete item;
  }

  void Validate(const std::string& key, const std::string& value) const {
    if (key.empty()) throw std::invalid_argument("KV key must not be empty");
    if (key.size() > max_key_bytes) {
      throw std::length_error("KV key exceeds configured DBx1000 row capacity");
    }
    if (value.size() > max_value_bytes) {
      throw std::length_error("KV value exceeds configured DBx1000 row capacity");
    }
  }

  row_t* Find(const std::string& key) const {
    itemid_t* item = nullptr;
    if (index.index_try_read(HashKey(key), item, 0) != RCOK) return nullptr;
    for (; item != nullptr; item = item->next) {
      row_t* row = static_cast<row_t*>(item->location);
      std::uint64_t key_size = 0;
      row->get_value(kKeySizeColumn, key_size);
      if (key_size == key.size() &&
          std::memcmp(row->get_value(kKeyColumn), key.data(), key_size) == 0) {
        return row;
      }
    }
    return nullptr;
  }

  row_t* FindOrCreate(const std::string& key) {
    if (row_t* row = Find(key)) return row;
    row_t* row = nullptr;
    std::uint64_t id = next_row_id++;
    if (table.get_new_row(row, 0, id) != RCOK || row == nullptr) {
      throw std::runtime_error("DBx1000 failed to allocate a KV row");
    }
    std::memset(row->get_data(), 0, schema.get_tuple_size());
    row->set_primary_key(HashKey(key));
    std::uint64_t key_size = key.size();
    row->set_value(kKeySizeColumn, key_size);
    row->set_value(kKeyColumn, const_cast<char*>(key.data()), key.size());

    itemid_t* item = new itemid_t();
    item->type = DT_row;
    item->location = row;
    item->next = nullptr;
    item->valid = true;
    if (index.index_insert(HashKey(key), item, 0) != RCOK) {
      delete item;
      row->free_row();
      _mm_free(row);
      throw std::runtime_error("DBx1000 failed to index a KV row");
    }
    rows.push_back(row);
    items.push_back(item);
    return row;
  }

  VersionedValue Read(row_t* row) const {
    if (row == nullptr) return VersionedValue{"", 0, false};
    std::uint64_t version = 0;
    std::uint64_t exists = 0;
    std::uint64_t value_size = 0;
    row->get_value(kVersionColumn, version);
    row->get_value(kExistsColumn, exists);
    row->get_value(kValueSizeColumn, value_size);
    if (!exists) return VersionedValue{"", version, false};
    return VersionedValue{
        std::string(row->get_value(kValueColumn), value_size), version, true};
  }

  void Write(row_t* row, const std::string& value, std::uint64_t version,
             bool exists) {
    std::uint64_t exists_value = exists ? 1 : 0;
    std::uint64_t value_size = exists ? value.size() : 0;
    row->set_value(kVersionColumn, version);
    row->set_value(kExistsColumn, exists_value);
    row->set_value(kValueSizeColumn, value_size);
    std::memset(row->get_value(kValueColumn), 0, max_value_bytes);
    if (exists && !value.empty()) {
      row->set_value(kValueColumn, const_cast<char*>(value.data()), value.size());
    }
  }

  std::size_t max_key_bytes;
  std::size_t max_value_bytes;
  mutable std::array<std::mutex, kLockShardCount> shard_mutexes;
  mutable std::mutex create_mutex;
  Catalog schema;
  table_t table;
  mutable IndexHash index;
  std::uint64_t next_row_id = 0;
  std::vector<row_t*> rows;
  std::vector<itemid_t*> items;
};

namespace {

std::size_t LockShard(const std::string& key) {
  return static_cast<std::size_t>(HashKey(key) % kLockShardCount);
}

std::vector<std::size_t> LockShards(
    const std::vector<VersionCheck>& checks,
    const std::vector<WriteOp>& writes) {
  std::vector<std::size_t> shards;
  shards.reserve(checks.size() + writes.size());
  for (const auto& check : checks) shards.push_back(LockShard(check.key));
  for (const auto& write : writes) shards.push_back(LockShard(write.key));
  std::sort(shards.begin(), shards.end());
  shards.erase(std::unique(shards.begin(), shards.end()), shards.end());
  return shards;
}

}  // namespace

Dbx1000VersionedKVStore::Dbx1000VersionedKVStore(
    std::size_t max_key_bytes, std::size_t max_value_bytes,
    std::size_t bucket_count)
    : impl_(std::make_unique<Impl>(max_key_bytes, max_value_bytes,
                                   bucket_count)) {}

Dbx1000VersionedKVStore::~Dbx1000VersionedKVStore() = default;

VersionedValue Dbx1000VersionedKVStore::Get(const std::string& key) const {
  std::lock_guard<std::mutex> lock(impl_->shard_mutexes[LockShard(key)]);
  return impl_->Read(impl_->Find(key));
}

std::uint64_t Dbx1000VersionedKVStore::GetVersion(
    const std::string& key) const {
  return Get(key).version;
}

void Dbx1000VersionedKVStore::Put(const std::string& key,
                                  const std::string& value) {
  impl_->Validate(key, value);
  std::lock_guard<std::mutex> lock(impl_->shard_mutexes[LockShard(key)]);
  std::lock_guard<std::mutex> create_lock(impl_->create_mutex);
  row_t* row = impl_->FindOrCreate(key);
  VersionedValue current = impl_->Read(row);
  impl_->Write(row, value, current.version + 1, true);
}

bool Dbx1000VersionedKVStore::PutIfVersion(const std::string& key,
                                           std::uint64_t expected,
                                           const std::string& value) {
  impl_->Validate(key, value);
  std::lock_guard<std::mutex> lock(impl_->shard_mutexes[LockShard(key)]);
  row_t* row = impl_->Find(key);
  VersionedValue current = impl_->Read(row);
  if (current.version != expected) return false;
  if (row == nullptr) {
    std::lock_guard<std::mutex> create_lock(impl_->create_mutex);
    row = impl_->FindOrCreate(key);
  }
  impl_->Write(row, value, expected + 1, true);
  return true;
}

bool Dbx1000VersionedKVStore::DeleteIfVersion(const std::string& key,
                                              std::uint64_t expected) {
  std::lock_guard<std::mutex> lock(impl_->shard_mutexes[LockShard(key)]);
  row_t* row = impl_->Find(key);
  VersionedValue current = impl_->Read(row);
  if (row == nullptr || !current.exists || current.version != expected) {
    return false;
  }
  impl_->Write(row, "", expected + 1, false);
  return true;
}

bool Dbx1000VersionedKVStore::ValidateVersions(
    const std::vector<VersionCheck>& checks) const {
  const auto shards = LockShards(checks, {});
  std::vector<std::unique_lock<std::mutex>> locks;
  locks.reserve(shards.size());
  for (std::size_t shard : shards) {
    locks.emplace_back(impl_->shard_mutexes[shard]);
  }
  for (const auto& check : checks) {
    if (impl_->Read(impl_->Find(check.key)).version != check.expected_version) {
      return false;
    }
  }
  return true;
}

bool Dbx1000VersionedKVStore::BatchPutIfVersion(
    const std::vector<VersionCheck>& checks,
    const std::vector<WriteOp>& writes) {
  std::unordered_set<std::string> write_keys;
  for (const auto& write : writes) {
    impl_->Validate(write.key, write.value);
    if (!write_keys.insert(write.key).second) return false;
  }

  const auto shards = LockShards(checks, writes);
  std::vector<std::unique_lock<std::mutex>> locks;
  locks.reserve(shards.size());
  for (std::size_t shard : shards) {
    locks.emplace_back(impl_->shard_mutexes[shard]);
  }
  for (const auto& check : checks) {
    if (impl_->Read(impl_->Find(check.key)).version != check.expected_version) {
      return false;
    }
  }
  for (const auto& write : writes) {
    row_t* row = impl_->Find(write.key);
    if (row == nullptr) {
      std::lock_guard<std::mutex> create_lock(impl_->create_mutex);
      row = impl_->FindOrCreate(write.key);
    }
    VersionedValue current = impl_->Read(row);
    impl_->Write(row, write.value, current.version + 1, true);
  }
  return true;
}

}  // namespace cast::storage
