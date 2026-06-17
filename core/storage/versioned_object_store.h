#pragma once
// 版本化对象存储：统一对象事务的底层边界。
//
// 边界声明：本层【只】提供基于版本的 key→(value, version) 读写原语，
// 【不】理解对象类型(row/text/counter)、事务、分支、winner/loser、并发策略或语义可合并性。
// 对象类型与语义解释、冲突策略全部由上层(intent/branch/txn)负责，随写意图与分支流转；
// 若上层需要"按 key 查类型"，应由上层的对象 catalog 维护，而非下沉到这里。
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/object/unified_object.h"  // 仅为 VersionedValue（值+版本），不引入类型语义

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

class VersionedObjectStore {
 public:
  VersionedValue Get(const std::string& key) const {
    auto it = data_.find(key);
    if (it == data_.end()) return VersionedValue{"", 0, false};
    return VersionedValue{it->second.value, it->second.version, true};
  }

  std::uint64_t GetVersion(const std::string& key) const { return Get(key).version; }

  // 无条件写（初始化用）。版本自增。
  void Put(const std::string& key, const std::string& value) {
    auto& e = data_[key];
    e.value = value;
    e.version += 1;
  }

  // 条件写：当前版本等于 expected 才写入，写后版本变为 expected+1。
  bool PutIfVersion(const std::string& key, std::uint64_t expected,
                    const std::string& value) {
    auto cur = Get(key);
    if (cur.version != expected) return false;
    auto& e = data_[key];
    e.value = value;
    e.version = expected + 1;
    return true;
  }

  bool DeleteIfVersion(const std::string& key, std::uint64_t expected) {
    auto cur = Get(key);
    if (!cur.exists || cur.version != expected) return false;
    data_.erase(key);
    return true;
  }

  // 批量条件写：所有版本检查通过才整体写入（多对象 all-or-nothing）。
  bool BatchPutIfVersion(const std::vector<VersionCheck>& checks,
                         const std::vector<WriteOp>& writes) {
    for (const auto& c : checks) {
      if (GetVersion(c.key) != c.expected_version) return false;
    }
    for (const auto& w : writes) {
      auto& e = data_[w.key];
      e.value = w.value;
      e.version += 1;
    }
    return true;
  }

 private:
  struct Entry {
    std::string value;
    std::uint64_t version = 0;
  };
  std::unordered_map<std::string, Entry> data_;
};

}  // namespace cast::storage
