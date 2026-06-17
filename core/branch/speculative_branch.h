#pragma once
// 成本标注的投机分支：一个候选解，携带读基线、缓冲写、写意图，以及生成成本与质量。
#include <cstdint>
#include <string>
#include <vector>

#include "core/intent/intent.h"
#include "core/object/unified_object.h"

namespace cast::branch {

// 一个分支对某对象的写：记录读到的基线(用于版本校验与语义 rebase)与缓冲的新值。
struct BranchWrite {
  std::string object_id;
  cast::object::ObjectType kind = cast::object::ObjectType::kGeneric;
  std::string base_value;            // 分支读到的基线值
  std::uint64_t base_version = 0;    // 分支读到的版本（读集）
  std::string branch_value;          // 分支缓冲的新值
  cast::intent::WriteIntent intent;  // 写意图（决定并发类与 rebase 方式）
};

struct SpeculativeBranch {
  std::string branch_id;
  std::vector<BranchWrite> writes;
  double gen_cost = 1.0;  // 该候选的生成成本（由算子注入）
  double quality = 0.0;   // winner 选择打分
};

}  // namespace cast::branch
