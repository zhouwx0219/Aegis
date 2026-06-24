#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/intent/intent.h"
#include "core/object/unified_object.h"

namespace cast::branch {

struct BranchRead {
  std::string object_id;
  std::uint64_t version = 0;
};

// Buffered write for one speculative branch. The base fields capture the
// branch snapshot; branch_value is the materialized candidate value; intent
// carries the semantic operation used by pluggable CC modules.
struct BranchWrite {
  std::string object_id;
  cast::object::ObjectType kind = cast::object::ObjectType::kGeneric;
  std::string base_value;
  std::uint64_t base_version = 0;
  std::string branch_value;
  cast::intent::WriteIntent intent;
};

struct SpeculativeBranch {
  std::string branch_id;
  std::vector<BranchRead> read_set;
  std::vector<BranchWrite> writes;
  double gen_cost = 1.0;
  double quality = 0.0;
};

}  // namespace cast::branch
