#pragma once

#include <string>

#include "core/branch/speculative_branch.h"
#include "core/object/unified_object.h"

namespace cast::concurrency {

enum class CCDecision { kApply, kSkip, kConflict, kReject };

struct CCResolveResult {
  CCDecision decision = CCDecision::kConflict;
  std::string value;
  std::string reason;
  bool rebased = false;
  bool merged = false;
};

class ConcurrencyControl {
 public:
  virtual ~ConcurrencyControl() = default;
  virtual const char* Name() const = 0;
  virtual const char* Family() const { return "custom"; }
  virtual const char* Description() const { return ""; }
  virtual bool AllowsSemanticRebase() const { return false; }
  virtual bool RequiresObjectLocks() const { return false; }
  virtual CCResolveResult Resolve(
      const branch::BranchWrite& write,
      const object::VersionedValue& current) const = 0;
};

}  // namespace cast::concurrency
