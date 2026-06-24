#pragma once

#include <string>
#include <utility>

#include "core/concurrency/concurrency_control.h"
#include "core/intent/policy_dispatcher.h"

namespace cast::concurrency {

class SemanticConcurrencyControl final : public ConcurrencyControl {
 public:
  const char* Name() const override { return "semantic"; }
  const char* Family() const override { return "semantic"; }
  const char* Description() const override {
    return "Intent-aware validation with semantic rebase for commutative, constrained, and conditional writes";
  }
  bool AllowsSemanticRebase() const override { return true; }

  CCResolveResult Resolve(
      const branch::BranchWrite& write,
      const object::VersionedValue& current) const override {
    return ResolveImpl(write, current, true);
  }

  static CCResolveResult ResolveImpl(
      const branch::BranchWrite& write,
      const object::VersionedValue& current, bool allow_rebase) {
    if (write.object_id.empty() ||
        (!write.intent.object_id.empty() &&
         write.intent.object_id != write.object_id)) {
      return {CCDecision::kReject, "", "intent object id does not match write", false,
              false};
    }

    const auto cc_class = intent::PolicyDispatcher::Classify(write.intent);
    if (cc_class == intent::PolicyDispatcher::ConcurrencyClass::kReadOnly) {
      return {CCDecision::kSkip, "", "", false, false};
    }

    const bool changed = current.version != write.base_version;
    if (changed &&
        (!allow_rebase ||
         (cc_class != intent::PolicyDispatcher::ConcurrencyClass::kCommutativeRebase &&
          cc_class != intent::PolicyDispatcher::ConcurrencyClass::kConstrainedCommutative &&
          cc_class != intent::PolicyDispatcher::ConcurrencyClass::kConditionalRebase))) {
      return {CCDecision::kConflict, "", "version conflict", false, false};
    }

    const bool needs_semantic_resolution =
        changed ||
        cc_class == intent::PolicyDispatcher::ConcurrencyClass::kConstrainedCommutative ||
        cc_class == intent::PolicyDispatcher::ConcurrencyClass::kConditionalRebase;
    if (!needs_semantic_resolution) {
      return {CCDecision::kApply, write.branch_value, "", false, false};
    }

    const auto resolved = intent::PolicyDispatcher::ResolveWrite(
        write.base_value, write.branch_value, write.intent, current.value);
    if (!resolved.success) {
      return {CCDecision::kReject, "", resolved.reason, false, false};
    }
    if (!resolved.should_write) {
      return {CCDecision::kSkip, "", "", changed, false};
    }
    const bool merged =
        changed &&
        (cc_class == intent::PolicyDispatcher::ConcurrencyClass::kCommutativeRebase ||
         cc_class == intent::PolicyDispatcher::ConcurrencyClass::kConstrainedCommutative);
    return {CCDecision::kApply, resolved.value, "", changed, merged};
  }
};

class StrictOccConcurrencyControl final : public ConcurrencyControl {
 public:
  const char* Name() const override { return "occ"; }
  const char* Family() const override { return "optimistic"; }
  const char* Description() const override {
    return "Strict optimistic validation over the agent read/write set";
  }

  CCResolveResult Resolve(
      const branch::BranchWrite& write,
      const object::VersionedValue& current) const override {
    return SemanticConcurrencyControl::ResolveImpl(write, current, false);
  }
};

class StrictValidationConcurrencyControl final : public ConcurrencyControl {
 public:
  explicit StrictValidationConcurrencyControl(
      std::string name = "strict",
      std::string family = "strict_validation",
      bool requires_object_locks = false,
      std::string description = "Strict version validation over the agent read/write set")
      : name_(std::move(name)),
        family_(std::move(family)),
        requires_object_locks_(requires_object_locks),
        description_(std::move(description)) {}

  const char* Name() const override { return name_.c_str(); }
  const char* Family() const override { return family_.c_str(); }
  const char* Description() const override { return description_.c_str(); }
  bool RequiresObjectLocks() const override { return requires_object_locks_; }

  CCResolveResult Resolve(
      const branch::BranchWrite& write,
      const object::VersionedValue& current) const override {
    return SemanticConcurrencyControl::ResolveImpl(write, current, false);
  }

 private:
  std::string name_;
  std::string family_;
  bool requires_object_locks_;
  std::string description_;
};

}  // namespace cast::concurrency
