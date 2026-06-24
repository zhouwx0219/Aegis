// Python bindings for the ASTRA transaction kernel.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

#include "core/branch/speculative_branch.h"
#include "core/concurrency/concurrency_control.h"
#include "core/concurrency/escrow.h"
#include "core/concurrency/semantic_cc.h"
#include "core/cost/cost_model.h"
#include "core/intent/intent.h"
#include "core/intent/policy_dispatcher.h"
#include "core/object/unified_object.h"
#include "core/storage/versioned_object_store.h"
#include "core/txn/commit_protocol.h"
#include "core/txn/cost_asymmetric_commit.h"

namespace py = pybind11;
using namespace cast;

PYBIND11_MODULE(cast_core, m) {
  m.doc() = "ASTRA: intent-aware agent transaction kernel";

  py::enum_<object::ObjectType>(m, "ObjectType")
      .value("kGeneric", object::ObjectType::kGeneric)
      .value("kRow", object::ObjectType::kRow)
      .value("kText", object::ObjectType::kText)
      .value("kCounter", object::ObjectType::kCounter)
      .value("kCandidateResult", object::ObjectType::kCandidateResult);

  py::class_<object::VersionedValue>(m, "VersionedValue")
      .def_readonly("value", &object::VersionedValue::value)
      .def_readonly("version", &object::VersionedValue::version)
      .def_readonly("exists", &object::VersionedValue::exists);

  py::enum_<intent::IntentType>(m, "IntentType")
      .value("kRead", intent::IntentType::kRead)
      .value("kOverwrite", intent::IntentType::kOverwrite)
      .value("kAppend", intent::IntentType::kAppend)
      .value("kDelta", intent::IntentType::kDelta)
      .value("kCas", intent::IntentType::kCas);

  py::enum_<intent::ConditionType>(m, "ConditionType")
      .value("kNone", intent::ConditionType::kNone)
      .value("kValueEquals", intent::ConditionType::kValueEquals);

  py::class_<intent::Condition>(m, "Condition")
      .def(py::init<>())
      .def_readwrite("type", &intent::Condition::type)
      .def_readwrite("expected_value", &intent::Condition::expected_value);

  py::class_<intent::WriteIntent>(m, "WriteIntent")
      .def(py::init<>())
      .def_readwrite("object_id", &intent::WriteIntent::object_id)
      .def_readwrite("intent_type", &intent::WriteIntent::intent_type)
      .def_readwrite("payload", &intent::WriteIntent::payload)
      .def_readwrite("commutative", &intent::WriteIntent::commutative)
      .def_readwrite("condition", &intent::WriteIntent::condition)
      .def_readwrite("constrained", &intent::WriteIntent::constrained)
      .def_readwrite("lower_bound", &intent::WriteIntent::lower_bound);

  py::class_<storage::VersionedKVStore>(m, "VersionedKVStore")
      .def("get", &storage::VersionedKVStore::Get)
      .def("get_version", &storage::VersionedKVStore::GetVersion)
      .def("put", &storage::VersionedKVStore::Put)
      .def("put_if_version", &storage::VersionedKVStore::PutIfVersion)
      .def("delete_if_version", &storage::VersionedKVStore::DeleteIfVersion)
      .def_property_readonly("backend_name", [](const storage::VersionedKVStore& store) {
        return std::string(store.BackendName());
      });

  py::class_<storage::Dbx1000VersionedKVStore, storage::VersionedKVStore>(
      m, "Dbx1000VersionedKVStore")
      .def(py::init<std::size_t, std::size_t, std::size_t>(),
           py::arg("max_key_bytes") = 512,
           py::arg("max_value_bytes") = 8192,
           py::arg("bucket_count") = 4096);
  m.attr("VersionedObjectStore") = m.attr("Dbx1000VersionedKVStore");

  py::class_<branch::BranchRead>(m, "BranchRead")
      .def(py::init<>())
      .def_readwrite("object_id", &branch::BranchRead::object_id)
      .def_readwrite("version", &branch::BranchRead::version);

  py::class_<branch::BranchWrite>(m, "BranchWrite")
      .def(py::init<>())
      .def_readwrite("object_id", &branch::BranchWrite::object_id)
      .def_readwrite("kind", &branch::BranchWrite::kind)
      .def_readwrite("base_value", &branch::BranchWrite::base_value)
      .def_readwrite("base_version", &branch::BranchWrite::base_version)
      .def_readwrite("branch_value", &branch::BranchWrite::branch_value)
      .def_readwrite("intent", &branch::BranchWrite::intent);

  py::class_<branch::SpeculativeBranch>(m, "SpeculativeBranch")
      .def(py::init<>())
      .def_readwrite("branch_id", &branch::SpeculativeBranch::branch_id)
      .def_readwrite("read_set", &branch::SpeculativeBranch::read_set)
      .def_readwrite("writes", &branch::SpeculativeBranch::writes)
      .def_readwrite("gen_cost", &branch::SpeculativeBranch::gen_cost)
      .def_readwrite("quality", &branch::SpeculativeBranch::quality);

  py::class_<cost::CostModel>(m, "CostModel")
      .def(py::init<>())
      .def(py::init([](double cg, double cm) {
             cost::CostModel cm_obj;
             cm_obj.c_gen = cg;
             cm_obj.c_merge = cm;
             return cm_obj;
           }),
           py::arg("c_gen"), py::arg("c_merge"))
      .def_readwrite("c_gen", &cost::CostModel::c_gen)
      .def_readwrite("c_merge", &cost::CostModel::c_merge);

  py::class_<cost::CostStats>(m, "CostStats")
      .def(py::init<>())
      .def_readwrite("n_tasks", &cost::CostStats::n_tasks)
      .def_readwrite("candidates_generated", &cost::CostStats::candidates_generated)
      .def_readwrite("n_merge", &cost::CostStats::n_merge)
      .def_readwrite("n_reselect", &cost::CostStats::n_reselect)
      .def_readwrite("n_regen", &cost::CostStats::n_regen)
      .def("wasted_compute", &cost::CostStats::WastedCompute, py::arg("model"))
      .def("total_compute", &cost::CostStats::TotalCompute, py::arg("model"));

  py::class_<concurrency::ConcurrencyControl>(m, "ConcurrencyControl")
      .def_property_readonly("name", [](const concurrency::ConcurrencyControl& cc) {
        return std::string(cc.Name());
      })
      .def_property_readonly("family", [](const concurrency::ConcurrencyControl& cc) {
        return std::string(cc.Family());
      })
      .def_property_readonly("description", [](const concurrency::ConcurrencyControl& cc) {
        return std::string(cc.Description());
      })
      .def_property_readonly("allows_semantic_rebase", [](const concurrency::ConcurrencyControl& cc) {
        return cc.AllowsSemanticRebase();
      })
      .def_property_readonly("requires_object_locks", [](const concurrency::ConcurrencyControl& cc) {
        return cc.RequiresObjectLocks();
      });
  py::class_<concurrency::SemanticConcurrencyControl,
             concurrency::ConcurrencyControl>(m, "SemanticConcurrencyControl")
      .def(py::init<>());
  py::class_<concurrency::StrictOccConcurrencyControl,
             concurrency::ConcurrencyControl>(m, "StrictOccConcurrencyControl")
      .def(py::init<>());
  py::class_<concurrency::StrictValidationConcurrencyControl,
             concurrency::ConcurrencyControl>(m, "StrictValidationConcurrencyControl")
      .def(py::init<std::string, std::string, bool, std::string>(),
           py::arg("name") = "strict",
           py::arg("family") = "strict_validation",
           py::arg("requires_object_locks") = false,
           py::arg("description") =
               "Strict version validation over the agent read/write set");

  py::class_<txn::CommitProtocol>(m, "CommitProtocol")
      .def_property_readonly("name", [](const txn::CommitProtocol& protocol) {
        return std::string(protocol.Name());
      })
      .def_property_readonly("family", [](const txn::CommitProtocol& protocol) {
        return std::string(protocol.Family());
      })
      .def_property_readonly("description", [](const txn::CommitProtocol& protocol) {
        return std::string(protocol.Description());
      });

  py::class_<txn::CommitOutcome>(m, "CommitOutcome")
      .def_readonly("committed", &txn::CommitOutcome::committed)
      .def_readonly("rejected", &txn::CommitOutcome::rejected)
      .def_readonly("needs_regeneration", &txn::CommitOutcome::needs_regeneration)
      .def_readonly("winner_branch_id", &txn::CommitOutcome::winner_branch_id)
      .def_readonly("action", &txn::CommitOutcome::action)
      .def_readonly("reason", &txn::CommitOutcome::reason)
      .def_readonly("conflict_object_ids", &txn::CommitOutcome::conflict_object_ids);

  py::class_<txn::CostAsymmetricCommit, txn::CommitProtocol>(m, "CostAsymmetricCommit")
      .def(py::init<storage::VersionedKVStore&, cost::CostModel>(), py::arg("store"),
           py::arg("model"), py::keep_alive<1, 2>())
      .def(
          "commit_task",
          [](txn::CostAsymmetricCommit& self, std::vector<branch::SpeculativeBranch> candidates,
             const concurrency::ConcurrencyControl& cc, cost::CostStats& stats) {
            return self.CommitTask(candidates, cc, &stats);
          },
          py::arg("candidates"), py::arg("cc"), py::arg("stats"));

  py::class_<concurrency::EscrowAccount>(m, "EscrowAccount")
      .def(py::init<>())
      .def(py::init<long long, long long>(), py::arg("capacity"), py::arg("lower_bound") = 0)
      .def("reserve", &concurrency::EscrowAccount::Reserve, py::arg("q"))
      .def("release", &concurrency::EscrowAccount::Release, py::arg("q"))
      .def("remaining", &concurrency::EscrowAccount::remaining)
      .def("lower_bound", &concurrency::EscrowAccount::lower_bound)
      .def("granted", &concurrency::EscrowAccount::granted)
      .def("rejected", &concurrency::EscrowAccount::rejected)
      .def("oversold", &concurrency::EscrowAccount::oversold);
}
