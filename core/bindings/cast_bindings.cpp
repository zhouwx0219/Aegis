// pybind11 桥接：把 C++ 事务内核暴露为 Python 扩展模块 cast_core。
// Python 层（算子/调度/workload）通过它构造候选并调用成本不对称提交。
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

#include "core/branch/speculative_branch.h"
#include "core/cost/cost_model.h"
#include "core/intent/intent.h"
#include "core/intent/policy_dispatcher.h"
#include "core/object/unified_object.h"
#include "core/storage/versioned_object_store.h"
#include "core/txn/cost_asymmetric_commit.h"

namespace py = pybind11;
using namespace cast;

PYBIND11_MODULE(cast_core, m) {
  m.doc() = "CAST: cost-asymmetric speculative transactions core (C++ kernel)";

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
      .def_readwrite("condition", &intent::WriteIntent::condition);

  py::class_<storage::VersionedObjectStore>(m, "VersionedObjectStore")
      .def(py::init<>())
      .def("get", &storage::VersionedObjectStore::Get)
      .def("get_version", &storage::VersionedObjectStore::GetVersion)
      .def("put", &storage::VersionedObjectStore::Put, py::arg("key"), py::arg("value"))
      .def("put_if_version", &storage::VersionedObjectStore::PutIfVersion);

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

  py::enum_<txn::CommitStrategy>(m, "CommitStrategy")
      .value("kStrictOCC", txn::CommitStrategy::kStrictOCC)
      .value("kCAST", txn::CommitStrategy::kCAST);

  py::class_<txn::CommitOutcome>(m, "CommitOutcome")
      .def_readonly("committed", &txn::CommitOutcome::committed)
      .def_readonly("winner_branch_id", &txn::CommitOutcome::winner_branch_id)
      .def_readonly("action", &txn::CommitOutcome::action)
      .def_readonly("reason", &txn::CommitOutcome::reason);

  py::class_<txn::CostAsymmetricCommit>(m, "CostAsymmetricCommit")
      .def(py::init<storage::VersionedObjectStore&, cost::CostModel>(), py::arg("store"),
           py::arg("model"), py::keep_alive<1, 2>())
      .def(
          "commit_task",
          [](txn::CostAsymmetricCommit& self, std::vector<branch::SpeculativeBranch> candidates,
             txn::CommitStrategy strategy, cost::CostStats& stats) {
            return self.CommitTask(candidates, strategy, &stats);
          },
          py::arg("candidates"), py::arg("strategy"), py::arg("stats"));
}
