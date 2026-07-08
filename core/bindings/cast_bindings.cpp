// Python bindings for the CAST-DAS storage kernel.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

#include "core/intent/intent.h"
#include "core/object/unified_object.h"
#include "core/storage/versioned_object_store.h"

namespace py = pybind11;
using namespace cast;

PYBIND11_MODULE(cast_core, m) {
  m.doc() = "CAST-DAS: versioned KV storage primitives for agent transactions";

  py::enum_<object::ObjectType>(m, "ObjectType")
      .value("kGeneric", object::ObjectType::kGeneric)
      .value("kRow", object::ObjectType::kRow)
      .value("kText", object::ObjectType::kText)
      .value("kCounter", object::ObjectType::kCounter);

  py::class_<object::VersionedValue>(m, "VersionedValue")
      .def_readonly("value", &object::VersionedValue::value)
      .def_readonly("version", &object::VersionedValue::version)
      .def_readonly("exists", &object::VersionedValue::exists);

  py::enum_<intent::IntentType>(m, "IntentType")
      .value("kRead", intent::IntentType::kRead)
      .value("kWrite", intent::IntentType::kWrite);

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

  py::class_<storage::VersionedKVStore>(m, "VersionedKVStore")
      .def("get", &storage::VersionedKVStore::Get)
      .def("get_version", &storage::VersionedKVStore::GetVersion)
      .def("put", &storage::VersionedKVStore::Put)
      .def("put_if_version", &storage::VersionedKVStore::PutIfVersion)
      .def(
          "batch_put_if_version",
          [](storage::VersionedKVStore& store,
             const std::vector<std::pair<std::string, std::uint64_t>>& checks,
             const std::vector<std::pair<std::string, std::string>>& writes) {
            std::vector<storage::VersionCheck> version_checks;
            version_checks.reserve(checks.size());
            for (const auto& check : checks) {
              version_checks.push_back({check.first, check.second});
            }
            std::vector<storage::WriteOp> write_ops;
            write_ops.reserve(writes.size());
            for (const auto& write : writes) {
              write_ops.push_back({write.first, write.second});
            }
            return store.BatchPutIfVersion(version_checks, write_ops);
          })
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

}
