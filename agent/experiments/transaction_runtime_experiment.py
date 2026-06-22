"""Correctness and lifecycle checks for the complete AgentTransaction runtime."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from agent.runtime import AgentTransactionManager, TransactionState

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def candidate(txn, branch_id, quality=1.0):
    return txn.add_candidate(branch_id, quality=quality, gen_cost=0.01)


def main():
    manager = AgentTransactionManager(c_gen=1.0, c_merge=0.01)
    for oid, value, kind in [
        ("stock:a", 1, "counter"),
        ("stock:b", 1, "counter"),
        ("counter", 0, "counter"),
        ("text", "base", "text"),
        ("row", "pending", "row"),
        ("cas", "pending", "row"),
    ]:
        manager.register_object(oid, value, kind=kind)

    results = {}

    tx = manager.begin("direct", {"case": "full lifecycle"})
    tx.record_model_call(model="controlled", latency_s=0.01, usage={"total_tokens": 10}, candidates=1)
    tx.record_tool_call("prepare", args={"object": "stock:a"})
    candidate(tx, "direct-a").delta("stock:a", -1, constrained=True)
    direct = tx.commit()
    assert direct.committed and direct.action == "direct"
    results["direct"] = direct.to_dict()

    tx = manager.begin("reselect")
    candidate(tx, "sold-out-a", quality=2).delta("stock:a", -1, constrained=True)
    candidate(tx, "fallback-b", quality=1).delta("stock:b", -1, constrained=True)
    reselect = tx.commit()
    assert reselect.committed and reselect.action == "reselect"
    assert reselect.n_reselect == 1
    results["reselect"] = reselect.to_dict()

    tx = manager.begin("reject")
    candidate(tx, "sold-out-a", quality=2).delta("stock:a", -1, constrained=True)
    candidate(tx, "sold-out-b", quality=1).delta("stock:b", -1, constrained=True)
    rejected = tx.commit()
    assert rejected.state == TransactionState.REJECTED
    assert rejected.n_regen == 0
    results["semantic_reject"] = rejected.to_dict()

    old = manager.begin("delta-merge-old")
    candidate(old, "old-delta").delta("counter", 1)
    new = manager.begin("delta-merge-new")
    candidate(new, "new-delta").delta("counter", 1)
    assert new.commit().committed
    merged = old.commit()
    assert merged.committed and merged.action == "merge"
    assert manager.value_of("counter") == "2"
    results["delta_merge"] = merged.to_dict()

    ordered_old = manager.begin("ordered-old")
    candidate(ordered_old, "ordered-a").append("text", "|A", commutative=False)
    ordered_new = manager.begin("ordered-new")
    candidate(ordered_new, "ordered-b").append("text", "|B", commutative=False)
    assert ordered_new.commit().committed
    ordered = ordered_old.commit()
    assert ordered.committed and ordered.action == "regenerate"
    assert manager.value_of("text") == "base|B|A"
    results["ordered_append"] = ordered.to_dict()

    cas_old = manager.begin("cas-old")
    candidate(cas_old, "cas-pending").cas("cas", "pending", "confirmed")
    cas_new = manager.begin("cas-new")
    candidate(cas_new, "cas-cancel").overwrite("cas", "cancelled")
    assert cas_new.commit().committed
    cas_reject = cas_old.commit()
    assert cas_reject.rejected and cas_reject.n_regen == 0
    assert manager.value_of("cas") == "cancelled"
    results["cas_reject"] = cas_reject.to_dict()

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "transaction_runtime.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "results": results,
                "final_values": manager.values(),
                "trace_count": len(manager.traces()),
                "traces": manager.traces(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("transaction runtime checks: PASS")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
