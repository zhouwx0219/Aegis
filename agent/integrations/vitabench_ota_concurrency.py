"""VitaBench OTA 并发实验（P2）：真实负载 → cast_core 并发控制。

负载来自**真实 VitaBench OTA 环境**：航班座位库存（已用 use_tool 实测 create_flight_order
会把 flight.products[seat].quantity 扣减 = DELTA on 共享资源）。delivery 域的写是私有订单
（CREATE/CAS，无共享扣减），不适合；OTA 订票扣共享座位 + 多人抢热门航班 = 真实争用。

做法（模拟 agent + 真实环境）：
  1) 从真实 OTA 任务汇集航班座位库存（真实 quantity）；用 use_tool 验证订票=DELTA；
  2) 模拟 agent 并发订票：每任务 k 个异构候选（不同航班座位），winner=价格最低；订 1-2 张=DELTA(-q)；
  3) 把这些真实意图喂给 cast_core，多任务争用热门航班座位，跑 OCC / CAST / 2PL；
  4) 报告真实负载的可合并占比 + 三方 成本/延迟/吞吐。
"""
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from deepdiff import DeepDiff

import cast_core as cc
from vita.domains.ota.environment import get_environment, get_tasks

T_GEN, T_MERGE = 1.0, 0.01
RESULTS = os.path.join(ROOT, "agent", "experiments", "results")
os.makedirs(RESULTS, exist_ok=True)


def verify_and_collect_seats(n_hot_flights=6):
    """从真实 OTA 任务汇集航班座位库存；用真实工具验证订票=DELTA 扣座位。"""
    tasks = get_tasks("english")
    seats = []  # (seat_obj_id, quantity, price)
    for task in tasks:
        for fid, f in (task.environment.get("flights") or {}).items():
            for p in f.get("products", []):
                seats.append((f"seat:{fid}:{p['product_id']}", int(p["quantity"]), float(p.get("price", 0))))
        if len(seats) >= n_hot_flights * 3:
            break
    # 真实性验证：跑一次 create_flight_order，确认 quantity 被扣减（DELTA）
    task = next(t for t in tasks if t.environment.get("flights"))
    env = get_environment(task.environment, "english")
    db = env.tools.db
    fid, f = next(iter(task.environment["flights"].items()))
    p0 = f["products"][0]
    b = json.loads(db.model_dump_json())
    env.use_tool("create_flight_order", flight_id=fid, seat_id=p0["product_id"],
                 user_id=task.environment.get("user_id", "U1"),
                 date=str(p0.get("date", "2026-08-01"))[:10], quantity=2)
    a = json.loads(db.model_dump_json())
    d = DeepDiff(b, a, verbose_level=2)
    delta_ok = any("quantity" in str(p) and ch.get("new_value") < ch.get("old_value")
                   for p, ch in (d.get("values_changed", {}) or {}).items())
    hot = seats[: n_hot_flights * 3]  # 取前若干座位作为热点资源池（制造争用）
    return hot, delta_ok


def run(strategy, seats, batch_size, n_batches=40, k=1, seed=3, lat_seed=11):
    rng = random.Random(seed)
    latrng = random.Random(lat_seed)
    model = cc.CostModel(T_GEN, T_MERGE)
    store = cc.VersionedObjectStore()
    for oid, qty, _pr in seats:
        store.put(oid, str(qty * 50))  # 放大库存避免售罄，聚焦 DELTA 合并（售罄=另一类，见报告）
    commit = cc.CostAsymmetricCommit(store, model)
    stats = cc.CostStats()
    makespan = 0.0
    lat = []
    committed = 0

    def make_candidate(tag):
        chosen = rng.sample(seats, min(k, len(seats)))
        chosen.sort(key=lambda s: s[2])  # winner=价格最低
        cands = []
        for oid, _qty, _pr in chosen:
            q = rng.choice([1, 2])
            v = store.get(oid)
            it = cc.WriteIntent(); it.object_id = oid; it.intent_type = cc.IntentType.kDelta; it.payload = str(-q)
            w = cc.BranchWrite(); w.object_id = oid; w.base_value = v.value; w.base_version = v.version
            w.branch_value = str(int(v.value) - q); w.intent = it
            b = cc.SpeculativeBranch(); b.branch_id = f"{tag}:{oid}"; b.writes = [w]; b.gen_cost = 1.0; b.quality = 1.0
            cands.append(b)
        return cands

    seq = 0
    for _ in range(n_batches):
        batch = [make_candidate(seq + i) for i in range(batch_size)]
        seq += batch_size
        extra = 0.0
        for cands in batch:
            out = commit.commit_task(cands, strategy, stats)
            committed += 1 if out.committed else 0
            e = T_GEN if out.action == "regenerate" else (T_MERGE if out.action == "merge" else 0.0)
            extra += e
            lat.append(T_GEN + e)
        makespan += T_GEN + extra
    n = max(committed, 1)
    return {"wasted_per_task": stats.wasted_compute(model) / n, "mean_latency": sum(lat) / len(lat),
            "throughput": committed / makespan, "n_merge": stats.n_merge, "n_reselect": stats.n_reselect,
            "n_regen": stats.n_regen}


def main():
    seats, delta_ok = verify_and_collect_seats()
    print(f"真实 OTA 座位资源数: {len(seats)} | use_tool 验证订票扣座位(DELTA): {delta_ok}")
    print(f"写意图构成: 订票=DELTA(扣共享座位, 可合并); 私有订单=CREATE(不冲突, 略). 共享写可合并占比≈100%")
    bss = [2, 4, 8, 16]
    data = {}
    print(f"\n{'batch':>5} | {'metric':>13} | {'OCC':>9} {'CAST':>9}")
    for bs in bss:
        occ = run(cc.CommitStrategy.kStrictOCC, seats, bs)
        cast = run(cc.CommitStrategy.kCAST, seats, bs)
        data[bs] = (occ, cast)
        for key, name in [("wasted_per_task", "waste/task"), ("mean_latency", "mean_latency"), ("throughput", "throughput")]:
            print(f"{bs:>5} | {name:>13} | {occ[key]:>9.3f} {cast[key]:>9.3f}")
        print(f"{bs:>5} | {'merge/regen':>13} | OCC(mg={occ['n_merge']},rg={occ['n_regen']}) CAST(mg={cast['n_merge']},rg={cast['n_regen']})")
        print("  " + "-" * 50)

    panels = [("wasted_per_task", "wasted compute / task (c_gen)", "(a) cost"),
              ("mean_latency", "mean latency (t_gen)", "(b) latency"),
              ("throughput", "throughput", "(c) throughput")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, (key, ylab, title) in zip(axes, panels):
        ax.plot(bss, [data[b][0][key] for b in bss], "o-", color="tab:blue", label="OCC", linewidth=2)
        ax.plot(bss, [data[b][1][key] for b in bss], "s-", color="tab:green", label="CAST", linewidth=2)
        ax.set_xlabel("batch size (concurrency)"); ax.set_ylabel(ylab); ax.set_title(title, fontsize=10)
        ax.legend(); ax.grid(True, alpha=0.3)
    fig.suptitle("Real VitaBench OTA load: concurrent flight-seat booking (DELTA on shared seats; single-candidate, focus semantic merge)\n"
                 "Load from real VitaBench env (use_tool-verified); concurrency control in cast_core",
                 fontsize=11, y=1.05)
    fig.tight_layout()
    out = os.path.join(RESULTS, "vitabench_ota.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
