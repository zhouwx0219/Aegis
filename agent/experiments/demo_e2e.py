"""端到端 demo：对比 strict OCC vs CAST 的浪费算力。

运行： python3 agent/experiments/demo_e2e.py   （在 cast-das 目录下）

展示两件事：
  场景1（可合并冲突）：候选对 counter/text 做 DELTA/APPEND，并发任务抢先改了它们。
                       OCC 必须 regenerate(昂贵)；CAST 用语义 rebase 合并(廉价)。
  场景2（strict 冲突）：候选对 row 做 OVERWRITE，并发改了它。
                       OVERWRITE 不可合并，CAST 也只能 regenerate —— 说明优势边界（诚实对照）。
"""
import os
import sys

# 把 cast-das 根目录加入 path，以便 import cast_core 与 agent.*
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import cast_core as cc
from agent.operators.mock_operator import MockOperator

MODEL = cc.CostModel(1.0, 0.01)  # c_gen=1.0（重跑一次候选）, c_merge=0.01（一次 rebase）


def run_scenario(title, build_candidates, apply_concurrent, strategy_name, strategy):
    store = cc.VersionedObjectStore()
    seed_store(store)
    op = MockOperator(store, c_gen=MODEL.c_gen, seed=1)
    candidates = build_candidates(op)   # 候选基于初始版本生成
    apply_concurrent(store)             # 并发任务抢先提交，相关对象版本前进
    stats = cc.CostStats()
    commit = cc.CostAsymmetricCommit(store, MODEL)
    outcome = commit.commit_task(candidates, strategy, stats)
    wasted = stats.wasted_compute(MODEL)
    print(f"  [{strategy_name:18}] committed={outcome.committed} action={outcome.action:11} "
          f"| merge={stats.n_merge} reselect={stats.n_reselect} regen={stats.n_regen} "
          f"| wasted={wasted:.3f}")
    return wasted


def seed_store(store):
    # 底层只存 key→(value, version)，不记类型；对象类型由上层写意图(DELTA/APPEND/OVERWRITE)体现。
    store.put("stock:item_8", "100")        # counter
    store.put("note:order_1", "base-note")  # text
    store.put("row:order_1", "status=pending")  # row


# ---- 场景1：可合并冲突（DELTA + APPEND） ----
def s1_candidates(op):
    w_stock = op.make_write("stock:item_8", cc.ObjectType.kCounter,
                            cc.IntentType.kDelta, payload="-5")
    w_note = op.make_write("note:order_1", cc.ObjectType.kText,
                           cc.IntentType.kAppend, payload=" | T1-confirmed")
    return [op.generate_candidate("T1-A", [w_stock, w_note], quality=0.9)]


def s1_concurrent(store):
    store.put_if_version("stock:item_8", 1, "97")                  # T2 扣减3, v1->v2
    store.put_if_version("note:order_1", 1, "base-note | T2-note")  # T2 追加, v1->v2


# ---- 场景2：strict 冲突（OVERWRITE） ----
def s2_candidates(op):
    w_row = op.make_write("row:order_1", cc.ObjectType.kRow,
                          cc.IntentType.kOverwrite, new_value="status=confirmed-by-T1")
    return [op.generate_candidate("T1-B", [w_row], quality=0.9)]


def s2_concurrent(store):
    store.put_if_version("row:order_1", 1, "status=confirmed-by-T2")  # T2 覆盖, v1->v2


def main():
    print("=" * 78)
    print("场景1：可合并冲突 —— 候选对 counter 做 DELTA、对 text 做 APPEND，并发任务抢先改了两者")
    print("=" * 78)
    w_occ_1 = run_scenario("s1", s1_candidates, s1_concurrent, "strict OCC", cc.CommitStrategy.kStrictOCC)
    w_cast_1 = run_scenario("s1", s1_candidates, s1_concurrent, "CAST (ours)", cc.CommitStrategy.kCAST)
    save = (1 - w_cast_1 / w_occ_1) * 100 if w_occ_1 > 0 else 0
    print(f"  => OCC wasted={w_occ_1:.3f} vs CAST wasted={w_cast_1:.3f}  |  CAST 省下 {save:.1f}% 浪费算力\n")

    print("=" * 78)
    print("场景2：strict 冲突 —— 候选对 row 做 OVERWRITE，并发任务抢先覆盖（优势边界对照）")
    print("=" * 78)
    w_occ_2 = run_scenario("s2", s2_candidates, s2_concurrent, "strict OCC", cc.CommitStrategy.kStrictOCC)
    w_cast_2 = run_scenario("s2", s2_candidates, s2_concurrent, "CAST (ours)", cc.CommitStrategy.kCAST)
    print(f"  => OCC wasted={w_occ_2:.3f} vs CAST wasted={w_cast_2:.3f}  |  "
          f"OVERWRITE 不可合并，CAST 无优势（诚实边界）\n")

    print("结论：CAST 的算力节省来自语义可合并的写（DELTA/APPEND/CAS-rebase）；")
    print("      对不可合并的 strict 写（OVERWRITE），CAST 退化到与 OCC 相同。")


if __name__ == "__main__":
    main()
