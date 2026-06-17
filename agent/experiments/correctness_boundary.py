"""正确性边界演示：可交换写"验证层放行"的安全条件（组会"讲死"用）。

结论：CAST 放行可交换写(DELTA)只保证【收敛性/不丢更新】——合并结果 = 各增量之和，
等价于某个串行执行（可串行化 w.r.t. 写）。但它【不保证带下界约束的不变量】（如库存>=0）：
  - 无下界约束（如计数器/点赞）：合并安全，与串行一致。
  - 有下界约束（如库存扣减）：纯 DELTA 放行合并会超卖（< 0），必须把"扣减且不破约束"
    建模为 conditional(CAS/escrow)，由 CAST 的 conditional-rebase 在提交时重检条件。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import cast_core as cc


def run_delta_batch(init, deltas):
    """一批并发 DELTA 任务（读同一基线后依次提交），CAST 合并，返回最终值。"""
    model = cc.CostModel(1.0, 0.01)
    store = cc.VersionedObjectStore()
    store.put("obj", str(init))
    commit = cc.CostAsymmetricCommit(store, model)
    stats = cc.CostStats()
    base = store.get("obj")  # 同一基线
    cands = []
    for i, d in enumerate(deltas):
        it = cc.WriteIntent(); it.object_id = "obj"; it.intent_type = cc.IntentType.kDelta; it.payload = str(d)
        w = cc.BranchWrite(); w.object_id = "obj"; w.base_value = base.value; w.base_version = base.version
        w.branch_value = str(int(base.value) + d); w.intent = it
        b = cc.SpeculativeBranch(); b.branch_id = f"t{i}"; b.writes = [w]; b.quality = 1.0
        cands.append([b])
    for c in cands:
        commit.commit_task(c, cc.CommitStrategy.kCAST, stats)
    return int(store.get("obj").value), stats.n_merge


def conditional_floor(init, n_tasks, qty=1):
    """正确建模：带下界的扣减用 conditional —— 提交时重检 库存>=qty，不足则拒绝（不超卖）。"""
    stock = init; ok = 0; rejected = 0
    for _ in range(n_tasks):
        if stock >= qty:
            stock -= qty; ok += 1
        else:
            rejected += 1
    return stock, ok, rejected


def main():
    print("=== 场景 A：无下界约束（计数器/点赞），8 个并发 DELTA(+1) ===")
    final, merges = run_delta_batch(100, [1] * 8)
    print(f"  CAST 合并最终值 = {final}（串行结果 = 108）→ {'一致, 可串行化, 安全' if final == 108 else '不一致!'}（merge={merges}）")

    print("\n=== 场景 B：有下界约束（库存>=0），初始 5，8 个并发 DELTA(-1) ===")
    final, merges = run_delta_batch(5, [-1] * 8)
    print(f"  纯 DELTA 放行合并最终值 = {final} → {'超卖! 违反库存>=0' if final < 0 else 'ok'}（merge={merges}）")
    print("  说明：合并值=5-8=-3 与串行一致（收敛正确），但违反非负约束——可交换性≠约束安全。")
    stock, ok, rej = conditional_floor(5, 8, 1)
    print(f"  正确做法 = conditional 扣减（库存>=1 才扣，CAST 的 conditional-rebase 提交时重检）：")
    print(f"     成功 {ok} 单、拒绝 {rej} 单、最终库存 {stock} → 不超卖。")

    print("\n=== 边界结论（讲死）===")
    print("  CAST 放行可交换写 ⇒ 保证不丢更新/收敛(可串行化 w.r.t. 写)；")
    print("  但【有下界/容量约束】的扣减必须用 conditional(CAS/escrow)，否则放行合并会破约束；")
    print("  且 CAST 放行读集 ⇒ 像 SI 一样允许 write-skew，不保证读依赖的可串行化（明确不保证项）。")


if __name__ == "__main__":
    main()
