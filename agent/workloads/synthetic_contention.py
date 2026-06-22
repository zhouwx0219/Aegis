"""受控争用负载（阶段1，重构为确定性 task-plan）。

并发模型：对象池 + 批内并发。批内 B 个任务读同一基线快照，再依次提交；
后提交的任务会看到先提交者造成的版本前进 -> 真实 OCC 冲突。
冲突强度由 (n_objects, batch_size, writes_per_task) 控制；
可合并写占比 p_mergeable 控制多少冲突可被语义 rebase 化解。

task-plan：在 build_plan() 里一次性确定每个任务访问哪些对象（确定性，可复现），
这样"真跑 OCC/CAST"与"SCC-kS 解析模型 / 结构性冲突深度"读的是同一套访问序列，
保证可对齐验证（OCC 实测 regen 应≈ 结构性冲突任务数）。
对象类型只体现在写意图（底层存储不记类型，符合纯版本化 KV 边界）。
"""
import random

import cast_core as cc


class ContentionWorkload:
    def __init__(self, n_objects=10, writes_per_task=3, p_mergeable=0.5, seed=0):
        self.n_objects = n_objects
        self.writes_per_task = writes_per_task
        self.rng = random.Random(seed)
        # 每个对象固定类型/意图：可合并(counter->DELTA / text->APPEND) 或 strict(row->OVERWRITE)
        self.obj_specs = []  # (oid, intent_type, init_value)
        for i in range(n_objects):
            if self.rng.random() < p_mergeable:
                if self.rng.random() < 0.5:
                    self.obj_specs.append((f"ctr:{i}", cc.IntentType.kDelta, "1000"))
                else:
                    self.obj_specs.append((f"txt:{i}", cc.IntentType.kAppend, "base"))
            else:
                self.obj_specs.append((f"row:{i}", cc.IntentType.kOverwrite, "v0"))

    def seed_store(self, store):
        for oid, _intent, init in self.obj_specs:
            store.put(oid, init)

    def build_plan(self, n_batches, batch_size):
        """确定性生成任务计划：返回 [[specs_of_task, ...]每批, ...]。"""
        plan = []
        n = min(self.writes_per_task, len(self.obj_specs))
        for _ in range(n_batches):
            batch = [self.rng.sample(self.obj_specs, n) for _ in range(batch_size)]
            plan.append(batch)
        return plan

    def _make_write(self, store, oid, intent_type, tag):
        v = store.get(oid)
        intent = cc.WriteIntent()
        intent.object_id = oid
        intent.intent_type = intent_type
        intent.commutative = intent_type == cc.IntentType.kAppend
        if intent_type == cc.IntentType.kDelta:
            intent.payload = "-1"
            branch_value = str(int(v.value) - 1)
        elif intent_type == cc.IntentType.kAppend:
            intent.payload = f"|{tag}"
            branch_value = v.value + f"|{tag}"
        else:
            intent.payload = ""
            branch_value = f"set-{tag}"
        w = cc.BranchWrite()
        w.object_id = oid
        w.base_value = v.value
        w.base_version = v.version
        w.branch_value = branch_value
        w.intent = intent
        return w

    def build_candidate(self, store, specs, tag):
        """根据一个任务的 specs（对象集）构造单个候选（读当前 store 基线）。"""
        writes = [self._make_write(store, oid, it, tag) for (oid, it, _init) in specs]
        b = cc.SpeculativeBranch()
        b.branch_id = f"t{tag}"
        b.writes = writes
        b.gen_cost = 1.0
        b.quality = 1.0
        return b
