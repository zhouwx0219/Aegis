"""探索式多候选负载（纯 strict，用于证明"异构多候选 reselect"的独立收益）。

场景：独占资源分配（如订会议室/选独家供应商/抢同一时段名额）——用 CAS（资源 free->taken，
条件=仍 free）。这是【纯 strict】场景，语义合并(merge)完全用不上，所以任何 CAST 的收益
都【只能】来自"异构多候选 + reselect"，从而把"探索式多候选"卖点与语义合并彻底分离。

每个任务：agent 生成 k 个【异构】候选——各占用一个【不同】资源（走不同解路径），
每个候选带规则质量分。winner=质量最高。若 winner 资源被并发占走（CAS 失败），
CAST 可 reselect 一个仍空闲的候选（不同资源）；OCC 无 reselect，只能重新探索（重跑 LLM）。
"""
import random

import cast_core as cc


class ExploreWorkload:
    def __init__(self, n_resources=60, k=4, seed=0):
        self.rng = random.Random(seed)
        self.n_resources = n_resources
        self.k = k
        self.resources = [f"res:{i}" for i in range(n_resources)]

    def seed_store(self, store):
        for r in self.resources:
            store.put(r, "free")

    def _candidate(self, store, res, tag, quality):
        v = store.get(res)
        intent = cc.WriteIntent()
        intent.object_id = res
        intent.intent_type = cc.IntentType.kCas
        cond = cc.Condition()
        cond.type = cc.ConditionType.kValueEquals
        cond.expected_value = "free"      # 只有资源仍空闲才能占用
        intent.condition = cond
        w = cc.BranchWrite()
        w.object_id = res
        w.base_value = v.value
        w.base_version = v.version
        w.branch_value = "taken"
        w.intent = intent
        b = cc.SpeculativeBranch()
        b.branch_id = f"{tag}:{res}"
        b.writes = [w]
        b.gen_cost = 1.0
        b.quality = quality
        return b

    def gen_candidates(self, store, tag):
        """读当前 store，生成 k 个异构候选（k 个不同资源），各带规则质量分。"""
        chosen = self.rng.sample(self.resources, min(self.k, len(self.resources)))
        return [self._candidate(store, res, tag, self.rng.random()) for res in chosen]
