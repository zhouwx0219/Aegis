"""VitaBench-derived 负载模型（阶段1）。

这是【基于 VitaBench 领域结构的负载模型】，不是真跑 VitaBench。
VitaBench(ICLR'26, 美团 LongCat) 覆盖外卖/到店/在线旅游三类真实交互任务。
本模型取其领域对象与自然写语义来构造并发事务负载；真实 LLM-in-the-loop 留作后续。

关键价值：可合并写占比【由领域语义自然产生】，不是人为调的参数——
  - 商家库存 stock:m       -> DELTA（下单扣减，可合并）
  - 酒店房态 room:h        -> DELTA（订房扣减，可合并）
  - 用户购物车 cart:u      -> APPEND（加购，可合并）
  - 用户行程 itin:u        -> APPEND（加行程段，可合并）
  - 订单状态 order:u       -> CAS  （pending->confirmed，条件提交）
热点争用：商家/酒店数量少、用户多 -> stock/room 是天然热点（高争用且可合并）。
对象类型只体现在写意图（底层存储仍是纯版本化 KV）。
"""
import random

import cast_core as cc


class VitaBenchWorkload:
    def __init__(self, n_merchants=4, n_hotels=3, n_users=20, seed=0):
        self.rng = random.Random(seed)
        self.n_merchants = n_merchants
        self.n_hotels = n_hotels
        self.n_users = n_users
        self.objs = {}  # oid -> (intent_type, init_value)
        for m in range(n_merchants):
            self.objs[f"stock:m{m}"] = (cc.IntentType.kDelta, "500")
        for h in range(n_hotels):
            self.objs[f"room:h{h}"] = (cc.IntentType.kDelta, "100")
        for u in range(n_users):
            self.objs[f"cart:u{u}"] = (cc.IntentType.kAppend, "cart")
            self.objs[f"itin:u{u}"] = (cc.IntentType.kAppend, "itin")
            self.objs[f"order:u{u}"] = (cc.IntentType.kCas, "pending")

    def seed_store(self, store):
        for oid, (_it, init) in self.objs.items():
            store.put(oid, init)

    def _spec(self, oid):
        it, init = self.objs[oid]
        return (oid, it, init)

    def _task(self):
        """随机一个跨场景任务（外卖下单 / 订行程 / 加购），混合可合并与 CAS 写。"""
        u = self.rng.randrange(self.n_users)
        kind = self.rng.choice(["food", "food", "travel", "cart_update"])  # 外卖权重更高
        if kind == "food":  # 扣库存(DELTA) + 加购(APPEND) + 确认订单(CAS)
            m = self.rng.randrange(self.n_merchants)
            return [self._spec(f"stock:m{m}"), self._spec(f"cart:u{u}"), self._spec(f"order:u{u}")]
        if kind == "travel":  # 扣房态(DELTA) + 加行程(APPEND)
            h = self.rng.randrange(self.n_hotels)
            return [self._spec(f"room:h{h}"), self._spec(f"itin:u{u}")]
        # 加购：扣库存(DELTA) + 加购(APPEND)
        m = self.rng.randrange(self.n_merchants)
        return [self._spec(f"stock:m{m}"), self._spec(f"cart:u{u}")]

    def build_plan(self, n_batches, batch_size):
        return [[self._task() for _ in range(batch_size)] for _ in range(n_batches)]

    def _make_write(self, store, oid, intent_type, tag):
        v = store.get(oid)
        intent = cc.WriteIntent()
        intent.object_id = oid
        intent.intent_type = intent_type
        if intent_type == cc.IntentType.kDelta:
            intent.payload = "-1"
            branch_value = str(int(v.value) - 1)
        elif intent_type == cc.IntentType.kAppend:
            intent.payload = f"|{tag}"
            branch_value = v.value + f"|{tag}"
        elif intent_type == cc.IntentType.kCas:
            cond = cc.Condition()
            cond.type = cc.ConditionType.kValueEquals
            cond.expected_value = v.value  # 条件：仍是读到时的值（如 pending）
            intent.condition = cond
            branch_value = "confirmed"
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
        writes = [self._make_write(store, oid, it, tag) for (oid, it, _init) in specs]
        b = cc.SpeculativeBranch()
        b.branch_id = f"t{tag}"
        b.writes = writes
        b.gen_cost = 1.0
        b.quality = 1.0
        return b
