"""可配置的模拟算子（MockOperator）。

第一版用确定性、可注入成本与冲突的模拟算子，先把事务/并发机制和评估跑通；
真实 LLM 算子将实现同一接口（read / make_write / generate_candidate）做后续验证。

关键点：算子只负责"产生候选 + 声明写意图 + 提供成本数字"，
所有事务/并发/提交都在 C++ 核（cast_core）里完成。
"""
import random

import cast_core as cc


class MockOperator:
    def __init__(self, store, c_gen=1.0, seed=0):
        self.store = store
        self.c_gen = c_gen
        self.rng = random.Random(seed)

    def read(self, object_id):
        """读对象，返回 (value, version)，并把版本记入读集（由调用方写进 BranchWrite.base_*）。"""
        v = self.store.get(object_id)
        return v.value, v.version

    def make_write(self, object_id, kind, intent_type, *, new_value=None,
                   payload="", cas_expected=None):
        """构造一个分支写：读基线 + 声明写意图 + 计算分支缓冲值。"""
        base_value, base_version = self.read(object_id)

        intent = cc.WriteIntent()
        intent.object_id = object_id
        intent.intent_type = intent_type
        intent.payload = payload
        if cas_expected is not None:
            cond = cc.Condition()
            cond.type = cc.ConditionType.kValueEquals
            cond.expected_value = cas_expected
            intent.condition = cond

        w = cc.BranchWrite()
        w.object_id = object_id
        w.kind = kind
        w.base_value = base_value
        w.base_version = base_version
        w.intent = intent

        # 分支缓冲值（基于读到的基线计算）
        if intent_type == cc.IntentType.kDelta:
            w.branch_value = str(int(base_value) + int(payload))
        elif intent_type == cc.IntentType.kAppend:
            w.branch_value = base_value + payload
        elif new_value is not None:
            w.branch_value = new_value
        else:
            w.branch_value = base_value
        return w

    def generate_candidate(self, branch_id, writes, quality=None):
        """产生一个候选（投机分支），携带生成成本与质量打分。"""
        b = cc.SpeculativeBranch()
        b.branch_id = branch_id
        b.writes = writes
        b.gen_cost = self.c_gen
        b.quality = quality if quality is not None else self.rng.random()
        return b
