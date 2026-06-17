"""CandidateScheduler（阶段1雏形）。

职责：决定为一个任务生成几个候选（k）以及如何生成；是候选生成的统一入口。
当前为规则版：
  - OCC / CAST：k=1（CAST 主要靠语义合并化解冲突，不需多开候选）。
  - SCC-kS：k=K（多 shadow 赌序列化顺序）——本阶段 SCC 以解析成本模型评估，
            故这里的 k 主要用于记录"投机度"，真跑路径仍取 winner 候选。
下一步（README 已标注）：成本不对称感知的自适应 k —— 当 strict 写占比高、
语义合并覆盖不到时，按"边际收益 < c_gen 即停"增开候选用 reselect 兜底。
"""


class CandidateScheduler:
    def __init__(self, workload, k=1):
        self.workload = workload
        self.k = k

    def candidates_for(self, store, specs, tag):
        """构造该任务的候选列表（当前 k=1，返回单元素列表）。"""
        return [self.workload.build_candidate(store, specs, tag)]
