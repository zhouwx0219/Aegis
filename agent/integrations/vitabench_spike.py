"""VitaBench P1 spike：无 LLM 跑通一个真实 VitaBench 写工具，并把 db 变化映射成 CAST 写意图。

前置（一次性）：
  bash setup_vitabench.sh            # git clone + pip install -e /tmp/vb
运行：
  python3 vitabench_spike.py

证明：模拟 agent（无 LLM）能加载真实 VitaBench 任务、执行写工具改 db，
并用 deepdiff 提取对象级写入，映射到 CAST 的写意图（CREATE/DELTA/APPEND/CAS/OVERWRITE）。
"""
import json

from deepdiff import DeepDiff
from vita.domains.delivery.environment import get_environment, get_tasks


def classify_diff(d):
    """把 deepdiff 结果映射成 CAST 写意图（P1 雏形：覆盖最常见几类）。"""
    intents = []
    for path in d.get("dictionary_item_added", []) or []:
        intents.append((str(path), "CREATE/OVERWRITE(new object)"))
    for path in d.get("iterable_item_added", {}) or {}:
        intents.append((str(path), "APPEND(list grew)"))
    for path, ch in (d.get("values_changed", {}) or {}).items():
        old, new = ch.get("old_value"), ch.get("new_value")
        if isinstance(old, (int, float)) and isinstance(new, (int, float)):
            intents.append((str(path), f"DELTA({new - old:+})" if new != old else "noop"))
        else:
            intents.append((str(path), f"CAS/OVERWRITE({old!r}->{new!r})"))
    return intents


def main():
    tasks = get_tasks("english")
    task = tasks[0]
    env = get_environment(task.environment, "english")
    db = env.tools.db

    # 动态取该任务的一个 store + 一个 product（不硬编码）
    stores = task.environment["stores"]
    sid = next(iter(stores))
    products = stores[sid]["products"]
    prod = products[0] if isinstance(products, list) else next(iter(products.values()))
    pid = prod["product_id"]
    loc = task.environment["location"][0]

    before = json.loads(db.model_dump_json())
    res = env.use_tool(
        "create_delivery_order",
        user_id=task.environment.get("user_id", "U000001"),
        store_id=sid,
        product_ids=[pid],
        product_cnts=[1],
        address=loc["address"],
        dispatch_time=task.environment.get("time"),
    )
    after = json.loads(db.model_dump_json())
    d = DeepDiff(before, after, verbose_level=2)

    print(f"task={task.id}  store={sid}  product={pid}")
    print(f"write tool 'create_delivery_order' -> {str(res)[:80]}...")
    print(f"db diff categories: {list(d.keys())}")
    print("==> mapped CAST write-intents:")
    for path, intent in classify_diff(d):
        print(f"   {intent:32} @ {path[:70]}")
    print("\nP1 SPIKE OK: 无 LLM 执行真实 VitaBench 写工具，diff 出对象级写入并映射为写意图。")


if __name__ == "__main__":
    main()
