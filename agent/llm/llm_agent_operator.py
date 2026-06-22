"""LLM agent 算子：把"订一张某航线的机票"交给真实 LLM，一次调用返回 K 个候选航班（严格 JSON）。

- 单次 LLM 调用返回 K 个候选 ⇒ 真实 c_gen/任务（与"1 次调用返回 K 选项"决策一致）。
- 每个候选映射为对所选航班共享座位库存的一次扣减写（DELTA -1，COMM_CONSTR/escrow 类）。
- K 个候选互为"备选航班" ⇒ 当首选航班售罄/争用时，提交内核可 reselect 到 LLM 给的备选（真实多候选收益线）。
- 提供 --mock 干跑：无 key 时用确定性合成候选，校验解析→映射→提交全链路。
"""
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from agent.llm import deepseek_client as ds
from agent.llm import ota_catalog as cat

SYS_PROMPT = (
    "You are a flight-booking agent. Given a user's route and preference and a list of "
    "available flights, pick the best candidate flights to book ONE economy seat. "
    "Return STRICT JSON only: {\"candidates\":[{\"flight_id\":\"<id>\",\"reason\":\"<short>\"}]}. "
    "Rank best-first. Propose DISTINCT alternative flights so a fallback exists if the top one is full."
)


def _user_prompt(task, flights, k):
    lines = [f"- {f['flight_id']}: depart {f['dep_time']}, price ¥{f['price']}" for f in flights]
    return (f"Route: {task['route']}\nPreference: {task['pref']}\n"
            f"Available flights:\n" + "\n".join(lines) +
            f"\nReturn up to {k} distinct candidate flight_id(s), best-first, as the JSON above.")


def _parse_candidates(text, flights, k):
    """从模型输出解析候选 flight_id，过滤为本航线合法航班、去重保序、截断到 k。"""
    valid = {f["flight_id"]: f for f in flights}
    picked = []
    try:
        obj = json.loads(text)
        for c in obj.get("candidates", []):
            fid = str(c.get("flight_id", "")).strip()
            if fid in valid and fid not in picked:
                picked.append(fid)
    except Exception:
        pass
    if not picked:  # 兜底：扫文本里出现的合法 flight_id
        for fid in valid:
            if fid in text and fid not in picked:
                picked.append(fid)
    return picked[:k] if picked else ([flights[0]["flight_id"]] if flights else [])


def make_tasks(n, seed=0, routes=None, hot_bias=0.6):
    """生成 n 个订票任务；hot_bias 把需求偏向前两条热门航线（制造争用）。"""
    rng = random.Random(seed)
    routes = routes or cat.builtin_routes()
    hot = routes[:2]
    prefs = ["cheapest", "earliest departure", "best value (price vs time)"]
    tasks = []
    for i in range(n):
        r = rng.choice(hot) if rng.random() < hot_bias else rng.choice(routes)
        tasks.append({"id": i, "route": r, "pref": rng.choice(prefs)})
    return tasks


def generate_candidates(task, k=3, model=ds.DEFAULT_MODEL, temperature=0.7, mock=False):
    """对一个任务产生 K 个候选航班。返回 dict（含真实 c_gen=latency_s）。"""
    flights = cat.flights_of(task["route"])
    if mock or not ds.have_key():
        # 干跑：确定性挑 K 个不同航班（按价格排序），c_gen 用占位
        rng = random.Random(task["id"] * 131 + 7)
        order = sorted(flights, key=lambda f: f["price"])
        chosen = [f["flight_id"] for f in order[:k]]
        rng.shuffle(chosen)
        c_gen = 0.0
        usage = {}
        raw = "MOCK"
    else:
        msgs = [{"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": _user_prompt(task, flights, k)}]
        r = ds.chat(msgs, model=model, temperature=temperature, max_tokens=512,
                    response_json=True)
        chosen = _parse_candidates(r["text"], flights, k)
        c_gen = r["latency_s"]
        usage = r["usage"]
        raw = r["text"]
    byid = {f["flight_id"]: f for f in flights}
    candidates = [{"flight_id": fid, "oid": byid[fid]["oid"], "price": byid[fid]["price"]}
                  for fid in chosen]
    return {"task_id": task["id"], "route": task["route"], "candidates": candidates,
            "c_gen": c_gen, "usage": usage, "n_parsed": len(candidates),
            "distinct_flights": len({c["oid"] for c in candidates}), "raw": raw}
