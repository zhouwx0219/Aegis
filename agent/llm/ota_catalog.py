"""OTA 资源目录（VitaBench-derived）：航班共享座位库存 + 价格，用于真实 LLM 决策空间。

默认用内置目录（自包含、无需装 VitaBench，便于复现）；设 source='vitabench' 时
从真实 VitaBench OTA 环境采集真实库存（需先 bash agent/integrations/setup_vitabench.sh）。
每个航班 = 一个共享座位库存对象（COMM_CONSTR，stock>=0），oid = flight:{route}:{flight_id}。
座位故意稀缺 ⇒ 多 agent 抢同一热门航班时产生真实争用（暴露超卖边界 + 触发多候选 reselect）。
"""

# 内置热门航线目录：(flight_id, dep_time, price_cny, seats)
_BUILTIN = {
    "PEK-SHA": [("CA1501", "08:00", 1280, 3), ("MU5102", "09:30", 1180, 2),
                ("HU7604", "12:00", 980, 4), ("CZ3902", "18:30", 880, 6)],
    "PEK-CAN": [("CA1801", "07:30", 1680, 2), ("CZ3104", "10:15", 1520, 3),
                ("MU5310", "14:00", 1390, 5)],
    "SHA-SZX": [("FM9201", "09:00", 1090, 2), ("MU2501", "13:30", 990, 3),
                ("ZH9805", "20:00", 860, 5)],
    "CTU-PEK": [("3U8801", "08:45", 1420, 3), ("CA4102", "11:20", 1310, 2),
                ("HU7302", "16:40", 1180, 4)],
}


def builtin_routes():
    return list(_BUILTIN.keys())


def flights_of(route, source="builtin"):
    """返回某航线的航班列表 [{flight_id, oid, dep_time, price, seats}]。"""
    flights = _BUILTIN.get(route, [])
    out = []
    for fid, dep, price, seats in flights:
        out.append({"flight_id": fid, "oid": f"flight:{route}:{fid}",
                    "dep_time": dep, "price": price, "seats": seats})
    return out


def all_flight_objects(routes=None):
    """返回全部航班对象（建库用）：{oid: seats}。"""
    routes = routes or builtin_routes()
    objs = {}
    for r in routes:
        for f in flights_of(r):
            objs[f["oid"]] = f["seats"]
    return objs


def try_load_vitabench(hot=12):
    """可选：从真实 VitaBench OTA 采集真实库存（需已 setup）。失败返回 None。"""
    try:
        from vita.domains.ota.environment import get_tasks
    except Exception:
        return None
    try:
        tasks = get_tasks("english")
    except Exception:
        return None
    objs = {}
    for t in tasks:
        for _id, obj in (t.environment.get("flights") or {}).items():
            for p in (obj.get("products") or []):
                q = int(p.get("quantity", 0))
                if q > 0:
                    objs[f"vbflight:{_id}:{p['product_id']}"] = q
        if len(objs) >= hot:
            break
    return objs or None
