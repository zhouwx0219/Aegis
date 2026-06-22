"""DeepSeek（OpenAI 兼容）客户端：真实 LLM 调用 + 真实墙钟计时。

安全约定：API key **只**从环境变量 `DEEPSEEK_API_KEY` 读取，绝不硬编码、绝不写盘、绝不打印。
零第三方依赖（urllib + json，stdlib），便于在任意环境复现。
"""
import json
import os
import time
import urllib.error
import urllib.request

API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-chat"  # 备选 deepseek-reasoner（更慢/更贵）


class DeepSeekError(RuntimeError):
    pass


def have_key():
    return bool(os.environ.get("DEEPSEEK_API_KEY"))


def _key():
    k = os.environ.get("DEEPSEEK_API_KEY")
    if not k:
        raise DeepSeekError("环境变量 DEEPSEEK_API_KEY 未设置（key 不写盘，只走 env）")
    return k


def chat(messages, model=DEFAULT_MODEL, temperature=0.7, max_tokens=1024,
         timeout=60, retries=3, response_json=False):
    """单次对话补全。返回 {text, latency_s, usage, raw}。

    latency_s = 真实墙钟（含网络往返）——即一次候选生成的真实 c_gen 代表量。
    response_json=True 时要求模型返回严格 JSON 对象（DeepSeek 支持 response_format）。
    对 429/5xx 做指数退避重试。
    """
    payload = {"model": model, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}
    if response_json:
        payload["response_format"] = {"type": "json_object"}
    body = json.dumps(payload).encode("utf-8")
    last = None
    for attempt in range(retries):
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(
                API_URL, data=body, method="POST",
                headers={"Authorization": f"Bearer {_key()}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            dt = time.perf_counter() - t0
            text = raw["choices"][0]["message"]["content"]
            return {"text": text, "latency_s": dt,
                    "usage": raw.get("usage", {}), "raw": raw}
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "ignore")[:300]
            last = DeepSeekError(f"HTTP {e.code}: {detail}")
            if e.code in (429, 500, 502, 503, 529) and attempt < retries - 1:
                time.sleep(1.5 * (2 ** attempt))
                continue
            raise last
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = DeepSeekError(f"network error: {e}")
            if attempt < retries - 1:
                time.sleep(1.5 * (2 ** attempt))
                continue
            raise last
    raise last if last else DeepSeekError("unknown")


def ping(model=DEFAULT_MODEL):
    """连通性 + 鉴权自检：返回 (ok, latency_s, msg)。"""
    try:
        r = chat([{"role": "user", "content": "reply with the single word: ok"}],
                 model=model, max_tokens=8, temperature=0.0, retries=1, timeout=30)
        return True, r["latency_s"], r["text"].strip()[:40]
    except Exception as e:  # noqa
        return False, 0.0, str(e)[:200]
