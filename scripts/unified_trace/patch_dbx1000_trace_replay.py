#!/usr/bin/env python3
"""Patch a DBx1000-family checkout to replay CAST-DAS fixed traces."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


TRACE_CONFIG_BLOCK = r"""
#ifndef CASTDAS_TRACE_REPLAY
#define CASTDAS_TRACE_REPLAY         false
#endif
#ifndef CASTDAS_TRACE_PATH
#define CASTDAS_TRACE_PATH           "./outputs/castdas_trace.tsv"
#endif
#ifndef CASTDAS_TRACE_MAX_OPS
#define CASTDAS_TRACE_MAX_OPS        64
#endif
"""


TRACE_HEADER = r"""#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <pthread.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "config.h"
#include "query.h"

#ifndef CASTDAS_TRACE_REPLAY
#define CASTDAS_TRACE_REPLAY false
#endif
#ifndef CASTDAS_TRACE_PATH
#define CASTDAS_TRACE_PATH "./outputs/castdas_trace.tsv"
#endif
#ifndef CASTDAS_TRACE_MAX_OPS
#define CASTDAS_TRACE_MAX_OPS 64
#endif

struct CastdasTraceOp {
  bool is_write;
  uint64_t key;
};

struct CastdasTraceTxn {
  bool valid;
  bool is_agent;
  uint32_t pre_delay_ms;
  uint32_t retry_delay_ms;
  uint32_t commit_delay_ms;
  std::string native_payload;
  std::vector<CastdasTraceOp> ops;
  CastdasTraceTxn()
      : valid(false), is_agent(false), pre_delay_ms(0), retry_delay_ms(0),
        commit_delay_ms(0), native_payload(), ops() {}
};

static inline std::vector<std::vector<CastdasTraceTxn> >& castdas_trace_rows() {
  static std::vector<std::vector<CastdasTraceTxn> > rows;
  return rows;
}

static inline std::vector<uint64_t>& castdas_trace_init_offsets() {
  static std::vector<uint64_t> offsets;
  return offsets;
}

static inline pthread_mutex_t& castdas_trace_mutex() {
  static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
  return lock;
}

static inline bool& castdas_trace_loaded() {
  static bool loaded = false;
  return loaded;
}

static inline std::vector<std::string> castdas_trace_split(const std::string& text, char delim) {
  std::vector<std::string> out;
  std::string item;
  std::stringstream ss(text);
  while (std::getline(ss, item, delim)) out.push_back(item);
  return out;
}

static inline uint64_t castdas_trace_u64(const std::string& text) {
  return strtoull(text.c_str(), NULL, 10);
}

static inline uint32_t castdas_trace_u32(const std::string& text) {
  return (uint32_t) strtoul(text.c_str(), NULL, 10);
}

static inline void castdas_trace_load_locked() {
  if (castdas_trace_loaded()) return;
  castdas_trace_rows().clear();
  castdas_trace_rows().resize(THREAD_CNT);
  castdas_trace_init_offsets().clear();
  castdas_trace_init_offsets().resize(THREAD_CNT, 0);
  std::ifstream input(CASTDAS_TRACE_PATH);
  if (!input.is_open()) {
    std::cerr << "cannot open CAST-DAS trace: " << CASTDAS_TRACE_PATH << std::endl;
    abort();
  }
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::vector<std::string> fields = castdas_trace_split(line, '\t');
    if (fields.size() < 7) continue;
    uint64_t worker_id = castdas_trace_u64(fields[0]);
    if (worker_id >= castdas_trace_rows().size()) continue;
    CastdasTraceTxn txn;
    txn.valid = true;
    txn.is_agent = fields[1] == "agent";
    txn.pre_delay_ms = castdas_trace_u32(fields[3]);
    txn.retry_delay_ms = castdas_trace_u32(fields[4]);
    txn.commit_delay_ms = castdas_trace_u32(fields[5]);
    if (fields.size() >= 8) txn.native_payload = fields[7];
    std::vector<std::string> ops = castdas_trace_split(fields[6], ',');
    for (size_t i = 0; i < ops.size(); ++i) {
      if (ops[i].size() < 3 || ops[i][1] != ':') continue;
      CastdasTraceOp op;
      op.is_write = ops[i][0] == 'W';
      op.key = castdas_trace_u64(ops[i].substr(2));
      txn.ops.push_back(op);
    }
    if (txn.ops.empty()) continue;
    castdas_trace_rows()[worker_id].push_back(txn);
  }
  for (uint64_t tid = 0; tid < castdas_trace_rows().size(); ++tid) {
    if (castdas_trace_rows()[tid].empty()) {
      std::cerr << "missing CAST-DAS trace rows for thread " << tid << std::endl;
      abort();
    }
  }
  castdas_trace_loaded() = true;
}

static inline void castdas_trace_ensure_loaded() {
  if (castdas_trace_loaded()) return;
  pthread_mutex_lock(&castdas_trace_mutex());
  castdas_trace_load_locked();
  pthread_mutex_unlock(&castdas_trace_mutex());
}

static inline uint64_t castdas_trace_next_index(uint64_t tid) {
  castdas_trace_ensure_loaded();
  pthread_mutex_lock(&castdas_trace_mutex());
  uint64_t index = castdas_trace_init_offsets()[tid]++;
  pthread_mutex_unlock(&castdas_trace_mutex());
  return index;
}

static inline const CastdasTraceTxn& castdas_trace_get(uint64_t tid, uint64_t index) {
  castdas_trace_ensure_loaded();
  const std::vector<CastdasTraceTxn>& rows = castdas_trace_rows()[tid];
  return rows[index % rows.size()];
}

static inline uint32_t castdas_trace_query_pre_delay_ms(base_query* query) {
#if CASTDAS_TRACE_REPLAY
  if (query == NULL || !query->castdas_trace_is_agent) return 0;
  return query->castdas_trace_pre_delay_ms;
#else
  (void) query;
  return 0;
#endif
}

static inline uint32_t castdas_trace_query_retry_delay_ms(base_query* query) {
#if CASTDAS_TRACE_REPLAY
  if (query == NULL || !query->castdas_trace_is_agent) return 0;
  return query->castdas_trace_retry_delay_ms;
#else
  (void) query;
  return 0;
#endif
}

static inline uint32_t castdas_trace_query_commit_delay_ms(base_query* query) {
#if CASTDAS_TRACE_REPLAY
  if (query == NULL || !query->castdas_trace_is_agent) return 0;
  return query->castdas_trace_commit_delay_ms;
#else
  (void) query;
  return 0;
#endif
}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    if not repo.exists():
        raise SystemExit(f"repo does not exist: {repo}")
    patch_repo(repo)
    return 0


def patch_repo(repo: Path) -> None:
    patch_config(repo / "config-std.h")
    write_if_changed(repo / "system" / "castdas_trace.h", TRACE_HEADER)
    patch_query_h(repo / "system" / "query.h")
    patch_ycsb_query(repo / "benchmarks" / "ycsb_query.cpp")
    patch_ycsb_txn(repo / "benchmarks" / "ycsb_txn.cpp")
    patch_tpcc_query(repo / "benchmarks" / "tpcc_query.cpp")
    patch_tpcc_txn(repo / "benchmarks" / "tpcc_txn.cpp")
    patch_thread(repo / "system" / "thread.cpp")


def patch_config(path: Path) -> None:
    text = path.read_text()
    if "CASTDAS_TRACE_REPLAY" in text:
        return
    marker = "/***********************************************/\n// Benchmark"
    if marker not in text:
        raise SystemExit(f"cannot find Benchmark marker in {path}")
    path.write_text(text.replace(marker, TRACE_CONFIG_BLOCK + "\n" + marker, 1))


def patch_query_h(path: Path) -> None:
    text = path.read_text()
    if "castdas_trace_pre_delay_ms" in text:
        return
    needle = "    bool rerun;\n"
    if needle not in text:
        needle = "\tbool rerun;\n"
    if needle not in text:
        raise SystemExit(f"cannot find base_query rerun field in {path}")
    repl = (
        needle
        + "#if CASTDAS_TRACE_REPLAY\n"
        + "    bool castdas_trace_is_agent;\n"
        + "    uint32_t castdas_trace_pre_delay_ms;\n"
        + "    uint32_t castdas_trace_retry_delay_ms;\n"
        + "    uint32_t castdas_trace_commit_delay_ms;\n"
        + "#endif\n"
    )
    path.write_text(text.replace(needle, repl, 1))


def patch_ycsb_query(path: Path) -> None:
    text = path.read_text()
    if '"castdas_trace.h"' not in text:
        text = text.replace('#include "table.h"\n', '#include "table.h"\n#include "castdas_trace.h"\n', 1)
    if "castdas_trace_apply_to_ycsb_query" not in text:
        helper = r'''
#if CASTDAS_TRACE_REPLAY
static void castdas_trace_apply_to_ycsb_query(ycsb_query *query, uint64_t thd_id) {
    const CastdasTraceTxn& trace = castdas_trace_get(thd_id, castdas_trace_next_index(thd_id));
    query->castdas_trace_is_agent = trace.is_agent;
    query->castdas_trace_pre_delay_ms = trace.pre_delay_ms;
    query->castdas_trace_retry_delay_ms = trace.retry_delay_ms;
    query->castdas_trace_commit_delay_ms = trace.commit_delay_ms;
    query->part_num = 1;
    query->part_to_access[0] = 0;
    query->request_cnt = trace.ops.size();
    query->local_req_per_query = query->request_cnt;
    query->is_long = false;
    assert(query->request_cnt > 0);
    assert(query->request_cnt <= CASTDAS_TRACE_MAX_OPS);
    for (uint64_t i = 0; i < query->request_cnt; ++i) {
        query->requests[i].rtype = trace.ops[i].is_write ? WR : RD;
        query->requests[i].key = trace.ops[i].key;
        query->requests[i].value = 0;
        query->requests[i].scan_len = 0;
    }
}
#endif

'''
        anchor = "double ycsb_query::denom = 0;\n\n"
        if anchor not in text:
            raise SystemExit(f"cannot find ycsb_query denom anchor in {path}")
        text = text.replace(anchor, anchor + helper, 1)
    if "castdas_trace_apply_to_ycsb_query(this, thd_id)" not in text:
        trace_block = (
            "#if CASTDAS_TRACE_REPLAY\n"
            "    castdas_trace_apply_to_ycsb_query(this, thd_id);\n"
            "    return;\n"
            "#endif\n"
        )
        normal_path = re.compile(
            r"(?P<zeta>[ \t]*zeta_2_theta = zeta\(2, g_zipf_theta\);\n)"
            r"(?P<tail>[ \t]*assert\(the_n != 0\);\n"
            r"[ \t]*assert\(denom != 0\);\n"
            r"[ \t]*gen_requests\(thd_id, h_wl\);)"
        )
        text, count = normal_path.subn(r"\g<zeta>" + trace_block + r"\g<tail>", text, count=1)
        if count != 1:
            raise SystemExit(f"cannot find normal ycsb query generation block in {path}")
    path.write_text(text)


def patch_ycsb_txn(path: Path) -> None:
    text = path.read_text()
    if '"castdas_agentic.h"' not in text:
        text = text.replace('#include "query.h"\n', '#include "query.h"\n#include "castdas_agentic.h"\n', 1)
    if '"castdas_trace.h"' not in text:
        text = text.replace('#include "castdas_agentic.h"\n', '#include "castdas_agentic.h"\n#include "castdas_trace.h"\n', 1)
    if "castdas_trace_query_commit_delay_ms(m_query)" not in text:
        def commit_delay_block(indent: str) -> str:
            inner = indent + "    "
            deeper = indent + "        "
            return (
                "#if CASTDAS_TRACE_REPLAY\n"
                f"{indent}if (rc == RCOK) {{\n"
                f"{inner}uint32_t castdas_commit_delay_ms = castdas_trace_query_commit_delay_ms(m_query);\n"
                f"{inner}castdas_sleep_ms(castdas_commit_delay_ms);\n"
                f"{inner}if (CASTDAS_RECORD_AGENT_STATS && castdas_commit_delay_ms > 0)\n"
                f"{deeper}INC_STATS(get_thd_id(), castdas_agent_delay_ns,\n"
                f"{deeper}    ((uint64_t) castdas_commit_delay_ms) * 1000000UL);\n"
                f"{indent}}}\n"
                "#endif\n"
            )

        finish_pattern = re.compile(r"(?m)^(?P<label>[ \t]*final:\n)(?P<indent>[ \t]*)rc\s*=\s*finish\(rc\);\s*$")

        def replace_finish(match: re.Match[str]) -> str:
            indent = match.group("indent")
            return match.group("label") + commit_delay_block(indent) + f"{indent}rc = finish(rc);"

        text, count = finish_pattern.subn(replace_finish, text, count=1)
        if count == 0:
            return_pattern = re.compile(r"(?m)^(?P<label>[ \t]*final:\n)(?P<indent>[ \t]*)return\s+rc;\s*$")

            def replace_return(match: re.Match[str]) -> str:
                indent = match.group("indent")
                return match.group("label") + commit_delay_block(indent) + f"{indent}return rc;"

            text, count = return_pattern.subn(replace_return, text, count=1)
        if count != 1:
            raise SystemExit(f"cannot find ycsb final block in {path}")
    path.write_text(text)


def patch_tpcc_query(path: Path) -> None:
    text = path.read_text()
    if '"castdas_trace.h"' not in text:
        text = text.replace('#include "table.h"\n', '#include "table.h"\n#include "castdas_trace.h"\n', 1)
    if "castdas_trace_apply_to_tpcc_query" not in text:
        helper = r'''
#if CASTDAS_TRACE_REPLAY
static std::string castdas_trace_payload_value(const std::string& payload, const std::string& key) {
    std::vector<std::string> fields = castdas_trace_split(payload, ';');
    std::string prefix = key + "=";
    for (size_t i = 0; i < fields.size(); ++i) {
        if (fields[i].compare(0, prefix.size(), prefix) == 0)
            return fields[i].substr(prefix.size());
    }
    return "";
}

static uint64_t castdas_trace_payload_u64(
    const std::string& payload, const std::string& key, uint64_t fallback) {
    std::string value = castdas_trace_payload_value(payload, key);
    if (value.empty())
        return fallback;
    uint64_t parsed = castdas_trace_u64(value);
    return parsed == 0 ? fallback : parsed;
}

static void castdas_trace_add_tpcc_part(tpcc_query *query, uint64_t wid) {
    uint64_t part = wh_to_part(wid);
    for (uint64_t i = 0; i < query->part_num; ++i) {
        if (query->part_to_access[i] == part)
            return;
    }
    if (query->part_num < g_part_cnt)
        query->part_to_access[query->part_num++] = part;
}

static bool castdas_trace_apply_to_tpcc_query(tpcc_query *query, uint64_t thd_id) {
    const CastdasTraceTxn& trace = castdas_trace_get(thd_id, castdas_trace_next_index(thd_id));
    query->castdas_trace_is_agent = trace.is_agent;
    query->castdas_trace_pre_delay_ms = trace.pre_delay_ms;
    query->castdas_trace_retry_delay_ms = trace.retry_delay_ms;
    query->castdas_trace_commit_delay_ms = trace.commit_delay_ms;
    std::string payload = trace.native_payload;
    std::string task = castdas_trace_payload_value(payload, "task");
    if (payload.empty() || task.empty())
        return false;

    query->w_id = castdas_trace_payload_u64(payload, "w", 1);
    query->d_id = castdas_trace_payload_u64(payload, "d", 1);
    query->c_id = castdas_trace_payload_u64(payload, "c", 1);
    query->part_num = 0;
    castdas_trace_add_tpcc_part(query, query->w_id);

    if (task == "payment") {
        query->type = TPCC_PAYMENT;
        query->d_w_id = castdas_trace_payload_u64(payload, "dw", query->w_id);
        query->c_w_id = castdas_trace_payload_u64(payload, "cw", query->w_id);
        query->c_d_id = castdas_trace_payload_u64(payload, "cd", query->d_id);
        query->h_amount = (double) castdas_trace_payload_u64(payload, "amount", 1);
        query->by_last_name = false;
        castdas_trace_add_tpcc_part(query, query->d_w_id);
        castdas_trace_add_tpcc_part(query, query->c_w_id);
        return true;
    }

    if (task != "new_order")
        return false;

    query->type = TPCC_NEW_ORDER;
    query->rbk = 50;
    query->remote = false;
    query->o_entry_d = 2013;
    std::vector<std::string> items = castdas_trace_split(castdas_trace_payload_value(payload, "items"), '|');
    if (items.empty() || (items.size() == 1 && items[0].empty()))
        items = std::vector<std::string>(1, "1:1:1");
    query->ol_cnt = items.size();
    query->items = (Item_no *) _mm_malloc(sizeof(Item_no) * query->ol_cnt, 64);
    for (uint64_t i = 0; i < query->ol_cnt; ++i) {
        std::vector<std::string> parts = castdas_trace_split(items[i], ':');
        uint64_t item_id = parts.size() > 0 ? castdas_trace_u64(parts[0]) : 1;
        uint64_t supply_w_id = parts.size() > 1 ? castdas_trace_u64(parts[1]) : query->w_id;
        uint64_t quantity = parts.size() > 2 ? castdas_trace_u64(parts[2]) : 1;
        query->items[i].ol_i_id = item_id == 0 ? 1 : item_id;
        query->items[i].ol_supply_w_id = supply_w_id == 0 ? query->w_id : supply_w_id;
        query->items[i].ol_quantity = quantity == 0 ? 1 : quantity;
        if (query->items[i].ol_supply_w_id != query->w_id)
            query->remote = true;
        castdas_trace_add_tpcc_part(query, query->items[i].ol_supply_w_id);
    }
    return true;
}
#endif

'''
        anchor = "void tpcc_query::init(uint64_t thd_id, workload * h_wl) {\n"
        if anchor not in text:
            raise SystemExit(f"cannot find tpcc_query init anchor in {path}")
        text = text.replace(anchor, helper + anchor, 1)
    if "castdas_trace_apply_to_tpcc_query(this, thd_id)" not in text:
        old = (
            "  part_to_access = (uint64_t *)\n"
            "      mem_allocator.alloc(sizeof(uint64_t) * g_part_cnt, thd_id);\n"
        )
        new = old + (
            "#if CASTDAS_TRACE_REPLAY\n"
            "  if (castdas_trace_apply_to_tpcc_query(this, thd_id))\n"
            "    return;\n"
            "#endif\n"
        )
        if old not in text:
            raise SystemExit(f"cannot find tpcc part_to_access allocation in {path}")
        text = text.replace(old, new, 1)
    path.write_text(text)


def patch_tpcc_txn(path: Path) -> None:
    text = path.read_text()
    if '"castdas_agentic.h"' not in text:
        text = text.replace(
            '#include "tpcc_const.h"\n',
            '#include "tpcc_const.h"\n#include "castdas_agentic.h"\n#include "castdas_trace.h"\n',
            1,
        )
    elif '"castdas_trace.h"' not in text:
        text = text.replace('#include "castdas_agentic.h"\n', '#include "castdas_agentic.h"\n#include "castdas_trace.h"\n', 1)
    if "CASTDAS_TPCC_TRACE_COMMIT_DELAY" not in text:
        helper = r'''
#if CASTDAS_TRACE_REPLAY
#define CASTDAS_TPCC_TRACE_COMMIT_DELAY(query) do { \
    if (rc == RCOK) { \
        uint32_t castdas_commit_delay_ms = castdas_trace_query_commit_delay_ms((base_query *) query); \
        castdas_sleep_ms(castdas_commit_delay_ms); \
        if (CASTDAS_RECORD_AGENT_STATS && castdas_commit_delay_ms > 0) \
            INC_STATS(get_thd_id(), castdas_agent_delay_ns, \
                ((uint64_t) castdas_commit_delay_ms) * 1000000UL); \
    } \
} while (0)
#else
#define CASTDAS_TPCC_TRACE_COMMIT_DELAY(query) do { } while (0)
#endif

'''
        anchor = "void tpcc_txn_man::init"
        if anchor not in text:
            raise SystemExit(f"cannot find tpcc txn init anchor in {path}")
        text = text.replace(anchor, helper + anchor, 1)
    if "CASTDAS_TPCC_TRACE_COMMIT_DELAY(query);\n  return finish(rc);" not in text and (
        "CASTDAS_TPCC_TRACE_COMMIT_DELAY(query);\n  return rc;" not in text
    ):
        pattern = re.compile(r"(?m)^([ \t]*assert\( rc == RCOK \);\n)([ \t]*)return (finish\(rc\)|rc);")
        replacements = 0

        def replace_final(match: re.Match[str]) -> str:
            nonlocal replacements
            replacements += 1
            if replacements > 2:
                return match.group(0)
            indent = match.group(2)
            return match.group(1) + f"{indent}CASTDAS_TPCC_TRACE_COMMIT_DELAY(query);\n{indent}return {match.group(3)};"

        text, count = pattern.subn(replace_final, text)
        if replacements < 2:
            raise SystemExit(f"cannot find payment/new-order final return blocks in {path}")
    path.write_text(text)


def patch_thread(path: Path) -> None:
    text = path.read_text()
    if '"castdas_trace.h"' not in text:
        text = text.replace('#include "castdas_agentic.h"\n', '#include "castdas_agentic.h"\n#include "castdas_trace.h"\n', 1)
    if "castdas_trace_query_pre_delay_ms(m_query)" not in text:
        old = (
            "#if CASTDAS_AGENTIC\n"
            "            uint32_t castdas_pre_delay_ms = castdas_agent_pre_exec_delay_ms(\n"
            "                get_thd_id(), m_txn->get_txn_id(), m_txn->abort_cnt);\n"
            "            castdas_sleep_ms(castdas_pre_delay_ms);\n"
            "            if (CASTDAS_RECORD_AGENT_STATS && castdas_pre_delay_ms > 0)\n"
            "                INC_STATS(get_thd_id(), castdas_agent_delay_ns,\n"
            "                    ((uint64_t) castdas_pre_delay_ms) * 1000000UL);\n"
            "#endif\n"
        )
        new = (
            "#if CASTDAS_TRACE_REPLAY\n"
            "            uint32_t castdas_pre_delay_ms = castdas_trace_query_pre_delay_ms(m_query);\n"
            "#elif CASTDAS_AGENTIC\n"
            "            uint32_t castdas_pre_delay_ms = castdas_agent_pre_exec_delay_ms(\n"
            "                get_thd_id(), m_txn->get_txn_id(), m_txn->abort_cnt);\n"
            "#else\n"
            "            uint32_t castdas_pre_delay_ms = 0;\n"
            "#endif\n"
            "#if CASTDAS_TRACE_REPLAY || CASTDAS_AGENTIC\n"
            "            castdas_sleep_ms(castdas_pre_delay_ms);\n"
            "            if (CASTDAS_RECORD_AGENT_STATS && castdas_pre_delay_ms > 0)\n"
            "                INC_STATS(get_thd_id(), castdas_agent_delay_ns,\n"
            "                    ((uint64_t) castdas_pre_delay_ms) * 1000000UL);\n"
            "#endif\n"
        )
        if old not in text:
            raise SystemExit(f"cannot find pre-delay block in {path}")
        text = text.replace(old, new, 1)
    if "castdas_trace_query_retry_delay_ms(m_query)" not in text:
        old = (
            "#if CASTDAS_AGENTIC\n"
            "            uint32_t castdas_retry_delay_ms = castdas_agent_phase_delay_ms(\n"
            "                get_thd_id(), m_txn->get_txn_id(), m_txn->abort_cnt + 1,\n"
            "                CASTDAS_PHASE_RETRY);\n"
            "            castdas_sleep_ms(castdas_retry_delay_ms);\n"
            "            if (CASTDAS_RECORD_AGENT_STATS && castdas_retry_delay_ms > 0)\n"
            "                INC_STATS(get_thd_id(), castdas_agent_delay_ns,\n"
            "                    ((uint64_t) castdas_retry_delay_ms) * 1000000UL);\n"
            "#endif\n"
        )
        new = (
            "#if CASTDAS_TRACE_REPLAY\n"
            "            uint32_t castdas_retry_delay_ms = castdas_trace_query_retry_delay_ms(m_query);\n"
            "#elif CASTDAS_AGENTIC\n"
            "            uint32_t castdas_retry_delay_ms = castdas_agent_phase_delay_ms(\n"
            "                get_thd_id(), m_txn->get_txn_id(), m_txn->abort_cnt + 1,\n"
            "                CASTDAS_PHASE_RETRY);\n"
            "#else\n"
            "            uint32_t castdas_retry_delay_ms = 0;\n"
            "#endif\n"
            "#if CASTDAS_TRACE_REPLAY || CASTDAS_AGENTIC\n"
            "            castdas_sleep_ms(castdas_retry_delay_ms);\n"
            "            if (CASTDAS_RECORD_AGENT_STATS && castdas_retry_delay_ms > 0)\n"
            "                INC_STATS(get_thd_id(), castdas_agent_delay_ns,\n"
            "                    ((uint64_t) castdas_retry_delay_ms) * 1000000UL);\n"
            "#endif\n"
        )
        if old not in text:
            raise SystemExit(f"cannot find retry-delay block in {path}")
        text = text.replace(old, new, 1)
    text = patch_trace_replay_thread_termination(text, path)
    path.write_text(text)


def patch_trace_replay_thread_termination(text: str, path: Path) -> str:
    if "CASTDAS_TRACE_REPLAY\n\t\t\treturn FINISH;" in text or "CASTDAS_TRACE_REPLAY\n                return FINISH;" in text:
        return text

    bamboo_pattern = re.compile(
        r"(?m)^([ \t]*assert\(txn_cnt == MAX_TXN_PER_PART\);\n)"
        r"(?P<indent>[ \t]*)if\( !ATOM_CAS\(_wl->sim_done, false, true\) \)\n"
        r"[ \t]*assert\( _wl->sim_done\);"
    )

    def replace_bamboo(match: re.Match[str]) -> str:
        indent = match.group("indent")
        return (
            match.group(1)
            + "#if CASTDAS_TRACE_REPLAY\n"
            + f"{indent}return FINISH;\n"
            + "#else\n"
            + f"{indent}if( !ATOM_CAS(_wl->sim_done, false, true) )\n"
            + f"{indent}\tassert( _wl->sim_done);\n"
            + "#endif"
        )

    text, count = bamboo_pattern.subn(replace_bamboo, text, count=1)
    if count == 1:
        return text

    polaris_line = "\t\t\t\t_wl->sim_done.store(true, std::memory_order_release);\n"
    if polaris_line in text:
        return text.replace(
            polaris_line,
            "#if CASTDAS_TRACE_REPLAY\n"
            "\t\t\t\treturn FINISH;\n"
            "#else\n"
            "\t\t\t\t_wl->sim_done.store(true, std::memory_order_release);\n"
            "#endif\n",
            1,
        )

    polaris_spaces = "                _wl->sim_done.store(true, std::memory_order_release);\n"
    if polaris_spaces in text:
        return text.replace(
            polaris_spaces,
            "#if CASTDAS_TRACE_REPLAY\n"
            "                return FINISH;\n"
            "#else\n"
            "                _wl->sim_done.store(true, std::memory_order_release);\n"
            "#endif\n",
            1,
        )

    raise SystemExit(f"cannot patch trace replay termination in {path}")


def write_if_changed(path: Path, text: str) -> None:
    if path.exists() and path.read_text() == text:
        return
    path.write_text(text)


if __name__ == "__main__":
    raise SystemExit(main())
