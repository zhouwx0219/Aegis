#!/usr/bin/env python3
"""Patch DBx1000-family repositories with CAST-DAS agentic benchmark hooks.

The patch is designed for disposable external checkouts, not for vendoring
third-party systems into CAST-DAS. It adds compile-time knobs and lightweight
agent/background delay simulation while preserving the external system's native
CC implementations.
"""

import argparse
import re
from pathlib import Path


CONFIG_BLOCK = r"""

/***********************************************/
// CAST-DAS external benchmark adapter
/***********************************************/
#define CASTDAS_AGENTIC              false
#define CASTDAS_AGENT_RATIO          1.0
// 0=low, 1=medium, 2=high
#define CASTDAS_AGENT_LEVEL          0
#define CASTDAS_REASONING_SCALE      1.0
#define CASTDAS_RECORD_AGENT_STATS   true
"""


AGENTIC_HEADER = r"""#pragma once

#include <stdint.h>
#include <unistd.h>
#include "config.h"

#ifndef CASTDAS_AGENTIC
#define CASTDAS_AGENTIC false
#endif
#ifndef CASTDAS_AGENT_RATIO
#define CASTDAS_AGENT_RATIO 1.0
#endif
#ifndef CASTDAS_AGENT_LEVEL
#define CASTDAS_AGENT_LEVEL 0
#endif
#ifndef CASTDAS_REASONING_SCALE
#define CASTDAS_REASONING_SCALE 1.0
#endif
#ifndef CASTDAS_RECORD_AGENT_STATS
#define CASTDAS_RECORD_AGENT_STATS true
#endif

enum CastdasPhase {
  CASTDAS_PHASE_EXPLORE = 0,
  CASTDAS_PHASE_REFINE = 1,
  CASTDAS_PHASE_COMMIT = 2,
  CASTDAS_PHASE_RETRY = 3
};

static inline uint64_t castdas_agent_thread_count() {
  double raw = ((double) THREAD_CNT) * ((double) CASTDAS_AGENT_RATIO);
  uint64_t count = (uint64_t) (raw + 0.5);
  if (CASTDAS_AGENT_RATIO > 0 && count == 0)
    count = 1;
  if (count > THREAD_CNT)
    count = THREAD_CNT;
  return count;
}

static inline bool castdas_is_agent_thread(uint64_t tid) {
  if (!CASTDAS_AGENTIC)
    return false;
  return tid < castdas_agent_thread_count();
}

static inline uint64_t castdas_hash64(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}

static inline uint32_t castdas_range_pick(
    uint32_t low, uint32_t high, uint64_t tid, uint64_t txn_id,
    uint64_t attempt, CastdasPhase phase) {
  if (high <= low)
    return low;
  uint64_t seed = txn_id ^ (tid << 32) ^ (attempt << 16) ^ (uint64_t) phase;
  uint64_t value = castdas_hash64(seed);
  return low + (uint32_t) (value % (high - low + 1));
}

static inline uint32_t castdas_base_delay_ms(
    CastdasPhase phase, uint64_t tid, uint64_t txn_id, uint64_t attempt) {
  uint32_t low = 0;
  uint32_t high = 0;
  int level = CASTDAS_AGENT_LEVEL;
  if (phase == CASTDAS_PHASE_RETRY) {
    if (level <= 0) {
      low = 2; high = 5;
    } else if (level == 1) {
      low = 20; high = 40;
    } else {
      low = 60; high = 120;
    }
  } else if (level <= 0) {
    if (phase == CASTDAS_PHASE_COMMIT) {
      low = 0; high = 1;
    } else {
      low = 1; high = 3;
    }
  } else if (level == 1) {
    if (phase == CASTDAS_PHASE_COMMIT) {
      low = 4; high = 8;
    } else {
      low = 8; high = 16;
    }
  } else {
    if (phase == CASTDAS_PHASE_COMMIT) {
      low = 10; high = 25;
    } else {
      low = 25; high = 50;
    }
  }
  uint32_t picked = castdas_range_pick(low, high, tid, txn_id, attempt, phase);
  double scaled = ((double) picked) * ((double) CASTDAS_REASONING_SCALE);
  if (scaled <= 0.0)
    return 0;
  return (uint32_t) (scaled + 0.5);
}

static inline uint32_t castdas_agent_phase_delay_ms(
    uint64_t tid, uint64_t txn_id, uint64_t attempt, CastdasPhase phase) {
  if (!castdas_is_agent_thread(tid))
    return 0;
  if (phase == CASTDAS_PHASE_RETRY && attempt == 0)
    return 0;
  return castdas_base_delay_ms(phase, tid, txn_id, attempt);
}

static inline uint32_t castdas_agent_pre_exec_delay_ms(
    uint64_t tid, uint64_t txn_id, uint64_t attempt) {
  return castdas_agent_phase_delay_ms(tid, txn_id, attempt, CASTDAS_PHASE_EXPLORE)
       + castdas_agent_phase_delay_ms(tid, txn_id, attempt, CASTDAS_PHASE_REFINE);
}

static inline void castdas_sleep_ms(uint32_t delay_ms) {
  if (delay_ms > 0)
    usleep((useconds_t) delay_ms * 1000);
}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    parser.add_argument(
        "--compat",
        action="store_true",
        help="Apply GCC 4.8 compatibility fixes used on node1.",
    )
    parser.add_argument(
        "--compiler",
        default="",
        help="Optional C++ compiler path to write into the external Makefile.",
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    if not repo.exists():
        raise SystemExit(f"repo does not exist: {repo}")
    patch_repo(repo, compat=args.compat, compiler=args.compiler)
    return 0


def patch_repo(repo: Path, *, compat: bool, compiler: str = "") -> None:
    normalize_schema_line_endings(repo)
    patch_config(repo / "config-std.h")
    write_if_changed(repo / "system" / "castdas_agentic.h", AGENTIC_HEADER)
    patch_stats(repo / "system" / "stats.h")
    patch_stats_cpp(repo / "system" / "stats.cpp")
    patch_thread(repo / "system" / "thread.cpp")
    patch_txn(repo / "system" / "txn.cpp")
    patch_makefile(repo / "Makefile", compat=compat, compiler=compiler)
    if compat:
        patch_static_asserts(repo / "config-std.h")


def normalize_schema_line_endings(repo: Path) -> None:
    for path in (repo / "benchmarks").glob("*_schema.txt"):
        data = path.read_bytes()
        normalized = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        if normalized != data:
            path.write_bytes(normalized)


def patch_config(path: Path) -> None:
    text = path.read_text()
    if "CAST-DAS external benchmark adapter" in text:
        return
    marker = "/***********************************************/\n// Benchmark"
    if marker not in text:
        raise SystemExit(f"cannot find Benchmark marker in {path}")
    text = text.replace(marker, CONFIG_BLOCK + "\n" + marker, 1)
    path.write_text(text)


def patch_stats(path: Path) -> None:
    text = path.read_text()
    if "castdas_agent_txn_cnt" in text:
        return
    needle = (
        "  y(uint64_t, lock_acquire_cnt) y(uint64_t, lock_directly_cnt) \\\n"
        "  TMP_METRICS(x, y)"
    )
    repl = (
        "  y(uint64_t, lock_acquire_cnt) y(uint64_t, lock_directly_cnt) \\\n"
        "  y(uint64_t, castdas_agent_txn_cnt) y(uint64_t, castdas_agent_abort_cnt) \\\n"
        "  y(uint64_t, castdas_background_txn_cnt) y(uint64_t, castdas_background_abort_cnt) \\\n"
        "  y(uint64_t, castdas_agent_delay_ns) \\\n"
        "  TMP_METRICS(x, y)"
    )
    if needle not in text:
        raise SystemExit(f"cannot patch ALL_METRICS in {path}")
    path.write_text(text.replace(needle, repl, 1))


def patch_stats_cpp(path: Path) -> None:
    text = path.read_text()
    if "CASTDAS stdout latency distribution" in text:
        return
    old = (
        "void Stats::print_lat_distr() {\n"
        "\tFILE * outf;\n"
        "\tif (output_file != NULL) {\n"
    )
    old_spaces = (
        "void Stats::print_lat_distr() {\n"
        "  FILE * outf;\n"
        "  if (output_file != NULL) {\n"
    )
    new = (
        "void Stats::print_lat_distr() {\n"
        "\t// CASTDAS stdout latency distribution for external trace adapters.\n"
        "\tfor (UInt32 tid = 0; tid < g_thread_cnt; tid ++) {\n"
        "\t\tprintf(\"[all_debug1 thd=%d] \", tid);\n"
        "\t\tfor (uint32_t tnum = 0; tnum < _stats[tid]->txn_cnt; tnum ++)\n"
        "\t\t\tprintf(\"%ld,\", _stats[tid]->all_debug1[tnum]);\n"
        "\t\tprintf(\"\\n\");\n"
        "\t}\n"
        "\tFILE * outf;\n"
        "\tif (output_file != NULL) {\n"
    )
    if old in text:
        path.write_text(text.replace(old, new, 1))
        return
    if old_spaces in text:
        path.write_text(text.replace(old_spaces, new, 1))
        return
    if old not in text:
        if "print_tail_latency" in text and "latency_record" in text:
            return
        raise SystemExit(f"cannot find print_lat_distr block in {path}")


def patch_thread(path: Path) -> None:
    text = path.read_text()
    if '"castdas_agentic.h"' not in text:
        text = text.replace('#include "test.h"\n', '#include "test.h"\n#include "castdas_agentic.h"\n', 1)

    if "castdas_agent_pre_exec_delay_ms" not in text:
        needle = "\t\tif (rc == RCOK)\n\t\t{\n"
        repl = (
            "\t\tif (rc == RCOK)\n\t\t{\n"
            "#if CASTDAS_AGENTIC\n"
            "            uint32_t castdas_pre_delay_ms = castdas_agent_pre_exec_delay_ms(\n"
            "                get_thd_id(), m_txn->get_txn_id(), m_txn->abort_cnt);\n"
            "            castdas_sleep_ms(castdas_pre_delay_ms);\n"
            "            if (CASTDAS_RECORD_AGENT_STATS && castdas_pre_delay_ms > 0)\n"
            "                INC_STATS(get_thd_id(), castdas_agent_delay_ns,\n"
            "                    ((uint64_t) castdas_pre_delay_ms) * 1000000UL);\n"
            "#endif\n"
        )
        if needle not in text:
            raise SystemExit(f"cannot find transaction execution block in {path}")
        text = text.replace(needle, repl, 1)

    if "castdas_agent_txn_cnt" not in text:
        bad_latency_block = (
            "#if CASTDAS_AGENTIC\n"
            "            bool castdas_latency_distribution_recorded = true;\n"
            "            stats.add_debug(get_thd_id(), timespan, 1);\n"
            "#endif\n"
        )
        text = text.replace(bad_latency_block, "")

        if "CASTDAS latency distribution sample" not in text:
            latency_repl = (
                r"\1"
                "#if CASTDAS_AGENTIC\n"
                "            // CASTDAS latency distribution sample for external trace adapters.\n"
                "            stats.add_debug(get_thd_id(), timespan, 1);\n"
                "#endif\n"
            )
            text, count = re.subn(
                r"([ \t]*INC_STATS\(get_thd_id\(\), commit_latency, timespan\);\n)",
                latency_repl,
                text,
                count=1,
            )
            if count != 1:
                raise SystemExit(f"cannot find commit latency counter in {path}")

        commit_repl = (
            r"\1"
            "#if CASTDAS_AGENTIC\n"
            "            if (CASTDAS_RECORD_AGENT_STATS) {\n"
            "                if (castdas_is_agent_thread(get_thd_id())) {\n"
            "                    INC_STATS(get_thd_id(), castdas_agent_txn_cnt, 1);\n"
            "                } else {\n"
            "                    INC_STATS(get_thd_id(), castdas_background_txn_cnt, 1);\n"
            "                }\n"
            "            }\n"
            "#endif\n"
        )
        text, count = re.subn(
            r"([ \t]*INC_STATS\(get_thd_id\(\), txn_cnt, 1\);\n)",
            commit_repl,
            text,
            count=1,
        )
        if count != 1:
            raise SystemExit(f"cannot find commit counter in {path}")

        abort_repl = (
            r"\1"
            "#if CASTDAS_AGENTIC\n"
            "            if (CASTDAS_RECORD_AGENT_STATS) {\n"
            "                if (castdas_is_agent_thread(get_thd_id())) {\n"
            "                    INC_STATS(get_thd_id(), castdas_agent_abort_cnt, 1);\n"
            "                } else {\n"
            "                    INC_STATS(get_thd_id(), castdas_background_abort_cnt, 1);\n"
            "                }\n"
            "            }\n"
            "#endif\n"
        )
        text, count = re.subn(
            r"([ \t]*INC_STATS\(get_thd_id\(\), abort_cnt, 1\);\n)",
            abort_repl,
            text,
            count=1,
        )
        if count != 1:
            raise SystemExit(f"cannot find abort counter in {path}")

        retry_repl = (
            "#if CASTDAS_AGENTIC\n"
            "            uint32_t castdas_retry_delay_ms = castdas_agent_phase_delay_ms(\n"
            "                get_thd_id(), m_txn->get_txn_id(), m_txn->abort_cnt + 1,\n"
            "                CASTDAS_PHASE_RETRY);\n"
            "            castdas_sleep_ms(castdas_retry_delay_ms);\n"
            "            if (CASTDAS_RECORD_AGENT_STATS && castdas_retry_delay_ms > 0)\n"
            "                INC_STATS(get_thd_id(), castdas_agent_delay_ns,\n"
            "                    ((uint64_t) castdas_retry_delay_ms) * 1000000UL);\n"
            "#endif\n"
            r"\1"
        )
        text, count = re.subn(
            r"([ \t]*m_txn->abort_cnt\+\+;\n)",
            retry_repl,
            text,
            count=1,
        )
        if count != 1:
            raise SystemExit(f"cannot find retry abort counter in {path}")

    path.write_text(text)


def patch_txn(path: Path) -> None:
    text = path.read_text()
    if '"castdas_agentic.h"' not in text:
        text = text.replace('#include "row_bamboo.h"\n', '#include "row_bamboo.h"\n#include "castdas_agentic.h"\n', 1)
    if "castdas_commit_delay_ms" not in text:
        needle = "#if CC_ALG == HSTORE\n    return RCOK;\n#endif\n    uint64_t starttime = get_sys_clock();\n"
        repl = (
            "#if CC_ALG == HSTORE\n    return RCOK;\n#endif\n"
            "#if CASTDAS_AGENTIC\n"
            "    if (rc == RCOK && castdas_is_agent_thread(get_thd_id())) {\n"
            "        uint32_t castdas_commit_delay_ms = castdas_agent_phase_delay_ms(\n"
            "            get_thd_id(), get_txn_id(), abort_cnt, CASTDAS_PHASE_COMMIT);\n"
            "        castdas_sleep_ms(castdas_commit_delay_ms);\n"
            "        if (CASTDAS_RECORD_AGENT_STATS && castdas_commit_delay_ms > 0)\n"
            "            INC_STATS(get_thd_id(), castdas_agent_delay_ns,\n"
            "                ((uint64_t) castdas_commit_delay_ms) * 1000000UL);\n"
            "    }\n"
            "#endif\n"
            "    uint64_t starttime = get_sys_clock();\n"
        )
        if needle not in text:
            raise SystemExit(f"cannot find finish validation block in {path}")
        text = text.replace(needle, repl, 1)
    path.write_text(text)


def patch_makefile(path: Path, *, compat: bool, compiler: str = "") -> None:
    text = path.read_text()
    if compat:
        text = text.replace(" -no-pie", "")
        text = text.replace("-std=c++17", "-std=c++0x")
    if compiler:
        text = re.sub(r"^CC=.*$", f"CC={compiler}", text, count=1, flags=re.MULTILINE)
    path.write_text(text)


def patch_static_asserts(path: Path) -> None:
    text = path.read_text()
    text = re.sub(
        r"static_assert\((SILO_PRIO_NUM_BITS_PRIO_VER\s*\+\s*SILO_PRIO_NUM_BITS_PRIO\s*\\\s*\n\s*\+\s*SILO_PRIO_NUM_BITS_REF_CNT\s*\+\s*SILO_PRIO_NUM_BITS_DATA_VER\s*\+\s*1\s*==\s*64)\s*\);",
        r'static_assert(\1, "silo prio tid width");',
        text,
    )
    text = re.sub(
        r"static_assert\((ARIA_NUM_BITS_BATCH_ID\s*\+\s*ARIA_NUM_BITS_PRIO\s*\\\s*\n\s*\+\s*ARIA_NUM_BITS_TXN_ID\s*==\s*64)\s*\);",
        r'static_assert(\1, "aria tid width");',
        text,
    )
    path.write_text(text)


def write_if_changed(path: Path, text: str) -> None:
    if path.exists() and path.read_text() == text:
        return
    path.write_text(text)


if __name__ == "__main__":
    raise SystemExit(main())
