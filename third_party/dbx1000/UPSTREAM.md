# Upstream

This directory vendors [yxymit/DBx1000](https://github.com/yxymit/DBx1000)
at commit `5f172a99bd57bed29745df81f72e93f91292bb31`.

ASTRA adds the `ASTRA_DBX1000_EMBEDDED` build mode and safe hash-index lookup
so DBx1000's `Catalog`, `table_t`, `row_t`, and `IndexHash` can be embedded as a
versioned KV engine without constructing DBx1000's compile-time row CC manager.

Local changes are limited to the embedding boundary:

- initialize and release catalog, table, row, and hash-index resources safely;
- allow non-asserting hash lookups and hash collisions;
- omit DBx1000's compile-time row CC manager in embedded mode; and
- accept both LF and CRLF benchmark schema files; and
- add modern GCC and non-PIE compatibility flags to the standalone build; and
- preserve the original standalone benchmark sources for provenance.

The ASTRA adapter is implemented in `core/storage/dbx1000_versioned_kv.cpp`.
Agent-facing YCSB and TPC-C conversions are separate Python workloads under
`agent/workloads`; they do not modify the upstream benchmark protocols.
