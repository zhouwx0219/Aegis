# External CC Agentic Workload Adapter

This folder contains repeatable adapters for running DBx1000-family external
systems, such as Polaris and Bamboo, with CAST-DAS-style agent/background
client behavior.

The adapter intentionally does not modify CAST-DAS runtime concurrency control.
It patches a disposable external repository checkout and keeps the comparison as
a native DB benchmark baseline.

## Scope

The first adapter ports the CAST-DAS experiment shape into DBx1000-family
systems:

- client split with `CASTDAS_AGENT_RATIO`;
- deterministic agent reasoning delay for explore/refine/commit/retry phases;
- CAST-DAS paper-style YCSB and TPC-C conflict profiles mapped to DBx1000
  parameters;
- parser output in CSV form for later merge with CAST-DAS matrix results.

This is a benchmark adapter, not a semantic port of CAST-DAS transaction
runtime or ATCC.

## Typical Use

From Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\external_cc\run_node1_external_cc.ps1 `
  -RemoteRoot /home/chenht/castdas_external_cc `
  -Systems bamboo,polaris `
  -Workloads ycsb,tpcc `
  -Levels low,medium,high `
  -ClientCounts 8,16,24,32,40,48 `
  -AgentRatios 1.0,0.8 `
  -Duration 5 `
  -Output results\external_cc_agentic.csv
```

The remote root must contain external checkouts named `Bamboo-Public` and/or
`polaris`. The wrapper creates patched disposable copies with `_castdas`
suffixes and copies the CSV back to the local workspace.

