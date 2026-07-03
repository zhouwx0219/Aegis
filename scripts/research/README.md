# Research Scripts

These scripts are retained for paper reproduction and historical experiment
workflows.  They are not the delivery entry points.

Use the delivery scripts first:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\smoke.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_benchmark.ps1
```

Research scripts:

```text
run_ycsb_compare.ps1       Historical YCSB comparison matrix.
run_tpcc_compare.ps1       Historical TPC-C comparison matrix.
run_atcc_ablation.ps1      ATCC ablation experiment wrapper.
summarize_retry_results.py Retry JSON to CSV summarizer.
```

