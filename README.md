# Aegis

Aegis is a compact research prototype for agent-side transaction semantics  
and concurrency control over a versioned KV backend. The storage layer is kept  
small on purpose: it exposes object reads, version reads, and conditional batch  
writes. Transaction boundaries, read/write sets, conflict validation, retry  
logic, and concurrency-control selection stay in the Python agent runtime.

