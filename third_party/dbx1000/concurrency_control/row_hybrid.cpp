#include "txn.h"
#include "row.h"
#include "row_hybrid.h"
#include "mem_alloc.h"

#if CC_ALG == HYBRID

void
Row_hybrid::init(row_t * row) {
	_row = row;
	int part_id = row->get_part_id();
	_latch = (pthread_mutex_t *)
		mem_allocator.alloc(sizeof(pthread_mutex_t), part_id);
	pthread_mutex_init(_latch, NULL);
	wts = 0;
}

RC
Row_hybrid::access(txn_man * txn, TsType type) {
	RC rc = RCOK;
	pthread_mutex_lock(_latch);
	if (type == R_REQ) {
		// HYBRID keeps a snapshot copy, like OCC, but read validation can be
		// relaxed at commit for CSI-style read semantics.
		txn->cur_row->copy(_row);
		rc = RCOK;
	} else {
		assert(false);
	}
	pthread_mutex_unlock(_latch);
	return rc;
}

void
Row_hybrid::latch() {
	pthread_mutex_lock(_latch);
}

bool
Row_hybrid::validate(uint64_t ts, bool in_write_set) {
	// Native DBx1000 does not carry ASTRA WriteIntent metadata. We therefore
	// model HYBRID's relaxed validation boundary here: reads are snapshot reads,
	// and writes are allowed to pass validation so semantic resolution can occur
	// in the commit layer / ASTRA runner. Strict workloads should use DBx OCC,
	// Silo, or TicToc baselines.
	(void)ts;
	(void)in_write_set;
	return true;
}

void
Row_hybrid::write(row_t * data, uint64_t ts) {
	_row->copy(data);
	wts = ts > wts ? ts : wts + 1;
}

void
Row_hybrid::release() {
	pthread_mutex_unlock(_latch);
}

#endif
