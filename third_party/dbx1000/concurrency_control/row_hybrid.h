#ifndef ROW_HYBRID_H
#define ROW_HYBRID_H

#include "global.h"

class row_t;
class txn_man;

// Row manager for DBx1000's ASTRA/HYBRID compile-time CC_ALG.
//
// The native DBx1000 path has no write-intent metadata, so this manager exposes
// versioned row access plus a merge-style write primitive. Full intent-aware
// constrained DELTA / reselect behavior is exercised by benchmarks/astra_vita.cpp
// against ASTRA's C++ commit kernel.
class Row_hybrid {
public:
	void 				init(row_t * row);
	RC 					access(txn_man * txn, TsType type);
	void 				latch();
	bool				validate(uint64_t ts, bool in_write_set);
	void				write(row_t * data, uint64_t ts);
	void 				release();
	uint64_t			version() const { return wts; }
private:
	pthread_mutex_t * 	_latch;
	row_t * 			_row;
	ts_t 				wts;
};

#endif
