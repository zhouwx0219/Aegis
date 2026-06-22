#include "global.h"
#include "helper.h"
#include "txn.h"
#include "row.h"
#include "manager.h"
#include "row_hybrid.h"

#if CC_ALG == HYBRID

RC
txn_man::validate_hybrid() {
	// Sort rows by table/key before latching, matching DBx1000's OCC style and
	// avoiding validation-time deadlock.
	for (int i = row_cnt - 1; i > 0; i--) {
		for (int j = 0; j < i; j++) {
			int tabcmp = strcmp(accesses[j]->orig_row->get_table_name(),
					accesses[j + 1]->orig_row->get_table_name());
			if (tabcmp > 0 || (tabcmp == 0 &&
					accesses[j]->orig_row->get_primary_key() >
					accesses[j + 1]->orig_row->get_primary_key())) {
				Access * tmp = accesses[j];
				accesses[j] = accesses[j + 1];
				accesses[j + 1] = tmp;
			}
		}
	}

	bool ok = true;
	int lock_cnt = 0;
	for (int i = 0; i < row_cnt && ok; i++) {
		accesses[i]->orig_row->manager->latch();
		lock_cnt++;
		ok = accesses[i]->orig_row->manager->validate(start_ts, accesses[i]->type == WR);
	}

	if (ok) {
		end_ts = glob_manager->get_ts(get_thd_id());
		cleanup(RCOK);
	} else {
		cleanup(Abort);
	}

	for (int i = 0; i < lock_cnt; i++)
		accesses[i]->orig_row->manager->release();
	return ok ? RCOK : Abort;
}

#endif
