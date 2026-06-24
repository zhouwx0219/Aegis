#pragma once

#include "global.h"
#include "helper.h"
#include "index_base.h"

class BucketNode {
 public:
  explicit BucketNode(idx_key_t value) { init(value); }
  void init(idx_key_t value) {
    key = value;
    next = NULL;
    items = NULL;
  }
  idx_key_t key;
  BucketNode* next;
  itemid_t* items;
};

class BucketHeader {
 public:
  void init();
  void insert_item(idx_key_t key, itemid_t* item, int part_id);
  void read_item(idx_key_t key, itemid_t*& item, const char* table_name);
  bool try_read_item(idx_key_t key, itemid_t*& item);
#ifdef ASTRA_DBX1000_EMBEDDED
  void clear();
#endif
  BucketNode* first_node;
  uint64_t node_cnt;
  bool locked;
};

class IndexHash : public index_base {
 public:
  IndexHash()
      : _buckets(NULL), _bucket_cnt(0), _bucket_cnt_per_part(0), _part_cnt(0) {}
#ifdef ASTRA_DBX1000_EMBEDDED
  ~IndexHash();
#endif
  using index_base::init;
  RC init(uint64_t bucket_cnt, int part_cnt);
  RC init(int part_cnt, table_t* table, uint64_t bucket_cnt);
  bool index_exist(idx_key_t key);
  RC index_insert(idx_key_t key, itemid_t* item, int part_id = -1);
  RC index_read(idx_key_t key, itemid_t*& item, int part_id = -1);
  RC index_read(idx_key_t key, itemid_t*& item, int part_id, int thd_id);
  RC index_try_read(idx_key_t key, itemid_t*& item, int part_id = 0);

 private:
  void get_latch(BucketHeader* bucket);
  void release_latch(BucketHeader* bucket);
  uint64_t hash(idx_key_t key) { return key % _bucket_cnt_per_part; }

  BucketHeader** _buckets;
  uint64_t _bucket_cnt;
  uint64_t _bucket_cnt_per_part;
  uint64_t _part_cnt;
};
