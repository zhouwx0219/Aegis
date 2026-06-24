#include "global.h"
#include "index_hash.h"
#ifndef ASTRA_DBX1000_EMBEDDED
#include "mem_alloc.h"
#endif
#include "table.h"

RC IndexHash::init(uint64_t bucket_cnt, int part_cnt) {
  assert(part_cnt > 0);
  assert(bucket_cnt >= static_cast<uint64_t>(part_cnt));
  _bucket_cnt = bucket_cnt;
  _bucket_cnt_per_part = bucket_cnt / part_cnt;
  _part_cnt = part_cnt;
  _buckets = new BucketHeader*[part_cnt];
  for (int part = 0; part < part_cnt; ++part) {
    _buckets[part] = static_cast<BucketHeader*>(
        _mm_malloc(sizeof(BucketHeader) * _bucket_cnt_per_part, 64));
    for (uint64_t bucket = 0; bucket < _bucket_cnt_per_part; ++bucket) {
      _buckets[part][bucket].init();
    }
  }
  return RCOK;
}

RC IndexHash::init(int part_cnt, table_t* value_table, uint64_t bucket_cnt) {
  init(bucket_cnt, part_cnt);
  table = value_table;
  return RCOK;
}

#ifdef ASTRA_DBX1000_EMBEDDED
IndexHash::~IndexHash() {
  if (_buckets == NULL) return;
  for (uint64_t part = 0; part < _part_cnt; ++part) {
    for (uint64_t bucket = 0; bucket < _bucket_cnt_per_part; ++bucket) {
      _buckets[part][bucket].clear();
    }
    _mm_free(_buckets[part]);
  }
  delete[] _buckets;
}
#endif

bool IndexHash::index_exist(idx_key_t key) {
  itemid_t* item = NULL;
  return index_try_read(key, item, 0) == RCOK;
}

void IndexHash::get_latch(BucketHeader* bucket) {
  while (!ATOM_CAS(bucket->locked, false, true)) {
  }
}

void IndexHash::release_latch(BucketHeader* bucket) {
  bool ok = ATOM_CAS(bucket->locked, true, false);
  assert(ok);
}

RC IndexHash::index_insert(idx_key_t key, itemid_t* item, int part_id) {
  if (part_id < 0 || static_cast<uint64_t>(part_id) >= _part_cnt) return ERROR;
  uint64_t bucket_index = hash(key);
  if (bucket_index >= _bucket_cnt_per_part) return ERROR;
  BucketHeader* bucket = &_buckets[part_id][bucket_index];
  get_latch(bucket);
  bucket->insert_item(key, item, part_id);
  release_latch(bucket);
  return RCOK;
}

RC IndexHash::index_read(idx_key_t key, itemid_t*& item, int part_id) {
  if (index_try_read(key, item, part_id) != RCOK) {
    M_ASSERT(false, "Key does not exist!");
  }
  return RCOK;
}

RC IndexHash::index_read(idx_key_t key, itemid_t*& item, int part_id,
                         int thd_id) {
  (void)thd_id;
  return index_read(key, item, part_id);
}

RC IndexHash::index_try_read(idx_key_t key, itemid_t*& item, int part_id) {
  if (part_id < 0 || static_cast<uint64_t>(part_id) >= _part_cnt) return ERROR;
  uint64_t bucket_index = hash(key);
  if (bucket_index >= _bucket_cnt_per_part) return ERROR;
  BucketHeader* bucket = &_buckets[part_id][bucket_index];
  return bucket->try_read_item(key, item) ? RCOK : ERROR;
}

void BucketHeader::init() {
  node_cnt = 0;
  first_node = NULL;
  locked = false;
}

void BucketHeader::insert_item(idx_key_t key, itemid_t* item, int part_id) {
  BucketNode* node = first_node;
  BucketNode* previous = NULL;
  while (node != NULL && node->key != key) {
    previous = node;
    node = node->next;
  }
  if (node == NULL) {
#ifdef ASTRA_DBX1000_EMBEDDED
    BucketNode* new_node = new BucketNode(key);
#else
    BucketNode* new_node = static_cast<BucketNode*>(
        mem_allocator.alloc(sizeof(BucketNode), part_id));
    new_node->init(key);
#endif
    new_node->items = item;
    if (previous == NULL) {
      new_node->next = first_node;
      first_node = new_node;
    } else {
      new_node->next = previous->next;
      previous->next = new_node;
    }
    ++node_cnt;
  } else {
    item->next = node->items;
    node->items = item;
  }
}

bool BucketHeader::try_read_item(idx_key_t key, itemid_t*& item) {
  BucketNode* node = first_node;
  while (node != NULL && node->key != key) node = node->next;
  if (node == NULL) {
    item = NULL;
    return false;
  }
  item = node->items;
  return true;
}

void BucketHeader::read_item(idx_key_t key, itemid_t*& item,
                             const char* table_name) {
  (void)table_name;
  bool found = try_read_item(key, item);
  M_ASSERT(found, "Key does not exist!");
}

#ifdef ASTRA_DBX1000_EMBEDDED
void BucketHeader::clear() {
  BucketNode* node = first_node;
  while (node != NULL) {
    BucketNode* next = node->next;
    delete node;
    node = next;
  }
  first_node = NULL;
  node_cnt = 0;
}
#endif
