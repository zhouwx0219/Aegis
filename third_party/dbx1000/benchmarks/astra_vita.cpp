// ASTRA/HybridCC Vita workload runner inside the DBx1000 tree.
//
// This is a DBx1000-facing in-memory OLTP benchmark driver for ASTRA's
// agent-side commit protocol. It keeps the workload and baseline semantics
// identical across policies and uses ASTRA's C++ commit kernel for HYBRID.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "core/branch/speculative_branch.h"
#include "core/cost/cost_model.h"
#include "core/intent/intent.h"
#include "core/storage/versioned_object_store.h"
#include "core/txn/cost_asymmetric_commit.h"

namespace {

using Clock = std::chrono::steady_clock;

struct Resource {
  std::string oid;
  std::string category;
  long long quantity = 1;
  double price = 0.0;
};

struct Task {
  std::vector<int> candidates;
  int quantity = 1;
};

struct Args {
  std::string resources = "agent/experiments/results/vitabench_authoritative_resources.csv";
  std::string out = "agent/experiments/results/dbx1000_vita.csv";
  int tasks = 3000;
  int threads = 16;
  int k = 4;
  int seeds = 3;
  int hot_per_category = 6;
  double hot_bias = 0.85;
  int capacity_multiplier = 20;
  double c_gen_ms = 2.0;
};

struct Row {
  long long stock = 0;
  std::uint64_t version = 0;
  std::mutex mu;
};

struct Metrics {
  long long booked = 0;
  long long no_stock = 0;
  long long oversell = 0;
  long long regen = 0;
  long long reselect = 0;
  long long merge = 0;
  long long lat_us_sum = 0;
  long long completed = 0;
  double wall_s = 0.0;
};

std::vector<std::string> split_csv_line(const std::string& line) {
  std::vector<std::string> out;
  std::string cur;
  bool quoted = false;
  for (char ch : line) {
    if (ch == '"') {
      quoted = !quoted;
    } else if (ch == ',' && !quoted) {
      out.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(ch);
    }
  }
  out.push_back(cur);
  return out;
}

std::unordered_map<std::string, std::size_t> header_index(const std::string& header) {
  std::unordered_map<std::string, std::size_t> idx;
  auto cols = split_csv_line(header);
  for (std::size_t i = 0; i < cols.size(); ++i) idx[cols[i]] = i;
  return idx;
}

std::vector<Resource> read_resources(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot open resources CSV: " + path);
  }
  std::string line;
  std::getline(in, line);
  auto idx = header_index(line);
  std::vector<Resource> resources;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    auto cols = split_csv_line(line);
    Resource r;
    r.oid = cols.at(idx.at("oid"));
    r.category = cols.at(idx.at("category"));
    r.quantity = std::max(1LL, std::stoll(cols.at(idx.at("quantity"))));
    if (idx.count("price")) r.price = std::stod(cols.at(idx.at("price")));
    resources.push_back(r);
  }
  if (resources.empty()) throw std::runtime_error("no resources loaded");
  return resources;
}

std::uint32_t mix_seed(int seed, int a, int b) {
  std::uint32_t x = static_cast<std::uint32_t>(seed);
  x = x * 1103515245u + 12345u + static_cast<std::uint32_t>(a) * 1000003u;
  x = x * 1103515245u + 12345u + static_cast<std::uint32_t>(b) * 9176u;
  return x;
}

std::vector<Task> make_tasks(const std::vector<Resource>& resources, const Args& args, int seed,
                             int candidate_limit) {
  std::unordered_map<std::string, std::vector<int>> by_cat;
  for (int i = 0; i < static_cast<int>(resources.size()); ++i) {
    by_cat[resources[i].category].push_back(i);
  }
  std::vector<std::string> cats;
  for (const auto& kv : by_cat) cats.push_back(kv.first);
  std::sort(cats.begin(), cats.end());
  for (auto& kv : by_cat) {
    auto& ids = kv.second;
    std::sort(ids.begin(), ids.end(), [&](int a, int b) {
      if (resources[a].price != resources[b].price) return resources[a].price < resources[b].price;
      return resources[a].oid < resources[b].oid;
    });
  }

  std::vector<Task> tasks;
  tasks.reserve(args.tasks);
  for (int tid = 0; tid < args.tasks; ++tid) {
    std::mt19937 rng(mix_seed(seed, tid, 31));
    const auto& pool = by_cat[cats[tid % cats.size()]];
    const int hot_n = std::max(1, std::min(args.hot_per_category, static_cast<int>(pool.size())));
    std::vector<int> base;
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    if (coin(rng) < args.hot_bias) {
      base.insert(base.end(), pool.begin(), pool.begin() + hot_n);
    } else {
      base = pool;
    }
    std::shuffle(base.begin(), base.end(), rng);
    Task t;
    const int limit = std::min(candidate_limit, static_cast<int>(base.size()));
    t.candidates.insert(t.candidates.end(), base.begin(), base.begin() + limit);
    tasks.push_back(t);
  }
  return tasks;
}

std::string policy_name(const std::string& policy) {
  if (policy == "HYBRID") return "ASTRA-HYBRID";
  if (policy == "HYBRID-K1") return "ASTRA-HYBRID-K1";
  if (policy == "branch-txn") return "branch-txn";
  return "DBX-" + policy;
}

Metrics run_dbx_policy(const std::string& policy, const std::vector<Resource>& resources,
                       const Args& args, int seed) {
  const int limit = (policy == "OCC-K1") ? 1 : args.k;
  auto tasks = make_tasks(resources, args, seed, limit);
  std::vector<Row> rows(resources.size());
  for (std::size_t i = 0; i < resources.size(); ++i) {
    rows[i].stock = std::max(1LL, resources[i].quantity * args.capacity_multiplier);
  }

  std::mutex store_mu;
  std::mutex metrics_mu;
  Metrics m;
  std::atomic<int> next{0};
  const auto sleep_dur = std::chrono::duration<double, std::milli>(args.c_gen_ms);
  auto start = Clock::now();

  auto worker = [&]() {
    while (true) {
      const int tid = next.fetch_add(1);
      if (tid >= static_cast<int>(tasks.size())) return;
      const auto& task = tasks[tid];
      auto t0 = Clock::now();
      long long booked = 0, no_stock = 0, oversell = 0, regen = 0, reselect = 0, merge = 0;

      if (policy == "2PL") {
        const int rid = task.candidates.front();
        std::lock_guard<std::mutex> row_lk(rows[rid].mu);
        std::this_thread::sleep_for(sleep_dur);
        if (rows[rid].stock >= task.quantity) {
          rows[rid].stock -= task.quantity;
          rows[rid].version += 1;
          booked = 1;
        } else {
          no_stock = 1;
        }
      } else if (policy == "merge-all") {
        const int rid = task.candidates.front();
        std::this_thread::sleep_for(sleep_dur);
        std::lock_guard<std::mutex> lk(store_mu);
        rows[rid].stock -= task.quantity;
        rows[rid].version += 1;
        booked = 1;
        if (rows[rid].stock < 0) oversell = 1;
      } else if (policy == "branch-txn") {
        std::uint64_t winner_base = 0;
        {
          std::lock_guard<std::mutex> lk(store_mu);
          winner_base = rows[task.candidates.front()].version;
        }
        std::this_thread::sleep_for(sleep_dur);
        bool need_regen = false;
        {
          std::lock_guard<std::mutex> lk(store_mu);
          const int rid = task.candidates.front();
          if (rows[rid].version == winner_base && rows[rid].stock >= task.quantity) {
            rows[rid].stock -= task.quantity;
            rows[rid].version += 1;
            booked = 1;
          } else {
            for (int cand : task.candidates) {
              if (rows[cand].stock >= task.quantity) {
                need_regen = true;
                break;
              }
            }
            if (!need_regen) no_stock = 1;
          }
        }
        if (need_regen) {
          std::this_thread::sleep_for(sleep_dur);
          regen = 1;
          std::lock_guard<std::mutex> lk(store_mu);
          for (int cand : task.candidates) {
            if (rows[cand].stock >= task.quantity) {
              rows[cand].stock -= task.quantity;
              rows[cand].version += 1;
              booked = 1;
              no_stock = 0;
              break;
            }
          }
          if (!booked) no_stock = 1;
        }
      } else {
        std::unordered_map<int, std::uint64_t> base;
        {
          std::lock_guard<std::mutex> lk(store_mu);
          for (int rid : task.candidates) base[rid] = rows[rid].version;
        }
        std::this_thread::sleep_for(sleep_dur);

        bool committed = false;
        {
          std::lock_guard<std::mutex> lk(store_mu);
          for (std::size_t i = 0; i < task.candidates.size(); ++i) {
            const int rid = task.candidates[i];
            const bool write_changed = rows[rid].version != base[rid];
            // DBx1000 native OCC/Silo/TicToc/MVCC all treat same-row writes as
            // write-write conflicts for this OTA stock decrement workload.
            if (!write_changed && rows[rid].stock >= task.quantity) {
              rows[rid].stock -= task.quantity;
              rows[rid].version += 1;
              booked = 1;
              reselect = i > 0 ? 1 : 0;
              committed = true;
              break;
            }
          }
          if (!committed) {
            bool any_stock = false;
            for (int rid : task.candidates) any_stock = any_stock || rows[rid].stock >= task.quantity;
            no_stock = any_stock ? 0 : 1;
          }
        }
        if (!committed && !no_stock) {
          std::this_thread::sleep_for(sleep_dur);
          regen = 1;
          std::lock_guard<std::mutex> lk(store_mu);
          for (std::size_t i = 0; i < task.candidates.size(); ++i) {
            const int rid = task.candidates[i];
            if (rows[rid].stock >= task.quantity) {
              rows[rid].stock -= task.quantity;
              rows[rid].version += 1;
              booked = 1;
              reselect = i > 0 ? 1 : 0;
              no_stock = 0;
              committed = true;
              break;
            }
          }
          if (!committed) no_stock = 1;
        }
      }

      auto t1 = Clock::now();
      const auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
      std::lock_guard<std::mutex> lk(metrics_mu);
      m.booked += booked;
      m.no_stock += no_stock;
      m.oversell += oversell;
      m.regen += regen;
      m.reselect += reselect;
      m.merge += merge;
      m.lat_us_sum += us;
      m.completed += 1;
    }
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < args.threads; ++i) threads.emplace_back(worker);
  for (auto& t : threads) t.join();
  m.wall_s = std::chrono::duration<double>(Clock::now() - start).count();
  return m;
}

Metrics run_hybrid(const std::string& policy, const std::vector<Resource>& resources,
                   const Args& args, int seed) {
  const int limit = (policy == "HYBRID-K1") ? 1 : args.k;
  auto tasks = make_tasks(resources, args, seed, limit);
  cast::storage::VersionedObjectStore store;
  for (const auto& r : resources) {
    store.Put(r.oid, std::to_string(std::max(1LL, r.quantity * args.capacity_multiplier)));
  }
  cast::cost::CostModel model;
  model.c_gen = args.c_gen_ms / 1000.0;
  model.c_merge = 0.0;
  cast::txn::CostAsymmetricCommit kernel(store, model);

  std::mutex store_mu;
  std::mutex metrics_mu;
  Metrics m;
  std::atomic<int> next{0};
  const auto sleep_dur = std::chrono::duration<double, std::milli>(args.c_gen_ms);
  auto start = Clock::now();

  auto worker = [&]() {
    while (true) {
      const int tid = next.fetch_add(1);
      if (tid >= static_cast<int>(tasks.size())) return;
      const auto& task = tasks[tid];
      auto t0 = Clock::now();
      std::vector<cast::branch::SpeculativeBranch> branches;
      {
        std::lock_guard<std::mutex> lk(store_mu);
        for (std::size_t i = 0; i < task.candidates.size(); ++i) {
          const auto& r = resources[task.candidates[i]];
          const auto base = store.Get(r.oid);
          cast::intent::WriteIntent intent;
          intent.object_id = r.oid;
          intent.intent_type = cast::intent::IntentType::kDelta;
          intent.payload = "-1";
          intent.constrained = true;
          intent.lower_bound = 0;

          cast::branch::BranchWrite w;
          w.object_id = r.oid;
          w.base_value = base.value;
          w.base_version = base.version;
          w.branch_value = std::to_string(std::stoll(base.value) - task.quantity);
          w.intent = intent;

          cast::branch::SpeculativeBranch b;
          b.branch_id = "t" + std::to_string(tid) + ":c" + std::to_string(i);
          b.writes.push_back(w);
          b.gen_cost = model.c_gen;
          b.quality = static_cast<double>(task.candidates.size() - i);
          branches.push_back(b);
        }
      }
      std::this_thread::sleep_for(sleep_dur);

      cast::cost::CostStats stats;
      cast::txn::CommitOutcome out;
      {
        std::lock_guard<std::mutex> lk(store_mu);
        out = kernel.CommitTask(branches, cast::txn::CommitStrategy::kCAST, &stats);
      }

      auto t1 = Clock::now();
      const auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
      std::lock_guard<std::mutex> lk(metrics_mu);
      m.booked += out.committed ? 1 : 0;
      m.no_stock += out.rejected ? 1 : 0;
      m.regen += stats.n_regen;
      m.reselect += stats.n_reselect;
      m.merge += stats.n_merge;
      m.lat_us_sum += us;
      m.completed += 1;
    }
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < args.threads; ++i) threads.emplace_back(worker);
  for (auto& t : threads) t.join();
  m.wall_s = std::chrono::duration<double>(Clock::now() - start).count();
  return m;
}

void write_header(std::ofstream& out) {
  out << "policy,seed,n_tasks,threads,k,hot_per_category,hot_bias,capacity_multiplier,"
         "c_gen_ms,throughput,task_throughput,latency_ms,booked,no_stock,oversell,"
         "regen,reselect,merge,generation_calls_per_task,wall_s\n";
}

void write_row(std::ofstream& out, const std::string& policy, int seed, const Args& args,
               const Metrics& m) {
  const double throughput = m.wall_s > 0 ? static_cast<double>(m.booked) / m.wall_s : 0.0;
  const double task_tp = m.wall_s > 0 ? static_cast<double>(m.completed) / m.wall_s : 0.0;
  const double latency_ms = m.completed > 0 ? (static_cast<double>(m.lat_us_sum) / m.completed) / 1000.0 : 0.0;
  const double gen_calls = static_cast<double>(m.completed + m.regen) / std::max(1LL, m.completed);
  out << policy_name(policy) << ',' << seed << ',' << args.tasks << ',' << args.threads << ','
      << args.k << ',' << args.hot_per_category << ',' << args.hot_bias << ','
      << args.capacity_multiplier << ',' << args.c_gen_ms << ',' << throughput << ','
      << task_tp << ',' << latency_ms << ',' << m.booked << ',' << m.no_stock << ','
      << m.oversell << ',' << m.regen << ',' << m.reselect << ',' << m.merge << ','
      << gen_calls << ',' << m.wall_s << '\n';
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string a(argv[i]);
    auto need = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + name);
      return std::string(argv[++i]);
    };
    if (a == "--resources") args.resources = need(a);
    else if (a == "--out") args.out = need(a);
    else if (a == "--tasks") args.tasks = std::stoi(need(a));
    else if (a == "--threads") args.threads = std::stoi(need(a));
    else if (a == "--k") args.k = std::stoi(need(a));
    else if (a == "--seeds") args.seeds = std::stoi(need(a));
    else if (a == "--hot-per-category") args.hot_per_category = std::stoi(need(a));
    else if (a == "--hot-bias") args.hot_bias = std::stod(need(a));
    else if (a == "--capacity-multiplier") args.capacity_multiplier = std::stoi(need(a));
    else if (a == "--c-gen-ms") args.c_gen_ms = std::stod(need(a));
    else if (a == "--help" || a == "-h") {
      std::cout << "astra_vita --resources CSV --out CSV --tasks N --threads N --k N "
                   "--seeds N --c-gen-ms MS\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + a);
    }
  }
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args args = parse_args(argc, argv);
    auto resources = read_resources(args.resources);
    std::ofstream out(args.out);
    if (!out) throw std::runtime_error("cannot open output CSV: " + args.out);
    write_header(out);

      const std::vector<std::string> policies = {
        "branch-txn", "OCC-K1", "OCC+K", "MVCC", "TICTOC", "SILO", "2PL", "merge-all", "HYBRID-K1", "HYBRID"};
    for (int seed = 1; seed <= args.seeds; ++seed) {
      for (const auto& policy : policies) {
        Metrics m = (policy == "HYBRID" || policy == "HYBRID-K1")
                        ? run_hybrid(policy, resources, args, seed)
                        : run_dbx_policy(policy, resources, args, seed);
        write_row(out, policy, seed, args, m);
        std::cout << policy_name(policy) << " seed=" << seed << " booked=" << m.booked
                  << " regen=" << m.regen << " reselect=" << m.reselect
                  << " oversell=" << m.oversell << " wall=" << m.wall_s << "s\n";
      }
    }
    std::cout << "saved " << args.out << "\n";
  } catch (const std::exception& e) {
    std::cerr << "astra_vita error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
