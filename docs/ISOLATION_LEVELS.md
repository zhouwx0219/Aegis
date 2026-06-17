# CAST 隔离级别形式化

> 目标：给 CAST 一个**精确的隔离级别**与可证明的保证，把组会"正确性边界讲死"升级为定理。
> 结论先行：CAST 提供一个**混合隔离级别**——读维度 = 快照隔离(SI)，写维度按意图分级：
> strict 写**冲突可串行化**、commutative 写**收敛(状态可串行化)**、CAS 写**条件安全**。
> 我们记之为 **CSI-SS（Convergent Snapshot Isolation with Serializable Strict writes）**。

---

## 1. 系统模型

- **对象**：每个对象 `o` 有版本化值 `(val(o), ver(o))`，`ver` 单调递增。底层只提供 `Get / PutIfVersion / BatchPutIfVersion`。
- **事务** `T`：一组读 `R(T) = {(o, rv_T(o))}`（`rv` = 读到的版本）+ 一组写意图 `W(T) = {(o, cls, payload)}`，
  `cls ∈ {strict, commutative, conditional}`（对应 OVERWRITE / {DELTA,APPEND} / CAS；READ 只进 `R(T)`）。
- **基线快照**：`T` 开始时读一致快照（记录每个读对象的版本 `rv_T(o)`）。
- **提交协议**：CAST 在提交点按 `cls` 处理每个写（见 `core/txn/cost_asymmetric_commit.h`）。
- **历史** `H`：已成功提交事务按提交时刻排成的序 `T_1, T_2, …, T_n`。

---

## 2. 各写意图的提交语义（形式化）

记 `T` 提交点对象 `o` 的当前值/版本为 `val_c(o)/ver_c(o)`。

- **strict（OVERWRITE）**：`T` 提交 `o` ⟺ `ver_c(o) = rv_T(o)`（版本自读后未变）。
  成功则 `val(o) ← new_T(o); ver(o)++`。否则视为冲突（→ reselect/regenerate）。**= first-committer-wins。**
- **commutative（DELTA/APPEND）**：提交时 `val(o) ← val_c(o) ⊕ Δ_T(o)`，
  其中 `Δ_T(o) = g(base_T(o), branch_T(o))`（DELTA：增量 `branch−base`；APPEND：追加片段）。
  `⊕` 满足**交换律 + 结合律**。`ver(o)++`。**不因版本变而拒绝**（验证层放行）。
- **conditional（CAS）**：`T` 提交 `o` ⟺ `cond_T(val_c(o))` 在**提交点**成立。成立则写入。

---

## 3. 三条定理

### 定理 1（strict 子历史可串行化）
**命题**：若所有事务仅含 strict 写（与读），CAST 产生的历史 `H` 是**冲突可串行化**的（与 OCC first-committer-wins 等价）。

**证明（草图）**：strict 写经版本校验提交——`T` 写 `o` 要求 `ver_c(o)=rv_T(o)`。若存在 `T'` 在 `T` 读 `o` 之后、`T` 提交之前提交了 `o`，则 `ver_c(o) > rv_T(o)`，`T` 该写被判冲突而不提交。
故对每个对象，已提交事务的"读版本→写版本"首尾相接、版本严格递增 ⟹ 无丢失更新且 first-committer-wins。
按提交时刻定序即得一个等价串行序（写写按版本序、读看到的是其串行前缀），故 `H` 冲突可串行化。∎

### 定理 2（commutative 子历史收敛 / 状态可串行化）
**命题**：若所有写为 commutative（`⊕` 可交换、结合），则无论提交序如何，
`val(o)` 终值 `= val_0(o) ⊕ Δ_{T_1}(o) ⊕ … ⊕ Δ_{T_n}(o)`，**与提交序无关**；且每个已提交 `T` 的 `Δ_T` 恰被计入一次（**不丢更新**）。

**证明（草图）**：归纳。CAST 的 commutative 提交是 `val ← val_c ⊕ Δ_T`（在**最新值**上应用本事务相对 base 的增量）。
第 `i` 次提交后 `val = val_0 ⊕ Δ_1 ⊕ … ⊕ Δ_i`。由 `⊕` 交换结合，结果与下标顺序无关；每个 `Δ_T` 出现且仅出现一次。
因此存在等价串行序（任取一个排列）产生相同终态 ⟹ **状态层可串行化（收敛）**。∎

**适用前提**：`⊕` 必须真可交换。DELTA 的 `⊕ = 整数加`（可交换）；APPEND 须取 **multiset/集合并**语义（可交换）——
**若 APPEND 结果依赖追加顺序（字符串顺序拼接），则 `⊕` 不可交换，定理 2 不适用**（该写应降级为 strict）。

### 定理 3（CAS 条件安全）
**命题**：CAS 写仅在提交点条件成立时落库 ⟹ 不产生违反该条件的状态转移。
**证明**：CAST 提交 CAS 写前重检 `cond_T(val_c(o))`（实现中即便版本未变也校验条件），不成立即判冲突不提交。∎

---

## 4. 主定理：CAST 的混合隔离级别 CSI-SS

**读维度**：事务从一致快照读、记录 `rv_T(o)`，**不强制读集验证** ⟹ 读维度达 **快照隔离 (SI)**。

**主定理**：在 (i) 所有放行的 commutative 写其 `⊕` 真可交换、(ii) 不依赖跨对象/带约束不变量 的前提下，
CAST 历史满足 **CSI-SS**：
- 历史的 **strict-only 投影**是冲突可串行化的（定理 1）；
- 历史的 **commutative-only 投影**是收敛的 / 状态可串行化（定理 2）；
- 读看到一致快照（SI）；CAS 写条件安全（定理 3）。

直觉：CAST 把"是否参与冲突验证"按写语义分级——strict 走最严（可串行化），commutative 走最松（放行+收敛），
CAS 走条件，读走快照。**这就是"语义感知验证减少冲突"在隔离级别上的精确含义。**

---

## 5. 明确**不保证**的（反例，讲死）

1. **write-skew（读依赖）**：读集不验证 ⟹ SI 经典反例——两事务读重叠数据、各写不同对象，破坏跨行约束。CAST 同样允许（与 SI 同级）。
2. **带下界/容量约束的 commutative 写**：定理 2 只保证收敛，不保证约束——库存 5、8×`DELTA(-1)` 合并 = **−3（超卖）**（实测 `correctness_boundary.py`）。
3. **跨对象不变量**：对象级合并不检测（如"多笔订单总额 ≤ 预算"）。
4. **顺序敏感 APPEND**：`⊕` 不可交换，定理 2 前提不满足（须降级 strict）。

> 需要上述任一保证时，把对应写从 commutative **降级**为 conditional / strict / escrow —— 这是分级验证的可调旋钮。

---

## 6. 与标准隔离级别的关系

- **Adya（基于现象）**：CSI-SS 的读维度 = **PL-SI**（禁脏读、禁丢失更新 P4，但允许 A5B write-skew）；
  strict 写维度达 **PL-3（可串行化）**。
- **commutative 收敛**超出 Adya 的 last-writer-wins 异常框架，需用**状态层可串行化 / CRDT 的强最终一致（SEC）**来刻画
  （Crooks et al. 的 state-based 隔离更适合描述）。
- 一句话：**CAST = 读 SI + strict 可串行化 + commutative 收敛(SEC) + CAS 条件安全 的混合点**，
  并提供"按写语义在这几档之间切换"的旋钮。

---

## 7. 把约束写拉回可合并：escrow（下一步，已实现见 escrow 实验）

定理 2 的缺口是"带约束的 commutative 写"。**escrow transactions（O'Neil, 1986）** 正是补这个缺口：
把"扣减且不破下界"表达为**额度预留**——并发事务各自预留 `q`（只要剩余额度 `≥ q`），提交时确认释放。
- 并发的预留互不阻塞（只要总预留 `≤` 库存）⟹ 保留 commutative 的**并发/合并收益**；
- 任何使库存 `< 0` 的预留被拒 ⟹ **不超卖**，约束被保持。

⟹ 带约束的扣减从"strict（需串行/重跑）或纯 DELTA（超卖）"升级为 **"可并发预留 + 约束保持的收敛"**，
**把第 5.2 的超卖边界从"不保证项"转成"可合并收益"，扩大 CAST 的适用面。** 详见 `escrow_experiment.py`。
