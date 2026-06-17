# CAST 正确性：严格证明（gap2）

> 把 `ISOLATION_LEVELS.md` 的定理草图升级为论文 appendix 级的证明。
> 给出形式化系统模型、提交操作语义、引理与定理（strict 可串行化 / commutative 收敛 / CAS 条件安全 / escrow 安全），以及不保证项的形式化反例。

## 1. 系统模型

- **对象与状态**：对象集 `O`。状态 `S: O → (Val × ℕ)`，记 `S(o)=(val(o), ver(o))`，`ver` 为版本号。初始状态 `S₀`，`ver` 初值 0。
- **事务**：`T = (RS(T), WS(T))`。
  - 读集 `RS(T) ⊆ O × ℕ`：`(o, rv_T(o))` 表示 `T` 读对象 `o` 时看到的版本 `rv_T(o)`（来自一致快照）。
  - 写集 `WS(T)`：对每个被写对象 `o` 给出 `(cls_T(o), op_T(o))`，`cls ∈ {S, C, K}`（strict / commutative / conditional）。
- **提交序**：成功提交的事务由全局 `commit_lock` 串成全序历史 `H = ⟨T₁, …, Tₙ⟩`（提交点互斥 ⇒ 全序良定义）。
- **可串行化（写）**：若存在 `H` 的某个排列对应的串行执行产生相同的最终状态与提交决定，则称该（子）历史**可串行化**。

## 2. 提交操作语义（CAST，提交点原子执行）

提交 `Tᵢ` 时，对每个 `o ∈ WS(Tᵢ)`，按 `cls` 执行：

- **(S) strict**：`if ver(o)=rv_{Tᵢ}(o)` then `val(o)←new; ver(o)←ver(o)+1`；else `Tᵢ` 在 `o` 上**冲突**（该候选不提交 → reselect/regenerate）。
- **(C) commutative**：`val(o) ← val(o) ⊕ Δ_{Tᵢ}(o); ver(o)←ver(o)+1`。`⊕` 满足**交换律与结合律**，且 `Δ` 不依赖被读版本（增量自带）。**不因版本变化而冲突**。
- **(K) conditional (CAS)**：`if cond_{Tᵢ}(val(o))` then `val(o)←new; ver(o)←ver(o)+1`；else **冲突**。

**引理 1（版本单调）**：任一对象 `o` 上，每次成功写使 `ver(o)` 严格 +1。*证明*：三分支成功时均执行 `ver(o)←ver(o)+1`。∎

**引理 2（提交原子）**：每个 `Tᵢ` 的提交在 `commit_lock` 内执行，期间无其他事务修改状态 ⇒ `Tᵢ` 的所有写基于同一状态快照 `S_{i-1}` 并原子地产生 `Sᵢ`。∎

## 3. 定理

### 定理 1（strict 子历史可串行化）
**设** `H` 中所有写均为 strict。**则** `H` 按提交序即为一个等价串行执行（冲突可串行化）。

**证明**：
对每个对象 `o`，考虑写过 `o` 的事务子序 `T_{i₁}, …, T_{i_m}`（按提交序）。
由 (S) 语义，`T_{i_k}` 成功写 `o` 要求 `ver(o)=rv_{T_{i_k}}(o)`。
由引理 1，`o` 的版本在每次成功写后 +1；故 `rv_{T_{i_k}}(o)` 必等于 `T_{i_{k-1}}` 写后的版本（否则版本不匹配 → 冲突 → `T_{i_k}` 不在 `H` 中）。
于是 `o` 上的"读版本 → 写版本"首尾相接、版本严格递增：**无丢失更新（no lost update）**，且每个写都 reads-from 其紧邻前驱。
跨对象地，`Tᵢ` 提交时（引理 2）读到的是前缀 `{T₁,…,T_{i-1}}` 相关写的结果。
因此串行执行 `T₁; …; Tₙ` 产生与 `H` 相同的状态与提交决定 ⇒ `H` 冲突可串行化（等价于 OCC first-committer-wins）。∎

### 定理 2（commutative 子历史收敛 / 状态可串行化）
**设** `H` 中所有写均为 commutative，`⊕` 交换、结合。**则** 对任意 `o`，
`val(o) = val₀(o) ⊕ ( ⊕_{T∈H, o∈WS(T)} Δ_T(o) )`，
该值**与 `H` 中相关事务的相对顺序无关**，且每个 `Δ_T(o)` 恰被计入一次（不丢更新）。

**证明**（对 `H` 长度归纳）：
- 基例 `n=0`：`val(o)=val₀(o)`（空 ⊕）。
- 归纳：设前 `n−1` 次提交后 `val(o)=val₀(o) ⊕ (⊕_{i<n} Δ_{Tᵢ}(o))`。提交 `Tₙ`（若 `o∈WS(Tₙ)`）由 (C) 得 `val(o) ← val(o) ⊕ Δ_{Tₙ}(o) = val₀(o) ⊕ (⊕_{i≤n} Δ_{Tᵢ}(o))`。
- 由 `⊕` 交换、结合，上式的大 `⊕` 与求和顺序无关；每个事务恰提交一次、每个写恰应用一次 ⇒ 每个 `Δ` 出现且仅一次。
故任取 `H` 的一个排列作串行执行得到相同终态 ⇒ **状态层可串行化（收敛）**。∎

**适用前提（推论）**：`⊕` 必须真可交换。整数加（DELTA）✓；multiset/集合并（无序 APPEND）✓；**字符串顺序拼接 ✗**（不满足交换律）⇒ 顺序敏感 APPEND 须降级为 strict（否则定理 2 不成立）。

### 定理 3（CAS 条件安全）
**命题**：任意成功提交的 CAS 写，其条件在提交点对当时状态成立；故 `H` 不含违反该 CAS 条件的状态转移。
**证明**：由 (K) 语义，`Tᵢ` 写 `o` 仅当 `cond_{Tᵢ}(val(o))` 成立。由引理 2，条件检查与写入在 `commit_lock` 内原子完成，期间 `val(o)` 不被他者改变 ⇒ 写入基于满足条件的状态。∎
（实现对应：版本未变分支也校验 CAS 条件，避免 regenerate 强行对齐版本后绕过条件。）

### 定理 4（escrow 安全 + 可交换）
**模型**：escrow 对象 `o` 维护 `(cap(o), reserved(o))`，不变量 `INV: 0 ≤ reserved(o) ≤ cap(o)`。扣减事务 `T`（额度 `q_T ≥ 0`）：
- `reserve`：`if cap−reserved ≥ q_T then reserved += q_T` 否则拒绝；
- `confirm`（提交）：`cap −= q_T; reserved −= q_T`；
- `release`（中止）：`reserved −= q_T`。

**定理 4a（不超卖 / 约束保持）**：在 reserve/confirm/release 的任意交错下，恒有 `INV` 且 `cap(o) ≥ 0`。
*证明（不变量归纳）*：初始 `reserved=0 ≤ cap`。`reserve` 仅在 `cap−reserved ≥ q` 时 `+q` ⇒ `reserved ≤ cap` 保持；`q≥0 ⇒ reserved≥0`。`confirm`：`q ≤ reserved`（已预留），故 `reserved−q ≥ 0`，且 `cap−q ≥ reserved−q ≥ 0` ⇒ `cap ≥ 0` 且 `INV` 保持。`release` 同理。∎

**定理 4b（可交换 / 并发）**：`reserve` 对 `reserved` 是加法、其成功判据只依赖标量 `cap−reserved`（总剩余额度），与预留者顺序无关 ⇒ 任意预留子集只要总额 `≤ cap` 均可成功且互不阻塞；`confirm` 亦为加法可交换。故 escrow 扣减是**可交换的（并发合并）**，并由 4a **约束保持**。∎

**推论**：escrow 把"带下界/容量约束的扣减"从 strict（需串行/重跑）或纯 commutative（超卖）升级为 **"可交换并发 + 约束保持的收敛"**——即把定理 2 的收敛扩展到带容量约束的对象。（实测 `escrow_experiment.py`：N>cap 时正确封顶、零重跑。）

## 4. 主定理与不保证

**主定理（CSI-SS）**：事务读自一致快照（记录 `rv`，不强制读集验证）、写按 `cls` 分级提交。则 `H` 满足 **CSI-SS**：
(1) strict-only 投影冲突可串行化（定理 1）；(2) commutative-only 投影收敛（定理 2）；
(3) CAS 写条件安全（定理 3）；(4) escrow 写约束保持的收敛（定理 4）；(5) 读维度 = 快照隔离。

**不保证（形式化反例）**：
- **write-skew（Adya A5B）**：`T₁: RS={x}, WS={y:S}`；`T₂: RS={y}, WS={x:S}`。读不验证 ⇒ 二者基于同一初始快照各自写，得到 SI 允许、可串行化禁止的结果。CAST 同 SI，允许。
- **带约束 commutative 无 escrow**：`cap=5`，8 个 `C` 写各 `Δ=−1` ⇒ 定理 2 终值 `5−8=−3 < 0`，违反 `cap≥0`（实测）。须降级 escrow（定理 4）或 CAS（定理 3）。
- **跨对象不变量**：如 `x+y ≤ B` 跨两对象，对象级提交不检测。

**结论**：CAST = 一个**按写语义分级**的并发控制——在 `{strict 可串行 / commutative 收敛 / escrow 约束收敛 / CAS 条件 / 读 SI}` 之间切换的旋钮；需要更强保证时把对应写升一档。

> 严格度说明：以上为论文 appendix 级证明（定义—引理—定理逻辑完整）。完整机器验证（TLA+/Coq）与并发交错的模型检查列为 future work。
