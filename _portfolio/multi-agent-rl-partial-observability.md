---
title: "Learning-Informed Masking for Multi-Agent Coordination"
excerpt: "Improving multi-agent reinforcement learning under partial observability with adaptive TD-error-based masking, achieving 2x faster learning and more stable policies"
collection: portfolio
permalink: /portfolio/learning-informed-masking-marl/
date: 2024-11-01
---

<div align="center">

<p><strong>Author:</strong><br>
<a href="https://ilyasbounoua.github.io">Ilyas Bounoua</a></p>

<p><strong>Institution:</strong><br>
Personal Research Project</p>

<h3>📋 Table of Contents</h3>
<p>
  <a href="#-introduction">📖 Introduction</a> •
  <a href="#-objectives">🎯 Objectives</a> •
  <a href="#️-methods">⚙️ Methods</a> •
  <a href="#-results">📊 Results</a> •
  <a href="#-discussion">💬 Discussion</a> •
  <a href="#-references">🔗 References</a>
</p>
</div>

---

## 📖 Introduction

Think about a team of autonomous drones trying to coordinate in a GPS-denied environment - maybe searching through a disaster zone where communication is spotty and each drone can only see what's directly around it. How do they work together effectively when nobody has the full picture?

This is the challenge of **partial observability** in multi-agent systems, and it's a tough one. Recent work has shown that masked auto-encoders can help agents "fill in the blanks" by reconstructing a global view from their limited local observations. But here's the thing - these methods typically use random masking strategies that ignore how well the agents are actually learning.

I wanted to see if we could do better. What if we used the agents' own training signals to guide which information gets masked? Turns out, this simple idea makes a pretty significant difference.

---

## 🎯 Objectives

1. **Identify the limitation** of policy-agnostic random masking in multi-agent representation learning
2. **Develop an adaptive masking strategy** that uses TD-error to focus on struggling agents
3. **Validate the approach** on diverse cooperative tasks requiring tight coordination

---

## ⚙️ Methods

### Problem Formulation: Partially Observable Multi-Agent Systems

Before diving into the solution, let's formalize what we're dealing with. Multi-agent coordination under partial observability is typically modeled as a **Decentralized Partially Observable Markov Decision Process (Dec-POMDP)**.

**Dec-POMDP Definition:**

A Dec-POMDP for $n$ agents is defined by the tuple $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \Omega, \mathcal{O}, \gamma \rangle$:

- $\mathcal{S}$: Global state space (not fully observable to any agent)
- $\mathcal{A} = \mathcal{A}_1 \times \dots \times \mathcal{A}_n$: Joint action space
- $\mathcal{P}: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow [0,1]$: State transition probability
- $\mathcal{R}: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$: Shared reward function
- $\Omega = \Omega_1 \times \dots \times \Omega_n$: Joint observation space
- $\mathcal{O}: \mathcal{S} \times \mathcal{A} \times \Omega \rightarrow [0,1]$: Observation probability
- $\gamma \in [0,1)$: Discount factor

Each agent $i$ receives only a local observation $o_i \in \Omega_i$ instead of the full state $s \in \mathcal{S}$. The goal is to learn a joint policy $\pi = \langle \pi_1, \dots, \pi_n \rangle$ that maximizes expected discounted return:

$$J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]$$

where $\tau$ represents a trajectory under policy $\pi$.

**The Partial Observability Challenge:**

The core difficulty is that each agent must make decisions based solely on its local action-observation history $h_i^t = (o_i^0, a_i^0, \dots, o_i^t)$, yet optimal coordination often requires reasoning about the global state $s$ and the hidden states/actions of teammates.

### The Core Problem: Random vs. Informed Masking

<div align="center">
  <img src="images/MARL/mae_original.PNG" alt="MAE Architecture" width="70%">
  <p><em>Standard Masked Auto-Encoder architecture - powerful, but uses random masking</em></p>
</div>

The baseline approach (MA²E) uses a masked auto-encoder where agent trajectories are randomly masked during training. The model learns to reconstruct these missing trajectories, helping agents infer what their teammates are doing even with partial observations.

**The limitation?** Random masking treats all agents equally, regardless of which ones are actually struggling with the task. This creates a disconnect between what the model is learning to reconstruct and what the agents actually need help with.

### CTDE and Value Decomposition Background

Our framework builds on the **Centralized Training with Decentralized Execution (CTDE)** paradigm, specifically using value decomposition methods like QMIX.

**Value Decomposition Principle:**

The key idea is to decompose the global Q-function $Q_{tot}(\tau, \mathbf{a})$ (where $\tau$ is the joint history and $\mathbf{a}$ is the joint action) into individual utility functions:

$$Q_{tot}(\tau, \mathbf{a}) = f_{\text{mix}}(Q_1(\tau_1, a_1), \dots, Q_n(\tau_n, a_n); s)$$

where:
- $Q_i(\tau_i, a_i)$ is agent $i$'s individual Q-value based only on its local history
- $f_{\text{mix}}$ is a mixing network that combines utilities (with access to global state $s$ during training)

**Monotonicity Constraint (IGM Principle):**

To ensure decentralized execution is optimal, QMIX enforces:

$$\frac{\partial Q_{tot}}{\partial Q_i} \geq 0, \quad \forall i$$

This guarantees that individual greedy action selection (choosing $\arg\max_{a_i} Q_i(\tau_i, a_i)$) corresponds to maximizing $Q_{tot}$.

**Temporal Difference Learning:**

During training, the QMIX algorithm minimizes the TD-error between predicted and target Q-values:

$$\delta = (r + \gamma \max_{\mathbf{a}'} Q_{tot}(\tau', \mathbf{a}', s'; \theta^-)) - Q_{tot}(\tau, \mathbf{a}, s; \theta)$$

where $r$ is the team reward, $\theta$ are the current network parameters, and $\theta^-$ are the parameters of a target network.

For our masking strategy, we compute per-agent TD-errors by decomposing this team-level signal. This error quantifies how surprised the model is by the observed reward and next state - a perfect measure of learning difficulty!

### Our Approach: Learning-Informed Masking

The key insight is pretty straightforward - if an agent is making large prediction errors (high TD-error), it's probably the one that needs the most help understanding the global state. So why not focus the auto-encoder's attention there?

#### Two-Phase Training

**Phase 1: Pre-training with Random Masking**
- Start with random policy data
- Use standard random masking
- Goal: Learn general agent representations

**Phase 2: Fine-tuning with TD-Error Masking**
- Switch to learned policy data
- Mask agents with highest TD-errors
- Goal: Focus on coordination challenges

This separation is important - you don't want to start with TD-error masking when the policy is completely random. Let the model learn the basics first, then get specific about what needs attention.

### Mathematical Formulation of the Masked Auto-Encoder

**Input Representation:**

The MAE operates on agent trajectory batches $\mathcal{T} = \{\tau_1, \tau_2, \dots, \tau_n\}$, where each trajectory is:

$$\tau_i = \{(o_i^{t-T+1}, a_i^{t-T+1}), \dots, (o_i^t, a_i^t)\}$$

representing observation-action pairs over a time window $T$.

**Masking Operation:**

We define a binary mask $M \in \{0,1\}^n$ where $M_i = 1$ indicates agent $i$'s trajectory should be masked. The corrupted input replaces masked trajectories with learnable $[\text{MASK}]$ tokens:

$$\tilde{\tau}_i = \begin{cases}
[\text{MASK}] & \text{if } M_i = 1 \\
\tau_i & \text{if } M_i = 0
\end{cases}$$

This agent-level masking forces the model to infer complete agent behaviors from partial team information.

**Reconstruction Objective:**

The MAE is trained to minimize the reconstruction loss:

$$\mathcal{L}_{MAE} = \frac{1}{|M|} \sum_{i: M_i=1} \|\tau_i - \hat{\tau}_i\|^2$$

where $\hat{\tau}_i = \text{Decoder}(\text{Encoder}(\tilde{\mathcal{T}}))$ is the reconstructed trajectory.

**Self-Attention Integration:**

For each agent $i$, the reconstructed trajectories are processed via self-attention:

$$\text{Context}_i = \text{Attention}(Q_i, K_{1:n}, V_{1:n})$$

where:
- Query: $Q_i = W_Q \hat{\tau}_i$ (agent's own reconstructed state)
- Keys: $K_j = W_K \hat{\tau}_j$ for all agents $j$
- Values: $V_j = W_V \hat{\tau}_j$ for all agents $j$

The attention mechanism computes:

$$\text{Attention}(Q_i, K, V) = \text{softmax}\left(\frac{Q_i K^T}{\sqrt{d_k}}\right) V$$

This allows agent $i$ to weight the importance of teammates' inferred states relative to its own situation.

**Transformer Architecture Details:**

The MAE uses an asymmetric encoder-decoder architecture:

**Encoder** (processes only visible trajectories):
- Input: Unmasked trajectories $\{\tau_i : M_i = 0\}$ 
- Embedding: $E_i = \text{Linear}(\tau_i) + PE_i$
  - $PE_i$ is positional encoding capturing both time-step and agent-ID
- $L_e$ Transformer layers with Multi-Head Self-Attention:
  - $\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$
  - Each head: $\text{head}_j = \text{Attention}(QW_j^Q, KW_j^K, VW_j^V)$
- Output: Latent representations $z_i$ for visible agents

**Decoder** (reconstructs full trajectory set):
- Input: $z_i$ for visible agents + learnable $[\text{MASK}]$ tokens for masked agents
- $L_d$ Transformer layers (typically $L_d < L_e$ for efficiency)
- Cross-attention to encoder outputs
- Output: Reconstructed trajectories $\hat{\tau}_{1:n}$ for all agents

The asymmetry is crucial - the encoder processes only partial information (mimicking execution), while the decoder leverages the full encoded context to reconstruct (mimicking centralized training).

### Adaptive Mean TD-Error Thresholding

The right scoring function wasn't obvious from the start. I tested several approaches, each with different theoretical motivations:

**What I Tried:**
1. **Exponential Moving Average (EMA)**: Tracked long-term performance with weighted history ($\alpha = 0.1$ for stability, $\alpha = 0.9$ for reactivity)
2. **Variance-Directional Score (VDS)**: Analyzed episode-level learning dynamics and "forgetting" patterns
3. **REFDS**: Focused on recent timesteps to catch immediate performance issues
4. **Observation-Weighted TD-Error**: Combined error magnitude with observation richness

**The Surprising Winner:**

After testing all these sophisticated approaches, the simplest method won - comparing each agent's TD-error to the current batch mean.

**TD-Error Per Agent:**

For each agent $i$ in the current training batch $B$, we compute its absolute TD-error from the individual utility function:

$$\delta_i = \left|Q_i(\tau_i, a_i) - \text{Target}_i\right|$$

where the target is the bootstrapped estimate from QMIX:

$$\text{Target}_i = r + \gamma \max_{a_i'} Q_i(\tau_i', a_i')$$

Note that while QMIX computes the global $Q_{tot}$ for training, we use the individual agent utilities $Q_i$ to determine which specific agents are struggling. This decomposition is natural since QMIX explicitly maintains per-agent Q-functions.

**Adaptive Threshold:**

The masking threshold is defined dynamically as the batch mean:

$$\theta_{\text{batch}} = \frac{1}{n} \sum_{i=1}^n \delta_i$$

**Masking Selection Rule:**

The mask $M$ is determined by:

$$M_i = \begin{cases}
1 & \text{if } \delta_i > \theta_{\text{batch}} \\
0 & \text{otherwise}
\end{cases}$$

**Top-k Fallback:** 

Late in training when all agents perform well ($\delta_i \approx \theta_{\text{batch}}$ for all $i$), we use a fallback mechanism:

$$\text{If } |\{i : M_i = 1\}| = 0, \text{ then mask top-}k \text{ agents by } \delta_i$$

This ensures the MAE continues fine-tuning even when performance converges.

**Algorithm:**

```
For each training batch B:
  1. Compute δᵢ = |TD-error| for each agent i
  2. Calculate θ_batch = mean(δ₁, ..., δₙ)
  3. M ← {i : δᵢ > θ_batch}
  4. If |M| = 0: M ← top-k agents by δᵢ
  5. Apply masking and train MAE on reconstructing M
```

**Why this works:**
- **Adaptive throughout training**: The threshold naturally adjusts as agents improve
- **No hyperparameters**: Only the top-k fallback value needs setting
- **Computationally cheap**: TD-errors already computed by MARL algorithm
- **Directly aligned**: Focuses on agents struggling with value estimation

**Mathematical Intuition:**

The key insight is that $\delta_i$ directly measures the Bellman residual for agent $i$. Agents with high $|\delta_i|$ are making poor predictions about future returns, indicating either:
1. Complex/uncertain local observations
2. Poor understanding of teammate behaviors
3. Challenging coordination requirements

By masking these agents, we force the MAE to prioritize reconstructing the global context precisely where coordination is most difficult.

### Theoretical Justification: Why TD-Error?

**Connection to Bellman Optimality:**

The TD-error $\delta_i$ is the temporal difference in Bellman backup:

$$\delta_i = \underbrace{r_i + \gamma V(s')}_{\text{Bellman target}} - \underbrace{V(s)}_{\text{Current estimate}}$$

In the limit of convergence, we have $\mathbb{E}[\delta_i | s] = 0$. Large $|\delta_i|$ indicates we're far from the Bellman optimality equation, meaning agent $i$ hasn't learned to predict returns accurately.

**Information-Theoretic View:**

We can interpret $|\delta_i|$ as quantifying the **predictive surprise** for agent $i$. Using KL divergence notation:

$$|\delta_i| \propto D_{KL}(P_{\text{target}} \| P_{\text{current}})$$

High surprise means the agent's current world model (encoded in $Q_i$) poorly matches reality. The MAE should focus its reconstruction capacity on these information-poor regions.

**Curriculum Learning Perspective:**

Our masking strategy implements a form of **automatic curriculum learning** \[Bengio et al., 2009\]:
- Early training: High variance in $\delta_i$ across agents → masks vary widely
- Mid training: $\theta_{\text{batch}}$ naturally decreases as policies improve → masking becomes more selective
- Late training: Top-k fallback ensures continued refinement on remaining challenges

This creates a natural progression from "learn general coordination" to "refine specific weaknesses."

**Comparison to Random Masking:**

Random masking uses:
$$M_i \sim \text{Bernoulli}(p)$$

This assigns equal probability to all agents regardless of learning state. In contrast, our approach uses:

$$P(M_i = 1) \propto \mathbb{I}[\delta_i > \theta_{\text{batch}}]$$

making masking probability directly proportional to learning difficulty. This aligns the self-supervised objective with the RL objective - a key theoretical advantage.

<div align="center">
  <img src="images/MARL/compare_scores.jpg" alt="Scoring strategies comparison" width="80%">
  <p><em>Comparison of all tested masking strategies - VDS actually hurt performance, while REFDS and Mean TD-Error Thresholding both worked well</em></p>
</div>

**Key Lessons:**
- EMA methods showed marginal improvement but had inertia issues
- VDS (based on episode variance) actually *degraded* performance significantly
- REFDS performed well but required hyperparameter tuning
- The batch-mean threshold won due to zero hyperparameters and direct alignment with the RL objective

### Integration with MARL Algorithms

The framework integrates cleanly with value-based CTDE methods like QMIX:

**Complete Architecture:**

1. **Backbone Network**: Each agent $i$ has a GRU that processes $h_i$ to produce hidden state $z_i$
2. **LI-MA²E Module**: Shared transformer-based MAE outputs $\hat{\tau}_{1:n}$
3. **Self-Attention**: Computes context-aware global representation $c_i = \text{Attention}(\hat{\tau}_i, \hat{\tau}_{1:n})$
4. **Aggregation**: Fuses local and global info: $\tilde{z}_i = [z_i; c_i]$
5. **Q-Network**: Maps $\tilde{z}_i$ to individual Q-values $Q_i(\tau_i, a_i)$

**Joint Training Objective:**

The total loss combines the MARL objective with MAE reconstruction:

$$\mathcal{L}_{total} = \mathcal{L}_{QMIX} + \lambda \mathcal{L}_{MAE}$$

where:

$$\mathcal{L}_{QMIX} = \mathbb{E}_{(s,\mathbf{a},r,s') \sim \mathcal{D}} \left[ \left(Q_{tot}(s, \mathbf{a}) - y \right)^2 \right]$$

with target $y = r + \gamma \max_{\mathbf{a}'} Q_{tot}(s', \mathbf{a}')$, and:

$$\mathcal{L}_{MAE} = \frac{1}{|M|} \sum_{i \in M} \left\| \tau_i - \hat{\tau}_i \right\|^2$$

The hyperparameter $\lambda$ balances task performance (via $\mathcal{L}_{QMIX}$) with representation quality (via $\mathcal{L}_{MAE}$).

**Two-Phase Training Protocol:**

**Phase 1 (Pre-training):**
- Sample trajectories from random policy $\pi_{\text{random}}$
- Use random masking: $M_i \sim \text{Bernoulli}(p_{\text{mask}})$
- Minimize only $\mathcal{L}_{MAE}$ to learn general representations
- Duration: $N_{\text{pre}}$ timesteps

**Phase 2 (Fine-tuning):**
- Sample from learning policy $\pi_\theta$
- Use TD-error adaptive masking (as defined above)
- Minimize $\mathcal{L}_{total}$ jointly
- The TD-errors $\delta_i$ guide which agents to mask
- Duration: $N_{\text{train}}$ timesteps

This two-phase approach prevents the instability that would arise from jointly training a random policy with a reconstruction task - you need competent base representations before the TD-error signal becomes meaningful.

**Computational Complexity:**

Let $n$ = number of agents, $T$ = trajectory length, $d$ = embedding dimension, $L_e$ = encoder layers, $L_d$ = decoder layers.

- **TD-error computation**: $O(n)$ - already done by MARL algorithm (no overhead!)
- **Masking selection**: $O(n \log n)$ for top-k, $O(n)$ for threshold (negligible)
- **MAE encoder**: $O(L_e \cdot (n-k)^2 \cdot T \cdot d)$ where $k$ is number masked
- **MAE decoder**: $O(L_d \cdot n^2 \cdot T \cdot d)$

The key insight: since we mask poorly-performing agents, $k$ tends to be small early (when errors are diverse) and the fallback top-k is small (typically $k \leq 2$). The transformer complexity is quadratic in sequence length, but our agent-level masking keeps sequences manageable even for moderate swarms ($n \leq 20$).

---

## 🛠️ Implementation & Experimental Setup

### Environment: StarCraft Multi-Agent Challenge (SMAC)

Tested on scenarios ranging from simple symmetric battles to complex asymmetric coordination:

| Scenario | Difficulty | Challenge | Key Skill Required |
|----------|-----------|-----------|-------------------|
| **3m** | Easy | 3v3 Marines | Focus-firing |
| **3s_vs_3z** | Medium | 3 Stalkers vs 3 Zealots | Kiting |
| **3s_vs_4z** | Medium | 3 Stalkers vs 4 Zealots | Kiting + Focus-fire |
| **3s_vs_5z** | Hard | 3 Stalkers vs 5 Zealots | Precise coordination |
| **8m** | Medium | 8v8 Marines | Large-scale focus-firing |

### Technical Details

- **Hardware**: NVIDIA RTX A5000 GPU (24.6GB VRAM), Intel Xeon W-2235 CPU
- **Training**: 1M+ timesteps per scenario
- **Framework**: PyMARL2 (built on PyTorch)
- **Baseline**: QMIX + MA²E with random masking

---

## 📊 Results

### Win Rate Comparison: Where It Really Matters

#### 3s_vs_5z (Hard Scenario)

<div align="center">
  <img src="images/MARL/test_battle_won_mean_3s_vs_5z_Mean_TD-Error_Thresholding_smoothed.png" alt="3s_vs_5z results" width="65%">
  <p><em>Performance on the hardest scenario - this is where the method shines</em></p>
</div>

**Key findings:**
- LI-MA²E reaches 90% win rate at ~600k timesteps
- Baseline needs 1.2M+ timesteps for similar performance
- **~2x speedup** in sample efficiency
- More stable convergence (baseline shows late-stage instability)

This map is brutal - three ranged units against five melee units. One mistake and you're done. The adaptive masking really helps here because it catches struggling agents early.

#### 3s_vs_3z (Medium Scenario)

<div align="center">
  <img src="images/MARL/comparison_plot_3s_vs_3z.png" alt="3s_vs_3z results" width="60%">
  <p><em>Medium difficulty - clear improvement in learning speed</em></p>
</div>

- LI-MA²E: Perfect win rate at ~400k timesteps
- Baseline: Needs ~600k timesteps
- Steeper learning curve = better sample efficiency

#### 3s_vs_4z (Medium Scenario)

<div align="center">
  <img src="images/MARL/comparison_plot_3s_vs_4z.png" alt="3s_vs_4z results" width="60%">
  <p><em>Interesting trade-off - baseline learns faster initially, but we converge more stably</em></p>
</div>

This one's interesting - the baseline actually learns faster initially, but our method achieves a more stable final policy. Sometimes the exploratory nature of random masking helps early on, but the guided approach wins out in the end.

#### 8m (Symmetric Scenario)

<div align="center">
  <img src="images/MARL/comparison_plot_8m.png" alt="8m results" width="60%">
  <p><em>Large homogeneous teams - baseline has the edge here</em></p>
</div>

Gotta be honest about this one - on the 8m map with homogeneous agents, random masking actually learns faster. This suggests the method works best in asymmetric scenarios where coordination is critical. Makes sense when you think about it.

### Policy Quality: Mean Test Return

Beyond just winning, we want to know if the policies are learning to win *efficiently* - with fewer casualties and in less time. The mean test return metric captures this.

<div align="center">
  <p><strong>Mean Test Return Across All Scenarios</strong></p>
</div>

**Key observations across scenarios:**

- **3s_vs_3z**: LI-MA²E rapidly converges to optimal return while baseline plateaus lower
- **3s_vs_4z**: Despite baseline's faster initial win rate, LI-MA²E learned the high-reward strategy almost immediately
- **3s_vs_5z**: Higher return with greater stability - avoids the baseline's late-stage drops
- **3m & 8m**: Reaches optimal return faster even on homogeneous maps

The return metric is more sensitive than win rate - it captures *how* you win. Across the board, LI-MA²E not only learns to complete the task but learns to do it optimally and efficiently.

### Summary of Results

| Metric | Simple Scenarios | Complex Asymmetric Scenarios |
|--------|-----------------|----------------------------|
| **Sample Efficiency** | Comparable | **~2x improvement** |
| **Final Win Rate** | Similar | **More stable** |
| **Policy Quality** | Faster convergence | **Significantly better** |
| **Learning Stability** | Good | **Superior** |

---

## 💬 Discussion

### What Worked Well

**The TD-Error Signal**: Using temporal difference error as the masking criterion turned out to be really effective. It's a direct measure of learning difficulty that's already computed by the MARL algorithm - no extra overhead.

**Adaptive Thresholding**: Instead of fixed thresholds or complex heuristics, just comparing to the batch mean works beautifully. It naturally adjusts throughout training without needing hyperparameter tuning.

**Two-Phase Training**: Pre-training with random masking before switching to TD-error masking is crucial. You need those general representations first.

### Where It's Most Useful

The results tell a clear story - this approach really shines in:
- **Asymmetric scenarios** where agents have different roles
- **Hard coordination tasks** where mistakes are costly
- **Scenarios requiring precise timing** (like kiting in 3s_vs_5z)

For simple symmetric tasks with homogeneous agents, random masking is sometimes sufficient or even faster initially. That's actually useful to know - tells you when the added complexity is worth it.

### Technical Insights

**Why Mean TD-Error Thresholding Won**: After testing exponential moving averages, variance-based scores, and observation-weighted metrics, the simple batch-mean threshold was the clear winner. Turns out trying to be too clever (tracking long-term history, analyzing episode dynamics) added instability. Sometimes simple really is better.

**The Stability Advantage**: On hard maps like 3s_vs_5z, the baseline shows late-stage performance drops that we don't see. I think this is because focusing on struggling agents helps the team learn more robust coordination patterns instead of brittle heuristics.

### Limitations and Future Work

**Scenario Dependence**: Performance varies across different map types. A meta-learning approach that adapts the masking strategy based on detected environment characteristics could help.

**Alternative Signals**: TD-error works well, but other signals might complement it - credit assignment metrics, communication patterns, or learned attention weights.

**Real-World Testing**: These are simulated environments. The real test will be deploying this on actual multi-robot systems with noisy sensors and communication delays.

**Computational Cost**: The transformer-based auto-encoder isn't free. For very large swarms, we'd need to think about efficiency optimizations.

---

## 🔗 References

[1] [Kang et al. "MA²E: Multi-Agent Masked Autoencoder for Efficient Multi-Agent Coordination"](https://arxiv.org)

[2] [Rashid et al. "QMIX: Monotonic Value Function Factorisation for Decentralised Multi-Agent Reinforcement Learning"](https://arxiv.org/abs/1803.11485)

[3] [Samvelyan et al. "The StarCraft Multi-Agent Challenge"](https://arxiv.org/abs/1902.04043)

[4] [He et al. "Masked Autoencoders Are Scalable Vision Learners"](https://arxiv.org/abs/2111.06377)

[5] [Vaswani et al. "Attention Is All You Need"](https://arxiv.org/abs/1706.03762)

---

<div align="center">
<p><em>Thanks for reading! Feel free to reach out if you want to chat about multi-agent RL or have ideas for improving this approach.</em></p>
</div>
