# Lab 3 : Mixed Precision Neural Architecture Search


## Task 1
> *1. In Tutorial 6, all layers allocated to IntegerLinear are allocated the same width and fractional width. This is suboptimal, as different layers may have different sensitivities to quantization.
>   a) Modify the code to allow different layers to have widths in the range [8, 16, 32] and fractional widths in the range [2, 4, 8]. Expose this choice as an additional hyperparameter for the Optuna sampler.
>   b) Run the search again, and plot a figure that has the number of trials on the x axis, and the maximum achieved accuracy up to that point on the y axis.*

### Setup

The search space is restricted to two layer types — `Linear` (full precision) and `LinearInteger` — using `restrict_search_space(search_space, ["Linear", "LinearInteger"])`. Each of BERT-tiny's ~14 Linear layers independently chooses between remaining at full precision (`Linear`) or being replaced with `LinearInteger`. When `LinearInteger` is chosen, the layer's `width` is sampled from {8, 16, 32} and `frac_width` from {2, 4, 8}, giving 9 possible integer configurations per layer. The per-layer model constructor (`make_model_constructor`) is used, so every layer makes independent decisions — no grouping is applied. The TPE sampler uses `multivariate=True`, `group=True`, and `n_startup_trials=15`.

```python
task1_space = restrict_search_space(search_space, ["Linear", "LinearInteger"])
obj_1 = make_objective(
    dataset, tokenizer, base_model, layer_map, task1_space,
    build_quant_config_fn=build_quant_config,
)
study_1 = run_search(obj_1, name="task1_integer", n_trials=20)
```

### Plot

![Quantization Effects](imgs/lab3/quantization%20effects%20-%20mixed%20precision%20search.png)

This cumulative maximum ("best accuracy so far") plot tracks the highest accuracy achieved across all trials up to each point. The x-axis is the trial number (1-20), and the y-axis is the running maximum accuracy. Note the narrow y-axis range: 0.830-0.860.

The curve shows three distinct phases:
1. **Trial 1** (~0.831): the very first random configuration already achieves high accuracy — well above the ~0.50 chance level that dominates early trials in the full multi-type search (Task 2).
2. **Trials 2-5** (0.831 → 0.859): rapid improvement, with the cumulative maximum climbing ~2.8 percentage points in just 4 trials. Trial 3 in particular makes a large jump to ~0.858.
3. **Trials 5-20** (plateau at ~0.859): the search converges completely. The remaining 15 trials fail to improve on the best configuration found at trial 5.

### Analysis

#### The integer-only search space is benign

The most striking feature of this plot is how narrow the accuracy range is. The worst trial achieves ~0.831 and the best ~0.859 — a spread of only ~0.028. This means that every random draw from the {Linear, LinearInteger} search space produces a viable model. No configuration collapses to chance level (~0.50), which stands in sharp contrast to Task 2's full 11-type search where ~37% of trials (32 out of 86) cluster near 0.50.

This happens because both available layer types are inherently safe:
- `Linear` preserves the pretrained weights exactly (zero quantisation noise).
- `LinearInteger` with any width in {8, 16, 32} preserves the majority of the pretrained information. Even the most aggressive setting (width=8, frac_width=2) provides 6 integer bits and 2 fractional bits 

There are no "trap" configurations in this space: unlike `LinearBinary` (which collapses weights to {-1, +1}) or `LinearLog` (which introduces high error near zero), every integer configuration retains sufficient representational capacity for fine-tuning.

#### Convergence 

With `n_startup_trials=15`, the first 15 trials use random sampling (no TPE guidance). Since the search converges by trial 5, all meaningful improvement comes from random exploration. TPE's density models only become active at trial 16, by which point the best accuracy has already plateaued. This makes intuitive sense: with ~14 layers, each choosing from {Linear + 9 integer configs} = 10 options, a randomly drawn configuration has a reasonable probability of assigning high-precision settings (width=16 or 32) to the most sensitive layers purely by chance. And even if some layers receive width=8, the overall model degradation is modest.



#### Implications for per-layer mixed precision

The rapid convergence also reveals that per-layer sensitivity differences, while real, are modest for integer quantisation. The gap between the worst random integer configuration (0.831) and the best (0.859) is only 2.8 points. This suggests that:

1. **Most layers are tolerant of 8-bit integer quantisation.** If the worst case (where some layers randomly get width=8) still achieves 0.831, the majority of layers must be robust to aggressive width reduction.
2. **A small number of sensitive layers drive the remaining improvement.** The jump from 0.831 to 0.859 likely comes from the search finding that specific layers (attention projections, pooler) benefit from width=16 or 32, matching what the Task 2 analysis found.
3. **Fractional width matters for attention layers.** The search space includes frac_width ∈ {2, 4, 8}, and the Task 2 top trials converged on `frac_width=8` for attention Q/K/V projections — consistent with the attention computation's sensitivity to numerical precision in the fractional part (which determines the resolution of dot-product similarity scores).

The plateau at trial 5 suggests the optimal per-layer integer configuration was found quickly, and the optimal assignment likely mirrors the pattern found in Task 2: higher width (16-32) for attention projections and the pooler head, with lower width (8-16) acceptable for FFN layers.

---

## Task 2

> *In Tutorial 6, all layers allocated to IntegerLinear are allocated the same width and fractional width. This is suboptimal, as different layers may have different sensitivities to quantization.
>  (a) Modify the code to allow different layers to have widths in the range [8, 16, 32] and fractional widths in the range [2, 4, 8]. Expose this choice as an additional hyperparameter for the Optuna sampler.
>  (b) Run the search again, and plot a figure that has the number of trials on the x axis, and the maximum achieved accuracy up to that point on the y axis.*

#### Layers Tested : 

```python
from chop.nn.quantized.modules.linear import (
  LinearInteger,
  LinearMinifloatDenorm,
  LinearMinifloatIEEE,
  LinearLog,
  LinearBlockFP,
  LinearBlockMinifloat,
  LinearBlockLog,
  LinearBinary,
  LinearBinaryScaling,
  LinearBinaryResidualSign,
)
```

#### Search Space : 

```python
{
  "linear_layer_choices": list(get_layer_map().keys()),

  # fixed-point
  "widths": [8, 16, 32],
  "frac_widths": [2, 4, 8],

  # exponent-based
  # Minifloat and Log layers both register <prefix>__exp_bias,
  # so their choice sets must be identical.
  "exponent_widths": [3, 4, 5],
  "exponent_biases": [3, 7, 15],
  "log_exponent_biases": [7, 15, 31],
  "exponent_bias_widths": [4, 5, 6],

  # block quant
  "block_sizes": [8, 16, 32],
  "skip_first_dim_choices": [True],

  # binary
  "binary_stochastic_choices": [False, True],
  "binary_bipolar_choices": [True],
  "binary_training_choices": [True],

  # residual sign
  # levels=1 is excluded: MASE's residual_sign_quantizer does not torch.stack the result for levels=1, so the tensor is 2-D, but
  # LinearBinaryResidualSign.forward() always indexes out_bin[l,:,:] (3 indices), causing IndexError on a 2-D tensor.
  # levels=4+ excluded: residual_sign_quantizer has no branch for them.
  "data_in_levels": [2, 3],
  "data_in_residual_sign": [True, False],
}
```


## Results

**Study:** `task2_grouped_tpe`

| Metric | Value |
|---|---|
| Total trials | 100 |
| Completed | 86 |
| Failed / pruned | 0 |

### Accuracy Statistics

| Statistic | Value |
|---|---|
| Mean | 0.6559 |
| Median | 0.5647 |
| Std | 0.1616 |
| Min | 0.4919 |
| Max | 0.8570 |
| Q25 | 0.5052 |
| Q75 | 0.8498 |

### Per-Precision Breakdown

| Dominant Precision Type | # Trials | Mean Acc | Best Acc | Worst Acc |
|---|:---:|:---:|:---:|:---:|
| Linear | 6 | 0.7550 | 0.8531 | 0.5265 |
| LinearBlockLog | 24 | 0.7483 | 0.8565 | 0.4984 |
| LinearInteger | 14 | 0.6795 | 0.8570 | 0.4949 |
| LinearMinifloatIEEE | 4 | 0.6588 | 0.8544 | 0.5120 |
| LinearBinaryResidualSign | 5 | 0.6430 | 0.8481 | 0.5043 |
| LinearBlockFP | 9 | 0.6305 | 0.8548 | 0.4986 |
| LinearMinifloatDenorm | 7 | 0.5862 | 0.8406 | 0.5002 |
| LinearLog | 4 | 0.5438 | 0.6574 | 0.4985 |
| LinearBlockMinifloat | 3 | 0.5348 | 0.6020 | 0.4979 |
| LinearBinary | 7 | 0.5004 | 0.5094 | 0.4919 |
| LinearBinaryScaling | 3 | 0.4973 | 0.5023 | 0.4929 |

### Top 5 Trials

| Rank | Trial # | Accuracy | Dominant Type | Layer Mix |
|:---:|:---:|:---:|---|---|
| 1 | 91 | **0.8570** | LinearInteger | Linear(1), LinearBlockLog(2), LinearBlockMinifloat(2), LinearInteger(3), LinearMinifloatIEEE(1) |
| 2 | 97 | 0.8565 | LinearBlockLog | Linear(1), LinearBlockFP(1), LinearBlockLog(2), LinearInteger(2), LinearMinifloatDenorm(2), LinearMinifloatIEEE(1) |
| 3 | 88 | 0.8557 | LinearBlockLog | LinearBlockFP(2), LinearBlockLog(3), LinearInteger(3), LinearMinifloatIEEE(1) |
| 4 | 90 | 0.8557 | LinearBlockLog | Linear(2), LinearBlockFP(1), LinearBlockLog(2), LinearBlockMinifloat(1), LinearInteger(1), LinearMinifloatDenorm(1), LinearMinifloatIEEE(1) |
| 5 | 96 | 0.8552 | LinearBlockLog | LinearBlockFP(1), LinearBlockLog(3), LinearBlockMinifloat(1), LinearInteger(3), LinearMinifloatIEEE(1) |

#### Best trial (#91) - per-layer configuration

| Layer | Type | Parameters |
|---|---|---|
| enc.early.attn_out | LinearBlockLog | block_size=32, exp_bias_width=5, width=16 |
| enc.early.attn_qkv | LinearInteger | frac_width=8, width=16 |
| enc.early.ffn_in | Linear | - |
| enc.early.ffn_out | LinearMinifloatIEEE | exp_width=3, width=16 |
| enc.mid.attn_out | LinearInteger | frac_width=2, width=32 |
| enc.mid.attn_qkv | LinearBlockMinifloat | block_size=16, exp_bias_width=6, exp_width=5, width=8 |
| enc.mid.ffn_in | LinearBlockMinifloat | block_size=16, exp_bias_width=5, exp_width=4, width=16 |
| enc.mid.ffn_out | LinearBlockLog | block_size=32, exp_bias_width=4, width=16 |
| head.pooler | LinearInteger | frac_width=2, width=32 |

---

#### Trial #97 - per-layer configuration

| Layer | Type | Parameters |
|---|---|---|
| enc.early.attn_out | LinearBlockLog | block_size=32, exp_bias_width=5, width=16 |
| enc.early.attn_qkv | LinearInteger | frac_width=8, width=16 |
| enc.early.ffn_in | Linear | - |
| enc.early.ffn_out | LinearMinifloatIEEE | exp_width=3, width=16 |
| enc.mid.attn_out | LinearMinifloatDenorm | exp_width=4, width=32 |
| enc.mid.attn_qkv | LinearBlockLog | block_size=16, exp_bias_width=6, width=8 |
| enc.mid.ffn_in | LinearMinifloatDenorm | exp_width=4, width=16 |
| enc.mid.ffn_out | LinearBlockFP | block_size=32, exp_width=3, width=16 |
| head.pooler | LinearInteger | frac_width=2, width=32 |

---

#### Trial #88 - per-layer configuration

| Layer | Type | Parameters |
|---|---|---|
| enc.early.attn_out | LinearBlockLog | block_size=32, exp_bias_width=5, width=16 |
| enc.early.attn_qkv | LinearInteger | frac_width=8, width=16 |
| enc.early.ffn_in | LinearBlockFP | block_size=16, exp_width=3, width=32 |
| enc.early.ffn_out | LinearMinifloatIEEE | exp_width=3, width=16 |
| enc.mid.attn_out | LinearInteger | frac_width=2, width=16 |
| enc.mid.attn_qkv | LinearBlockLog | block_size=16, exp_bias_width=6, width=16 |
| enc.mid.ffn_in | LinearBlockLog | block_size=32, exp_bias_width=5, width=8 |
| enc.mid.ffn_out | LinearBlockFP | block_size=32, exp_width=3, width=16 |
| head.pooler | LinearInteger | frac_width=2, width=32 |

---

#### Bottom 5 Trials (excluding accuracy=0)

| Rank | Trial # | Accuracy | Dominant Type | Layer Mix |
|:---:|:---:|:---:|---|---|
| 1 | 15 | 0.4919 | LinearBinary | LinearBinary(2), LinearBinaryScaling(1), LinearBlockFP(2), LinearBlockLog(1), LinearLog(1), LinearMinifloatDenorm(2) |
| 2 | 5 | 0.4929 | LinearBinaryScaling | LinearBinaryResidualSign(1), LinearBinaryScaling(2), LinearBlockFP(1), LinearBlockLog(1), LinearLog(2), LinearMinifloatDenorm(1), LinearMinifloatIEEE(1) |
| 3 | 8 | 0.4930 | LinearBinary | LinearBinary(2), LinearBinaryScaling(1), LinearBlockFP(2), LinearBlockLog(1), LinearLog(1), LinearMinifloatDenorm(2) |
| 4 | 78 | 0.4949 | LinearInteger | LinearBinaryResidualSign(1), LinearBinaryScaling(1), LinearBlockFP(1), LinearBlockLog(1), LinearBlockMinifloat(1), LinearInteger(2), LinearMinifloatDenorm(1), LinearMinifloatIEEE(1) |
| 5 | 13 | 0.4966 | LinearBinaryScaling | LinearBinaryResidualSign(1), LinearBinaryScaling(2), LinearBlockFP(1), LinearBlockLog(1), LinearLog(2), LinearMinifloatDenorm(1), LinearMinifloatIEEE(1) |

---
---

## Plots

### 1. Accuracy per Trial

![Accuracy per Trial](../imgs/lab3/accuracy%20per%20trial%20lab%204.2.png)

This plot shows the raw search trajectory: each point is one completed Optuna trial in chronological order, with dashed reference lines for the mean and median accuracy across all trials. The early trials (roughly 0-15, corresponding to the `n_startup_trials=15` random-exploration phase) show high variance and frequent low-accuracy outcomes, as expected from uniform random sampling over the full 11-type search space. After the startup phase, the plot shows a visible shift: later trials are increasingly concentrated in the high-accuracy band (>0.80), with far fewer trials falling near chance level (~0.50). This is direct evidence that TPE's density models are learning to focus on productive regions of the search space. The fact that accuracy does not steadily increase trial-by-trial (there are still occasional low-accuracy trials late in the search) is expected: TPE's acquisition function (Expected Improvement) naturally samples some points from under-explored regions of the search space, and with 11 possible layer types per group, some combinations remain relatively unexplored even after many trials.

### 2. Best Accuracy So Far (Cumulative Maximum)

![Best Accuracy So Far](../imgs/lab3/best%20acc%20so%20far.png)

This plot tracks the cumulative maximum accuracy as a function of the number of completed trials. A steep initial rise followed by a plateau indicates the search is efficiently converging, while a continuously rising curve indicates the search is still discovering better configurations. The curve shows rapid improvement within the first ~20 trials, reaching >0.85 accuracy relatively quickly, followed by a long plateau with only marginal gains. This suggests two things: (1) the grouped search space dimensionality reduction was effective - TPE found the high-performing region fast, even during the random startup phase, because the space was tractable enough for a good configuration to appear early; and (2) the top of the accuracy landscape is relatively flat - many different mixed-precision configurations achieve similar peak accuracy (~0.855-0.857), so further gains require fine-tuning rather than discovering entirely new architectural patterns. The best trial (#91, accuracy 0.8570) appeared late in the 100-trial sequence, indicating TPE was still making marginal improvements toward the end of the search budget.

### 3. Accuracy Distribution

![Accuracy Distribution](../imgs/lab3/acc%20dist.png)

This histogram shows the **distribution of final accuracy values** across all completed trials, with vertical markers for the mean and median. It reveals the overall character of the search space - whether most configurations are viable or whether only a narrow slice of the space produces good results. The distribution is clearly bimodal: a large cluster of trials near ~0.50 (chance-level for binary classification) and a second cluster above ~0.83. The mean (0.6559) is notably higher than the median (0.5647), confirming a right-skewed distribution where a minority of high-performing trials pull the mean up. This bimodality is likely driven by binary and log quantisation families dominating the left mode: when any of `LinearBinary`, `LinearBinaryScaling`, or `LinearLog` are the dominant type in a trial, accuracy collapses to near-chance. The right mode corresponds to trials where higher-precision formats (Integer, BlockLog, BlockFP, Minifloat) dominate. The gap between the two modes (~0.55-0.80) contains very few trials, meaning the search space has an almost binary outcome - a configuration either preserves enough representational capacity to learn, or it doesn't.

### 4. Accuracy by Precision Type 

![Accuracy by Precision](../imgs/lab3/acc%20by%20precision.png)

This box plot groups trials by their dominant precision type - defined as the most frequently occurring layer type among the trial's 9 searched functional groups (each group may contain multiple actual Linear layers, e.g. Q/K/V share one group). Note: this is an approximation; a trial labelled "LinearBlockLog-dominant" may still use Integer or Minifloat in other groups. The plot shows the accuracy distribution within each dominant-type group, sorted by median accuracy.

The plot, together with the per-precision breakdown table, reveals a clear precision hierarchy:

- **Top tier** (highest mean accuracy): `Linear` (full precision baseline, mean 0.7550), `LinearBlockLog` (mean 0.7483), and `LinearInteger` (mean 0.6795). `Linear` and `LinearBlockLog` also achieve best accuracies above 0.85, indicating they reliably preserve BERT's pretrained knowledge through quantisation. `LinearInteger` shows wider variance (worst 0.4949, best 0.8570) - it can match the best when well-configured but is more sensitive to its hyperparameters.
- **Mid tier** (wide spread): `LinearMinifloatIEEE` (mean 0.6588), `LinearBinaryResidualSign` (mean 0.6430), `LinearBlockFP` (mean 0.6305), and `LinearMinifloatDenorm` (mean 0.5862). These can achieve good accuracy (bests of 0.84-0.85) when paired with the right configuration and placed on the right layers, but are less robust - their worst-case accuracies are near chance.
- **Bottom tier** (mean ~0.50): `LinearLog` (mean 0.5438, best only 0.6574), `LinearBlockMinifloat` (mean 0.5348, best only 0.6020), `LinearBinary` (mean 0.5004), and `LinearBinaryScaling` (mean 0.4973). These formats consistently fail to learn beyond random chance when they dominate the model. The extremely tight ranges for `LinearBinary` (0.4919-0.5094) and `LinearBinaryScaling` (0.4929-0.5023) confirm that 1-bit quantisation fundamentally destroys the pretrained representations, regardless of other hyperparameter choices. By contrast, `LinearBlockLog`'s high mean and narrow best-worst spread (relative to other quantised types) indicates it is both high-performing and robust.

---
---


## Analysis

### TPE Sampler Effectiveness


1. **Trial concentration shifts over time.** The per-trial plot shows early trials scattered across the full accuracy range (0.49-0.85) during the 15-trial random startup, then increasingly concentrated in the high-accuracy regime (>0.80) afterwards - showing TPE learning productive density regions.

2. **Top trials appear late.** All five best trials (#88, 90, 91, 96, 97) occur after trial 85. TPE's multivariate density models have accumulated enough observations by this point to model the joint distribution of layer-type choices and their quantisation parameters accurately. 

3. **Converged layer-level preferences.** Across all top-5 trials, TPE converged on consistent per-group choices:

   | Functional Group | Converged Choice (all top-5 trials) |
   |---|---|
   | enc.early.attn_out | LinearBlockLog (block_size=32, exp_bias_width=5, width=16) |
   | enc.early.attn_qkv | LinearInteger (frac_width=8, width=16) |
   | enc.early.ffn_out | LinearMinifloatIEEE (exp_width=3, width=16) |
   | head.pooler | LinearInteger (frac_width=2, width=32) - 4 out of 5 |

   This convergence validates two design decisions: (a) the grouped search correctly identified that layers within the same functional role prefer the same quantisation, and (b) TPE's density models successfully captured these role-specific preferences. The remaining groups (`enc.early.ffn_in`, `enc.mid.*`) show more variation across top trials, suggesting the model is less sensitive to quantisation choices at those positions.

4. **Efficient trial allocation.** LinearBlockLog was the dominant type in 24 trials (the most of any type), while LinearBinary and LinearBinaryScaling dominated only 7 and 3 trials respectively. Since the "dominant" label reflects which type TPE selected most frequently across a trial's groups, this indicates TPE increasingly favoured BlockLog for multiple groups within the same trial - consistent with its density models learning that BlockLog is broadly effective. Conversely, binary types appeared as dominant less often because TPE learned to avoid them after early poor results.

---

### Layer and Hyperparameter Picks

**Attention Q/K/V projections prefer fixed-point integer quantisation.** 

Every top trial uses `LinearInteger` with `width=16, frac_width=8` for the early-stage Q/K/V projections. The Q/K/V matrices produce the query, key, and value vectors whose dot products determine attention weights via `softmax(QK^T / sqrt(d))`. This computation is sensitive to numerical precision for two reasons: (1) the dot product sums over the hidden dimension, so per-element quantisation errors accumulate across all d terms, and (2) the softmax function exponentiates these dot products, meaning even moderate additive errors in the logits can significantly shift the resulting attention distribution. Fixed-point 16-bit with 8 fractional bits provides a good balance of dynamic range (8 integer bits) and precision (8 fractional bits) to keep these errors small.

**Attention output projections prefer block logarithmic quantisation.** 

`LinearBlockLog` with `block_size=32, exp_bias_width=5, width=16` dominates this position. The attention output projection concatenates multi-head outputs and projects them back to the model dimension. Block-level log quantisation is well-suited here because each block of 32 values receives its own exponent bias, allowing the format to adapt to local weight and activation magnitudes within the matrix. This per-block adaptation accommodates heterogeneity in the weight distribution - different regions of the projection matrix may operate at different scales - whereas a single global quantisation scheme would have to clip large values or waste bits covering an unnecessarily wide range.

**FFN output prefers minifloat IEEE format.** 

`LinearMinifloatIEEE` with `exp_width=3, width=16` consistently appears at `enc.early.ffn_out`. The FFN output projection maps the intermediate activations (after GeLU) back to the model dimension. The minifloat format quantises both the weights and the incoming activations (data_in). On the activation side, GeLU produces an asymmetric distribution (positive values pass through, negative values are suppressed near zero), and a floating-point representation handles this better than fixed-point because it allocates finer resolution near zero where most suppressed values lie, while still representing larger positive values. On the weight side, floating-point formats are more tolerant of occasional outlier weights than fixed-point, which must either clip them or waste bits on a wide integer range.

**The pooler head needs high precision.** 

4 of 5 top trials use `LinearInteger` with `width=32, frac_width=2` - the widest bit-width available (32 bits) - at `head.pooler`. This is the layer that directly feeds the classification head and determines the model's output logits. Quantisation errors here propagate directly to the cross-entropy loss with no opportunity for subsequent layers to compensate. At 32 bits total, the quantisation error is already extremely small regardless of how those bits are split between integer and fractional parts, so the specific choice of `frac_width=2` matters less than the overall width - TPE's key finding here is that the pooler requires the maximum available bit-width.

**FFN intermediate and mid-stage layers are less sensitive.** 

The `enc.early.ffn_in` and `enc.mid.*` groups show the most variation across top trials - they use Linear, BlockFP, BlockLog, BlockMinifloat, or MinifloatDenorm interchangeably without significant accuracy impact. This suggests these positions are tolerant of aggressive quantisation, making them prime candidates for bit-width reduction in a deployment scenario where minimising model size is the goal.

---

### Combinations Leading to Training Instability or Failure to Learn

**Binary quantisation at attention layers destroys the attention mechanism.** 

`LinearBinary` and `LinearBinaryScaling` reduce weights to `{-1, +1}` (1-bit). In bottom trial #15 (accuracy 0.4919), `LinearBinary` is used at `enc.early.attn_out` and `head.pooler`. Binarising the attention output projection means the multi-head concatenation is projected through a matrix of only +1 and -1 values - this cannot preserve the fine-grained per-head information that attention relies on. Similarly, binarising the pooler collapses the model's final representation to a 1-bit signal before classification. The result is that the model cannot produce meaningfully different logits for different inputs, so it converges to always predicting the majority class (accuracy ~0.50 = random chance on a balanced 2-class task).

**LinearLog struggles because BERT's weight distributions are poorly suited to log-space representation.** 

Log quantisation maps values to powers of two, which is efficient for data with large dynamic range but introduces high relative error for values near zero. BERT's pretrained linear-layer weights are roughly normally distributed around zero - the regime where log quantisation is weakest. The per-precision breakdown shows `LinearLog`'s best accuracy is only 0.6574 (far below other formats' bests of 0.85+), and its mean is 0.5438. Even the best-case log configuration cannot preserve the pretrained weight information well enough.

**LinearBlockMinifloat underperforms despite a correct STE gradient patch.** 

The original MASE `BlockMinifloatQuantize.backward()` declared extra positional arguments that PyTorch's autograd never passes, causing crashes. The fix replaces it with a standard straight-through estimator (`return grad_output, None, None, None, None, None`) - the same STE approach used successfully by other quantisation layers. The patch itself is correct, so the poor performance (best 0.6020 across only 3 dominant trials) is more likely attributable to the block minifloat format itself: it introduces both block-level and element-level quantisation noise (via the per-block exponent bias), and the interaction of these two sources of noise may make the loss landscape harder to optimise during fine-tuning. With only 3 trials dominated by this type, the sample size is also too small to draw strong conclusions - TPE may simply not have found a good hyperparameter configuration for it.

**Mixing too many different precision families in one model corrupts the forward-pass signal.** 

Bottom trial #78 (accuracy 0.4949) uses 8 different layer types across 9 groups. The fundamental problem is not gradient-magnitude mismatch (all quantisation layers use the straight-through estimator, which passes gradients through approximately unchanged). Rather, the issue is that each quantisation format introduces a different kind and magnitude of forward-pass noise: binary layers inject extreme noise (rounding to {-1, +1}), while 32-bit integer layers introduce negligible noise. When these layers are composed in sequence, the heavy noise from binary layers propagates through the entire network, overwhelming the precise representations maintained by higher-precision layers. The result is that the model's forward pass produces unreliable loss signals, making optimisation ineffective regardless of the learning rate. The grouped search mitigates this by keeping the number of distinct types per model low, but it cannot entirely prevent such combinations when TPE explores the boundary of the search space.

**Early-stage layers are more critical than mid-stage layers.** 

Comparing top and bottom trials reveals that the early encoder stage quantisation choices drive accuracy more than mid-stage choices. With only 2 encoder layers in BERT-tiny (layer 0 = early, layer 1 = mid), the error-compounding depth difference is modest. However, the early encoder layer processes the raw token embeddings, whose statistical properties differ from the residual-stream representations at layer 1 (which have already been refined by attention and FFN). Quantising these initial embeddings aggressively introduces errors before any representational structure has been established, and the residual connections propagate these errors through the rest of the network. The top-trial convergence table shows all 5 best trials agree on early-stage attention choices (LinearInteger for Q/K/V, LinearBlockLog for output) while mid-stage choices vary freely - directly confirming that early-stage sensitivity is higher.

---
---

## Code & Methodology

```python
def run_search(
    objective,
    name: str = "search",
    sampler=None,
    direction: str = "maximize",
    n_trials: int = 50,
    timeout: int = 60 * 60,
) -> optuna.Study:

  storage = make_storage(name)

  if sampler is None:
    sampler = make_tpe_sampler(SEED)

  study = optuna.create_study(
    direction=direction,
    study_name=name,
    sampler=sampler,
    storage=storage,
    load_if_exists=True,
  )

  existing = len(study.trials)
  remaining = max(0, n_trials - existing)

  if remaining > 0:
    study.optimize(objective, n_trials=remaining, timeout=timeout)
  else:
    print(f"[skip] Study '{name}' already has {existing}/{n_trials} trials.")

  return study
```

---


### Dimensionality Reduction via Grouped Search

Grouping the possible choices transforms the search into a manageable structured one. If each module got its own categorical choice of layer type, and if the chosen type required quantization parameters, those were sampled independently as well. On BERT-tiny, this creates dozens of independent decisions. On BERT-base, it creates over a hundred.

**Approach: per-functional-group decisions.** Instead of treating every linear layer as unique, the approach uses the fact that linear layers within a Transformer serve specific functional roles, and layers with the same role are likely to benefit from the same quantization strategy. The model's linear layers are organized into a small number of functional groups:

- **Attention Q/K/V projections** - the three projections that create queries, keys, and values for self-attention. These operate on similar data distributions and have similar computational characteristics.
- **Attention output projection** - the projection applied after multi-head attention is concatenated. This has a different role from Q/K/V and may benefit from different precision.
- **FFN intermediate** - the first (typically larger) linear layer in the feed-forward network. This is often the most parameter-heavy layer and a prime candidate for aggressive quantization.
- **FFN output** - the second linear layer in the feed-forward network, projecting back to the model dimension.
- **Pooler and classifier head** - the final layers that produce task-specific outputs. These are better kept at full precision because they directly influence the loss.

Optuna samples one layer type per group and one set of quantization knobs per group, rather than per module. All linear layers within a group share the same configuration. Optionally, groups can be further split by depth stage (early, middle, late encoder layers), though this is less important for small models like BERT-tiny.

With this approach, the effective dimensionality of the search space collapses from dozens or hundreds of independent decisions to roughly 5-12 meaningful ones. TPE's density models can now better capture real patterns, such as "FFN layers prefer block floating-point formats" or "attention projections work best with 8-bit integer quantization" or "the classifier head should remain at full precision." The objective also becomes less noisy because each trial changes a small number of coherent architectural decisions rather than randomly perturbing every layer independently.


```python
def make_grouped_model_constructor(
    base_model: nn.Module,
    layer_map: dict[str, type],
    search_space: dict,
    build_quant_config_fn=None,
    *,
    stage_mode: str = "none",  # "none" or "3stage"
    group_fn: Callable[..., Optional[str]] = bert_linear_group,
    force_full_precision_groups: set[str] | None = None,
):

    if build_quant_config_fn is None:
        build_quant_config_fn = _default_build_quant_config

    if force_full_precision_groups is None:
        force_full_precision_groups = {"head.classifier"}

    layer_choices_key = "linear_layer_choices"
    num_hidden_layers = getattr(getattr(base_model, "config", None), "num_hidden_layers", None)

    def construct_model(trial: optuna.trial.Trial) -> nn.Module:
        trial_model = deepcopy(base_model)

        # Collect Linear layers and their groups
        linear_entries: list[tuple[str, nn.Linear, str]] = []
        group_counts: dict[str, int] = {}

        for name, layer in list(trial_model.named_modules()):
            if not isinstance(layer, torch.nn.Linear):
                continue

            group = group_fn( name, num_hidden_layers=num_hidden_layers, stage_mode=stage_mode,) or "other.linear"
            linear_entries.append((name, layer, group))
            group_counts[group] = group_counts.get(group, 0) + 1


        # Sample one choice per group
        group_choices: dict[str, dict[str, Any]] = {}
        for group in sorted(set(g for _, _, g in linear_entries)):
            if group in force_full_precision_groups:
                group_choices[group] = {"type": "Linear", "config": None}
                continue

            layer_cls_name = trial.suggest_categorical(f"{sanitize_prefix(group)}__type",search_space[layer_choices_key])

            config = None
            new_layer_cls = layer_map[layer_cls_name]
            if new_layer_cls is not torch.nn.Linear and ctor_accepts_arg(new_layer_cls, "config"):
                config = build_quant_config_fn(trial, layer_cls_name, prefix=group)

            group_choices[group] = {"type": layer_cls_name, "config": config}


        # Apply to every Linear layer
        layer_choices: list[dict] = []
        for name, layer, group in linear_entries:
            choice = group_choices[group]
            layer_cls_name = choice["type"]
            new_layer_cls = layer_map[layer_cls_name]

            choice_info: dict = {
                "name": name,
                "group": group,
                "type": layer_cls_name,
                "shape": (layer.in_features, layer.out_features, layer.bias is not None),
                "config": choice.get("config"),
            }
            layer_choices.append(choice_info)

            if new_layer_cls is torch.nn.Linear:
                continue

            kwargs: dict[str, Any] = {
                "in_features": layer.in_features,
                "out_features": layer.out_features,
                "bias": layer.bias is not None,
            }

            if ctor_accepts_arg(new_layer_cls, "config"):
                kwargs["config"] = choice.get("config")

            inner = new_layer_cls(**kwargs)

            # Copy pretrained weights
            with torch.no_grad():
                inner.weight.copy_(layer.weight)
                if layer.bias is not None and getattr(inner, "bias", None) is not None:
                    inner.bias.copy_(layer.bias)

            # ResidualSign often wants 2D input; adapter handles flatten/reshape.
            if layer_cls_name == "LinearBinaryResidualSign":
                new_layer = QuantLinearAdapter(
                    inner,
                    out_features=layer.out_features,
                    force_2d_input=True,
                    auto_retry=True,
                )
            else:
                new_layer = inner

            deepsetattr(trial_model, name, new_layer)

        return trial_model

    return construct_model
```

---

### Conditional Hyperparameters

The code has conditional parameter sampling - it only built a quantization config if the chosen layer type's constructor accepted one. 

For each functional group, the conditional sampling works as follows:

- If the group is assigned `LinearInteger`, Optuna samples `width` and `frac_width` for that group.
- If the group is assigned `LinearLog`, Optuna samples `width` and `exp_bias` for that group.
- If the group is assigned `LinearMinifloatDenorm` or `LinearMinifloatIEEE`, Optuna samples `width` and `exp_width` for that group (the exponent bias is derived automatically as `2^(exp_width-1) - 1`, following the IEEE 754 convention).
- If the group is assigned standard `Linear`, no additional parameters are sampled.

TPE was designed for tree-structured conditional spaces. When a parameter only exists conditionally (e.g., `frac_width` only matters when the layer type is `LinearInteger`), TPE can model it cleanly by only considering it in the density estimation when its parent condition is met. Removing parameters that don't apply to the current layer type reduces the effective dimensionality further and prevents the density models from being diluted by irrelevant dimensions.

```python
_CONFIG_DISPATCH: dict[str, callable] = {
  "LinearInteger": _fixed_point_config,
  "LinearMinifloatDenorm": _minifloat_config,
  "LinearMinifloatIEEE": _minifloat_config,
  "LinearLog": _log_config,
  "LinearBlockFP": _block_fp_config,
  "LinearBlockMinifloat": _block_minifloat_config,
  "LinearBlockLog": _block_log_config,
  "LinearBinary": _binary_config,
  "LinearBinaryScaling": _binary_scaling_config,
  "LinearBinaryResidualSign": _residual_sign_config,
}


def build_quant_config(
  trial,
  layer_type: str,
  prefix: str,
  search_space: dict | None = None,
) -> dict:
  if search_space is None:
    search_space = get_search_space()
  builder = _CONFIG_DISPATCH.get(layer_type, _fixed_point_config)
  return builder(trial, prefix, search_space)
```

---

### Sampler Configuration for Correlated Parameters

The sampler configuration enables several features that are useful for this type of search space.

**`multivariate=True`:** By default, TPE models each hyperparameter independently - it builds separate one-dimensional density models for each parameter. With `multivariate=True`, TPE builds a joint density model over multiple parameters simultaneously. This allows it to capture interactions between parameters, such as the correlation between `width` and `frac_width` in integer quantization, or the relationship between `block_size` and `exp_width` in block floating-point formats. Without this, TPE might learn that `width=8` is generally good and `frac_width=4` is generally good, but it cannot learn that the specific combination `(8, 4)` is particularly effective.

**`group=True`:** When supported by the Optuna version, this feature groups correlated parameters together for density estimation. Parameters that are sampled together (like the quantization knobs for a single functional group) are treated as a unit, which improves the quality of the density models.

**`n_startup_trials=15`:** This sets the number of initial random trials before TPE begins using its density models to guide the search. The default is typically 10, but with a more structured search space, a slightly larger startup phase ensures that TPE has enough diverse observations to build reliable initial models. Too few startup trials can cause TPE to prematurely converge on a suboptimal region.

Quantization parameters are inherently correlated. The bit-width, fractional width, exponent width, and block size of a quantization format interact to determine its representational capacity and precision characteristics. Modeling these parameters independently throws away critical information.

---

### Pruning 

The original objective function trained every trial for a full epoch, regardless of how poorly the model was performing during training. A trial that achieves 10% accuracy after processing half the training data is almost certainly not going to produce a competitive result at the end of the epoch, but the original code would dutifully continue training it until completion. The code integrates an Optuna pruning callback (when the training framework supports it). Pruning works by periodically reporting intermediate objective values during training. If the optimizer determines that the current trial is unlikely to beat the best-known result (based on its intermediate performance relative to other trials), it terminates the trial early and moves on to the next one. Pruning allows the same computational budget to produce more completed trials. Since TPE's effectiveness is directly proportional to the number of informative data points it can use for density estimation, getting more trials within the same wall-clock time translates directly into better optimization performance. Additionally, pruned trials still contribute partial information to the optimizer, so the data isn't entirely wasted.

**Note:** In this particular study, 0 trials were pruned (86 completed, 0 failed/pruned out of 100 total). The 14 non-completed trials timed out or were interrupted during execution. The lack of pruned trials may be because single-epoch training is short enough that the intermediate evaluation checkpoints reported to the pruner did not trigger early stopping - the pruning callback needs sufficient intermediate reports to make a confident decision, and a single epoch may not provide enough.

```python
def make_objective(
    dataset,
    tokenizer,
    base_model: nn.Module,
    layer_map: dict[str, type],
    search_space: dict,
    num_train_epochs: int = 1,
    build_quant_config_fn=None,
    *,
    construct_model_fn=None,
    constructor_factory=make_model_constructor,
    enable_pruning: bool = True,
    prune_metric: str = "eval_accuracy",
):

  if build_quant_config_fn is None:
    build_quant_config_fn = _default_build_quant_config

  if construct_model_fn is None:
    construct_model = constructor_factory(
        base_model, layer_map, search_space, build_quant_config_fn,
    )
  else:
    construct_model = construct_model_fn


  def objective(trial: optuna.trial.Trial) -> float:
      trainer = None
      model = None
      acc = 0.0
      metrics = None
      exception_str = None
      try:
        model = construct_model(trial)

        trainer = get_trainer(
          model=model,
          tokenized_dataset=dataset,
          tokenizer=tokenizer,
          evaluate_metric="accuracy",
          num_train_epochs=num_train_epochs,
        )

        if enable_pruning:
            try:
              from optuna.integration import HuggingFacePruningCallback
              if hasattr(trainer, "add_callback"):
                trainer.add_callback(
                  HuggingFacePruningCallback(trial, prune_metric)
                )
            except Exception:
              pass

        trainer.train()
        metrics = trainer.evaluate()

        save_model_for_trial(trial, trainer.model, tag="final")

        acc = float(metrics.get("eval_accuracy", 0.0))
        return acc

      except Exception as e:
        traceback.print_exc()
        exception_str = repr(e)
        trial.set_user_attr("exception", exception_str)
        return 0.0

      finally:
        # Persist config n result JSON to Drive after every trial
        try:
          save_trial_result(
            trial,
            accuracy=acc,
            metrics=metrics,
            exception=exception_str,
          )
        except Exception:
          pass  # don't let a save failure kill the study

        del trainer, model
        if torch.cuda.is_available():
          torch.cuda.empty_cache()

  return objective
```

---

###  Adapters to Prevent Trial Failures

The adapter supports bidirectional shape conversion:

- **3D → 2D flattening:** When `force_2d_input` is enabled, the adapter reshapes `[batch, sequence, hidden]` tensors to `[batch * sequence, hidden]` before passing them to the quantization layer. After the forward pass, it reshapes the output back to `[batch, sequence, out_features]`.
- **2D → 3D unsqueezing:** The original behavior is preserved for layers that require 3D input.
- **Automatic retry:** If a layer raises a shape-related assertion error, the adapter can attempt the alternative shape convention before failing.

Sparse-to-dense conversion (for layers that produce sparse tensor outputs) is also retained from the original implementation. Every trial that fails due to a shape mismatch produces a garbage data point - typically recorded as 0 accuracy. These garbage points distort TPE's density models by making certain regions of the search space appear catastrophically bad when they're actually just buggy. By ensuring that trials succeed or fail based on the actual quality of the quantization approach rather than implementation artifacts, the density models become more reliable and TPE can make better-informed sampling decisions.


```python
class QuantLinearAdapter(nn.Module):
    """
    Shape + sparsity adapter for quantized linear layers.

    Handles:
    - Optional 3-D input enforcement (some quant layers want [B,1,H]).
    - Optional 2-D input enforcement (some quant layers assert input.dim()==2).
    - Flattened-output reshaping (some quant layers collapse B*S).
    - Sparse-to-dense conversion (HF attention expects dense tensors).
    """

    def __init__(
        self,
        inner: nn.Module,
        out_features: int,
        force_3d_input: bool = False,
        force_2d_input: bool = False,
        auto_retry: bool = True,
    ):
        super().__init__()
        if force_3d_input and force_2d_input:
            raise ValueError("force_3d_input and force_2d_input cannot both be True.")

        self.inner = inner
        self.out_features = int(out_features)
        self.force_3d_input = bool(force_3d_input)
        self.force_2d_input = bool(force_2d_input)
        self.auto_retry = bool(auto_retry)

    @property
    def weight(self):
        return getattr(self.inner, "weight", None)

    @property
    def bias(self):
        return getattr(self.inner, "bias", None)

    def _call_inner(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)

    def _to_2d(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten everything except the last dim.
        # [B,S,H] -> [B*S,H], [B,1,H] -> [B,H], etc.
        if x.dim() == 2:
            return x
        return x.reshape(-1, x.shape[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_shape = (*x.shape[:-1], self.out_features)

        x_in = x
        squeezed_seq = False

        # Primary (requested) reshape
        if self.force_2d_input:
            x_in = self._to_2d(x_in)
        elif self.force_3d_input and x_in.dim() == 2:
            x_in = x_in.unsqueeze(1)
            squeezed_seq = True

        # Call inner, with optional fallback if it asserts on shape
        try:
            y = self._call_inner(x_in)
        except AssertionError:
            if not self.auto_retry:
                raise
            # Common failure: layer asserts input is 2D, but BERT supplies 3D.
            # Retry with flattened 2D input.
            y = self._call_inner(self._to_2d(x))
        except RuntimeError:
            if not self.auto_retry:
                raise
            # Retry path for "expected 2D/3D" type runtime errors
            if x.dim() >= 3:
                y = self._call_inner(self._to_2d(x))
            elif x.dim() == 2:
                y = self._call_inner(x.unsqueeze(1))
                squeezed_seq = True
            else:
                raise

        # Sparse -> dense (HF often expects dense)
        if getattr(y, "is_sparse", False):
            y = y.to_dense()

        # Undo forced 3D for 2D callers
        if squeezed_seq and y.dim() == 3 and y.shape[1] == 1:
            y = y.squeeze(1)

        # Fix shape if the quant kernel flattened B*S
        if tuple(y.shape) != tuple(target_shape):
            if y.dim() == 2 and len(target_shape) == 3:
                B, S = target_shape[0], target_shape[1]
                if y.shape[0] == B * S and y.shape[1] == self.out_features:
                    y = y.reshape(B, S, self.out_features)
                elif y.numel() == B * S * self.out_features:
                    y = y.reshape(B, S, self.out_features)
                else:
                    raise RuntimeError(f"Cannot reshape {tuple(y.shape)} -> {target_shape}")
            elif y.numel() == math.prod(target_shape):
                y = y.reshape(*target_shape)
            else:
                raise RuntimeError(f"Cannot reshape {tuple(y.shape)} -> {target_shape}")

        return y
```

#### Comparison with Task 1


| Property | Task 1 (Integer-only) | Task 2 (All 11 types) |
|---|---|---|
| Trials run | 20 | 100 |
| Best accuracy | ~0.859 | 0.857 |
| Worst-case accuracy | ~0.831 | 0.4919 |
| Accuracy spread | ~0.028 | 0.365 |
| Trials to reach peak | ~5 | ~91 |
| First trial accuracy | ~0.831 | ~0.50 |
| Cummax at trial 15 | ~0.859 | ~0.51 |
