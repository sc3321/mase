# Lab 3 : Mixed Precision Neural Architecture Search

## Part 1 : Tasks 

### Task 1
> *1. In Tutorial 6, all layers allocated to IntegerLinear are allocated the same width and fractional width. This is suboptimal, as different layers may have different sensitivities to quantization.
>   a) Modify the code to allow different layers to have widths in the range [8, 16, 32] and fractional widths in the range [2, 4, 8]. Expose this choice as an additional hyperparameter for the Optuna sampler.
>   b) Run the search again, and plot a figure that has the number of trials on the x axis, and the maximum achieved accuracy up to that point on the y axis.*


#### Plots 

![Quantization Effects](imgs/quantization%20effects%20-%20mixed%20precision%20search.png)


#### Analysis

---

### Task 2

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

#### Analysis


---

## Code 

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

## Optimization Techniques 

### Dimensionality Reduction via Grouped Search

Grouping the possible choices transforms the search into a manageable structured one. If each module got its own categorical choice of layer type, and if the chosen type required quantization parameters, those were sampled independently as well. On BERT-tiny, this creates dozens of independent decisions. On BERT-base, it creates over a hundred.

**Approach: per-functional-group decisions.** Instead of treating every linear layer as unique, the approach uses the fact that linear layers within a Transformer serve specific functional roles, and layers with the same role are likely to benefit from the same quantization strategy. The model's linear layers are organized into a small number of functional groups:

- **Attention Q/K/V projections** — the three projections that create queries, keys, and values for self-attention. These operate on similar data distributions and have similar computational characteristics.
- **Attention output projection** — the projection applied after multi-head attention is concatenated. This has a different role from Q/K/V and may benefit from different precision.
- **FFN intermediate** — the first (typically larger) linear layer in the feed-forward network. This is often the most parameter-heavy layer and a prime candidate for aggressive quantization.
- **FFN output** — the second linear layer in the feed-forward network, projecting back to the model dimension.
- **Pooler and classifier head** — the final layers that produce task-specific outputs. These are better kept at full precision because they directly influence the loss.

Optuna samples one layer type per group and one set of quantization knobs per group, rather than per module. All linear layers within a group share the same configuration. Optionally, groups can be further split by depth stage (early, middle, late encoder layers), though this is less important for small models like BERT-tiny.

With this approach, the effective dimensionality of the search space collapses from dozens or hundreds of independent decisions to approximately maybe 5–12 meaningful ones. TPE's density models can now better capture real patterns, such as "FFN layers prefer block floating-point formats" or "attention projections work best with 8-bit integer quantization" or "the classifier head should remain at full precision." The objective also becomes less noisy because each trial changes a small number of coherent architectural decisions rather than randomly perturbing every layer independently.


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

### Conditional Hyperparameters

The code has conditional parameter sampling — it only built a quantization config if the chosen layer type's constructor accepted one. 

For each functional group, the conditional sampling works as follows:

- If the group is assigned `LinearInteger`, Optuna samples `width` and `frac_width` for that group.
- If the group is assigned `LinearLog`, Optuna samples the log bias for that group.
- If the group is assigned `LinearMinifloat`, Optuna samples `exp_width` and `man_width` (with bias derived automatically).
- If the group is assigned standard `Linear`, no additional parameters are sampled.

TPE was explicitly designed for tree-structured conditional spaces. When a parameter only exists conditionally (e.g., `frac_width` only matters when the layer type is `LinearInteger`), TPE can model it cleanly by only considering it in the density estimation when its parent condition is met. Removing parameters that don't apply to the current layer type reduces the effective dimensionality further and prevents the density models from being diluted by irrelevant dimensions.

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

### Sampler Configuration for Correlated Parameters

The sampler configuration enables several features that are useful for this type of search space.

**`multivariate=True`:** By default, TPE models each hyperparameter independently — it builds separate one-dimensional density models for each parameter. With `multivariate=True`, TPE builds a joint density model over multiple parameters simultaneously. This allows it to capture interactions between parameters, such as the correlation between `width` and `frac_width` in integer quantization, or the relationship between `block_size` and `exp_width` in block floating-point formats. Without this, TPE might learn that `width=8` is generally good and `frac_width=4` is generally good, but it cannot learn that the specific combination `(8, 4)` is particularly effective.

**`group=True`:** When supported by the Optuna version, this feature groups correlated parameters together for density estimation. Parameters that are sampled together (like the quantization knobs for a single functional group) are treated as a unit, which improves the quality of the density models.

**`n_startup_trials=15`:** This sets the number of initial random trials before TPE begins using its density models to guide the search. The default is typically 10, but with a more structured search space, a slightly larger startup phase ensures that TPE has enough diverse observations to build reliable initial models. Too few startup trials can cause TPE to prematurely converge on a suboptimal region.

Quantization parameters are inherently correlated. The bit-width, fractional width, exponent width, and block size of a quantization format interact to determine its representational capacity and precision characteristics. Modeling these parameters independently throws away critical information.

### Pruning 

The original objective function trained every trial for a full epoch, regardless of how poorly the model was performing during training. A trial that achieves 10% accuracy after processing half the training data is almost certainly not going to produce a competitive result at the end of the epoch, but the original code would dutifully continue training it until completion. The code integrates an Optuna pruning callback (when the training framework supports it). Pruning works by periodically reporting intermediate objective values during training. If the optimizer determines that the current trial is unlikely to beat the best-known result (based on its intermediate performance relative to other trials), it terminates the trial early and moves on to the next one. Pruning allows the same computational budget to produce more completed trials. Since TPE's effectiveness is directly proportional to the number of informative data points it can use for density estimation, getting more trials within the same wall-clock time translates directly into better optimization performance. Additionally, pruned trials still contribute partial information to the optimizer, so the data isn't entirely wasted.

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

### 3.6  Adapters to Prevent Trial Failures

The adapter supports bidirectional shape conversion:

- **3D → 2D flattening:** When `force_2d_input` is enabled, the adapter reshapes `[batch, sequence, hidden]` tensors to `[batch * sequence, hidden]` before passing them to the quantization layer. After the forward pass, it reshapes the output back to `[batch, sequence, out_features]`.
- **2D → 3D unsqueezing:** The original behavior is preserved for layers that require 3D input.
- **Automatic retry:** If a layer raises a shape-related assertion error, the adapter can attempt the alternative shape convention before failing.

Sparse-to-dense conversion (for layers that produce sparse tensor outputs) is also retained from the original implementation. Every trial that fails due to a shape mismatch produces a garbage data point — typically recorded as 0 accuracy. These garbage points distort TPE's density models by making certain regions of the search space appear catastrophically bad when they're actually just buggy. By ensuring that trials succeed or fail based on the actual quality of the quantization approach rather than implementation artifacts, the density models become more reliable and TPE can make better-informed sampling decisions.


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