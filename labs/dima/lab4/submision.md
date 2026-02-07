# Lab 4 (Software Stream) 



## 1. torch.compile: Why Can the Compiled Model Be Slower?

The lab observes that `torch.compile` can produce a model that runs *slower* than the original eager-mode model, depending on the machine. The first invocation of a compiled model triggers graph capture, code generation, kernel compilation, and sometimes autotuning. If the number of timed iterations is small, this one-time cost dominates the average and makes the compiled model appear slower. The fix is to always run warmup iterations that are excluded from timing. TorchDynamo inserts runtime guards that check assumptions about tensor properties such as shape, dtype, strides, and whether `requires_grad` is set. If inputs change between iterations — even subtly, such as a stride change from a view or transpose — Dynamo may recompile new graph variants or fall back to eager execution. Each guard check and potential recompilation adds overhead that can exceed any compilation benefit. When Dynamo encounters Python constructs it cannot capture (data-dependent control flow, side effects, unsupported operations), it introduces a graph break. The execution then alternates between compiled and eager segments: compiled -> eager -> compiled. Each boundary forces tensor materialisation and prevents cross-segment fusion, adding overhead with little optimisation upside. In eager mode, PyTorch dispatches to highly optimised vendor libraries such as oneDNN and MKL for convolutions and matrix multiplications. TorchInductor's generated code may not match these hand-tuned implementations for large, compute-heavy operations. Inductor excels at fusing many small elementwise operations, but for workloads dominated by large GEMMs or convolutions, its generated kernels can be slower than the vendor alternatives. On CPU, MKL and OpenMP each spawn their own thread pools. If Inductor also introduces parallelism, the system can end up with more runnable threads than physical cores, causing cache thrashing and excessive context switching. This could be reduced by explicitly controlling thread counts via `torch.set_num_threads()` and the `OMP_NUM_THREADS` / `MKL_NUM_THREADS` environment variables. If the model is dominated by large cuDNN convolutions or cuBLAS matrix multiplications, eager execution is already calling near-optimal kernels. `torch.compile` can only help by reducing dispatch overhead and fusing the small operations between these large kernels. If the model has few such fusible operations, the compile overhead may outweigh any gains.

### Benchmarking Methodology Errors

- Timing GPU operations without synchronisation
- Not calling `model.eval()` (which changes dropout and batch normalisation behaviour)
- Leaving gradient computation enabled
- Measuring only the first run.

---

## 2. MXINT8: Benefits for Custom Hardware

**Question:** How does MXINT8 benefit custom hardware if both the activations and weights in a linear layer are quantised to MXINT8?

Each mantissa is a signed fixed-point integer, and the shared exponent provides a power-of-two scale factor for the entire group. This means that within a group, every element's value is computed as:

> value = 2^(exponent − 127) x mantissa

When both activations and weights are in MXINT8, each element can be decomposed into a small fixed-point mantissa and a shared group exponent. For a dot product between activation group *g_x* and weight group *g_w*, each product becomes:

> x_i x w_i = 2^(e_x + e_w) x (m_x(i) x m_w(i))

The mantissa products `m_x(i) x m_w(i)` are int8 x int8 -> int16 multiplications, and they can be accumulated into an int32 accumulator. The combined exponent `e_x + e_w` is a single power-of-two scaling applied once per group, which in hardware is simply a bit shift rather than a full floating-point multiply. This leads to :

- **Smaller, denser compute units:** Integer MAC (multiply-accumulate) units are significantly smaller and consume less power than FP16 or FP32 fused multiply-add units. This can enable more MACs per unit area and higher throughput/watt.
- **Reduced exponent handling:** The exponent decode and addition happens once per group rather than once per element -> reducing control logic.
- **Better dynamic range than plain INT8:** A plain 8-bit integer has a fixed representable range. MXINT8's shared exponent lets each group adapt its range to the local data distribution -> reduces clipping and saturation errors.
- **Higher overall throughput:** The combination of smaller compute units and reduced per-element overhead means custom hardware can achieve more ops per second and per watt than a design based on floating-point pipelines.


Current GPUs are architected around FP16/BF16/TF32 datapaths and vendor GEMM libraries. They lack native MXINT compute units, so they must dequantise MXINT values back to a floating-point format before computation. Custom ASICs, FPGAs, or NPUs can be designed with native MXINT datapaths, avoiding this dequantisation step

---

## 3. Dequantization: Purpose of `dont_need_abs` and `bias`

**Question:** What is the purpose of the `dont_need_abs` variable and the `bias` variable in the C++ dequantisation code? Note that unlike IEEE floating-point, MXINT has no implicit leading bit for the mantissa.


BFloat16 is an IEEE-style floating-point format. For normalised BF16 numbers, the value is:

> value = (−1)^s x 2^(E−127) x (1 + f)

The fraction field `f` only stores the bits after this leading one, so BF16 always assumes the mantissa is at least 1.0. MXINT mantissas dont have an implicit leading bit. The mantissa is a signed fixed-point value with the radix placed after the second leftmost bit (format: `si.iiiiii`), meaning it can represent values in the range [0, 2) - this includes values < 1.0, where the leading integer bit is 0 

### What `dont_need_abs` Checks

The variable `dont_need_abs` is set by testing `mantissa_abs & 0x40`, which checks bit 6 of the absolute mantissa value. Given the MXINT radix placement:

- **If bit 6 = 1** : The mantissa magnitude is in [1.0, 2.0). This matches BF16's normalised form `1.xxxx`, and the constructed BF16 bit pattern is correct. No correction is needed.
- **If bit 6 = 0** : The mantissa magnitude is in [0.0, 1.0). BF16 would incorrectly interpret the fraction as `1.xxxx` when the true value is `0.xxxx`. A correction is required.

`dont_need_abs` is effectively a `has_leading_one` flag that indicates whether the mantissa already has the implicit leading bit that BF16 assumes.

### What `bias` Represents

The variable `bias` is constructed as:

```cpp
auto bias = cutlass::bfloat16_t::bitcast(sign | exp | uint16_t(0));
```

This is a BF16 number with the same sign and exponent but a zero fraction, encoding the value:

> bias = (−1)^s x 2^(E−127) x 1.0

This is the implicit leading-one term that BF16 adds to every normalised number.

### Corrections

The code constructs a candidate output `out` by placing the sign, exponent, and fraction bits directly into a BF16 bit pattern. Due to BF16's implicit leading one, this `out` represents:

> out = (−1)^s x 2^E x (1 + f)

For mantissas where the leading bit is already 1, this is correct. But for mantissas where the leading bit is 0, the desired value is:

> desired = (−1)^s x 2^E x f

Subtracting `bias` achieves exactly this:

> out − bias = (−1)^s x 2^E x (1 + f) − (−1)^s x 2^E x 1.0 = (−1)^s x 2^E x f

This also handles the zero mantissa correctly: when `mantissa_abs = 0`, the constructed `out` equals ±2^E, and subtracting `bias` yields exactly zero.

---

## 4. CUTE Kernel: How `cta_tiler` Partitions Data for Copy

**Question (Challenging):** How does `cta_tiler` partition the data for copy?


The CUDA dequantisation kernel treats the 1D MXINT8 data as a logical 2D matrix of shape `[num_groups, group_size]` using `group_tiler`. This reshaping aligns each group of mantissas (which share a single exponent) into one row, making the dequantisation pattern regular and tileable. In CUTE, a tiler is a shape specification that defines the tile dimensions a single CTA (Cooperative Thread Array, i.e. threadblock) is responsible for. `cta_tiler` defines a tile shape such as `(CTA_G, CTA_S)`, where `CTA_G` is the number of groups (rows) and `CTA_S` is the number of within-group elements (columns) handled by one CTA. The function `local_tile(tensor, cta_tiler, cta_coord)` computes the sub-tensor assigned to a specific CTA. Given a CTA coordinate `(b_g, b_s)` derived from the CUDA block index:

- The tile origin is `(b_g x CTA_G, b_s x CTA_S)`.
- The tile covers groups `b_gxCTA_G` through `(b_g+1)xCTA_G − 1` and within-group indices `b_sxCTA_S` through `(b_s+1)xCTA_S − 1`.

This partitions the full `[num_groups, group_size]` matrix into a grid of `⌈num_groups / CTA_G⌉ x ⌈group_size / CTA_S⌉` tiles, with each CTA owning exactly one tile. Once the CTA-local view is established, the kernel's cooperative copy operation moves that tile from global memory into shared memory - aligning `CTA_S` with the contiguous memory dimension, which enables coalesced global memory reads. Predication masks handle boundary tiles where the data dimensions are not exact multiples of the tile shape to prevent out-of-bounds accesses.


---

## 5. CUTE Kernel: How `layout_sX` Partitions Threads for Computation

**Question (Challenging):** How does `layout_sX` partition the threads in a threadblock for computation?


`layout_sX` is a CUTE layout object that defines a mapping from thread identifiers (e.g. `threadIdx.x`) to element coordinates within the shared-memory tile `sX`. It specifies how the tile's elements are distributed across the threads in the CTA (effectively a thread-to-data assignment) The function `local_partition(sX, layout_sX, thread_id)` returns a per-thread tensor fragment: a view into the shared-memory tile consisting of exactly those elements assigned to the calling thread. The layout ensures:

- **Complete coverage:** Every element in the tile is assigned to exactly one thread, with no gaps or overlaps.
- **Strided access patterns:** Each thread's assigned elements follow a regular stride through shared memory, enabling the compiler to vectorise loads and exploit ILP


The specific layout selected for `layout_sX` serves the following:

- **Bank-conflict avoidance:** Shared memory is divided into banks. If threads in the same warp access the same bank simultaneously, accesses are serialised - a good layout ensures threads access distinct banks.
- **Vectorisation and ILP:** If each thread's elements are contiguous or regularly strided, the compiler can issue wider memory operations and overlap computation with memory access.
- **Balanced workload:** The layout ensures each thread processes roughly the same number of elements, avoiding idle threads.

For the dequantisation kernel specifically, each thread reads its assigned int8 elements from `sX` in shared memory, applies the sign extraction, exponent shifting, fraction construction, and implicit-leading-bit correction, then writes the resulting BF16 values to global memory.

---

## 6. Saved GPU Memory: Why Not Exactly 74.2%?

**Question:** Why is the saved GPU memory not exactly (32 − (8 + 8/32)) / 32 = 74.2%?


An FP32 weight occupies 32 bits. An MXINT8 weight with `group_size=32` has an effective bitwidth of 8 + 8/32 = 8.25 bits (8 bits for the mantissa plus the amortised cost of the shared 8-bit exponent). The idealised saving is therefore 1 − 8.25/32 ~ 74.2%. The measurement uses `torch.cuda.max_memory_allocated()`, which captures peak GPU memory during inference — not just weight storage. Several factors reduce the observed saving below the theoretical maximum. The code only replaces `torch.nn.Linear` layers with quantised equivalents, and explicitly skips layers whose name contains `"classifier"`. This means embedding tables, LayerNorm parameters, biases, and non-Linear projection weights all remain in FP32. Since these unquantised parameters still consume the same memory as before, the total weight memory reduction is less than 74.2%. Peak allocated memory during inference includes intermediate activations (attention maps, MLP hidden states, layer outputs), temporary workspaces for GEMM and attention kernels, and PyTorch internal buffers. Quantising weights has no effect on these components. If baseline peak memory is W_fp32 + A (weights plus activations/temporaries), then after weight-only quantisation it becomes W_mxint8 + A. The achievable percentage saving is:

> 1 − (W_mxint8 + A) / (W_fp32 + A)

Because A is non-trivial, this is necessarily less than the pure weight-ratio saving. Pracitcally, you would store scale/exponent tensors separately and may pad dimensions to multiples of the group size or to alignment boundaries (16, 32, or 128 bytes). Memory allocs are byte-granular at minimum, so the theoretical "8.25 bits per weight" cannot be achieved exactly. These overheads slightly inflate the actual memory consumed by quantised weights. PyTorch also uses a caching memory allocator. `max_memory_allocated` reflects allocations from this allocator, which can include internal fragmentation and transient allocations that inflate the observed peak. This is not a precise measurement of payload bytes.
