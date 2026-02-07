# Lab 2: Quantization Aware Training and Pruning

## Task 1

> In Tutorial 3 you quantized every Linear layer in the model to the provided configuration. Now, explore a range of fixed point widths from 4 to 32.
> (a) Plot a figure where the x-axis is the fixed point width and the y-axis is the highest achieved accuracy on the IMDb dataset, following the procedure in Tutorial 3.
> (b) Plot separate curves for PTQ and QAT at each precision to show the effect of post-quantization finetuning.

### Results

| Width | PTQ Accuracy | QAT Accuracy |
|------:|-------------:|-------------:|
|     4 |       0.5000 |       0.5000 |
|     5 |       0.5000 |       0.5000 |
|     6 |       0.5000 |       0.5000 |
|     7 |       0.5000 |       0.5000 |
|     8 |       0.4945 |       0.8579 |
|     9 |       0.4945 |       0.8572 |
|    10 |       0.4774 |       0.8593 |
|    11 |       0.4774 |       0.8583 |
|    12 |       0.5000 |       0.8577 |
|    13 |       0.4669 |       0.8601 |
|    14 |       0.4508 |       0.8595 |
|    15 |       0.4508 |       0.8592 |
|    16 |       0.4599 |       0.8605 |
|    17 |       0.4599 |       0.8610 |
|    18 |       0.4584 |       0.8603 |
|    19 |       0.4584 |       0.8602 |
|    20 |       0.5000 |       0.8596 |
|    21 |       0.4579 |       0.8604 |
|    22 |       0.4580 |       0.8605 |
|    23 |       0.4580 |       0.8604 |
|    24 |       0.4581 |       0.8600 |
|    25 |       0.4581 |       0.8600 |
|    26 |       0.4578 |       0.8603 |
|    27 |       0.4578 |       0.8603 |
|    28 |       0.4581 |       0.8601 |
|    29 |       0.4581 |       0.8601 |
|    30 |       0.4577 |       0.8601 |
|    31 |       0.4577 |       0.8601 |
|    32 |       0.4579 |       0.8601 |

**Best configuration:** Width = 17, QAT Accuracy = **0.8610**

### Plot

![PTQ vs QAT Accuracy across Fixed-Point Widths](imgs/PTQ%20vs%20QAT.png)

### Analysis

---

## Task 2

> Take your best obtained model from Task 1 and rerun the pruning procedure, this time varying the sparsity from 0.1 to 0.9.
> (a) Plot a figure where the x-axis is the sparsity and the y-axis is the highest achieved accuracy on the IMDb dataset, following the procedure in Tutorial 4.
> (b) Plot separate curves for Random and L1-Norm methods to evaluate the effect of different pruning strategies.

### Plot
![Pruning Performance : L1 Norm vs Random](imgs/pruning%20perf.png)


### Analysis
