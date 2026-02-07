# Lab 2: Quantization Aware Training and Pruning

## Task 1

> In Tutorial 3 you quantized every Linear layer in the model to the provided configuration. Now, explore a range of fixed point widths from 4 to 32.
> (a) Plot a figure where the x-axis is the fixed point width and the y-axis is the highest achieved accuracy on the IMDb dataset, following the procedure in Tutorial 3.
> (b) Plot separate curves for PTQ and QAT at each precision to show the effect of post-quantization finetuning.



### Plot

![PTQ vs QAT Accuracy across Fixed-Point Widths](imgs/ptq_vs_qat_accuracy.png)

### Analysis

---

## Task 2

> Take your best obtained model from Task 1 and rerun the pruning procedure, this time varying the sparsity from 0.1 to 0.9.
> (a) Plot a figure where the x-axis is the sparsity and the y-axis is the highest achieved accuracy on the IMDb dataset, following the procedure in Tutorial 4.
> (b) Plot separate curves for Random and L1-Norm methods to evaluate the effect of different pruning strategies.

### Plot
![Pruning Performance : L1 Norm vs Random](imgs/pruning_accuracy_vs_sparsity.png)


### Analysis
