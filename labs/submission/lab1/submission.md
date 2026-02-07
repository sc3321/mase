# Lab 1: Quantization Aware Training and Pruning

This lab investigates the effects of **quantization precision** and **model pruning** on the performance of a BERT-based sentiment classifier on the IMDb dataset. Following **Tutorial 3 (Quantization)** and **Tutorial 4 (Pruning)**, we first compare Post-Training Quantization (PTQ) and Quantization Aware Training (QAT) across a range of fixed-point widths, and then evaluate different pruning strategies at increasing sparsity levels.

---

## Task 1: Quantization Precision Sweep (PTQ vs QAT)

### Aim
In Tutorial 3, all `Linear` layers were quantized using a fixed configuration. Here, we sweep the fixed-point width from **4 to 32 bits** and measure the highest achieved accuracy on the IMDb dataset.

### Method
For each fixed-point width W, the fractional width is set to F = floor(W/2). All Linear layers are quantized using an integer quantizer. Two evaluation modes are considered:

- **PTQ:** quantize → evaluate  
- **QAT:** quantize → finetune → evaluate  

Simplified quantization sweep logic:

    for W in range(4, 33):
        F = W // 2
        qcfg = update_quantization_config(base_qcfg, W, F)
        mg = MaseGraph.from_checkpoint(float_ckpt)
        mg, _ = passes.quantize_transform_pass(mg, pass_args=qcfg)
        ptq_acc[W] = trainer.evaluate()["eval_accuracy"]
        trainer.train()
        qat_acc[W] = trainer.evaluate()["eval_accuracy"]

### Plot
![PTQ vs QAT Accuracy across Fixed-Point Widths](imgs/ptq_vs_qat_accuracy.png)

### Analysis
At very low fixed-point widths (4–6 bits), PTQ accuracy collapses toward random-guessing performance, indicating severe quantization noise. In contrast, QAT maintains high accuracy by adapting the model parameters to quantized arithmetic during training.

As the fixed-point width increases into the mid range (7–10 bits), PTQ accuracy improves rapidly and the performance gap between PTQ and QAT narrows. Beyond approximately 12–16 bits, both methods plateau and converge, showing diminishing returns from additional precision.

These results demonstrate that **QAT is essential for aggressive low-bit quantization**, while **PTQ is sufficient at higher precisions**.

---

## Task 2: Pruning Sweep (Random vs L1-Norm)

### Aim
Using the best-performing quantized model from Task 1, pruning is applied with sparsity levels ranging from **0.1 to 0.9**, following the procedure in Tutorial 4.

### Method
Two pruning strategies are compared:

- **Random pruning**
- **L1-Norm pruning**

For each sparsity level, pruning is followed by finetuning and evaluation.

    for s in sparsities:
        pcfg = update_pruning_config(base_pcfg, s, method)
        mg = MaseGraph.from_checkpoint(best_ckpt)
        mg, _ = passes.prune_transform_pass(mg, pass_args=pcfg)
        trainer.train()
        acc[s] = trainer.evaluate()["eval_accuracy"]

### Plot
![Pruning Performance : L1 Norm vs Random](imgs/pruning_accuracy_vs_sparsity.png)

### Analysis
At low sparsity levels (0.1–0.3), both pruning strategies retain high accuracy, indicating substantial redundancy in the model. L1-Norm pruning slightly outperforms random pruning by removing low-magnitude parameters first.

At moderate sparsity levels (0.4–0.6), random pruning degrades sharply as important parameters are removed indiscriminately, while L1-Norm pruning degrades more gradually. At high sparsity levels (≥0.7), both methods eventually collapse due to insufficient model capacity, although L1-Norm pruning remains effective for higher sparsity levels than random pruning.

---

## Summary
This lab shows that **QAT combined with magnitude-based pruning** provides an effective strategy for compressing transformer models while maintaining strong predictive performance.
