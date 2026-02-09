"""Model & dataset helpers: loading, graph construction, evaluation."""

from __future__ import annotations

from transformers import AutoModelForSequenceClassification

from chop import MaseGraph
import chop.passes as passes
from chop.tools import get_tokenized_dataset, get_trainer

from .constants import CHECKPOINT, TOKENIZER_CHECKPOINT, DATASET_NAME, HF_INPUT_NAMES
from .io_utils import mount_gdrive


# Dataset

def load_dataset_and_tokenizer(
    checkpoint: str = TOKENIZER_CHECKPOINT,
    dataset_name: str = DATASET_NAME,
):
    dataset, tokenizer = get_tokenized_dataset(
        dataset=dataset_name,
        checkpoint=checkpoint,
        return_tokenizer=True,
    )
    return dataset, tokenizer


# MaseGraph construction

def build_masegraph(checkpoint: str = CHECKPOINT) -> MaseGraph:
    """Create a fresh, metadata-initialised MaseGraph from a pretrained checkpoint."""
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.config.problem_type = "single_label_classification"
    model.config.use_cache = False
    model = model.to("cpu")

    mg = MaseGraph(model, hf_input_names=HF_INPUT_NAMES)
    mg, _ = passes.init_metadata_analysis_pass(mg)
    mg, _ = passes.add_common_metadata_analysis_pass(mg)
    return mg


# Evaluation

def evaluate_model(
    mg: MaseGraph,
    dataset,
    tokenizer,
    num_epochs: int = 1,
) -> tuple[float, float, dict, dict]:
    """
    Evaluate a MaseGraph model on the IMDb dataset.

    Returns :

        ptq_acc : float
            Accuracy before fine-tuning.
        qat_acc : float
            Accuracy after fine-tuning.
        ptq_metrics : dict
            Full evaluation metrics dict (pre fine-tune).
        qat_metrics : dict
            Full evaluation metrics dict (post fine-tune).
    """
    trainer = get_trainer(
        model=mg.model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=num_epochs,
    )

    ptq_metrics = trainer.evaluate()
    ptq_acc = float(ptq_metrics["eval_accuracy"])

    trainer.train()

    qat_metrics = trainer.evaluate()
    qat_acc = float(qat_metrics["eval_accuracy"])

    return ptq_acc, qat_acc, ptq_metrics, qat_metrics


# One-call environment bootstrap

def prep_env():
    mount_gdrive()
    dataset, tokenizer = load_dataset_and_tokenizer()
    base_model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, num_labels=2,
    )
    return dataset, tokenizer, base_model
