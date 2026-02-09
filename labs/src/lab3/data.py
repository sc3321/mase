"""Dataset loading, tokenisation, and environment bootstrap."""
from __future__ import annotations

from transformers import AutoModelForSequenceClassification

from chop.tools.huggingface import get_tokenized_dataset

from .constants import CHECKPOINT, DATASET_NAME, TOKENIZER_CHECKPOINT
from .layers import get_layer_map
from .search_space import get_search_space


def mount_gdrive(mount_point: str = "/content/drive") -> None:
    try:
        from google.colab import drive  # noqa: F811
        drive.mount(mount_point, force_remount=False)
    except ImportError:
        pass


def load_dataset_and_tokenizer(
    checkpoint: str = TOKENIZER_CHECKPOINT,
    dataset_name: str = DATASET_NAME,
):
    dataset, tokenizer = get_tokenized_dataset(
        dataset=dataset_name,
        checkpoint=checkpoint,
        return_tokenizer=True,
    )
    keep_cols = {"input_ids", "attention_mask", "token_type_ids", "label"}
    for split in dataset.keys():
        drop = list(set(dataset[split].column_names) - keep_cols)
        if drop:
            dataset[split] = dataset[split].remove_columns(drop)
    return dataset, tokenizer


def prep_env():
    mount_gdrive()
    dataset, tokenizer = load_dataset_and_tokenizer()
    search_space = get_search_space()
    layer_map = get_layer_map()
    base_model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, num_labels=2,
    )
    return dataset, tokenizer, search_space, layer_map, base_model
