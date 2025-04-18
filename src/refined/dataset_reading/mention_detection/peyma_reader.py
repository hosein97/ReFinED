import json
from random import random
from typing import List, Set, Tuple, Dict, Optional

import torch
from refined.utilities.md_dataset_utils import (
    create_collate_fn,
    tokenize_and_preserve_labels,
)
from torch import Tensor
from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer
from refined.utilities.general_utils import batch_items


class PeymaNER(Dataset):
    def __init__(
        self,
        data_dir: str,
        ner_tag_to_num: Dict[str, int],
        data_split: str = "train",
        bio_only: bool = True,
        sentence_level: bool = True,
        transformer_name: str = "HooshvareLab/bert-base-parsbert-uncased",
        max_seq: int = 100,
        random_lower_case_prob: float = 0.15,
        filter_types: Set[str] = None,
        random_replace_question_mark: float = 0.15,
        lower: bool = False,
        use_mention_tag: bool = False,
        convert_types: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        max_seq -= 2
        assert data_split in ["train", "dev", "test"]
        assert max_seq <= 510, "max seq must be below 512"
        self.lower = lower
        self.random_lower_case_prob = random_lower_case_prob
        self.random_replace_question_mark = random_replace_question_mark
        # if filter_types is None:
        #     self.filter_types = {"TIME", "DURATION", "NUMBER", "ORDINAL", "DATE"}
        # else:
        self.filter_types = filter_types
        self.max_seq = max_seq
        data_split_to_filename = {
            "train": "peyma_training.txt",
            "dev": "peyma_development.txt",
            "test": "peyma_test.txt",
        }
        self.special_tag_to_text = {
            "-LRB-": "(",
            "-RRB-": ")",
            "-LCB-": "{",
            "-RCB-": "}",
            "-LSB-": "[",
            "-RSB-": "]",
            "``": '"',
            "''": '"',
        }
        self.file_path = f'{data_dir.rstrip("/")}/{data_split_to_filename[data_split]}'
        self.ner_tag_to_num = ner_tag_to_num
        self.use_mention_tag = use_mention_tag
        self.convert_types = convert_types
        self.num_labels = len(self.ner_tag_to_num)
        self.bio_only = bio_only
        self.sentence_level = sentence_level
        self.transformer_name = transformer_name
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_name, use_fast=True)
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.pad_id = self.tokenizer.pad_token_id
        # [token, ner_tag]
        self.batch_elements: List[List[Tuple[str, str]]] = self.get_batch_elements()
        self.collate = create_collate_fn(self.pad_id)

    def get_batch_elements(self) -> List[List[Tuple[str, str]]]:

        batch_elements: List[List[Tuple[str, str]]] = []

        for line in self.read_files():

            words_t, ner_t, _ = tokenize_and_preserve_labels(line["words"], line["ners"], self.tokenizer)

            for batch_element in batch_items(zip(words_t, ner_t), n=self.max_seq):
                batch_elements.append(batch_element)

        return batch_elements

    def read_files(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            words, ners = [], []

            for line in f:
                line = line.strip()
                if not line:  # end of a sentence
                    if words:
                        words = [
                            self.special_tag_to_text[word] if word in self.special_tag_to_text else word
                            for word in words
                        ]

                        # Filter or transform NERs
                        processed_ners = []
                        for ner in ners:
                            if ner == "O" or (self.filter_types is not None and ner in self.filter_types):
                                processed_ners.append("O")
                            else:
                                processed_ners.append(self.get_ner_tag(ner))

                        if all(ner == "O" for ner in processed_ners):
                            # Skip sentences with no entities
                            words, ners = [], []
                            continue

                        yield {
                            "words": words,
                            "ners": processed_ners,
                        }

                        words, ners = [], []

                    continue  # skip the empty line

                # Parse word and ner
                if "|" in line:
                    word, ner = line.split("|")
                    words.append(word.strip())
                    ners.append(ner.strip())
                else:
                    # malformed line
                    continue


    def get_ner_tag(self, ner: str):
        start_tag, ner_type = ner.split("_")

        if self.convert_types is not None and ner_type in self.convert_types:
            ner_type = self.convert_types[ner_type]

        if self.bio_only:
            return start_tag  # Just return 'B' or 'I'

        if self.use_mention_tag:
            if f"B-{ner_type}" in self.ner_tag_to_num:
                new_ner = f"{start_tag}-{ner_type}"
            else:
                new_ner = f"{start_tag}-MENTION"
        else:
            if f"B-{ner_type}" in self.ner_tag_to_num:
                new_ner = f"{start_tag}-{ner_type}"
            else:
                new_ner = "O"

        return new_ner

    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        item: List[Tuple[str, str]] = self.batch_elements[index]
        tokens, ners = zip(*item)

        token_ids = torch.tensor(
            [self.cls_id] + self.tokenizer.convert_tokens_to_ids(tokens) + [self.sep_id],
            dtype=torch.long,
        )
        labels = torch.tensor(
            [self.ner_tag_to_num["O"]]
            + list(map(lambda x: self.ner_tag_to_num[x], ners))
            + [self.ner_tag_to_num["O"]],
            dtype=torch.long,
        )
        return token_ids, labels

    def __len__(self) -> int:
        return len(self.batch_elements)
