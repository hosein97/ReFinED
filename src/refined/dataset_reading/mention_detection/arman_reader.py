import json
from random import random
from typing import List, Set, Tuple, Dict, Optional

import torch
from refined.utilities.md_dataset_utils import (
    create_collate_fn,
    tokenize_and_preserve_labels,
    bio_to_offset_pairs
)
from torch import Tensor
from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer
from refined.utilities.general_utils import batch_items
from tqdm import tqdm

from refined.inference.standalone_md import MentionDetector

from typing import List, Tuple


import re

class ArmanNER(Dataset):
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
        additional_filename: Optional[str] = None,
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
            "train": "arman_training.txt",
            "dev": "arman_development.txt",
            "test": "arman_test.txt",
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
        self.data_split = data_split
        self.data_dir = data_dir
        self.data_split_to_filename = data_split_to_filename
        self.file_path = self.get_filepath(data_split, additional_filename)
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
        separator = r" "
        begin_sign = "B-"
        in_sign = "I-"
        pattern = re.compile(rf'^(.*){separator}({begin_sign}\w+|{in_sign}\w+|O)$')
        
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
                match = pattern.match(line)
                if match:
                    word, ner = match.group(1).strip(), match.group(2).strip()
                    words.append(word)
                    ners.append(ner)
                else:
                    # malformed line
                    continue


    def get_ner_tag(self, ner: str):
        start_tag, ner_type = ner.split("-")

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



    def get_filepath(self, split: str, additional_filename: Optional[str]) -> str:
        file_path = self.data_split_to_filename[split]

        if additional_filename is not None:
            file_path = file_path[:-4] + additional_filename + '.txt'

        return f'{self.data_dir.rstrip("/")}/{file_path}'




    def read_file_as_sentences(
        self,
        file_path,
        bio_only=True,
        leave_all_mentions: bool = False
    ) -> List[List[Tuple[str, str]]]:
        """
        Reads a token-NER file where each line is: token [space] NER
        Sentences are separated by a blank line.
        Returns: List of sentences. Each sentence is a list of (token, ner) tuples.
        """
        sentences: List[List[Tuple[str, str]]] = []
        current_sent: List[Tuple[str, str]] = []

        with open(file_path, 'r', encoding='utf-8') as f:
            separator = r" "
            begin_sign = "B-"
            in_sign = "I-"
            pattern = re.compile(rf'^(.*){separator}({begin_sign}\w+|{in_sign}\w+|O)$')
        
            for line in f:
                line = line.strip()
                if not line:
                    if current_sent:
                        sentences.append(current_sent)
                        current_sent = []
                    continue

                match = pattern.match(line)
                if match:
                    word, ner = match.group(1).strip(), match.group(2).strip()
                    if not leave_all_mentions:
                        if bio_only:
                            ner = ner[0]  # B-pers -> B, I-loc -> I, O -> O

                    current_sent.append((word, ner))

            if current_sent:
                sentences.append(current_sent)

        return sentences




    def relabel_dataset(self, additional_filename: str, ner_types_to_add: Set[str],
                        mention_detector: MentionDetector) -> None:
        """
        Create a new version of a dataset using MentionDetector to add additional NER labels (e.g. for DATE spans)
        """
        assert not self.lower, "Lowercasing text whilst re-labelling will lose casing information when file is written" \
                               "back to disk"
        assert not self.sentence_level, "Should relabel at article level to preserve article structure of dataset"

        new_file_path = self.get_filepath(self.data_split, additional_filename=additional_filename)

        with open(new_file_path, "w") as write_file:

            for sent in tqdm(
                    self.read_file_as_sentences(self.file_path, bio_only=False,
                                           leave_all_mentions=True)):

                # Relabel the doc using the trained NER model
                new_sent = self.relabel_sent(doc=sent, mention_detector=mention_detector,
                                           ner_types_to_add=ner_types_to_add)


                # Write the sentence to file in the original format
                for word, ner in new_sent:
                    write_file.write(f"{word} {ner}\n")
                write_file.write("\n")  # Sentence separator

    
    @staticmethod
    def relabel_sent(sent: List[Tuple[str, str]],
                    mention_detector: MentionDetector,
                    ner_types_to_add: Set[str]) -> List[Tuple[str, str]]:
        """
        Add new NER labels to a sentence using a trained MentionDetector.
        Only labels from ner_types_to_add are added, and only on spans that are currently labeled as 'O'.

        Each sentence is a list of (word, ner) tuples.
        """
        words, ners = zip(*sent)
        ners = list(ners)

        ner_preds = mention_detector.process_words(list(words))

        ner_pred_tuples = bio_to_offset_pairs(ner_preds, use_labels=True)

        for start_ix, end_ix, ner_type in ner_pred_tuples:
            if ner_type not in ner_types_to_add:
                continue
            if set(ners[start_ix:end_ix]) != {'O'}:
                continue
            new_labels = ['B-' + ner_type] + ['I-' + ner_type for _ in range(end_ix - start_ix - 1)]
            ners[start_ix:end_ix] = new_labels

        return [(word, new_ner) for word, new_ner in zip(words, ners)]


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
