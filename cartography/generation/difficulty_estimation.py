import os
import warnings
warnings.filterwarnings("ignore", True)
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from typing import Union, Optional

from rouge_score import rouge_scorer

from datasets import DatasetDict, Dataset
from transformers import TrainingArguments, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq

class DatasetCartographyGenerativeTask:
    def __init__(self, 
                 model_id: str, 
                 tokenizer_id: str,
                 rouge_scorer_object: Optional[rouge_scorer.RougeScorer]=None
                 ) -> None:
        self.model_id = model_id or "t5-base"
        self.tokenizer_id = tokenizer_id or "t5-base"
        self.scorer = rouge_scorer_object or rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.on = "epoch"
        self._is_fitted = False

        # These fields will be assigned by the _load_data() method
        self.input_col_name = None
        self.output_col_name = None
        self.hf_data = None
        self.input_files = None
        self.input_labels = None
        
        # These fields are assigned by the _train_model() method
        self.output_weight_path = None


    def _load_data(self, input_file_or_path, input_col_name, output_col_name):
        if isinstance(input_file_or_path, str):
            df = pd.read_csv(input_file_or_path)
        elif isinstance(input_file_or_path, pd.DataFrame):
            df = input_file_or_path
        self.hf_data = DatasetDict({"train":Dataset.from_pandas(df)})
        self.input_col_name = input_col_name
        self.output_col_name = output_col_name
        try:
            self.input_files = df[self.input_col_name].values
        except:
            raise Exception(f"The input_col_name {self.input_col_name} is not a valid column name.")
        
        try:
            self.input_labels = df[self.output_col_name].values
        except:
            raise Exception(f"The output_col_name {self.output_col_name} is not a valid column name.")


    def _train_model(self, training_arguments):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id)

        def tokenize_function(sample):
            model_inputs = tokenizer(sample[self.input_col_name], max_length=512, truncation=True, padding=True)
            labels = tokenizer(sample[self.output_col_name], max_length=512, truncation=True, padding=True)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_datasets = self.hf_data.map(tokenize_function, batched=True)

        data_collator = DataCollatorForSeq2Seq(tokenizer)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)

        training_args = training_arguments or \
            Seq2SeqTrainingArguments(output_dir="./output_weights", num_train_epochs=5, save_strategy="epoch", logging_strategy="epoch")

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            data_collator=data_collator
        )

        trainer.train()
        self._is_fitted = True
        self.output_weight_path = training_args.output_dir


    def _get_average_confidence(self, model_weights_path, batch_size):
        ckpt_return = {}
        for ckpt_i in tqdm(os.listdir(model_weights_path)):
            if ckpt_i not in ["runs", ".DS_Store"]:
                ckpt_return[ckpt_i] = {}
                model_ckpt_i = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(model_weights_path, ckpt_i))
                try:
                    tokenizer_ckpt_i = AutoTokenizer.from_pretrained(os.path.join(model_weights_path, ckpt_i))
                except:
                    tokenizer_ckpt_i = AutoTokenizer.from_pretrained(self.tokenizer_id)

                batches = [self.input_files[i:i+batch_size] for i in range(0, len(self.input_files), batch_size)]
                perplexity_ckpt_i = []

                for batch in batches:
                    input_batch = tokenizer_ckpt_i(batch.tolist(), return_tensors="pt", truncation=True, padding=True, max_length=512)
                    outputs = model_ckpt_i.generate(**input_batch, max_new_tokens=200, return_dict_in_generate=True, output_scores=True)
                    transition_scores = model_ckpt_i.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
                    perplexity_ckpt_i.extend(transition_scores.exp().mean(dim=1).numpy().tolist())
                ckpt_return[ckpt_i] = perplexity_ckpt_i
        return pd.DataFrame(ckpt_return).to_numpy()


    def _get_variability(self, average_perplexity_across_epochs):
        return np.std(average_perplexity_across_epochs, axis=1)


    def _get_correctness(self, model_weights_path, batch_size):
        ckpt_return = {}
        for ckpt_i in tqdm(os.listdir(model_weights_path)):
            if ckpt_i not in ["runs", ".DS_Store"]:
                ckpt_return[ckpt_i] = {}
                model_ckpt_i = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(model_weights_path, ckpt_i))
                try:
                    tokenizer_ckpt_i = AutoTokenizer.from_pretrained(os.path.join(model_weights_path, ckpt_i))
                except:
                    tokenizer_ckpt_i = AutoTokenizer.from_pretrained("t5-base")

                batches_input = [self.input_files[i:i+batch_size] for i in range(0, len(self.input_files), batch_size)]
                batches_output = [self.input_labels[i:i+batch_size] for i in range(0, len(self.input_labels), batch_size)]
                rouge_l_ckpt_i = []

                for batch_in, batch_out in zip(batches_input, batches_output):
                    input_batch = tokenizer_ckpt_i(batch_in.tolist(), return_tensors="pt", truncation=True, padding=True, max_length=512)
                    outputs = model_ckpt_i.generate(**input_batch, max_new_tokens=200, return_dict_in_generate=True)
                    batch_gen = tokenizer_ckpt_i.batch_decode(outputs.sequences, skip_special_tokens=True)
                    for batch_out_i, batch_gen_i in zip(batch_out, batch_gen):
                        scores = self.scorer.score(batch_out_i, batch_gen_i)
                        score = scores["rougeL"].fmeasure
                        rouge_l_ckpt_i.append(score)
                ckpt_return[ckpt_i] = rouge_l_ckpt_i
        return pd.DataFrame(ckpt_return).mean(axis=1).to_numpy()


    def fit(self, 
            input_data_or_path: Union[str, pd.DataFrame],
            input_col_name: Optional[str]=None,
            output_col_name: Optional[str]=None,
            training_arguments: Optional[TrainingArguments]=None
            ):
        """
        Load dataset and assign instance variables related to data.
        Train the model
        """
        self._load_data(input_file_or_path=input_data_or_path, 
                        input_col_name=input_col_name or "input", 
                        output_col_name=output_col_name or "output")
        self._train_model(training_arguments=training_arguments)


    def transform(self,
                  input_data_or_path: Optional[Union[str, pd.DataFrame]]=None,
                  input_col_name: Optional[str]=None,
                  output_col_name: Optional[str]=None,
                  model_weights_path: Optional[str]=None,
                  batch_size: Optional[int]=None
                  ):
        """
        Using the trained model checkpoints, do the following steps:
        1. Get Average Confidence Across self.on
        2. Get Variability
        3. Get Correctness
        """
        if not self._is_fitted:
            if input_data_or_path is None or model_weights_path is None:
                raise Exception("""One of input_data_or_path or model_weights_path has not been assigned.
                                Since the model has not been fit, instance fields cannot be derived.
                                Pass values for the above mentioned arguments.""")
            else:
                self._load_data(input_file_or_path=input_data_or_path,
                                input_col_name=input_col_name or "input",
                                output_col_name=output_col_name or "output")
        print("[INFO] Computing average confidence across epochs")
        avg_confidence_across_epochs_list = self._get_average_confidence(model_weights_path=model_weights_path, batch_size=batch_size or 4)
        print("[INFO] Computing variance")
        variability_list = self._get_variability(average_perplexity_across_epochs=avg_confidence_across_epochs_list)
        print("[INFO] Computing ")
        correctness_list = self._get_correctness(model_weights_path=model_weights_path, batch_size=batch_size or 4)
        return avg_confidence_across_epochs_list, variability_list, correctness_list