# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train the model on forward reasoning only."""


import argparse
import json
import os

import peft
import torch
from huggingface_hub import login

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

Dataset = torch.utils.data.Dataset
login(os.getenv("HF_TOKEN"))

class ForwardDataset(Dataset):
  def __init__(self, ex):
    self.data = ex

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    item = self.data[idx]
    return fr_template.format(question=item['question'],
                             answer=item['reasoning'])


class ForwardDataCollator:
  """Collate the data for training."""

  def __init__(self,
               tokenizer,
               label_pad_token_id=-100):
    self.tokenizer = tokenizer
    self.label_pad_token_id = label_pad_token_id

  def __call__(self, features):
    texts = features  # features is now a list of formatted strings
    inputs = self.tokenizer(texts,
                           padding=True,
                           truncation=True,
                           return_tensors='pt')

    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'labels': inputs['input_ids']  # For causal LM, labels are the same as input_ids
    }

class CastOutputToFloat(torch.nn.Sequential):
  def forward(self, x):
    return super().forward(x).to(torch.float32)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--task', default='CSQA', type=str)
  parser.add_argument('--model', default='gemma-7b', type=str)
  parser.add_argument('--model_dir', default='', type=str)
  parser.add_argument('--data_dir', default=None, type=str)

  args = parser.parse_args()

  if args.model == 'mistral-7b':
    base_model = 'mistralai/Mistral-7B-Instruct-v0.3'
  elif args.model == 'gemma-2b':
    base_model = 'google/gemma-2b-it'
  elif args.model == 'gemma-7b':
    base_model = 'google/gemma-7b-it'
  else:
    raise ValueError(f'Unsupported model: {args.model}')

  if 'mistral' in args.model:
    fr_template = """<s>[INST] Answer the following question:\n### Question: {question} [/INST] ### Answer: {answer}</s>"""
  elif 'gemma' in args.model:
    fr_template = """<bos><start_of_turn>user\nAnswer the following question:\n### Question: {question}<end_of_turn>\n<start_of_turn>model\n### Answer: {answer}<eos>"""

  tokenizer = AutoTokenizer.from_pretrained(
      base_model,
      model_max_length=1024,
      padding_side='right'
  )
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.add_bos_token = False
  tokenizer.add_eos_token = False

  model = AutoModelForCausalLM.from_pretrained(
      base_model,
      device_map='auto',
      cache_dir=args.model_dir
  )

  model.config.use_cache = False
  model.gradient_checkpointing_enable()
  model.enable_input_require_grads()
  model.lm_head = CastOutputToFloat(model.lm_head)

  lora_config = peft.LoraConfig(
      r=16,
      lora_alpha=32,
      target_modules=['q_proj', 'v_proj'],
      lora_dropout=0.05,
      bias='none',
      task_type='CAUSAL_LM'
  )

  model = peft.get_peft_model(model, lora_config)
  
  # Chceck if the data file is passed
  assert args.data_dir
  print(args.data_dir)
  with open(args.data_dir, 'r') as f:
    data = json.load(f)


  #print(f'Using {args.n}% of data. ({num_samples}/{len(data)})')
  #print(len(training_data))
  dataset = ForwardDataset(data)
  data_collator = ForwardDataCollator(tokenizer)

  lr = 5e-6 if 'mistral' in args.model else 2e-4
  training_args = TrainingArguments(
      output_dir=f'./outputs/{args.model}_{args.task}_{args.n}_forward_only',
      save_strategy='epoch',
      num_train_epochs=10,
      per_device_train_batch_size=4,
      gradient_accumulation_steps=4,
      learning_rate=lr,
      weight_decay=0.001,
      logging_dir='./logs',
      logging_steps=100,
      remove_unused_columns=False,
      fp16=False,
      bf16=True,
      warmup_ratio=0.3,
      lr_scheduler_type='constant'
  )

  trainer = Trainer(  # Use standard Trainer
      model=model,
      args=training_args,
      train_dataset=dataset,
      data_collator=data_collator
  )

  trainer.train()
  save_path = f'./checkpoints/{args.model}_{args.task}_{args.n}_forward_only'
  os.makedirs(save_path, exist_ok=True)
  trainer.model.save_pretrained(save_path)