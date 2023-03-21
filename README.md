# Evolutionary Verbalizer Search

The repository contains the code for Paper "Evolutionary Verbalizer Search for
Prompt-based Few Shot Text Classification"

## Introduction

This repo includes various verbalziers in prompt-tuning, (based on [OpenPrompt](https://github.com/thunlp/OpenPrompt)) and our **Eoverb**. We conduct experiments in five different datasets: AGNews, DBPedia, Yahoo Answers, IMDB, Amazon. More details can be seen in our paper.

## Usage

```
python train.py
```

## Yahoo Download

You can download [Yahoo](https://www.heywhale.com/mw/dataset/5d9ff886037db3002d417c5f) dataset and get the compressed file `
yahoo_answers_csv.tgz`.

```sh
tar -zxvf yahoo_answers_csv.tgz -C ./datasets
cd datasets
mv yahoo_answers_csv yahoo
```

## Main APP
| Arguments                | Description                                                  |
| ------------------------ | ------------------------------------------------------------ |
| dataset                | The input dataset: agnews, dbpedia, amazon, yahoo, imdb                                     |
| model_type              | Plms: bert, roberta                                          |
| model_name_or_path             | Path to pretrained model or shortcut name of the model       |
| verb | verbalizer function: one, many, know, auto, soft, proto, evo  |
| device_id | which gpu to be used |
| temp_id | which template to be used, you can input 0~3 |
| k | k-shot experiment setting |
| max_seq_length | The maximum total input sequence length after tokenization. |
| batch_size | Batch size per GPU/CPU for training. |
| learning_rate | The initial learning rate for AdamW.|
| num_train_epochs | Total number of training epochs to perform. |
| seed | random seed|
| Nl | label_word_num_per_class |
| Nc | the number of candidates|
|output_dir| the output directory.|

