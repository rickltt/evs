import argparse
import torch
import os
import random
import numpy as np
import copy
from tqdm import tqdm
from openprompt.plms import load_plm
from openprompt import PromptForClassification,PromptDataLoader
from openprompt.prompts import AutomaticVerbalizer, ProtoVerbalizer, KnowledgeableVerbalizer, SoftVerbalizer
from evoverb import EvoVerbalizer
from transformers import AdamW
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from openprompt.data_utils.text_classification_dataset import PROCESSORS
from openprompt.data_utils.data_sampler import FewShotSampler
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, eval_dataloader, device):
    alllabels = []
    allpreds = []
    model.eval()
    with torch.no_grad():
        epoch_iterator = tqdm(eval_dataloader, desc="Test")
        for inputs in epoch_iterator:
            inputs = inputs.to(device)
            logits = model(inputs)
            labels = inputs['label']
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    return acc


def train(args, model, train_dataloader, dev_dataloader):
    
    no_decay = ['bias', 'LayerNorm.weight']
    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    loss_func = torch.nn.CrossEntropyLoss()

    if args.verb == 'proto':
        optimizer_grouped_parameters_proto = [
            {'params': model.verbalizer.group_parameters_proto, "lr":3e-5}
        ]
        optimizer_proto = AdamW(optimizer_grouped_parameters_proto)
        model.verbalizer.train_proto(model, train_dataloader, args.device)

    max_val_acc = 0.0
    path = None
    model.train()
    for epoch in range(args.num_train_epochs):
        logger.info("Epoch:{}".format(epoch+1))
        tot_loss = 0
        epoch_iterator = tqdm(train_dataloader, desc="Train")
        for inputs in epoch_iterator:
            inputs = inputs.to(args.device)
            logits = model(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            epoch_iterator.set_description('Loss: {}'.format(round(loss.item(), 6)))
            optimizer.step()
            optimizer.zero_grad()
            
            if args.verb == 'proto':
                optimizer_proto.step()
                optimizer_proto.zero_grad()
        
        if args.verb == 'evo':
            model.verbalizer.optimize(model, dev_dataloader, args.device)
        elif args.verb == 'auto':
            model.verbalizer.optimize_to_initialize()

        val_acc = evaluate(model, dev_dataloader, args.device)
        
        print('val_acc:', val_acc)
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            if not os.path.exists(args.output_dir):
                os.mkdir(args.output_dir)
            path = 'output/{}_{}_acc_{}'.format(args.verb, args.dataset, round(val_acc, 4))
            torch.save(model.state_dict(), path)
            print('>> saved model: {}'.format(path)) 
        if val_acc > max_val_acc:
            max_val_acc = val_acc
    return path

def main():
    parser = argparse.ArgumentParser(description="Command line interface for Prompt-based text classification.")
    
    parser.add_argument(
        "--dataset",
        default='agnews',
        type=str,
        help="agnews, dbpedia, amazon, yahoo, imdb"
    )

    parser.add_argument(
        "--model_type",
        default='roberta',
        type=str,
        help="Select the model type selected to be used from bert, roberta, albert, now only support roberta."
    )
    parser.add_argument(
        "--model_name_or_path",
        default='roberta-large',
        type=str,
        help="Path to pretrained model or shortcut name of the model."
    )

    parser.add_argument(
        "--verb",
        default='one',
        type=str,
        help="verbalizer function: one, many, know, auto, soft, proto, evo"
    )

    parser.add_argument("--device_id", default='0', type = str , help="which gpu to be used.")
    parser.add_argument("--temp_id", default=0, type=int, help="which template to be used.")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--k", default=4, type=int, help="k-shot experimental setting.")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for AdamW.")
    parser.add_argument(
        "--num_train_epochs", default=5, type=int, help="Total number of training epochs to perform."
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")

    parser.add_argument("--Nl", default=10, type=int, help="label_word_num_per_class")
    parser.add_argument("--Nc", default=100, type=int, help="the number of candidates")

    

    parser.add_argument(
        "--output_dir", default='output', type=str, help="the output directory."
    )

    args = parser.parse_args()

    set_seed(42)
    os.system('rm -rf {}'.format(args.output_dir))

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    plm, tokenizer, _, WrapperClass = load_plm(args.model_type, args.model_name_or_path)
    dataset = {}
    data_dir = os.path.join('./datasets',args.dataset)
    logger.info("dataset:{}".format(args.dataset))

    Processor = PROCESSORS[args.dataset]()
    train_data = Processor.get_train_examples(data_dir)
    logger.info("k:{}".format(args.k))
    sampler  = FewShotSampler(num_examples_per_label = args.k, num_examples_per_label_dev = args.k, also_sample_dev=True)
    dataset['train'], dataset['dev'] = sampler(train_data)
    if args.dataset == 'amazon':
        test_dataset = Processor.get_test_examples(data_dir)
        random.shuffle(test_dataset)
        dataset['test'] = test_dataset[:10000]
    else:
        dataset['test'] = Processor.get_test_examples(data_dir)

    logger.info("train size:{}".format(len(dataset['train'])))
    logger.info("dev size:{}".format(len(dataset['dev'])))
    logger.info("test size:{}".format(len(dataset['test'])))

    prompt_dir = os.path.join('./prompts',args.dataset)
    manual_template = ManualTemplate(tokenizer=tokenizer).from_file(os.path.join(prompt_dir , 'manual_template.txt'),choice=args.temp_id)

    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    max_seq_length = args.max_seq_length
    batch_size = args.batch_size
    num_classes = Processor.get_num_labels()

    train_dataloader = PromptDataLoader(dataset=dataset["train"], template=manual_template, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_length,batch_size=batch_size,shuffle=True)
    dev_dataloader = PromptDataLoader(dataset=dataset["dev"], template=manual_template, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_length, batch_size=batch_size,shuffle=False)
    test_dataloader = PromptDataLoader(dataset=dataset["test"], template=manual_template, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_length, batch_size=batch_size,shuffle=False)
 
    if args.verb == 'one':
        verbalizer = ManualVerbalizer(tokenizer,num_classes=num_classes).from_file(os.path.join(prompt_dir , 'manual_verbalizer.txt'))
    elif args.verb == 'many':
        verbalizer = ManualVerbalizer(tokenizer,num_classes=num_classes).from_file(os.path.join(prompt_dir , 'multiwords_verbalizer.txt'))
    elif args.verb == 'auto':
        verbalizer = AutomaticVerbalizer(tokenizer, label_word_num_per_class= 10, num_searches = 5, num_classes=num_classes, classes = Processor.get_labels() )
    elif args.verb == 'soft':
        verbalizer = SoftVerbalizer(tokenizer, plm, num_classes=num_classes)
    elif args.verb == 'proto':
        verbalizer = ProtoVerbalizer(tokenizer, plm, num_classes=num_classes,multi_verb='proto').from_file(os.path.join(prompt_dir , 'manual_verbalizer.txt'))
    elif args.verb == 'know':
        verbalizer = KnowledgeableVerbalizer(tokenizer,num_classes=num_classes).from_file(os.path.join(prompt_dir, 'knowledgeable_verbalizer.txt'))
    elif args.verb == 'evo':
        verbalizer = EvoVerbalizer(tokenizer, label_word_num_per_class = args.Nl, num_candidates = args.Nc, num_classes=num_classes, classes = Processor.get_labels())
    else:
        print('Not correct verbalizer!')

    logger.info("Verbalizer:{}".format(verbalizer))
    model = PromptForClassification(plm = plm, template = manual_template, verbalizer = verbalizer, freeze_plm=False)
    
    model.to(args.device)

    best_model_path = train(args, model, train_dataloader, dev_dataloader)
    model.load_state_dict(torch.load(best_model_path))
    test_acc = evaluate(model, test_dataloader, args.device)
    res = 'verb:{}, temp_id:{}, k:{}, acc:{}'.format(args.verb, args.temp_id, args.k, test_acc)
    print(res)
    with open(os.path.join('./{}_result.txt'.format(args.dataset)), 'a') as f:
        f.write(res+'\n')

if __name__ == '__main__':
    main()