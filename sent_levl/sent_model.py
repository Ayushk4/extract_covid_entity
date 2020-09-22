from transformers import BertTokenizer, BertPreTrainedModel, BertModel, BertConfig, AdamW, get_linear_schedule_with_warmup
from torch import nn
import torch

RANDOM_SEED = 901
import random
random.seed(RANDOM_SEED)

import numpy as np
from collections import Counter
import pickle

import os
import argparse
import time
import datetime
import string
import re
import collections
import json
import sys
sys.path.insert(0, "../model")
from utils import make_dir_if_not_exists

DO_TASKS_DICT = {"tested_positive": ["part1.Response", "part2-gender.Response", "part2-relation.Response"],
                "tested_negative": ["part1.Response", "part2-gender.Response", "part2-relation.Response"],
                "can_not_test": ["part1.Response", "part2-symptoms.Response", "part2-relation.Response"],
                "death": ["part1.Response", "part2-relation.Response"],
                "cure": ["part1.Response", "part2-opinion.Response"]
                }

NUM_TASKS_DICT = {"tested_positive": 3, "tested_negative": 3, "can_not_test": 3, "death": 2, "cure": 2}
parser = argparse.ArgumentParser()

parser.add_argument("--data_file", type=str, required=True)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--batch_size", help="Train batch size for BERT model", type=int, default=32)
parser.add_argument("--n_epochs", help="Number of epochs", type=int, default=10)
parser.add_argument("--device", type=str, default="cuda", help="name of the device to be used for training")
parser.add_argument("--wandb",  dest="wandb", action="store_true", default= False)
parser.add_argument("--run",  type=str, help="name of the run", default="default_")

args = parser.parse_args()
device = args.device
DO_TASKS_RAW = DO_TASKS_DICT[args.task]
DO_TASKS = [t.split('.')[0] for t in DO_TASKS_RAW]
POSSIBLE_BATCH_SIZE = 8

if args.wandb:
    import wandb
    wandb.init(project="wnut", name=args.run+"_"+args.task, config=args)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

class SentBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2
        self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob * 2)
        self.subtasks = config.subtasks

        # Part 1 stuff
        self.sentence_classifier_1 = nn.Sequential(nn.Linear(config.hidden_size, 50),
                                        nn.Dropout(0.1), nn.Tanh())
        self.sentence_classifier_2 = nn.Linear(50, 2)

        # We will create a dictionary of classifiers based on the number of subtasks
        self.classifier_in_size = config.hidden_size + 50
        self.classifiers = nn.ModuleDict()
        for subtask in self.subtasks:
            if subtask == "part2-gender":
                self.classifiers[subtask] = nn.Linear(self.classifier_in_size, 3)
            elif subtask != "part1":
                self.classifiers[subtask] = nn.Linear(self.classifier_in_size, 2)

        self.att_linears = nn.ModuleDict()
        for subtask in self.subtasks:
            if subtask != "part1":
                self.att_linears[subtask] = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids, labels=None):
        attention_mask = (input_ids != 0) * 1 # No weights for <PAD> input
        outputs = self.bert(input_ids, attention_mask=attention_mask,)

        sentence_pooled = outputs[0][:, 0, :]
        sentence_classifier_feats = self.sentence_classifier_1(sentence_pooled)
        sentence_logits = self.sentence_classifier_2(sentence_classifier_feats)

        all_output = outputs[0]
        all_output = self.dropout(all_output)

        # Get logits for each subtask
        logits = dict()
        for subtask in self.att_linears.keys():
            att_weights = self.att_linears[subtask](all_output)
            att_weights = att_weights + ((attention_mask.unsqueeze(-1) - 1) * 10000.0) # additive mask
            att_weights = torch.softmax(att_weights, 1)
            pooled_output = torch.sum(all_output * att_weights, 1)

            classify_pooled_feats = torch.cat([pooled_output, sentence_classifier_feats], 1)
            logits[subtask] = self.classifiers[subtask](classify_pooled_feats)
        logits["part1"] = sentence_logits
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss = 0
            for i, subtask in enumerate(self.subtasks):
                num_labels = 3 if subtask == "gender" else 2
                this_loss = loss_fct(logits[subtask], labels[subtask].view(-1))
                loss += this_loss.mean()
            outputs = (loss,) + outputs

        return outputs # (loss), logits, (hidden_states), (attentions)


class COVID19SentDataset:
    def __init__(self, datapath, split_ratio, tokenizer, dummy=False):
        self.tokenizer = tokenizer
        self.split_ratio = split_ratio
        self.tasks_raw = DO_TASKS_RAW
        self.tasks = DO_TASKS
        self.cnt = 0

        fo = open(datapath, "r")
        tweets = json.load(fo)
        fo.close()
        if dummy:
            tweets = tweets[:10]
            split_ratio = 0.5
        self.len_dataset = len(tweets)

        len_train = int(self.len_dataset * split_ratio)
        self.train_dataset = self.batch_dataset(self.pick_label_txt_dataset(tweets[:len_train]))
        self.dev_dataset = self.batch_dataset(self.pick_label_txt_dataset(tweets[len_train:]))

        print("Train stats:")
        self.gimme_stats(self.train_dataset)
        print("Dev stats:")
        self.gimme_stats(self.dev_dataset)
        print("No consensus:", self.cnt)

    def batch_dataset(self, dataset): # each data of the format (tid, text, labels)
        batch_size = args.batch_size
        batches = []
        this_batch = [[], [], {task:[] for task in self.tasks}] # tid, text_indices, labels
        for tid, text, labels in dataset:
            if len(this_batch[0]) == batch_size:
                this_batch[1] = self.tokenizer.batch_encode_plus(this_batch[1],pad_to_max_length=True,
                                                        return_tensors="pt")['input_ids'].to(device)
                for k in this_batch[2].keys():
                    this_batch[2][k] = torch.LongTensor(this_batch[2][k]).to(device)
                batches.append(this_batch)

                this_batch = [[], [], {task:[] for task in self.tasks}] # tid, text_indices, labels

            this_batch[0].append(tid)
            this_batch[1].append(text)
            for task in self.tasks:
                this_batch[2][task].append(labels[task])

        if len(this_batch) != 0 :
            this_batch[1] = self.tokenizer.batch_encode_plus(this_batch[1],pad_to_max_length=True,
                                                    return_tensors="pt")['input_ids'].to(device)
            for k in this_batch[2].keys():
                this_batch[2][k] = torch.LongTensor(this_batch[2][k]).to(device)
            batches.append(this_batch)

        return batches            

    def map_label_to_num(self, label):
        if label == "NO_CONSENSUS":
            self.cnt += 1
            label = ["no"]
        assert len(label) == 1, label
        label = label[0].lower()
        if label == "yes":
            return 1
        elif label == "male":
            return 1
        elif label == "female":
            return 2
        elif label == "effective":
            return 1
        elif label in ['no_opinion', 'n', 'not_effective', 'no_cure', 'no', 'not specified']:
            return 0
        else:
            print(label)
            raise "wrong label"
        return

    def reverse_map(self, label_num, task, allow_part1 = False):
        assert type(label_num) == int
        if task != 'part2-gender':
            assert label_num < 2
            if task in ['part2-relation', 'part2-symptoms']:
                label = "yes" if label_num == 1 else "not specified" 
            else:
                assert (task == "part2-opinion") or (allow_part1 and task == "part1"), task
                label = 'effective' if label_num == 1 else 'not_effective'
        else:
            assert label_num < 3
            if label_num == 0:
                label = "not specified"
            else:
                label = "male" if label_num == 1 else "female"
        return label

    def pick_label_txt_dataset(self, dataset):
        self.cnt = 0
        this_dataset = []
        for data in dataset:
            labels = {t: self.map_label_to_num(data['annotation'][t_raw]) for t_raw, t in zip(self.tasks_raw, self.tasks)}
            tid = data['id']
            text = data['prepro']
            this_dataset.append((tid,text,labels))
        print("No consensus:", self.cnt)
        self.cnt = 0

        return this_dataset

    def gimme_stats(self, dataset):
        for task in self.tasks:
            if task == "part2-gender":
                labels_cnt = [0, 0, 0]
                for batch in dataset:
                    labels = batch[2][task].cpu().tolist()
                    for l in labels:
                        labels_cnt[l] += 1
                print(task, ":", self.reverse_map(0, task), "=", labels_cnt[0], "||",
                                self.reverse_map(1, task), "=", labels_cnt[1], "||",
                                self.reverse_map(2, task), "=", labels_cnt[2])
            else:
                labels_cnt = [0, 0]
                for batch in dataset:
                    labels = batch[2][task].cpu().tolist()
                    for l in labels:
                        labels_cnt[l] += 1
                print(task, ":", self.reverse_map(0, task, True), "=", labels_cnt[0], "||",
                                self.reverse_map(1, task, True), "=", labels_cnt[1])


def make_predictions(dataset, model):
    model.eval()
    subtasks = model.subtasks
    predicted_labels, gold_labels = {st:[] for st in subtasks}, {st:[] for st in subtasks}
    softmax_func = nn.Softmax(dim=1)
    with torch.no_grad():
        for batch in dataset:
            logits = model(batch[1])[0]
            for subtask in subtasks:
                predicted_labels[subtask].extend(logits[subtask].max(1)[1].cpu().tolist())
                gold_labels[subtask].extend(batch[2][subtask].cpu().tolist())
    return predicted_labels, gold_labels

def F1_P_R_TP_FP_FN(preds, gold_labels, task):
    print(task)
    if task == 'part2-gender':
        conf_mat = [[0, 0, 0], [0, 0, 0], [0, 0, 0]] 
        for pred, gold in zip(preds, gold_labels):
            conf_mat[pred][gold] += 1
        print(np.array(conf_mat, dtype=np.int))
        TP = conf_mat[1][1] + conf_mat[2][2]
        FP = conf_mat[1][2] + conf_mat[1][0] + conf_mat[2][0] + conf_mat[2][1]
        FN = conf_mat[2][1] + conf_mat[0][1] + conf_mat[0][2] + conf_mat[1][2]


    else:
        conf_mat = [[0, 0], [0, 0]]
        for pred, gold in zip(preds, gold_labels):
            conf_mat[pred][gold] += 1
        print(np.array(conf_mat, dtype=np.int))
        TP = conf_mat[1][1]
        FP = conf_mat[1][0]
        FN = conf_mat[0][1]

    P, R, F1 = 0, 0, 0
    if TP + FP != 0:
        P = TP/(TP+FP)
    if TP + FN != 0:
        R = TP/(TP+FN)
    if P + R != 0:
        F1 = 2*P*R/(P+R)
    return F1, P, R, TP, FP, FN

if __name__ == "__main__":
    # Read all the data instances
    model_name = "digitalepidemiologylab/covid-twitter-bert"
    print("Creating and training the model from '" + model_name + "'")

    # Initialize tokenizer and model with pretrained weights
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Add new tokens in tokenizer
    new_special_tokens_dict = {"additional_special_tokens": ["<user>"]}
    tokenizer.add_special_tokens(new_special_tokens_dict)

    dataset = COVID19SentDataset(args.data_file, 0.7, tokenizer)
    print(len(dataset.train_dataset), dataset.train_dataset[0])
    print(len(dataset.dev_dataset), dataset.dev_dataset[0])
    print(len(dataset.train_dataset[0][0]), dataset.train_dataset[0][1].shape)
    [print(k,v.shape) for k,v in dataset.train_dataset[0][2].items()]

    save_dir = "./" + args.run + "_" + args.task
    make_dir_if_not_exists(save_dir)

    config = BertConfig.from_pretrained(model_name)
    config.subtasks = DO_TASKS
    model = SentBert.from_pretrained(model_name, config=config)
    # Add the new embeddings in the weights
    print("Embeddings type:", model.bert.embeddings.word_embeddings.weight.data.type())
    print("Embeddings shape:", model.bert.embeddings.word_embeddings.weight.data.size())
    embedding_size = model.bert.embeddings.word_embeddings.weight.size(1)
    new_embeddings = torch.FloatTensor(len(new_special_tokens_dict["additional_special_tokens"]), embedding_size).uniform_(-0.1, 0.1)

    print("new_embeddings shape:", new_embeddings.size())
    new_embedding_weight = torch.cat((model.bert.embeddings.word_embeddings.weight.data,new_embeddings), 0)
    model.bert.embeddings.word_embeddings.weight.data = new_embedding_weight
    print("Embeddings shape:", model.bert.embeddings.word_embeddings.weight.data.size())

    # Update model config vocab size
    model.config.vocab_size = model.config.vocab_size + len(new_special_tokens_dict["additional_special_tokens"])
    model.to(device)
    if args.wandb:
        wandb.watch(model)

    print(f"Task dataset for task: {args.task} loaded from {args.data_file}.")
    train_dataset = dataset.train_dataset
    dev_dataset = dataset.dev_dataset

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    print("Created model optimizer")

    epochs = args.n_epochs
    # Total number of training steps is [number of batches] x [number of epochs]. 
    total_steps = len(train_dataset) * epochs

    # Create the learning rate scheduler.
    # NOTE: num_warmup_steps = 0 is the Default value in run_glue.py
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    # We'll store a number of quantities such as training and validation loss, validation accuracy, and timings.
    print("\n\n\n ====== Training for task", args.task, "=============\n\n\n")
    print(f"Initiating training loop for {args.n_epochs} epochs...")
    accumulation_steps = int(args.batch_size/POSSIBLE_BATCH_SIZE)
    # Dev validation
    best_dev_F1 = 0
    for epoch in range(epochs):
        print(f"Initiating Epoch {epoch+1}:")
        start_time = time.time()
        model.train()

        dev_log_frequency = 5
        n_steps = len(train_dataset)
        dev_steps = int(n_steps / dev_log_frequency)
        total_train_loss = 0
        for step, batch in enumerate(train_dataset):
            # Forward
            loss, logits = model(batch[1], batch[2])
            total_train_loss += loss.item()
            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                elapsed = format_time(time.time() - start_time)
                avg_train_loss = total_train_loss/(step+1)

                # keep track of changing avg_train_loss
                print(f"Epoch:{epoch+1}|Batch:{step}/{len(train_dataset)}|Time:{elapsed}|Avg. Loss:{avg_train_loss:.4f}|Loss:{loss.item():.4f}")

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # Clean the model's previous gradients
                model.zero_grad()
                scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataset)

        print("\nRunning Validation...")
        model.eval()
        predicted_labels, gold_labels = make_predictions(dev_dataset, model)
        for task in gold_labels.keys():
            print(task, ":", sum([l == 0 for l in predicted_labels[task]]),
                        ",", sum([l == 1 for l in predicted_labels[task]]))
            print(task, ":", sum([l == 0 for l in gold_labels[task]]),
                        ",", sum([l == 1 for l in gold_labels[task]]))

        wandb_log_dict = {"Train Loss":avg_train_loss}
        print("Dev Set:")
        collect_TP_FP_FN = {"TP": 0.00001, "FP": 0.00001, "FN": 0.00001}
        for subtask in model.subtasks:
            dev_F1, dev_P, dev_R, dev_TP, dev_FP, dev_FN = F1_P_R_TP_FP_FN(predicted_labels[subtask],
                                                                    gold_labels[subtask], task=subtask)
            if subtask == 'part2-gender':
                collect_TP_FP_FN["TP"] += (dev_TP * 1.5)
                collect_TP_FP_FN["FP"] += (dev_FP * 1.5)
                collect_TP_FP_FN["FN"] += (dev_FN * 1.5)
            elif subtask != 'part1':
                collect_TP_FP_FN["TP"] += dev_TP
                collect_TP_FP_FN["FP"] += dev_FP
                collect_TP_FP_FN["FN"] += dev_FN

            print(f"Subtask:{subtask:>15}\tN={dev_TP + dev_FN}\tF1={dev_F1}\tP={dev_P}\tR={dev_R}\tTP={dev_TP}\tFP={dev_FP}\tFN={dev_FN}")

            wandb_log_dict["Dev_ " + subtask + "_F1"] = dev_F1
            wandb_log_dict["Dev_ " + subtask + "_P"] = dev_P
            wandb_log_dict["Dev_ " + subtask + "_R"] = dev_R

        dev_macro_P = collect_TP_FP_FN["TP"] / (collect_TP_FP_FN["TP"] + collect_TP_FP_FN["FP"])
        dev_macro_R = collect_TP_FP_FN["TP"] / (collect_TP_FP_FN["TP"] + collect_TP_FP_FN["FN"])
        dev_macro_F1 = (2 * dev_macro_P * dev_macro_R) / (dev_macro_P + dev_macro_R)
        print(collect_TP_FP_FN)
        print("dev_macro_P:", dev_macro_P, "\ndev_macro_R:", dev_macro_R, "\ndev_macro_F1:", dev_macro_F1,"\n")
        wandb_log_dict["Dev_macro_F1"] = dev_macro_F1
        wandb_log_dict["Dev_macro_P"] = dev_macro_P
        wandb_log_dict["Dev_macro_R"] = dev_macro_R

        if args.wandb:
            wandb.log(wandb_log_dict)

        if dev_macro_F1 > best_dev_F1:
            best_dev_F1 = dev_macro_F1
            print("NEW BEST F1:", best_dev_F1, " Saving checkpoint now.")
            torch.save(model.state_dict(), save_dir + "/ckpt.pth")
        model.train()

    model.load_state_dict(torch.load(save_dir + "/ckpt.pth"))
    model.eval()


