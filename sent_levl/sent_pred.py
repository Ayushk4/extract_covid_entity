from transformers import BertTokenizer, BertPreTrainedModel, BertModel, BertConfig, AdamW, get_linear_schedule_with_warmup
from torch import nn
import torch

import json
import sys
import argparse
from sent_model import SentBert
import numpy as np

sys.path.insert(0, "../model")

DO_TASKS_DICT = {"tested_positive": ["part1.Response", "part2-gender.Response", "part2-relation.Response"],
                "tested_negative": ["part1.Response", "part2-gender.Response", "part2-relation.Response"],
                "can_not_test": ["part1.Response", "part2-symptoms.Response", "part2-relation.Response"],
                "death": ["part1.Response", "part2-relation.Response"],
                "cure": ["part1.Response", "part2-opinion.Response"]
                }

parser = argparse.ArgumentParser()

parser.add_argument("--data_file", type=str, required=True)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--batch_size", help="Train batch size for BERT model", type=int, default=32)
parser.add_argument("--device", type=str, default="cuda", help="name of the device to be used for training")
parser.add_argument("--run", type=str, help="name of the run", default="default_")

args = parser.parse_args()
device = args.device = "cuda"

DO_TASKS_RAW = DO_TASKS_DICT[args.task]
DO_TASKS = [t.split('.')[0] for t in DO_TASKS_RAW]

class COVID19SentDataset:
    def __init__(self, datapath, tokenizer):
        self.tokenizer = tokenizer
        self.tasks_raw = DO_TASKS_RAW
        self.tasks = DO_TASKS

        fo = open(datapath, "r")
        tweets = json.load(fo)
        fo.close()
        self.dataset = self.batch_dataset(self.pick_label_txt_dataset(tweets))

    def batch_dataset(self, dataset): # each data of the format (tid, text, labels)
        batch_size = args.batch_size
        batches = []
        this_batch = [[], [], []] # tid, text_indices, labels
        for tid, text, labels in dataset:
            if len(this_batch[0]) == batch_size:
                this_batch[1] = self.tokenizer.batch_encode_plus(this_batch[1],pad_to_max_length=True,
                                                        return_tensors="pt")['input_ids'].to(device)
                batches.append(this_batch)

                this_batch = [[], [], []] # tid, text_indices, labels

            this_batch[0].append(tid)
            this_batch[1].append(text)

        if len(this_batch) != 0 :
            this_batch[1] = self.tokenizer.batch_encode_plus(this_batch[1],pad_to_max_length=True,
                                                    return_tensors="pt")['input_ids'].to(device)
            batches.append(this_batch)
        return batches            

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
        this_dataset = []
        for data in dataset:
            tid = data['id']
            text = data['prepro']
            this_dataset.append((tid,text,{}))
        return this_dataset

def make_predictions(dataset, model):
    model.eval()
    subtasks = model.subtasks
    tids = []
    predicted_labels = {st:[] for st in subtasks}
    softmax_func = nn.Softmax(dim=1)
    with torch.no_grad():
        for batch in dataset:
            tids.extend(batch[0])
            logits = model(batch[1])[0]
            for subtask in subtasks:
                predicted_labels[subtask].extend(logits[subtask].max(1)[1].cpu().tolist())
    return predicted_labels, tids

if __name__ == "__main__":
    # Read all the data instances
    model_name = "digitalepidemiologylab/covid-twitter-bert"
    print("Creating and training the model from '" + model_name + "'")

    # Initialize tokenizer and model with pretrained weights
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Add new tokens in tokenizer
    new_special_tokens_dict = {"additional_special_tokens": ["<user>"]}
    tokenizer.add_special_tokens(new_special_tokens_dict)

    dataset = COVID19SentDataset(args.data_file, tokenizer)
    print(len(dataset.dataset), dataset.dataset[0])
    save_dir = "./" + args.run + "_" + args.task
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
    model.load_state_dict(torch.load(save_dir + "/ckpt.pth"))
    model.to(device)

    predicted_labels, tweet_ids = make_predictions(dataset.dataset, model)
    print([len(p) for p in predicted_labels.values()], len(tweet_ids))
    predictions = {tid: {} for tid in tweet_ids}
    for subtask in model.subtasks:
        if subtask == "part1":
            continue
        for tid, pred in zip(tweet_ids, predicted_labels[subtask]):
            predictions[tid][subtask] = dataset.reverse_map(pred, subtask)

    fo = open("../preds/sent_"+args.task, "w+")
    json.dump(predictions, fo)
    fo.close()

