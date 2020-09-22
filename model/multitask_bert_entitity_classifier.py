from transformers import BertTokenizer, BertPreTrainedModel, BertModel, BertConfig, AdamW, get_linear_schedule_with_warmup
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch

RANDOM_SEED = 901
import random
random.seed(RANDOM_SEED)

import numpy as np
from collections import Counter
import pickle
from pprint import pprint

from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

import os
from tqdm import tqdm
import argparse
import time
import datetime
import string
import re
import collections

from utils import log_list, make_dir_if_not_exists, save_in_pickle, load_from_pickle, get_multitask_instances_for_valid_tasks, split_multitask_instances_in_train_dev, log_data_statistics, save_in_json, get_raw_scores, get_TP_FP_FN, add_marker_for_loss_ignore

IGNORE_TASKS_DICT = {"tested_positive": ["part1.Response", "gender_male", "gender_female", "relation"],
                "tested_negative": ["part1.Response", "how_long", "gender_male", "gender_female", "relation"],
                "can_not_test": ["part1.Response", "symptoms", "relation"],
                "death": ["part1.Response", "symptoms", "relation"],
                "cure": ["part1.Response", "opinion"]
                }

NUM_TASKS_DICT = {"tested_positive": 10, "tested_negative": 9, "can_not_test": 5, "death": 6, "cure": 3}

IGNORE_TASKS = ["part1.Response"]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data_file", help="Path to the pickle file that contains the training instances", type=str, required=True)
    parser.add_argument("-t", "--task", help="Event for which we want to train the baseline", type=str, required=True)
    parser.add_argument("-s", "--save_directory", help="Path to the directory where we will save model and the tokenizer", type=str, required=True)
    parser.add_argument("-o", "--output_dir", help="Path to the output directory where we will save all the model results", type=str, required=True)
    parser.add_argument("-rt", "--retrain", help="Flag that will indicate if the model needs to be retrained or loaded from the existing save_directory", action="store_true")
    parser.add_argument("-bs", "--batch_size", help="Train batch size for BERT model", type=int, default=32)
    parser.add_argument("-e", "--n_epochs", help="Number of epochs", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda", help="name of the device to be used for training")
    parser.add_argument("--wandb",  dest="wandb", action="store_true", default= False)
    parser.add_argument("--run",  type=str, help="name of the run", required=True)
    parser.add_argument("--large_bert", dest="large_bert", action="store_true", default=False)
    parser.add_argument("--covid_bert", dest="covid_bert", action="store_true", default=False)
    parser.add_argument("--loss_for_no_consensus", dest="loss_for_no_consensus", action="store_true", default= False)
    parser.add_argument("--sentence_level_classify", dest="sentence_level_classify", action="store_true", default= False)

    args = parser.parse_args()
    assert (args.large_bert and args.covid_bert) == False # Not both can be true at same time
    IGNORE_TASKS = IGNORE_TASKS_DICT[args.task]

    import logging
    # Ref: https://stackoverflow.com/a/49202811/4535284
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Also add the stream handler so that it logs on STD out as well
    # Ref: https://stackoverflow.com/a/46098711/4535284
    make_dir_if_not_exists(args.output_dir)
    if args.retrain:
        logfile = os.path.join(args.output_dir, "train_output.log")
    else:
        logfile = os.path.join(args.output_dir, "output.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.FileHandler(logfile, mode='w'), logging.StreamHandler()])

    device = args.device
    logging.info(f"Using {device} to train")

    if args.wandb:
        import wandb
        wandb.init(project="wnut_" + args.task, name=args.run, config=args)

    if args.sentence_level_classify:
        LOSS_SCALE = {'part1': 1, 'others': 1}
        print(sum([LOSS_SCALE['part1'], LOSS_SCALE['others'] * (NUM_TASKS_DICT[args.task] - len(IGNORE_TASKS_DICT[args.task]))]))
    else:
        raise
        LOSS_SCALE = {'part1': 0, 'others': 1}

Q_TOKEN = "<Q_TARGET>"
URL_TOKEN = "<URL>"
RANDOM_SEED = 901
torch.manual_seed(RANDOM_SEED)
POSSIBLE_BATCH_SIZE = 8

def make_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        logging.info("Creating new directory: {}".format(directory))
        os.makedirs(directory)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

class MultiTaskBertForCovidEntityClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)

        if args.large_bert or args.covid_bert:
            self.dropout = nn.Dropout(config.hidden_dropout_prob * 2)
        else:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.subtasks = config.subtasks
        self.classifier_in_size = config.hidden_size

        # For sentence level classification
        if args.sentence_level_classify:
            self.sentence_classification_att = nn.Linear(config.hidden_size, 1)
            self.sentence_classifier_1 = nn.Sequential(nn.Linear(config.hidden_size, 50), nn.Dropout(0.1))
            self.sentence_classifier_2 = nn.Linear(50, 2)
            self.sent_fuse_skip = nn.Sequential(nn.Linear(2,4), nn.LeakyReLU(), nn.Linear(4,2))
            # self.classifier_in_size += 50

        # We will create a dictionary of classifiers based on the number of subtasks

        self.classifiers = nn.ModuleDict()
        for subtask in self.subtasks:
            if subtask != "part1.Response":
                self.classifiers[subtask] = nn.Linear(self.classifier_in_size, config.num_labels)

        self.context_vectors = nn.ModuleDict()
        for subtask in self.subtasks:
            if subtask != "part1.Response":
                self.context_vectors[subtask] = nn.Embedding(1,config.hidden_size)
            # self.att_taskwise_mlp[subtask] = nn.Linear(config.

        # self.subtask_to_id = {s:i for i,s in enumerate(self.subtasks)}
        # self.id_to_subtask = {i:s for s,i in self.subtask_to_id.items()}

        # self.ids_long_tensor = torch.LongTensor([i for i in id_to_subtask.keys()])

        for task in IGNORE_TASKS:
            if task == "part1.Response":
                continue
            self.classifiers[task].weight.requires_grad = False
            self.classifiers[task].bias.requires_grad = False

            torch.nn.init.zeros_(self.classifiers[task].weight)

            devic = self.classifiers[task].weight.device
            self.classifiers[task].bias.data = torch.tensor([10.0, -10.0]).to(devic) # Only predict negative class.

        self.init_weights()
        self.norm_probs = lambda x: x / x.sum(1).unsqueeze(-1)

    def forward(
        self,
        input_ids,
        entity_start_positions,
        entity_end_positions,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        label_weight=None
    ):
        assert attention_mask == None
        attention_mask = (input_ids != 0) * 1 # No weights for <PAD> input
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        # Sentence level classification:
        if args.sentence_level_classify:
            # pad_mask = (input_ids != 0) # No weights for <PAD> input
            # att_scores = self.sentence_classification_att(outputs[0]) * pad_mask.unsqueeze(-1) # Calculate score
            # att_scores = torch.softmax(att_scores, dim=1) # Softmax across the words.

            sentence_pooled = outputs[0][:, 0, :] # 
            # sentence_pooled = (outputs[0] * att_scores).sum(1) # Get a single sentence vector for each sentence.
            sentence_classifier_feats = self.sentence_classifier_1(sentence_pooled)
            sentence_logits = self.sentence_classifier_2(sentence_classifier_feats)
            detached_sent_logits = sentence_logits.detach()
            sent_log_skip_add = self.sent_fuse_skip(sentence_logits)

        # NOTE: outputs[0] has all the hidden dimensions for the entire sequence
        # We will extract the embeddings indexed with entity_start_positions
        all_output = outputs[0]
        all_output = self.dropout(all_output)


        # Entity_mask
        entity_mask = torch.arange(input_ids.shape[1]).expand(input_ids.shape[0], -1).to(input_ids.device)
        entity_mask = (entity_mask >= entity_start_positions[:, 1:2]) & (entity_mask <= entity_end_positions[:, 1:2])
        entity_mask = ~ entity_mask.unsqueeze(-1)

        # Get logits for each subtask
        if args.sentence_level_classify:
            logits = dict() # {subtask: self.classifiers[subtask](pooled_output) for subtask in self.classifiers.keys()}
            for subtask in self.context_vectors.keys():
                att_weights = torch.matmul(all_output, self.dropout(self.context_vectors[subtask].weight.T))
                att_weights = att_weights.masked_fill(entity_mask, -1000)
                att_weights = torch.softmax(att_weights, 1)
                pooled_output = torch.sum(all_output * att_weights, 1)

                logits[subtask] = self.classifiers[subtask](pooled_output)
                if subtask not in IGNORE_TASKS:
                    logits[subtask] = sent_log_skip_add + logits[subtask] # self.fuse_classify[subtask](sentence_logits) + logits[subtask]
        # else:
            # logits = {subtask: torch.softmax(self.classifiers[subtask](pooled_output), 1) for subtask in self.classifiers.keys()}

        if args.sentence_level_classify:
            logits["part1.Response"] = sentence_logits

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='none')

            loss = 0
            for i, subtask in enumerate(self.subtasks):
                if (args.sentence_level_classify and subtask == "part1.Response") or subtask not in IGNORE_TASKS:
                    # if loss == None:
                    #     this_loss = loss_fct(logits[subtask].view(-1, self.num_labels)   , labels[subtask].view(-1)) * label_weight[subtask]
                    #     if subtask == "part1.Response":
                    #         this_loss *= LOSS_SCALE['part1']
                    #     else:
                    #         this_loss *= LOSS_SCALE['others']
                    #     loss = this_loss.mean()
                    # else:
                    this_loss = loss_fct(logits[subtask].view(-1, self.num_labels), labels[subtask].view(-1)) * label_weight[subtask]
                    if subtask == "part1.Response":
                        this_loss *= LOSS_SCALE['part1']
                    else:
                        this_loss *= LOSS_SCALE['others']
                    loss += this_loss.mean()
            outputs = (loss,) + outputs

        return outputs # (loss), logits, (hidden_states), (attentions)

class COVID19TaskDataset(Dataset):
    """COVID19TaskDataset is a generic dataset class which will read data related to different questions"""
    def __init__(self, instances):
        super(COVID19TaskDataset, self).__init__()
        self.instances = instances
        self.nsamples = len(self.instances)

    def __getitem__(self, index):
        return self.instances[index]

    def __len__(self):
        return self.nsamples

class TokenizeCollator():
    def __init__(self, tokenizer, subtasks, entity_start_token_id, entity_end_token_id):
        self.tokenizer = tokenizer
        self.subtasks = subtasks
        self.entity_start_token_id = entity_start_token_id
        self.entity_end_token_id = entity_end_token_id

    def fix_user_mentions_in_tokenized_tweet(self, tokenized_tweet):
        return ' '.join(["@USER" if word.startswith("@") else word for word in tokenized_tweet.split()])

    def __call__(self, batch):
        all_bert_model_input_texts = list()
        gold_labels = {subtask: list() for subtask in self.subtasks}
        labels_weight = {subtask: list() for subtask in self.subtasks}
        # text :: candidate_chunk :: candidate_chunk_id :: chunk_start_text_id :: chunk_end_text_id :: tokenized_tweet :: tokenized_tweet_with_masked_q_token :: tagged_chunks :: question_label
        for text, chunk, chunk_id, chunk_start_text_id, chunk_end_text_id, tokenized_tweet, tokenized_tweet_with_masked_chunk, subtask_labels_dict in batch:
            tokenized_tweet_with_masked_chunk = self.fix_user_mentions_in_tokenized_tweet(tokenized_tweet_with_masked_chunk)
            if chunk in ["AUTHOR OF THE TWEET", "NEAR AUTHOR OF THE TWEET"]:
                # First element of the text will be considered as AUTHOR OF THE TWEET or NEAR AUTHOR OF THE TWEET
                bert_model_input_text = tokenized_tweet_with_masked_chunk.replace(Q_TOKEN, "<E> </E>")
            else:
                bert_model_input_text = tokenized_tweet_with_masked_chunk.replace(Q_TOKEN, "<E> " + chunk + " </E>")
            all_bert_model_input_texts.append(bert_model_input_text)
            # Add subtask labels in the gold_labels dictionary
            for subtask in self.subtasks:
                gold_labels[subtask].append(subtask_labels_dict[subtask][1])
                labels_weight[subtask].append(subtask_labels_dict[subtask][2])

        # Tokenize
        all_bert_model_inputs_tokenized = self.tokenizer.batch_encode_plus(all_bert_model_input_texts, pad_to_max_length=True, return_tensors="pt")
        input_ids, token_type_ids, attention_mask = all_bert_model_inputs_tokenized['input_ids'], all_bert_model_inputs_tokenized['token_type_ids'], all_bert_model_inputs_tokenized['attention_mask']

        # Extract the indices of <E> and </E> token in each sentence and save it in the batch
        entity_start_positions = (input_ids == self.entity_start_token_id).nonzero()
        assert torch.sum(entity_start_positions[:, 0] == torch.arange(input_ids.shape[0])).item() == input_ids.shape[0]
        entity_end_positions = (input_ids == self.entity_end_token_id).nonzero()
        assert torch.sum(entity_end_positions[:, 0] == torch.arange(input_ids.shape[0])).item() == input_ids.shape[0]

        # Also extract the gold labels and their weights (weight = 0 for NO CONSENSUS)
        labels = {subtask: torch.LongTensor(subtask_gold_labels) for subtask, subtask_gold_labels in gold_labels.items()}
        labels_weight = {subtask: torch.Tensor(subtask_label_wt) for subtask, subtask_label_wt in labels_weight.items()}

        if entity_start_positions.size(0) == 0:
            raise
            # Send entity_start_positions to [CLS]'s position i.e. 0
            entity_start_positions = torch.zeros(input_ids.size(0), 2).long()

        # Verify that the number of labels for each subtask is equal to the number of instances
        for subtask in self.subtasks:
            try:
                assert input_ids.size(0) == labels[subtask].size(0)
                assert input_ids.size(0) == labels_weight[subtask].size(0)
            except AssertionError:
                logging.error(f"Error Bad batch: Incorrect number of labels given for the batch of size: {len(batch)}")
                logging.error(f"{subtask}, {labels[subtask]}, {labels[subtask].size(0)}, {labels_weight[subtask].size(0)}")
                exit()
        return {"input_ids": input_ids, "entity_start_positions": entity_start_positions, "entity_end_positions": entity_end_positions, "gold_labels": labels,
                "batch_data": batch, "label_ignore_loss": labels_weight}

def make_predictions_on_dataset(dataloader, model, device, dataset_name, dev_flag = False):
    # Create tqdm progressbar
    if dev_flag:
        pbar = dataloader
    else:
        logging.info(f"Predicting on the dataset {dataset_name}")
        pbar = tqdm(dataloader)
    # Setting model to eval for predictions
    # NOTE: assuming that model is already in the given device
    model.eval()
    subtasks = model.subtasks
    all_predictions = {subtask: list() for subtask in subtasks}
    all_prediction_scores = {subtask: list() for subtask in subtasks}
    all_labels = {subtask: list() for subtask in subtasks}
    softmax_func = nn.Softmax(dim=1)
    with torch.no_grad():
        for step, batch in enumerate(pbar):
            # Create testing instance for model
            input_dict = {"input_ids": batch["input_ids"].to(device), "entity_start_positions": batch["entity_start_positions"].to(device),
                          "entity_end_positions": batch["entity_end_positions"].to(device)}
            labels = batch["gold_labels"]
            logits = model(**input_dict)[0]

            # Apply softmax on each subtask's logits			
            softmax_logits = {subtask: softmax_func(logits[subtask]) for subtask in subtasks}

            # Extract predicted labels and predicted scores for each subtask
            predicted_labels = {subtask: None for subtask in subtasks}
            prediction_scores = {subtask: None for subtask in subtasks}
            for subtask in subtasks:
                _, predicted_label = softmax_logits[subtask].max(1)
                prediction_score = softmax_logits[subtask][:, 1]
                prediction_scores[subtask] = prediction_score.cpu().tolist()
                predicted_labels[subtask] = predicted_label.cpu().tolist()

                # Save all the predictions and labels in lists
                all_predictions[subtask].extend(predicted_labels[subtask])
                all_prediction_scores[subtask].extend(prediction_scores[subtask])
                all_labels[subtask].extend(labels[subtask])
    return all_predictions, all_prediction_scores, all_labels

def plot_train_loss(loss_trajectory_per_epoch, trajectory_file):
    plt.cla()
    plt.clf()

    fig, ax = plt.subplots()
    x = [epoch * len(loss_trajectory) + j + 1 for epoch, loss_trajectory in enumerate(loss_trajectory_per_epoch) for j, avg_loss in enumerate(loss_trajectory) ]
    x_ticks = [ "(" + str(epoch + 1) + "," + str(j + 1) + ")" for epoch, loss_trajectory in enumerate(loss_trajectory_per_epoch) for j, avg_loss in enumerate(loss_trajectory) ]
    full_train_trajectory = [avg_loss for epoch, loss_trajectory in enumerate(loss_trajectory_per_epoch) for j, avg_loss in enumerate(loss_trajectory)]
    ax.plot(x, full_train_trajectory)

    ax.set(xlabel='Epoch, Step', ylabel='Loss',
            title='Train loss trajectory')
    step_size = 100
    ax.xaxis.set_ticks(range(0, len(x_ticks), step_size), x_ticks[::step_size])
    ax.grid()

    fig.savefig(trajectory_file)

def split_data_based_on_subtasks(data, subtasks): # split the data into data_instances based on subtask_labels
    subtasks_data = {subtask: list() for subtask in subtasks}
    for text, chunk, chunk_id, chunk_start_text_id, chunk_end_text_id, tokenized_tweet, tokenized_tweet_with_masked_chunk, subtask_labels_dict in data:
        for subtask in subtasks:
            subtasks_data[subtask].append((text, chunk, chunk_id, chunk_start_text_id, chunk_end_text_id, tokenized_tweet, tokenized_tweet_with_masked_chunk, subtask_labels_dict[subtask][0], subtask_labels_dict[subtask][1]))
    return subtasks_data

def log_multitask_data_statistics(data, subtasks):
    logging.info(f"Total instances in the data = {len(data)}")
    pos_counts = {subtask: sum(subtask_labels_dict[subtask][1] for _,_,_,_,_,_,_,subtask_labels_dict in data) for subtask in subtasks}
    neg_counts = dict()
    for subtask in subtasks:
        neg_counts[subtask] = len(data) - pos_counts[subtask]
        logging.info(f"Subtask:{subtask:>15}\tPositive labels = {pos_counts[subtask]}\tNegative labels = {neg_counts[subtask]}")
    return len(data), pos_counts, neg_counts

def main():
    # Read all the data instances
    task_instances_dict, tag_statistics, question_keys_and_tags = load_from_pickle(args.data_file)
    data, subtasks_list = get_multitask_instances_for_valid_tasks(task_instances_dict, tag_statistics)
    data = add_marker_for_loss_ignore(data, 1.0 if args.loss_for_no_consensus else 0.0)

    if args.retrain:
        if args.large_bert:
            model_name = "bert-large-cased"
        elif args.covid_bert:
            model_name = "digitalepidemiologylab/covid-twitter-bert"
        else:
            model_name = "bert-base-cased"

        logging.info("Creating and training the model from '" + model_name + "'")
        # Create the save_directory if not exists
        make_dir_if_not_exists(args.save_directory)

        # Initialize tokenizer and model with pretrained weights
        tokenizer = BertTokenizer.from_pretrained(model_name)
        config = BertConfig.from_pretrained(model_name)
        config.subtasks = subtasks_list
        model = MultiTaskBertForCovidEntityClassification.from_pretrained(model_name, config=config)

        # Add new tokens in tokenizer
        new_special_tokens_dict = {"additional_special_tokens": ["<E>", "</E>", "<URL>", "@USER"]}
        tokenizer.add_special_tokens(new_special_tokens_dict)
        
        # Add the new embeddings in the weights
        print("Embeddings type:", model.bert.embeddings.word_embeddings.weight.data.type())
        print("Embeddings shape:", model.bert.embeddings.word_embeddings.weight.data.size())
        embedding_size = model.bert.embeddings.word_embeddings.weight.size(1)
        new_embeddings = torch.FloatTensor(len(new_special_tokens_dict["additional_special_tokens"]), embedding_size).uniform_(-0.1, 0.1)
        # new_embeddings = torch.FloatTensor(2, embedding_size).uniform_(-0.1, 0.1)
        print("new_embeddings shape:", new_embeddings.size())
        new_embedding_weight = torch.cat((model.bert.embeddings.word_embeddings.weight.data,new_embeddings), 0)
        model.bert.embeddings.word_embeddings.weight.data = new_embedding_weight
        print("Embeddings shape:", model.bert.embeddings.word_embeddings.weight.data.size())
        # Update model config vocab size
        model.config.vocab_size = model.config.vocab_size + len(new_special_tokens_dict["additional_special_tokens"])
    else:
        # Load the tokenizer and model from the save_directory
        tokenizer = BertTokenizer.from_pretrained(args.save_directory)
        model = MultiTaskBertForCovidEntityClassification.from_pretrained(args.save_directory)
        # Load from individual state dicts
        for subtask in model.subtasks:
            model.classifiers[subtask].load_state_dict(torch.load(os.path.join(args.save_directory, f"{subtask}_classifier.bin")))
    model.to(device)
    if args.wandb:
        wandb.watch(model)

    # Explicitly move the classifiers to device
    for subtask, classifier in model.classifiers.items():
        classifier.to(device)
    for subtask, classifier in model.context_vectors.items():
        classifier.to(device)

    entity_start_token_id = tokenizer.convert_tokens_to_ids(["<E>"])[0]
    entity_end_token_id = tokenizer.convert_tokens_to_ids(["</E>"])[0]

    logging.info(f"Task dataset for task: {args.task} loaded from {args.data_file}.")
    
    model_config = dict()
    results = dict()

    # Split the data into train, dev and test and shuffle the train segment
    train_data, dev_data = split_multitask_instances_in_train_dev(data)
    random.shuffle(train_data)		# shuffle happens in-place
    logging.info("Train Data:")
    total_train_size, pos_subtasks_train_size, neg_subtasks_train_size = log_multitask_data_statistics(train_data, model.subtasks)
    logging.info("Dev Data:")
    total_dev_size, pos_subtasks_dev_size, neg_subtasks_dev_size = log_multitask_data_statistics(dev_data, model.subtasks)
    #logging.info("Test Data:")
    #total_test_size, pos_subtasks_test_size, neg_subtasks_test_size = log_multitask_data_statistics(test_data, model.subtasks)
    logging.info("\n")
    model_config["train_data"] = {"size":total_train_size, "pos":pos_subtasks_train_size, "neg":neg_subtasks_train_size}
    model_config["dev_data"] = {"size":total_dev_size, "pos":pos_subtasks_dev_size, "neg":neg_subtasks_dev_size}
    #model_config["test_data"] = {"size":total_test_size, "pos":pos_subtasks_test_size, "neg":neg_subtasks_test_size}

    # Extract subtasks data for dev and test
    train_subtasks_data = split_data_based_on_subtasks(train_data, model.subtasks)
    dev_subtasks_data = split_data_based_on_subtasks(dev_data, model.subtasks)
    #test_subtasks_data = split_data_based_on_subtasks(test_data, model.subtasks)

    # Load the instances into pytorch dataset
    train_dataset = COVID19TaskDataset(train_data)
    dev_dataset = COVID19TaskDataset(dev_data)
    #test_dataset = COVID19TaskDataset(test_data)
    logging.info("Loaded the datasets into Pytorch datasets")

    tokenize_collator = TokenizeCollator(tokenizer, model.subtasks, entity_start_token_id, entity_end_token_id)
    train_dataloader = DataLoader(train_dataset, batch_size=POSSIBLE_BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=tokenize_collator)
    dev_dataloader = DataLoader(dev_dataset, batch_size=POSSIBLE_BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=tokenize_collator)
    #test_dataloader = DataLoader(test_dataset, batch_size=POSSIBLE_BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=tokenize_collator)
    logging.info("Created train and test dataloaders with batch aggregation")

    # Only retrain if needed
    if args.retrain:
        optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
        logging.info("Created model optimizer")
        #if args.sentence_level_classify:
        #    args.n_epochs += 2
        epochs = args.n_epochs

        # Total number of training steps is [number of batches] x [number of epochs]. 
        total_steps = len(train_dataloader) * epochs

        # Create the learning rate scheduler.
        # NOTE: num_warmup_steps = 0 is the Default value in run_glue.py
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        # We'll store a number of quantities such as training and validation loss, validation accuracy, and timings.
        training_stats = []
        print("\n\n\n ====== Training for task", args.task, "=============\n\n\n")
        logging.info(f"Initiating training loop for {args.n_epochs} epochs...")
        print(model.state_dict().keys())


        total_start_time = time.time()

        # Find the accumulation steps
        accumulation_steps = args.batch_size/POSSIBLE_BATCH_SIZE

        # Dev validation trajectory
        epoch_train_loss = list()
        train_subtasks_validation_statistics = {subtask: list() for subtask in model.subtasks}
        dev_subtasks_validation_statistics = {subtask: list() for subtask in model.subtasks}
        best_dev_F1 = 0
        for epoch in range(epochs):

            logging.info(f"Initiating Epoch {epoch+1}:")

            # Reset the total loss for each epoch.
            total_train_loss = 0
            train_loss_trajectory = list()

            # Reset timer for each epoch
            start_time = time.time()
            model.train()

            dev_log_frequency = 5
            n_steps = len(train_dataloader)
            dev_steps = int(n_steps / dev_log_frequency)
            for step, batch in enumerate(train_dataloader):
                # Upload labels of each subtask to device
                for subtask in model.subtasks:
                    subtask_labels = batch["gold_labels"][subtask]
                    subtask_labels = subtask_labels.to(device)
                    batch["gold_labels"][subtask] = subtask_labels
                    batch["label_ignore_loss"][subtask] = batch["label_ignore_loss"][subtask].to(device)

                # Forward
                input_dict = {"input_ids": batch["input_ids"].to(device),
                                "entity_start_positions": batch["entity_start_positions"].to(device),
                                "entity_end_positions": batch["entity_end_positions"].to(device),
                                "labels": batch["gold_labels"],
                                "label_weight": batch["label_ignore_loss"]
                            }

                input_ids = batch["input_ids"]
                entity_start_positions = batch["entity_start_positions"]
                gold_labels = batch["gold_labels"]
                batch_data = batch["batch_data"]
                loss, logits = model(**input_dict)

                # Accumulate loss
                total_train_loss += loss.item()

                # Backward: compute gradients
                loss.backward()
                
                if (step + 1) % accumulation_steps == 0:
                    # Calculate elapsed time in minutes and print loss on the tqdm bar
                    elapsed = format_time(time.time() - start_time)
                    avg_train_loss = total_train_loss/(step+1)

                    # keep track of changing avg_train_loss
                    train_loss_trajectory.append(avg_train_loss)
                    if (step + 1) % (accumulation_steps*20) == 0:
                        print(f"Epoch:{epoch+1}|Batch:{step}/{len(train_dataloader)}|Time:{elapsed}|Avg. Loss:{avg_train_loss:.4f}|Loss:{loss.item():.4f}")

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    # Clean the model's previous gradients
                    model.zero_grad()
                    scheduler.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)

            # Perform validation with the model and log the performance
            print("\n")
            logging.info("Running Validation...")
            # Put the model in evaluation mode--the dropout layers behave differently during evaluation.
            model.eval()
            dev_predicted_labels, dev_prediction_scores, dev_gold_labels = make_predictions_on_dataset(dev_dataloader, model, device, args.task + "_dev", True)

            wandb_log_dict = {"Train Loss":avg_train_loss}
            print("Dev Set:")
            collect_TP_FP_FN = {"TP": 0, "FP": 0, "FN": 0}
            for subtask in model.subtasks:
                dev_subtask_data = dev_subtasks_data[subtask]
                dev_subtask_prediction_scores = dev_prediction_scores[subtask]
                dev_F1, dev_P, dev_R, dev_TP, dev_FP, dev_FN = get_TP_FP_FN(dev_subtask_data, dev_subtask_prediction_scores, task=subtask)
                if subtask not in IGNORE_TASKS:
                    collect_TP_FP_FN["TP"] += dev_TP
       	       	    collect_TP_FP_FN["FP"] += dev_FP
       	       	    collect_TP_FP_FN["FN"] += dev_FN
                else:
                    print("IGNORE: ", end = "")

                print(f"Subtask:{subtask:>15}\tN={dev_TP + dev_FN}\tF1={dev_F1}\tP={dev_P}\tR={dev_R}\tTP={dev_TP}\tFP={dev_FP}\tFN={dev_FN}")
                dev_subtasks_validation_statistics[subtask].append((epoch + 1, step + 1, dev_TP + dev_FN, dev_F1, dev_P, dev_R, dev_TP, dev_FP, dev_FN))

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
                torch.save(model.state_dict(), args.output_dir + "/ckpt.pth")
                #print(model.state_dict().keys())
                #model.save_pretrained(args.save_directory)
            model.train()

            training_time = format_time(time.time() - start_time)

            # Record all statistics from this epoch.
            training_stats.append({
                    'epoch': epoch + 1,
                    'Training Loss': avg_train_loss,
                    'Training Time': training_time})

            # Save the loss trajectory
            epoch_train_loss.append(train_loss_trajectory)
            print("\n\n")

        logging.info(f"Training complete with total Train time:{format_time(time.time()- total_start_time)}")
        log_list(training_stats)

        model.load_state_dict(torch.load(args.output_dir + "/ckpt.pth"))
        model.eval()
        # Save the model and the Tokenizer here:
        #logging.info(f"Saving the model and tokenizer in {args.save_directory}")
        #model.save_pretrained(args.save_directory)

        # Save each subtask classifiers weights to individual state dicts
        #for subtask, classifier in model.classifiers.items():
        #    classifier_save_file = os.path.join(args.save_directory, f"{subtask}_classifier.bin")
        #    logging.info(f"Saving the model's {subtask} classifier weights at {classifier_save_file}")
        #    torch.save(classifier.state_dict(), classifier_save_file)
        #tokenizer.save_pretrained(args.save_directory)

        # Plot the train loss trajectory in a plot
        #train_loss_trajectory_plot_file = os.path.join(args.output_dir, "train_loss_trajectory.png")
        #logging.info(f"Saving the Train loss trajectory at {train_loss_trajectory_plot_file}")
        #print(epoch_train_loss)

        # TODO: Plot the validation performance
        # Save dev_subtasks_validation_statistics
    else:
        raise
        logging.info("No training needed. Directly going to evaluation!")

    # Save the model name in the model_config file
    model_config["model"] = "MultiTaskBertForCovidEntityClassification"
    model_config["epochs"] = args.n_epochs

    # Find best threshold for each subtask based on dev set performance
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #test_predicted_labels, test_prediction_scores, test_gold_labels = make_predictions_on_dataset(test_dataloader, model, device, args.task, True)
    dev_predicted_labels, dev_prediction_scores, dev_gold_labels = make_predictions_on_dataset(dev_dataloader, model, device, args.task + "_dev", True)

    best_test_thresholds = {subtask: 0.5 for subtask in model.subtasks}
    best_dev_thresholds = {subtask: 0.5 for subtask in model.subtasks}
    best_test_F1s = {subtask: 0.0 for subtask in model.subtasks}
    best_dev_F1s = {subtask: 0.0 for subtask in model.subtasks}
    #test_subtasks_t_F1_P_Rs = {subtask: list() for subtask in model.subtasks}
    dev_subtasks_t_F1_P_Rs = {subtask: list() for subtask in model.subtasks}

    for subtask in model.subtasks:	
        dev_subtask_data = dev_subtasks_data[subtask]
        dev_subtask_prediction_scores = dev_prediction_scores[subtask]
        for t in thresholds:
            dev_F1, dev_P, dev_R, dev_TP, dev_FP, dev_FN = get_TP_FP_FN(dev_subtask_data, dev_subtask_prediction_scores, THRESHOLD=t, task=subtask)
            dev_subtasks_t_F1_P_Rs[subtask].append((t, dev_F1, dev_P, dev_R, dev_TP + dev_FN, dev_TP, dev_FP, dev_FN))
            if dev_F1 > best_dev_F1s[subtask]:
                best_dev_thresholds[subtask] = t
                best_dev_F1s[subtask] = dev_F1

        logging.info(f"Subtask:{subtask:>15}")
        log_list(dev_subtasks_t_F1_P_Rs[subtask])
        logging.info(f"Best Dev Threshold for subtask: {best_dev_thresholds[subtask]}\t Best dev F1: {best_dev_F1s[subtask]}")

    # Save the best dev threshold and dev_F1 in results dict
    results["best_dev_threshold"] = best_dev_thresholds
    results["best_dev_F1s"] = best_dev_F1s
    results["dev_t_F1_P_Rs"] = dev_subtasks_t_F1_P_Rs

    # Evaluate on Test
    logging.info("Testing on eval dataset")
    predicted_labels, prediction_scores, gold_labels = make_predictions_on_dataset(dev_dataloader, model, device, args.task)

    # Test 
    for subtask in model.subtasks:
        logging.info(f"\nTesting the trained classifier on subtask: {subtask}")

        results[subtask] = dict()
        cm = metrics.confusion_matrix(gold_labels[subtask], predicted_labels[subtask])
        classification_report = metrics.classification_report(gold_labels[subtask], predicted_labels[subtask], output_dict=True)
        logging.info(cm)
        logging.info(metrics.classification_report(gold_labels[subtask], predicted_labels[subtask]))
        results[subtask]["CM"] = cm.tolist()			# Storing it as list of lists instead of numpy.ndarray
        results[subtask]["Classification Report"] = classification_report

        # SQuAD style EM and F1 evaluation for all test cases and for positive test cases (i.e. for cases where annotators had a gold annotation)
        EM_score, F1_score, total = get_raw_scores(dev_subtasks_data[subtask], prediction_scores[subtask])
        logging.info("Word overlap based SQuAD evaluation style metrics:")
        logging.info(f"Total number of cases: {total}")
        logging.info(f"EM_score: {EM_score}")
        logging.info(f"F1_score: {F1_score}")
        results[subtask]["SQuAD_EM"] = EM_score
        results[subtask]["SQuAD_F1"] = F1_score
        results[subtask]["SQuAD_total"] = total
        pos_EM_score, pos_F1_score, pos_total = get_raw_scores(dev_subtasks_data[subtask], prediction_scores[subtask], positive_only=True)
        logging.info(f"Total number of Positive cases: {pos_total}")
        logging.info(f"Pos. EM_score: {pos_EM_score}")
        logging.info(f"Pos. F1_score: {pos_F1_score}")
        results[subtask]["SQuAD_Pos. EM"] = pos_EM_score
        results[subtask]["SQuAD_Pos. F1"] = pos_F1_score
        results[subtask]["SQuAD_Pos. EM_F1_total"] = pos_total

        # New evaluation suggested by Alan
        F1, P, R, TP, FP, FN = get_TP_FP_FN(dev_subtasks_data[subtask], prediction_scores[subtask], THRESHOLD=best_dev_thresholds[subtask], task=subtask)
        logging.info("New evaluation scores:")
        logging.info(f"F1: {F1}")
        logging.info(f"Precision: {P}")
        logging.info(f"Recall: {R}")
        logging.info(f"True Positive: {TP}")
        logging.info(f"False Positive: {FP}")
        logging.info(f"False Negative: {FN}")
        results[subtask]["F1"] = F1
        results[subtask]["P"] = P
        results[subtask]["R"] = R
        results[subtask]["TP"] = TP
        results[subtask]["FP"] = FP
        results[subtask]["FN"] = FN
        N = TP + FN
        results[subtask]["N"] = N

    # Save model_config and results
    model_config_file = os.path.join(args.output_dir, "model_config.json")
    results_file = os.path.join(args.output_dir, "results.json")
    logging.info(f"Saving model config at {model_config_file}")
    save_in_json(model_config, model_config_file)
    logging.info(f"Saving results at {results_file}")
    save_in_json(results, results_file)

if __name__ == '__main__':
    main()
