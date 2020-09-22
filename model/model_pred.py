import torch
from transformers import BertTokenizer, BertPreTrainedModel, BertModel, BertConfig, AdamW, get_linear_schedule_with_warmup
from torch import nn
from torch.utils.data import Dataset, DataLoader
from multitask_bert_entitity_classifier import COVID19TaskDataset, TokenizeCollator, make_predictions_on_dataset, split_data_based_on_subtasks
from utils import load_from_pickle, get_multitask_instances_for_valid_tasks, split_multitask_instances_in_train_dev, get_TP_FP_FN, add_marker_for_loss_ignore
import random
import json
import argparse

TEXT_TO_TWEET_ID_PATH = "../test/txt_2_tid.json"
POSSIBLE_BATCH_SIZE=8
IGNORE_TASKS_DICT = {"tested_positive": ["part1.Response", "gender_male", "gender_female", "relation"],
                "tested_negative": ["part1.Response", "how_long", "gender_male", "gender_female", "relation"],
                "can_not_test": ["part1.Response", "symptoms", "relation"],
                "death": ["part1.Response", "symptoms", "relation"],
                "cure": ["part1.Response", "opinion"]
                }

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_file", help="Path to the pickle file that contains the training instances", type=str, required=True)
parser.add_argument("-t", "--task", help="Event for which we want to train the baseline", type=str, required=True)
parser.add_argument("-s", "--save_path", help="Path to the directory where saved model is.", type=str, required=True)
parser.add_argument("-o", "--output_dir", help="Path to the output directory where we will save all the model results", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda", help="name of the device to be used for training")
parser.add_argument("--sentence_level_classify", dest="sentence_level_classify", action="store_true", default=True)
args = parser.parse_args()
device = args.device="cuda"
args.sentence_level_classify=True
IGNORE_TASKS = IGNORE_TASKS_DICT[args.task]

TASK_TO_TID_KEY = {'tested_positive': 'positive', 'tested_negative': 'negative', 'can_not_test': 'can_not_test',
                   'cure': 'cure', 'death': 'death'}
text_to_tweetid = json.load(open(TEXT_TO_TWEET_ID_PATH, 'r'))[TASK_TO_TID_KEY[args.task]]

def log_multitask_data_statistics(data, subtasks):
    print(f"Total instances in the data = {len(data)}")
    pos_counts = {subtask: sum(subtask_labels_dict[subtask][1] for _,_,_,_,_,_,_,subtask_labels_dict in data) for subtask in subtasks}
    neg_counts = dict()
    for subtask in subtasks:
        neg_counts[subtask] = len(data) - pos_counts[subtask]
        print(f"Subtask:{subtask:>15}\tPositive labels = {pos_counts[subtask]}\tNegative labels = {neg_counts[subtask]}")
    return len(data), pos_counts, neg_counts

class MultiTaskBertForCovidEntityClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob * 2)

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

def get_chunk_tweet_id(data, prediction_scores, THRESHOLD=0.5): # code from get_TP_FP_FN in utils.py
    predicted_chunks_for_each_instance = dict()
    for (text, chunk, _, _, _, _, _, _, _), prediction_score in zip(data, prediction_scores):
        original_text = text
        predicted_chunks_for_each_instance.setdefault(original_text, set())
        predicted_chunks = predicted_chunks_for_each_instance[original_text]
        if prediction_score > THRESHOLD:
        # Save this prediction in the predicted chunks
            # print(chunk, prediction_score)
            predicted_chunks.add(chunk)
            predicted_chunks_for_each_instance[original_text] = predicted_chunks

    return {text_to_tweetid[text]:preds for text,preds in predicted_chunks_for_each_instance.items()}

def json_save_predicts(dev_pred_chunks, pred_save_path, question_keys_and_tags):
    guesses = {}
    i = 0
    current_subtasks = dev_pred_chunks.keys()
    for subtask in current_subtasks:
        question = question_keys_and_tags[subtask]
        if i == 0:
            for id, pred in dev_pred_chunks[subtask].items():
                guesses[id] = {question: list(pred)}
        else:
            for id, pred in dev_pred_chunks[subtask].items():
                guesses[id][question] = list(pred)
        i += 1

    fo = open(pred_save_path, "w")
    json.dump(guesses, fo)
    fo.close()
    print("Saved predicts as JSON:", pred_save_path)


def main_try(args):
    task_instances_dict, tag_statistics, question_keys_and_tags = load_from_pickle(args.data_file)
    data, subtasks_list = get_multitask_instances_for_valid_tasks(task_instances_dict, tag_statistics)
    data = add_marker_for_loss_ignore(data, 1.0 if False else 0.0)
    model_name = "digitalepidemiologylab/covid-twitter-bert"
    print("\n\n===========\n\n", subtasks_list, "\n\n===========\n\n")

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

    model.load_state_dict(torch.load(args.save_path + "ckpt.pth"))
    print("loaded_model")
    model.to("cuda")

    entity_start_token_id = tokenizer.convert_tokens_to_ids(["<E>"])[0]
    entity_end_token_id = tokenizer.convert_tokens_to_ids(["</E>"])[0]

    print(f"Task dataset for task: {args.task} loaded from {args.data_file}.")

    model_config = dict()
    results = dict()

    # Split the data into train, dev and test and shuffle the train segment
    dev_data = data
    print("Dev Data:")
    total_dev_size, pos_subtasks_dev_size, neg_subtasks_dev_size = log_multitask_data_statistics(dev_data, model.subtasks)
    model_config["dev_data"] = {"size":total_dev_size, "pos":pos_subtasks_dev_size, "neg":neg_subtasks_dev_size}

    # Extract subtasks data for dev and test
    dev_subtasks_data = split_data_based_on_subtasks(dev_data, model.subtasks)

    # Load the instances into pytorch dataset
    dev_dataset = COVID19TaskDataset(dev_data)

    tokenize_collator = TokenizeCollator(tokenizer, model.subtasks, entity_start_token_id, entity_end_token_id)
    dev_dataloader = DataLoader(dev_dataset, batch_size=POSSIBLE_BATCH_SIZE, collate_fn=tokenize_collator)
    print("Created dev dataloaders with batch aggregation")

    dev_predicted_labels, dev_prediction_scores, dev_gold_labels = make_predictions_on_dataset(dev_dataloader, model, device, args.task + "_dev", True)
    # print(dev_predicted_labels['age'][0], dev_prediction_scores['age'][0], dev_gold_labels['age'][0])
    assert dev_predicted_labels.keys() == dev_prediction_scores.keys()
    assert dev_predicted_labels.keys() == dev_gold_labels.keys()
    
    for st in dev_gold_labels.keys():
        print(st,":", len(dev_predicted_labels[st]), len(dev_prediction_scores[st]), len(dev_gold_labels[st]))

    dev_threshold = json.load(open(args.save_path + "/results.json", "r"))['best_dev_threshold']
    print(dev_threshold)

    # [print(k, v) for k,v in get_chunk_tweet_id(dev_subtasks_data['age'], dev_prediction_scores['age'], dev_threshold['age']).items()]
    dev_pred_chunks = {}
    for subtask in subtasks_list:
        if subtask not in IGNORE_TASKS:
            dev_pred_chunks[subtask] = get_chunk_tweet_id(dev_subtasks_data[subtask], dev_prediction_scores[subtask], 
                                                        dev_threshold[subtask])

    json_save_predicts(dev_pred_chunks, args.output_dir + "/" + args.task +".json",
                        {k:v for k,v in question_keys_and_tags})

    collect_TP_FP_FN = {"TP": 0.0001, "FP": 0.0001, "FN": 0.0001}
    for subtask in model.subtasks:
        dev_subtask_data = dev_subtasks_data[subtask]
        dev_subtask_prediction_scores = dev_prediction_scores[subtask]
        dev_F1, dev_P, dev_R, dev_TP, dev_FP, dev_FN = get_TP_FP_FN(dev_subtask_data, dev_subtask_prediction_scores, 
                                                                    dev_threshold[subtask], task=subtask)
        if subtask not in IGNORE_TASKS:
            collect_TP_FP_FN["TP"] += dev_TP
            collect_TP_FP_FN["FP"] += dev_FP
            collect_TP_FP_FN["FN"] += dev_FN
        else:
            print("IGNORE: ", end = "")

        print(f"Subtask:{subtask:>15}\tN={dev_TP + dev_FN}\tF1={dev_F1}\tP={dev_P}\tR={dev_R}\tTP={dev_TP}\tFP={dev_FP}\tFN={dev_FN}")

    dev_macro_P = collect_TP_FP_FN["TP"] / (collect_TP_FP_FN["TP"] + collect_TP_FP_FN["FP"])
    dev_macro_R = collect_TP_FP_FN["TP"] / (collect_TP_FP_FN["TP"] + collect_TP_FP_FN["FN"])
    dev_macro_F1 = (2 * dev_macro_P * dev_macro_R) / (dev_macro_P + dev_macro_R)
    print(collect_TP_FP_FN)
    print("dev_macro_P:", dev_macro_P, "\ndev_macro_R:", dev_macro_R, "\ndev_macro_F1:", dev_macro_F1,"\n")


main_try(args)

