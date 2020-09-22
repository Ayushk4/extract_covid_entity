# We will add all the common utility functions over here

import os
import re
import string
import collections
import json
import pickle


import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

MIN_POS_SAMPLES_THRESHOLD = -1
Q_TOKEN = "<Q_TARGET>"
URL_TOKEN = "<URL>"

def print_list(l):
	for e in l:
		print(e)
	print()

def log_list(l):
	for e in l:
		logging.info(e)
	logging.info("")

def save_in_pickle(save_object, save_file):
	with open(save_file, "wb") as pickle_out:
		pickle.dump(save_object, pickle_out)

def load_from_pickle(pickle_file):
	with open(pickle_file, "rb") as pickle_in:
		return pickle.load(pickle_in)

def save_in_json(save_dict, save_file):
	with open(save_file, 'w') as fp:
		json.dump(save_dict, fp)

def load_from_json(json_file):
	with open(json_file, 'r') as fp:
		return json.load(fp)

def read_json_line(path):
	output = []
	i = 0
	with open(path, 'r') as f:
		for line in f:
			output.append(json.loads(line))
		i += 1
		print(i)
	return output

def write_json_line(data, path):
	with open(path, 'w') as f:
		for i in data:
			f.write("%s\n" % json.dumps(i))
	return None

def make_dir_if_not_exists(directory):
	if not os.path.exists(directory):
		logging.info("Creating new directory: {}".format(directory))
		os.makedirs(directory)

def extract_instances_for_current_subtask(task_instances, sub_task):
	return task_instances[sub_task]

def get_multitask_instances_for_valid_tasks(task_instances, tag_statistics):
	# Extract instances and labels from all the sub-task
	# Align w.r.t. instances and merge all the sub-task labels
	subtasks = list()
	for subtask in task_instances.keys():
		#current_question_tag_statistics = tag_statistics[0][subtask]
		#if len(current_question_tag_statistics) > 1 and current_question_tag_statistics[1] >= MIN_POS_SAMPLES_THRESHOLD:
		subtasks.append(subtask)
	
	# For each tweet we will first extract all its instances from each task and their corresponding labels
	text_to_subtask_instances = dict()
	original_text_list = list()
	for subtask in subtasks:
		# get the instances for current subtask and add it to a set
		for text, chunk, chunk_id, chunk_start_text_id, chunk_end_text_id, tokenized_tweet, tokenized_tweet_with_masked_chunk, gold_chunk, label in task_instances[subtask]:
			instance = (text, chunk, chunk_id, chunk_start_text_id, chunk_end_text_id, tokenized_tweet, tokenized_tweet_with_masked_chunk)
			if text not in text_to_subtask_instances:
				original_text_list.append(text)
				text_to_subtask_instances[text] = dict()
			text_to_subtask_instances[text].setdefault(instance, dict())
			text_to_subtask_instances[text][instance][subtask] = (gold_chunk, label)

	# For each instance we need to make sure that it has all the subtask labels
	# For missing subtask labels we will give a default label of 0
	for text in original_text_list:
		for instance, subtasks_labels_dict in text_to_subtask_instances[text].items():
			for subtask in subtasks:
				if subtask not in subtasks_labels_dict:
					# Adding empty label for this instance
					subtasks_labels_dict[subtask] = ([], 0)
			# update the subtask labels_dict in the text_to_subtask_instances data structure
			assert len(subtasks_labels_dict) == len(subtasks)
			text_to_subtask_instances[text][instance] = subtasks_labels_dict

	# Merge all the instances into one list
	all_multitask_instances = list()
	for text in original_text_list:
		for instance, subtasks_labels_dict in text_to_subtask_instances[text].items():
			all_multitask_instances.append((*instance, subtasks_labels_dict))
	return all_multitask_instances, subtasks

def split_multitask_instances_in_train_dev_test(multitask_instances, TRAIN_RATIO = 0.6, DEV_RATIO = 0.15):
	# Group the multitask_instances by original tweet
	original_tweets = dict()
	original_tweets_list = list()
	# text :: candidate_chunk :: candidate_chunk_id :: chunk_start_text_id :: chunk_end_text_id :: tokenized_tweet :: tokenized_tweet_with_masked_q_token :: tagged_chunks :: question_label
	for tweet,_,_,_,_,_,_,_ in multitask_instances:
		if tweet not in original_tweets:
			original_tweets[tweet] = 1
			original_tweets_list.append(tweet)
		else:
			original_tweets[tweet] += 1

	train_size = int(len(original_tweets_list) * TRAIN_RATIO)
	dev_size = int(len(original_tweets_list) * DEV_RATIO)
	train_tweets = original_tweets_list[:train_size]
	dev_tweets = original_tweets_list[train_size:train_size + dev_size]
	test_tweets = original_tweets_list[train_size + dev_size:]
	segment_multitask_instances = {"train": list(), "dev": list(), "test": list()}

	# A dictionary that stores the segment each tweet belongs to
	tweets_to_segment = dict()
	for tweet in train_tweets:
		tweets_to_segment[tweet] = "train"
	for tweet in dev_tweets:
		tweets_to_segment[tweet] = "dev"
	for tweet in test_tweets:
		tweets_to_segment[tweet] = "test"

	# Get multitask_instances
	for instance in multitask_instances:
		tweet = instance[0]
		segment_multitask_instances[tweets_to_segment[tweet]].append(instance)

	return segment_multitask_instances['train'], segment_multitask_instances['dev'], segment_multitask_instances['test']

def split_multitask_instances_in_train_dev(multitask_instances, TRAIN_RATIO = 0.7):
        # Group the multitask_instances by original tweet
        original_tweets = dict()
        original_tweets_list = list()
        # text :: candidate_chunk :: candidate_chunk_id :: chunk_start_text_id :: chunk_end_text_id :: tokenized_tweet :: tokenized_tweet_with_masked_q_token :: tagged_chunks :: question_label
        for tweet,_,_,_,_,_,_,_ in multitask_instances:
                if tweet not in original_tweets:
                        original_tweets[tweet] = 1
                        original_tweets_list.append(tweet)
                else:
                     	original_tweets[tweet] += 1

        train_size = int(len(original_tweets_list) * TRAIN_RATIO)
        train_tweets = original_tweets_list[:train_size]
        dev_tweets = original_tweets_list[train_size:]
        segment_multitask_instances = {"train": list(), "dev": list()}

        # A dictionary that stores the segment each tweet belongs to
        tweets_to_segment = dict()
        for tweet in train_tweets:
                tweets_to_segment[tweet] = "train"
        for tweet in dev_tweets:
                tweets_to_segment[tweet] = "dev"

        # Get multitask_instances
        for instance in multitask_instances:
                tweet = instance[0]
                segment_multitask_instances[tweets_to_segment[tweet]].append(instance)

        return segment_multitask_instances['train'], segment_multitask_instances['dev']

def split_instances_in_train_dev_test(instances, TRAIN_RATIO = 0.6, DEV_RATIO = 0.15):
	# Group the instances by original tweet
	original_tweets = dict()
	original_tweets_list = list()
	# text :: candidate_chunk :: candidate_chunk_id :: chunk_start_text_id :: chunk_end_text_id :: tokenized_tweet :: tokenized_tweet_with_masked_q_token :: tagged_chunks :: question_label
	for tweet,_,_,_,_,_,_,_,_ in instances:
		if tweet not in original_tweets:
			original_tweets[tweet] = 1
			original_tweets_list.append(tweet)
		else:
			original_tweets[tweet] += 1

	train_size = int(len(original_tweets_list) * TRAIN_RATIO)
	dev_size = int(len(original_tweets_list) * DEV_RATIO)
	train_tweets = original_tweets_list[:train_size]
	dev_tweets = original_tweets_list[train_size:train_size + dev_size]
	test_tweets = original_tweets_list[train_size + dev_size:]
	segment_instances = {"train": list(), "dev": list(), "test": list()}

	# A dictionary that stores the segment each tweet belongs to
	tweets_to_segment = dict()
	for tweet in train_tweets:
		tweets_to_segment[tweet] = "train"
	for tweet in dev_tweets:
		tweets_to_segment[tweet] = "dev"
	for tweet in test_tweets:
		tweets_to_segment[tweet] = "test"

	# Get instances
	for instance in instances:
		tweet = instance[0]
		segment_instances[tweets_to_segment[tweet]].append(instance)

	return segment_instances['train'], segment_instances['dev'], segment_instances['test']

def log_data_statistics(data):
	logging.info(f"Total instances in the data = {len(data)}")
	pos_count = sum(label for _,_,_,_,_,_,_,_,label in data)
	logging.info(f"Positive labels = {pos_count} Negative labels = {len(data) - pos_count}")
	return len(data), pos_count, (len(data) - pos_count)

def add_marker_for_loss_ignore(data, no_consensus_weight = 0.0): # We ignore NO_CONSENSUS
	loss_marked_data = []
	for e0,e1,e2,e3,e4,e5,e6,e7 in data:
		loss_marked_labels = dict()
		for key in e7.keys():
			loss_marked_labels[key] = (e7[key][0], e7[key][1],
										no_consensus_weight if len(e7[key][0]) > 0 and e7[key][0] == ['NO_CONSENSUS'] else 1.0)
		loss_marked_data.append((e0,e1,e2,e3,e4,e5,e6,loss_marked_labels))
	return loss_marked_data

# SQuAD F-1 evaluation
def normalize_answer(s):
	"""Lower text and remove punctuation, articles and extra whitespace."""
	def remove_articles(text):
		regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
		return re.sub(regex, ' ', text)
	def white_space_fix(text):
		return ' '.join(text.split())
	def remove_punc(text):
		exclude = set(string.punctuation)
		return ''.join(ch for ch in text if ch not in exclude)
	def lower(text):
		return text.lower()
	return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
	if not s: return []
	return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
	return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
	gold_toks = get_tokens(a_gold)
	pred_toks = get_tokens(a_pred)
	common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
	num_same = sum(common.values())
	if len(gold_toks) == 0 or len(pred_toks) == 0:
		# If either is no-answer, then F1 is 1 if they agree, 0 otherwise
		return int(gold_toks == pred_toks)
	if num_same == 0:
		return 0
	precision = 1.0 * num_same / len(pred_toks)
	recall = 1.0 * num_same / len(gold_toks)
	f1 = (2 * precision * recall) / (precision + recall)
	return f1

def get_raw_scores(data, prediction_scores, positive_only=False):
	predicted_chunks_for_each_instance = dict()
	# text :: candidate_chunk :: candidate_chunk_id :: chunk_start_text_id :: chunk_end_text_id :: tokenized_tweet :: tokenized_tweet_with_masked_q_token :: tagged_chunks :: question_label
	for (text, chunk, chunk_id, chunk_start_text_id, chunk_end_text_id, tokenized_tweet, tokenized_tweet_with_masked_chunk, gold_chunk, label), prediction_score in zip(data, prediction_scores):
		original_text = text
		predicted_chunks_for_each_instance.setdefault(original_text, ('', 0.0, set(), set()))
		current_predicted_chunk, current_predicted_chunk_score, predicted_chunks, gold_chunks = predicted_chunks_for_each_instance[original_text]
		if gold_chunk != ['Not Specified']:
			gold_chunks = gold_chunks.union(set(gold_chunk))
		if prediction_score > 0.5:
			# Save this prediction in the predicted chunks
			predicted_chunks.add(chunk)
			current_predicted_chunk_score = prediction_score
			current_predicted_chunk = chunk
			predicted_chunks_for_each_instance[original_text] = current_predicted_chunk, current_predicted_chunk_score, predicted_chunks, gold_chunks
		elif prediction_score > current_predicted_chunk_score:
			# only update the current_predicted_chunk and its score
			current_predicted_chunk_score = prediction_score
			current_predicted_chunk = chunk
			predicted_chunks_for_each_instance[original_text] = current_predicted_chunk, current_predicted_chunk_score, predicted_chunks, gold_chunks
	total = 0.0
	exact_scores, f1_scores = 0.0, 0.0
	for original_text, (current_predicted_chunk, current_predicted_chunk_score, predicted_chunks, gold_chunks) in predicted_chunks_for_each_instance.items():
		if len(gold_chunks) > 0:
			if len(predicted_chunks) > 0:
				for predicted_chunk in predicted_chunks:
					# Get best exact_score and f1_score compared to all the gold_chunks
					best_exact_score, best_f1_score = 0.0, 0.0
					for gold_chunk in gold_chunks:
						best_exact_score = max(best_exact_score, compute_exact(gold_chunk, predicted_chunk))
						best_f1_score = max(best_f1_score, compute_f1(gold_chunk, predicted_chunk))
					exact_scores += best_exact_score
					f1_scores += best_f1_score
					total += 1.0
			else:
				# Assume the top prediction for this (text, gold_chunk) pair as final prediction
				# Get best exact_score and f1_score compared to all the gold_chunks
				best_exact_score, best_f1_score = 0.0, 0.0
				exact_scores += best_exact_score
				f1_scores += best_f1_score
				total += 1.0
			
		elif len(gold_chunks) == 0 and not positive_only:
			if len(predicted_chunks) > 0:
				# Model predicted something and the gold is also nothing
				for i in range(len(predicted_chunks)):
					# Penalize for every incorrect predicted chunk
					best_exact_score, best_f1_score = 0.0, 0.0
					exact_scores += best_exact_score
					f1_scores += best_f1_score
					total += 1.0
			else:
				# Model predicted nothing and the gold is also nothing
				best_exact_score, best_f1_score = 1.0, 1.0
				exact_scores += best_exact_score
				f1_scores += best_f1_score
				total += 1.0
		
	if total == 0:
		predictions_exact_score = total
		predictions_f1_score = total
	else:
		predictions_exact_score = exact_scores * 100.0 / total
		predictions_f1_score = f1_scores * 100.0 / total
	return predictions_exact_score, predictions_f1_score, total

def get_TP_FP_FN(data, prediction_scores, THRESHOLD=0.5, task=None):
	if task == None:
		raise
	if task == 'part1.Response':
		TP, FP, FN = 0.0, 0.0, 0.0
		for d, score in zip(data, prediction_scores):
			_, _, _, _, _, _, _, _, label = d
			if score > THRESHOLD:
				if label == 1:
					TP += 1
				else:
					FP += 1
			else:
				if label == 1:
					FN += 1
		if TP + FP == 0:
			P = 0.0
		else:
			P = TP / (TP + FP)

		if TP + FN == 0:
			R = 0.0
		else:
			R = TP / (TP + FN)

		if P + R == 0:
			F1 = 0.0
		else:
			F1 = 2.0 * P * R / (P + R)
		return F1, P, R, TP, FP, FN


	predicted_chunks_for_each_instance = dict()
	for (text, chunk, chunk_id, chunk_start_text_id, chunk_end_text_id, tokenized_tweet, tokenized_tweet_with_masked_chunk, gold_chunk, label), prediction_score in zip(data, prediction_scores):
		original_text = text
		predicted_chunks_for_each_instance.setdefault(original_text, ('', 0.0, set(), set()))
		current_predicted_chunk, current_predicted_chunk_score, predicted_chunks, gold_chunks = predicted_chunks_for_each_instance[original_text]

		if gold_chunk != ['Not Specified'] and label == 1:
			gold_chunks = gold_chunks.union(set(gold_chunk))
			predicted_chunks_for_each_instance[original_text] = current_predicted_chunk, current_predicted_chunk_score, predicted_chunks, gold_chunks
		if prediction_score > THRESHOLD:
			# Save this prediction in the predicted chunks
			predicted_chunks.add(chunk)
			current_predicted_chunk_score = prediction_score
			current_predicted_chunk = chunk
			predicted_chunks_for_each_instance[original_text] = current_predicted_chunk, current_predicted_chunk_score, predicted_chunks, gold_chunks
		elif prediction_score > current_predicted_chunk_score:
			# only update the current_predicted_chunk and its score
			current_predicted_chunk_score = prediction_score
			current_predicted_chunk = chunk
			predicted_chunks_for_each_instance[original_text] = current_predicted_chunk, current_predicted_chunk_score, predicted_chunks, gold_chunks
	
	# Take every span that is predicted by the model, and every gold span in the data.
	# Then, you can easily calculate the number of True Positives, False Positives and False negatives.
	# True positives are predicted spans that appear in the gold labels.
	# False positives are predicted spans that don't appear in the gold labels.
	# False negatives are gold spans that weren't in the set of spans predicted by the model.
	# then you can compute P/R using the standard formulas: P= TP/(TP + FP). R = TP/(TP+FN)

	TP, FP, FN = 0.0, 0.0, 0.0
	total_gold_chunks = 0
	for original_text, (current_predicted_chunk, current_predicted_chunk_score, predicted_chunks, gold_chunks) in predicted_chunks_for_each_instance.items():
		total_gold_chunks += len(gold_chunks)
		if len(gold_chunks) > 0:
			# print(f"{len(gold_chunks)} Gold chunks: {gold_chunks} for tweet {original_text}")
			if len(predicted_chunks) > 0:
				for predicted_chunk in predicted_chunks:
					if predicted_chunk in gold_chunks:
						TP += 1		# True positives are predicted spans that appear in the gold labels.
					else:
						FP += 1		# False positives are predicted spans that don't appear in the gold labels.
			for gold_chunk in gold_chunks:
				if gold_chunk not in predicted_chunks:
					FN += 1			# False negatives are gold spans that weren't in the set of spans predicted by the model.
		else:
			if len(predicted_chunks) > 0:
				for predicted_chunk in predicted_chunks:
					FP += 1			# False positives are predicted spans that don't appear in the gold labels.

	if TP + FP == 0:
		P = 0.0
	else:
		P = TP / (TP + FP)
	
	if TP + FN == 0:
		R = 0.0
	else:
		R = TP / (TP + FN)

	if P + R == 0:
		F1 = 0.0
	else:
		F1 = 2.0 * P * R / (P + R)
	return F1, P, R, TP, FP, FN
