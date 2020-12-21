# Extract COVID Entities

Leveraging Event Specific and Chunk Span features to Extract COVID Events - 1st at the leaderboard for the EMNLP 2020 workshop [WNUT Shared Task-3](http://noisy-text.github.io/2020/extract_covid19_event-shared_task.html).

This Repo contains
- Code for Models
- Trained models used for final submission and reporting
- Steps to replicate the results, including the dependencies.

Relevant Links: [arxiv-pdf](https://arxiv.org/pdf/2012.10052.pdf), [slides](https://docs.google.com/presentation/d/13DDY6VSmrVPBddTjWb3rThYRFlRDE_9fi4iyBrhJev4/edit?usp=sharing), [poster](https://github.com/noisy-text/noisy-text.github.io/blob/master/2020/posters/WNUT2020_91_poster%20-%20Tejas%20vaidhya.pdf)

Please cite with the following BiBTeX code:

```@article{2012.10052,
Author = {Ayush Kaushal and Tejas Vaidhya},
Title = {Leveraging Event Specific and Chunk Span features to Extract COVID Events from tweets},
Year = {2020},
Eprint = {arXiv:2012.10052},
Doi = {10.18653/v1/2020.wnut-1.79},
}
```

**Authors**: [Ayush Kaushal](https://github.com/Ayushk4) and [Tejas Vaidhya](https://github.com/tejasvaidhyadev)

## Overview

### Abstract

Twitter has acted as an important source of information during disasters and pandemic, especially during the times of COVID-19. In this paper, we describe our system entry for WNUT 2020 Shared Task-3. The task was aimed at automating the extraction of a variety of COVID-19 related events from Twitter, such as individuals who recently contracted the virus, someone with symptoms who were denied testing and believed remedies against the infection. The system consists of separate multi-task models for slot-filling subtasks and sentence-classification subtasks while leveraging the useful sentence-level information for the corresponding event. The system uses COVID-Twitter-Bert with attention-weighted pooling of candidate slot-chunk features to capture the useful information chunks. The system ranks 1st at the leader-board with F1 of 0.6598, without using any ensembles or additional datasets. The code and trained models
are available at this [https url](https://github.com/Ayushk4/extract_covid_entity/).

### System overview

Our system contains two models, one for sentence classification and one for slot-filling task, both with the following enhancements:
- An event-prediction task as auxiliary subtask
- Fuse event-prediction features for all the event-specific subtasks
- Weighted pooling over the candidate chunk span enabling the model to attend to subtask specific cues
- Domain-specific Covid-Twitter Bert

Refer [our paper](https://arxiv.org/pdf/2012.10052.pdf) for complete details.

**Slot-Filling**
----------------

![](images/slot_filling.png)

**Classification**
----------------

![](images/sent_classification.png)

## Dependencies and set-up

| Dependency | Version | Installation Command |
| ---------- | ------- | -------------------- |
| Python     | 3.8     | `conda create --name covid_entities python=3.8` and `conda activate covid_entities` |
| PyTorch, cudatoolkit    | 1.5.0, 10.1   | `conda install pytorch==1.5.0 cudatoolkit=10.1 -c pytorch` |
| Transformers (Huggingface) | 2.9.0 | `pip install transformers==2.9.0` |
| Scikit-learn | 0.23.1 | `pip install scikit-learn==0.23.1` |
| scipy        | 1.5.0  | `pip install scipy==1.5.0` |
| ekphrasis    | 0.5.1  | `pip install ekphrasis==0.5.1` |
| wandb        | -      | `pip install wandb`
<!--
- python 3.8
```conda create --name covid_entities python=3.8``` & ```conda activate covid_entities```
- PyTorch 1.5.0, cudatoolkit=10.1
```conda install pytorch==1.5.0 cudatoolkit=10.1 -c pytorch```
- Huggingface transformers - 2.9.0
```pip install transformers==2.9.0```
- scikit-learn 0.23.1
```pip install scikit-learn==0.23.1```
- scipy 1.5.0
```pip install scipy==1.5.0```
- ekphrasis 0.5.1
```pip install ekphrasis==0.5.1```
- wandb
```pip install wandb```
-->


## Instructions

0. Set up the codebase and requirements
   - `git clone https://github.com/Ayushk4/extract_covid_entity` & `cd extract_covid_entity`.
   - Follow the instructions from the `Dependencies and set-up` above to install the dependencies.
   - If you are interested in logging your runs, Set up your wandb. `wandb login`.
1. Set up the dataset: Follow instructions given in `data/README.md`
2. Recreating the experiments for our final submission:
   - **Slot-filling:** `python automate_multitask_bert_entity_classifier_experiments.py --sentence_level`.
   - **Sentence classification:** First pre_process by `python3 pre_process.py` (required only once) and then `python3 sent_model.py --data <PREPROCESSED-FILE-LOCATION> --task " + <TASK-NAME>`
   - You may add the following optional flags depending on which experiment you would like to replicate. Run_name - `--run=<YOUR_RUN_NAME>`; Use COVID_Twitter BERT - `--covid`. Track runs on Wandb - `--wandb`.


## Trained Models

Our model weights used in the submission have been [released now](https://github.com/Ayushk4/extract_covid_entity/releases).

##### Slot-filling models

| Task | Link |
| ------ | ------ |
| Tested Positive | [positive.tar.gz](https://github.com/Ayushk4/extract_covid_entity/releases/download/v0.0.1/positive.tar.gz) |
| Tested Negative | [negative.tar.gz](https://github.com/Ayushk4/extract_covid_entity/releases/download/v0.0.1/negative.tar.gz) |
| Denied Testing | [can_not_test.tar.gz](https://github.com/Ayushk4/extract_covid_entity/releases/download/v0.0.1/can_not_test.tar.gz) |
| Death | [death.tar.gz](https://github.com/Ayushk4/extract_covid_entity/releases/download/v0.0.1/death.tar.gz) |
| Cure/Prevention | [cure.tar.gz](https://github.com/Ayushk4/extract_covid_entity/releases/download/v0.0.1/cure.tar.gz) |

##### Sentence classification models

| Task | Link |
| ------ | ------ |
| Tested Positive | [sent_positive.tar.gz](https://github.com/Ayushk4/extract_covid_entity/releases/download/v0.0.1/sent_positive.tar.gz) |
| Tested Negative | [sent_negative.tar.gz](https://github.com/Ayushk4/extract_covid_entity/releases/download/v0.0.1/sent_negative.tar.gz) |
| Denied Testing | [sent_can_not_test.tar.gz](https://github.com/Ayushk4/extract_covid_entity/releases/download/v0.0.1/sent_can_not_test.tar.gz) |
| Death | [sent_death.tar.gz](https://github.com/Ayushk4/extract_covid_entity/releases/download/v0.0.1/sent_death.tar.gz) |
| Cure/Prevention | [sent_cure.tar.gz](https://github.com/Ayushk4/extract_covid_entity/releases/download/v0.0.1/sent_cure.tar.gz) |

##### Model performances on test set.

We stand first overall as well as on `Denied Testing`, `Death`, `Cure/Prevention` categories. To get the 


| Task | Micro-F1 | Micro-Precision | Micro-Recall |
| ------ | ------ | ------ | ------ |
| Tested Positive | 0.676 | 0.802 | 0.584 |
| Tested Negative | 0.663 | 0.659 | 0.667 |
| Denied Testing | 0.652 | 0.666 | 0.640 |
| Death | 0.694 | 0.724 | 0.667 |
| Cure/Prevention | 0.621 | 0.745 | 0.532 |
| **Overall** | **0.660** | **0.727** | **0.604** |


## Miscellanous

- You may contact us by opening an issue on this repo. Please allow 2-3 days of time to address the issue.

- For the slot-filling model, the starter code was obtained from [here](https://github.com/viczong/extract_COVID19_events_from_Twitter)

- License: MIT

**Update: Dec 2020**: The dataset is no longer public due to Twitter Privacy Policy. To get access to the dataset, please mail `zong.56@osu.edu` and cc `alan.ritter@cc.gatech.edu`.

