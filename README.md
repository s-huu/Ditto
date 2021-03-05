<img width="200" height="150" src="https://user-images.githubusercontent.com/14993256/109053987-54418f80-76ab-11eb-98bd-2c119d8a61ce.gif">

# Ditto: Fair and Robust Federated Learning Through Personalization

This repository contains the code and experiments for the manuscript:

> [Ditto: Fair and Robust Federated Learning Through Personalization](https://arxiv.org/abs/2012.04221)
>

Fairness and robustness are two important concerns for federated learning systems.
In this work, we identify that *robustness* to data and model poisoning attacks and *fairness*, measured as the uniformity of performance across devices, are competing constraints in statistically heterogeneous networks. 
To address these constraints, we propose employing a simple, general framework for personalized federated learning, Ditto, and develop a scalable solver for it. 
Theoretically, we  analyze the ability of Ditto to achieve
fairness and robustness simultaneously on a class of linear problems.
Empirically, across a suite of federated datasets, we show that Ditto not only achieves competitive performance relative to recent personalization methods, but also enables more accurate, robust, and fair models relative to state-of-the-art fair or robust baselines.



### *We also provide [Tensorflow implementation](https://github.com/litian96/ditto)*



## Preparation


### Downloading dependencies

```
pip3 install -r requirements.txt
``` 

## Run on federated benchmarks

(A subset of) Options in `models/run.sh`:

* `dataset` chosen from `[so]`, where so is short for StackOverflow.
* `aggregation` chosen from `['mean', 'median', 'krum']`.
* `attack` chosen from `['label_poison', 'random', 'model_replacement']`. 
* `num_mali_devices` is the number of malicious devices. 
* `personalized` indicates whether we want to train personalized models.
* `clipping` indicates whether we want to clip the model updates while training the global model.
* `k_aggregator` indicates whether we want to run k-loss/k-norm.

