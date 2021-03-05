"""Script to run the baselines."""

import argparse
import copy
import importlib
import gc
import math
import pickle as pkl
import random
import os
import sys
import time
import json
from datetime import timedelta

import numpy as np
import pandas as pd
from PIL import Image
import torch

from baseline_constants import MAIN_PARAMS, MODEL_PARAMS, \
        SIM_TIMES, AVG_LOSS_KEY, ACCURACY_KEY, TRAINING_KEYS
from baseline_constants import OptimLoggingKeys
from baseline_constants import AGGR_MEAN, AGGR_MEDIAN, AGGR_KRUM, AGGR_MULTIKRUM
from baseline_constants import REGULARIZATION_PARAMS
from client import Client
from model import ServerModel
from server import Server
from utils.constants import DATASETS
from utils.model_utils import read_so_data
from metrics.writer import writer_print_metrics, writer_get_metrics_names

STAT_METRICS_PATH = 'metrics/stat_metrics.csv'
SYS_METRICS_PATH = 'metrics/sys_metrics.csv'
SUMMARY_METRICS_PATH = 'outputs/summary_metrics'
SUMMARY_METRICS_PATH_REG = 'outputs/reg/'

def reinit_clients():
        for c in train_clients:
            c.reinit_model()
        for c in test_clients:
            c.reinit_model()

def main():
    args = parse_args()
    global_start_time = start_time = time.time()

    # set torch seed
    torch_seed = (args.seed + 25) // 2
    torch.random.manual_seed(torch_seed)

    model_path = '%s/%s.py' % (args.dataset, args.model)
    if not os.path.exists(model_path):
        print('Please specify a valid dataset and a valid model.')
    model_path = '%s.%s' % (args.dataset, args.model)

    print('############################## %s ##############################' % model_path)
    mod = importlib.import_module(model_path)
    ClientModel = getattr(mod, 'ClientModel')

    tup = MAIN_PARAMS[args.dataset][args.t]
    num_rounds = args.num_rounds if args.num_rounds != -1 else tup[0]
    eval_every = args.eval_every if args.eval_every != -1 else tup[1]
    clients_per_round = args.clients_per_round if args.clients_per_round != -1 else tup[2]

    # Create 2 different models: one for server and one shared by all clients
    print('Obtained Model Path : ' + str(model_path))
    model_params = MODEL_PARAMS[model_path]
    if args.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = args.lr
        model_params = tuple(model_params_list)
    else:
        args.lr = model_params[0]
    if args.decay_lr_every is None:
        args.decay_lr_every = 50
    print(args)
    client_model = ClientModel(*model_params, seed=args.seed)
    server_model = ServerModel(ClientModel(*model_params, seed=args.seed - 1))


    # Create server
    server = Server(server_model)

    # Create clients
    clients = setup_clients(args.dataset, model_name=args.model,
                            model=client_model, validation=args.validation, 
                            seed=args.seed)
    train_clients = clients['train_clients']
    test_clients = clients['test_clients']
    validation_clients = []

    

    print('#Clients = %d ; setup time = %s' % (len(train_clients) + len(test_clients),
                                               timedelta(seconds=round(time.time() - start_time))))
    gc.collect()  # free discarded memory in setup_clients


    # Logging utilities
    train_ids, train_groups, train_clients_num_train_samples, train_clients_num_test_samples = server.get_clients_info(train_clients)
    test_ids, test_groups, test_clients_num_train_samples, test_clients_num_test_samples = server.get_clients_info(test_clients)
    if args.validation:
        validation_ids, validation_groups, validation_clients_num_train_samples, validation_clients_num_test_samples = server.get_clients_info(
        validation_clients)


    summary = None
    
    def log_helper(iteration, path_for_validation, comm_rounds=None, malicious_devices=[], personalized_models=dict(),norm_diff=[],test_accs=[]):

        if comm_rounds is None:
            comm_rounds = iteration
        nonlocal summary
        start_time = time.time()
        if args.no_logging:
            stat_metrics = None
        else:
            train_clients_stat_metrics = server.test_model(clients_to_test=train_clients, train_and_test=False,
                                                            train_users=True,
                                                            malicious_devices=malicious_devices,
                                                            personalized=args.personalized,
                                                            personalized_models=personalized_models)
            test_clients_stat_metrics = server.test_model(clients_to_test=test_clients, train_and_test=False,
                                                            train_users=False,
                                                            malicious_devices=malicious_devices,
                                                            personalized=args.personalized,
                                                            personalized_models=personalized_models)


        summary_iter_train, str_output_train, norm_diff, _ = print_metrics(iteration, comm_rounds, 
                                            train_clients_stat_metrics,
                                            train_clients_num_train_samples, train_clients_num_test_samples,
                                            time.time() - start_time,
                                            train_users=True, full_record=args.full_record,
                                            malicious_devices=malicious_devices,
                                            norm_differences=norm_diff,
                                            test_accs=test_accs)
        summary_iter_test, str_output_test,_,test_accs = print_metrics(iteration, comm_rounds, test_clients_stat_metrics,
                                    test_clients_num_train_samples, test_clients_num_test_samples,
                                    time.time() - start_time,
                                            train_users=False, full_record=args.full_record,
                                            malicious_devices=malicious_devices,
                                            test_accs=test_accs)
        print(str_output_train + str_output_test)

        if iteration == 0:

            summary = pd.Series(summary_iter_train).to_frame().T   
            summary_iter = {**summary_iter_train, **summary_iter_test}
            summary = summary.append(summary_iter, ignore_index=True)
        else:
                
            summary_iter = {**summary_iter_train, **summary_iter_test}
            summary = summary.append(summary_iter, ignore_index=True)

        return summary_iter_train, norm_diff,test_accs


    # Simulate training
    def main_training(path_for_validation, regularization_param=None):
        # Required for initial logging to be correct when initializing from non-zero weights

        # Initialize diagnostics
        s,norm_diff,test_accs = log_helper(0, path_for_validation)
        initial_loss = s.get(OptimLoggingKeys.TRAIN_LOSS_KEY[0], None)
        initial_avg_loss = None
        num_no_progress = 0
        client_ids = []
        possible_clients = online(train_clients)
        for c in possible_clients:
            client_ids.append(c.id)
        malicious_devices = random.sample(client_ids,args.num_mali_devices)
        personalized_models = dict()
        benign_ratio = 1-args.num_mali_devices / len(client_ids)

        for i in range(num_rounds):
            local_finetune = (i >= args.start_finetune_rounds)
            #             print(malicious_devices) 
            print('--- Round %d of %d: Training %d Clients ---' % (i + 1, num_rounds, clients_per_round))
            sys.stdout.flush()
            start_time = time.time()
            # Select clients to train this round
            s_c = server.select_clients(possible_clients, num_clients=clients_per_round)
            c_ids, c_groups, c_num_train_samples, c_num_test_samples = server.get_clients_info()




            # Simulate server model training on selected clients' data
            sys_metrics, avg_loss, losses, personalized_models = server.train_model(num_epochs=args.num_epochs,
                                                               batch_size=args.batch_size,
                                                               minibatch=args.minibatch,
                                                               lr=args.lr,
                                                               malicious_devices=malicious_devices,
                                                               personalized=args.personalized,
                                                               personalized_models=personalized_models,
                                                               q=args.q,
                                                               qffl=args.qffl,
                                                               local_finetune=local_finetune,
                                                               attack=args.attack,
                                                               lmbda=args.lmbda)
            updates = server.updates
            writer_print_metrics(i, c_ids, sys_metrics, c_groups, c_num_test_samples, args.output_sys_file)

            if initial_avg_loss is None:
                initial_avg_loss = avg_loss
            total_num_comm_rounds=0
            # Update server model
            if not local_finetune:
                total_num_comm_rounds, is_updated = server.update_model(aggregation=args.aggregation,
                                                                        losses=losses,
                                                                        clipping=args.clipping,
                                                                        k_aggregator=args.k_aggregator,
                                                                        k_aggr_ratio=int(benign_ratio*len(server.updates)))


                if is_updated:
                    num_no_progress = 0
                else:
                    num_no_progress += 1
                if num_no_progress > args.patience_iter:
                    print('No progress made in {} iterations. Quitting.'.format(num_no_progress))
                    sys.exit(1)

                norm = np.linalg.norm(server_model.model.optimizer.w)
                print('\t\t\tRound: {} AvgLoss: {:.3f} Norm: {:.2f} Time: {} Tot_time {}'.format(
                    i + 1, avg_loss, norm,
                    timedelta(seconds=round(time.time() - start_time)),
                    timedelta(seconds=round(time.time() - global_start_time))
                ), flush=True)

            # Test model on all clients
            if (
                    (i + 1) in [5, 10, 15, 20, 75]
                    or (i + 1) % eval_every == 0
                    or (i + 1) == num_rounds
            ):
                s,norm_diff,test_accs = log_helper(i + 1, path_for_validation, total_num_comm_rounds, malicious_devices=malicious_devices, personalized_models=personalized_models,norm_diff=norm_diff)

                if OptimLoggingKeys.TRAIN_LOSS_KEY in s:
                    if initial_loss is not None and s[OptimLoggingKeys.TRAIN_LOSS_KEY][0] > 3 * initial_loss:
                        print('Loss > 3 * initial_loss. Exiting')
                        break
                    if math.isnan(s[OptimLoggingKeys.TRAIN_LOSS_KEY][0]):
                        print('Loss NaN. Exiting')
                        break
                if args.no_logging:
                    # save model
                    save_model(server_model, args.dataset, args.model,
                               '{}_iteration{}'.format(args.output_summary_file + ".csv", i + 1))
                if args.save_updates:
                    fn = '{}_updates_{}'.format(args.output_summary_file + '.csv', i+1)
                    selected_client_ids = [c.id for c in server.selected_clients]
                    with open(fn, 'wb') as f:
                        pkl.dump([[], selected_client_ids, updates], f)

            if (i + 1) % args.decay_lr_every == 0:
                args.lr /= args.lr_decay

        s,norm_diff,test_accs = log_helper(num_rounds, path_for_validation, total_num_comm_rounds, malicious_devices=malicious_devices, personalized_models=personalized_models,norm_diff=norm_diff)
        # Save server model
        if args.validation:
            summary.to_csv(path_for_validation, mode='w', header=True, index=False)
        else:
            summary.to_csv(path_for_validation, mode='w', header=True, index=False)


    main_training(args.output_summary_file + '.csv', regularization_param=args.reg_param)

    print('Job complete. Total time taken:', timedelta(seconds=round(time.time() - global_start_time)))



def online(clients):
    """We assume all users are always online."""
    return clients


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset',
                        help='name of dataset;',
                        type=str,
                        choices=DATASETS,
                        required=True)
    parser.add_argument('-model',
                        help='name of model;',
                        type=str,
                        required=True)
    parser.add_argument('--full_record',
                        help='name of model;',
                        type=bool,
                        default=False)
    parser.add_argument('--num-rounds',
                        help='number of rounds to simulate;',
                        type=int,
                        default=-1)
    parser.add_argument('--eval-every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=-1)
    parser.add_argument('--clients-per-round',
                        help='number of clients trained per round;',
                        type=int,
                        default=-1)
    parser.add_argument('--seed',
                        help='random seed for reproducibility;',
                        type=int)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=32)
    parser.add_argument('--minibatch',
                        help='None for FedAvg, else fraction;',
                        type=float,
                        default=None)
    parser.add_argument('--num_epochs',
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=1)
    parser.add_argument('-t',
                        help='simulation time: small, medium, or large;',
                        type=str,
                        choices=SIM_TIMES,
                        default='large')
    parser.add_argument('-lr',
                        help='learning rate for local optimizers;',
                        type=float,
                        default=1.0,
                        required=False)
    parser.add_argument('--lr-decay',
                        help='decay in learning rate',
                        type=float,
                        default=1.0)
    parser.add_argument('--decay-lr-every',
                        help='number of iterations to decay learning rate',
                        type=int,
                        default=200)
    parser.add_argument('-reg_param',
                        help='regularization learning parameter',
                        type=float,
                        default=None)
    parser.add_argument('--output_stat_file',
                        help='Filename to log stat metrics in CSV',
                        default=STAT_METRICS_PATH)
    parser.add_argument('--output_sys_file',
                        help='Filename to log system metrics in CSV',
                        default=SYS_METRICS_PATH)
    parser.add_argument('--output_summary_file',
                        help='Filename to log summary of optimization performance in CSV',
                        default=SUMMARY_METRICS_PATH)
    parser.add_argument('--validation',
                        help='If specified, hold out part of training data to use as a dev set for parameter search',
                        type=bool,
                        default=False)
    parser.add_argument('--patience-iter',
                        help='Number of patience rounds of no updates to wait for before giving up',
                        type=int,
                        default=20)
    parser.add_argument('--aggregation',
                        help='Aggregation technique used to combine updates or gradients',
                        choices=[AGGR_MEAN,AGGR_MEDIAN,AGGR_KRUM,AGGR_MULTIKRUM],
                        default=AGGR_MEAN)
    parser.add_argument('--attack',
                        help='the attack we are using',
                        type=str,
                        default="label_poison")
    parser.add_argument('--lmbda',
                        help='the lambda we are using',
                        default=0.05)
    parser.add_argument('--no-logging',
                        help='if specified, do not perform testing. Instead save model to disk.',
                        action='store_true')
    parser.add_argument('--save-updates',
                        help='if specified, save updates.',
                        action='store_true')
    parser.add_argument('--gpu',
                        help='Which gpu to use. Unspecified (=None) means CPU',
                        type=int)
    parser.add_argument('--num_mali_devices',
                        help='number of malicious devices',
                        type=int,
                        default=0)
    parser.add_argument('--personalized',
                        help='run with personalization',
                        action='store_true')
    parser.add_argument('--clipping',
                        help='run with update clipping',
                        action='store_true')
    parser.add_argument('--q',
                        help='qffl parameter',
                        type=float,
                        default=0.0)
    parser.add_argument('--qffl',
                        help='run with qffl',
                        action='store_true')
    parser.add_argument('--k_aggregator',
                        help='use k aggregator',
                        action='store_true')
    parser.add_argument('--start_finetune_rounds',
                        help='start of round of local finetuning',
                        type=int,
                        default=1000)
    

    args = parser.parse_args()
    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 2)
        print('Random seed not provided. Using {} as seed'.format(args.seed))
    return args


def setup_clients(dataset, model_name=None, model=None, validation=False, seed=-1, subsample_fraction=0.5):
    """Instantiates clients based on given train and test data directories.
        If validation is True, use part of training set as validation set
    Return:
        all_clients: list of Client objects.
    """
    
    clients, groups, train_data, test_data = read_so_data()
    
    if seed != -1:
        np.random.seed(seed)
    else:
        np.random.seed(42)

    train_users = clients['train_users']
    test_users = clients['test_users']


    train_groups = [[] for _ in train_users]
    test_groups = [[] for _ in test_users]

    

    print('------>', len(train_users))

    train_clients = []
    test_clients = []

    

    for u, g in zip(train_users, train_groups):
        train_data_u_x = (train_data[u]['x'])
        train_data_u_y = (train_data[u]['y'])
        train_data_u = {'x': train_data_u_x, 'y': train_data_u_y}
        train_clients.append(Client(u, g, train_data=train_data_u, model=model, dataset=dataset))

    for u, g in zip(test_users, test_groups):
        test_data_u_x = (test_data[u]['x'])
        test_data_u_y = (test_data[u]['y'])
        test_data_u = {'x': test_data_u_x, 'y': test_data_u_y}
        test_clients.append(Client(u, g,  eval_data=test_data_u, model=model, dataset=dataset))

    all_clients = {
        'train_clients': train_clients,
        'test_clients': test_clients
    }

    if validation:
        validation_clients = []
        for u, g in zip(validation_users, validation_groups):
            validation_data_u_x = (
                preprocess_data_x(validation_data[u]['x'],
                                    dataset=dataset,
                                    model_name=model_name)
                if do_preprocess else validation_data[u]['x']
            )
            validation_data_u_y = (
                preprocess_data_y(validation_data[u]['y'],
                                    dataset=dataset,
                                    model_name=model_name)
                if do_preprocess else validation_data[u]['y']
            )
            validation_data_u = {'x': validation_data_u_x, 'y': validation_data_u_y}
            validation_clients.append(Client(u, g, train_data=validation_data_u, model=model, dataset=dataset))

        all_clients['validation_clients'] = validation_clients

    return all_clients 


def save_model(server_model, dataset, model, output_summary_file):
    """Saves the given server model on checkpoints/dataset/model.ckpt."""
    # Save server model
    start_time = time.time()
    ckpt_path = os.path.join('checkpoints', *(output_summary_file.split(os.path.sep)[1:]))
    print(ckpt_path)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    save_path = server_model.save('%s.ckpt' % ckpt_path)
    print('Model saved in path: {} in time {:.2f} sec'.format(save_path, time.time() - start_time))


def print_metrics(iteration, comm_rounds, metrics, train_weights, test_weights, elapsed_time=0.0,
                  train_users=True, validation=False, full_record=False, malicious_devices=None, norm_differences=[],
                  test_accs=[]):
    """Prints weighted averages of the given metrics.

    Args:
        iteration: current iteration number
        comm_rounds: number of communication rounds
        metrics: dict with client ids as keys. Each entry is a dict
            with the metrics of that client.
        train_weights: dict with client ids as keys. Each entry is the weight
            for that client for training metrics.
        test_weights: dict with client ids as keys. Each entry is the weight
            for that client for testing metrics
        elapsed_time: time taken for testing
    """
    output = {'iteration': iteration}
    if metrics is None:
        print(iteration, comm_rounds)
    else:
        str_output = str(iteration) + ', '
        if train_users:
            ordered_tr_weights = [train_weights[c] for c in sorted(train_weights)]
            ordered_tr_weights_benign = [train_weights[c] for c in sorted(train_weights) if c not in malicious_devices]
            if full_record:  # do not use together with validation
                output['train_clients'] = {}
                for key, value in metrics.items():

                    output['train_clients'][key] = {
                        'loss': value['train_loss'],
                        'accuracy': value['train_accuracy']
                    }
                ordered_metric = [metrics[c]['train_loss'] for c in sorted(metrics)]
                avg_metric = np.average(ordered_metric, weights=ordered_tr_weights)
                str_output += 'train_loss : ' + str(round(avg_metric, 3)) + ' '
                
                ordered_metric = [metrics[c]['train_accuracy'] for c in sorted(metrics)]
                avg_metric = np.average(ordered_metric, weights=ordered_tr_weights)
                str_output += 'train_accuracy : ' + str(round(avg_metric, 3)) + ' '
                
                ordered_metric = [metrics[c]['train_accuracy'] for c in sorted(metrics) if c not in malicious_devices]
                avg_metric = np.average(ordered_metric, weights=ordered_tr_weights_benign)
                str_output += 'train_accuracy_benign : ' + str(round(avg_metric, 3)) + ' '
                
                str_output += "train accuracy distribution: " + str(np.array(ordered_metric).std()) + ' '
                
                ordered_metric = [metrics[c]['norm_of_difference'] for c in sorted(metrics) if c not in malicious_devices]
                avg_metric = np.average(ordered_metric[:5])
                norm_differences.append(avg_metric)
                str_output += 'average_norm_difference: ' + str(round(avg_metric, 3)) + ' '
                
                ordered_metric = [metrics[c]['train_loss'] for c in sorted(metrics) if c not in malicious_devices]
                print('train loss distribution for benign devices: ', ordered_metric)
                
                ordered_metric = [metrics[c]['train_loss'] for c in sorted(metrics) if c in malicious_devices]
                print('train loss distribution for malicious devices: ', ordered_metric)

            else:

                metric_names = writer_get_metrics_names(metrics)
                for metric in metric_names:
                    if metric == ACCURACY_KEY:  # Do not print
                        continue
                    if metric == "test_loss":
                        break
                    ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
                    avg_metric = np.average(ordered_metric, weights=ordered_tr_weights)

                    if validation:
                        status = 'validation_'
                    else:
                        status = ''

                    output[status + metric] = [avg_metric]

                    if metric == 'train_loss':
                        for i in [60, 70, 80, 90]:
                            output[status + metric].append(np.percentile(ordered_metric, i))
                        output[status + metric].append(np.max(ordered_metric))

                    if metric == 'train_accuracy':
                        for i in [10, 20, 30, 40]:
                            output[status + metric].append(np.percentile(ordered_metric, i))
                        output[status + metric].append(np.min(ordered_metric))

                    str_output += metric + ': ' + str(round(avg_metric, 3)) + ' '

        else:
            # split by user, test
            ordered_te_weights = [test_weights[c] for c in sorted(test_weights)]
            ordered_te_weights_benign = [test_weights[c] for c in sorted(test_weights) if c not in malicious_devices]
            if full_record:  # do not use together with validation
                output['test_clients'] = {}
                for key, value in metrics.items():

                    output['test_clients'][key] = {
                        'loss': value['test_loss'],
                        'accuracy': value['test_accuracy']
                    }

                ordered_metric = [metrics[c]['test_loss'] for c in sorted(metrics)]
                avg_metric = np.average(ordered_metric, weights=ordered_te_weights)
                str_output += 'test_loss : ' + str(round(avg_metric, 3)) + ' '

                ordered_metric = [metrics[c]['test_accuracy'] for c in sorted(metrics)]
                avg_metric = np.average(ordered_metric, weights=ordered_te_weights)
                str_output += 'test_accuracy : ' + str(round(avg_metric, 3)) + ' '
                
                ordered_metric = [metrics[c]['test_accuracy'] for c in sorted(metrics) if c not in malicious_devices]
                avg_metric = np.average(ordered_metric, weights=ordered_te_weights_benign)
                str_output += 'test_accuracy_benign : ' + str(round(avg_metric, 3)) + ' '
                test_accs.append(avg_metric)
                
                str_output += "test accuracy distribution: " + str(np.array(ordered_metric).std())

            else:
                metric_names = writer_get_metrics_names(metrics)
                for metric in metric_names:
                    if metric == ACCURACY_KEY or metric == 'train_loss' or metric == 'train_accuracy':  # Do not print
                        continue
                    ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
                    avg_metric = np.average(ordered_metric, weights=ordered_te_weights)
                    output[metric] = [avg_metric]
                    if metric == 'test_loss':
                        for i in [60, 70, 80, 90]:
                            output[metric].append(np.percentile(ordered_metric, i))
                        output[metric].append(np.max(ordered_metric))

                    if metric == 'test_accuracy':
                        for i in [10, 20, 30, 40]:
                            output[metric].append(np.percentile(ordered_metric, i))
                        output[metric].append(np.min(ordered_metric))

                    str_output += metric + ': ' + str(round(avg_metric, 3)) + ' '

        # sys.stdout.flush()
        return output, str_output, norm_differences, test_accs
        
    return output, norm_of_differences, test_accs

if __name__ == '__main__':
    main()
