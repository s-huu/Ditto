"""Writes the given metrics in a csv."""

import os
import sys

import numpy as np
import pandas as pd

models_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(models_dir)

from baseline_constants import CLIENT_ID_KEY, NUM_ROUND_KEY, NUM_SAMPLES_KEY

COLUMN_NAMES = [
    CLIENT_ID_KEY, NUM_ROUND_KEY, 'hierarchy', NUM_SAMPLES_KEY]

def writer_print_metrics(
        round_number,
        client_ids,
        metrics,
        hierarchies,
        num_samples,
        path):
    """Prints or appends the given metrics in a csv.

    The resulting dataframe is of the form:
        client_id, round_number, hierarchy, num_samples, metric1, metric2
        twebbstack, 0, , 18, 0.5, 0.89

    Args:
        round_number: Number of the round the metrics correspond to. If
            0, then the file in path is overwritten. If not 0, we append to
            that file.
        client_ids: Ids of the clients. Not all ids must be in the following
            dicts.
        metrics: Dict keyed by client id. Each element is a dict of metrics
            for that client in the specified round. The dicts for all clients
            are expected to have the same set of keys.
        hierarchies: Dict keyed by client id. Each element is a list of hierarchies
            to which the client belongs.
        num_samples: Dict keyed by client id. Each element is the number of test
            samples for the client.
        path: Full path of output CSV file.
    """
    columns = COLUMN_NAMES + writer_get_metrics_names(metrics)
    client_data = pd.DataFrame(columns=columns)
    for i, c_id in enumerate(client_ids):
        current_client = {
            'client_id': c_id,
            'round_number': round_number,
            'hierarchy': ','.join(hierarchies.get(c_id, [])),
            'num_samples': num_samples.get(c_id, np.nan)
        }

        current_metrics = metrics.get(c_id, {})
        # TODO: fix training metrics
        for metric, metric_value in current_metrics.items():
            current_client[metric] = metric_value
        client_data.loc[len(client_data)] = current_client

    mode = 'w' if round_number == 0 else 'a'
    writer_print_dataframe(client_data, path, mode)


def writer_print_dataframe(df, path, mode='w'):
    """Writes the given dataframe in path as a csv"""
    header = mode == 'w'
    df.to_csv(path, mode=mode, header=header, index=False)

def writer_get_metrics_names(metrics):
    """Gets the names of the metrics.

    Args:
        metrics: Dict keyed by client id. Each element is a dict of metrics
            for that client in the specified round. The dicts for all clients
            are expected to have the same set of keys."""
    if len(metrics) == 0:
        return []
    metrics_dict = next(iter(metrics.values()))
    return list(metrics_dict.keys())
