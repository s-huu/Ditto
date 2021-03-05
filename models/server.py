import numpy as np
import random
import copy

from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY, AVG_LOSS_KEY
from baseline_constants import MAX_UPDATE_NORM
from utils.torch_utils import numpy_to_torch, torch_to_numpy


class Server:

    def __init__(self, model):
        self.model = model  # global model of the server.
        self.selected_clients = []
        self.updates = []
        self.rng = model.rng  # use random number generator of the model
        self.total_num_comm_rounds = 0
        self.eta = None

    def select_clients(self, possible_clients, num_clients=20):
        """Selects num_clients clients randomly from possible_clients.
        
        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).

        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        Return:
            list of (num_train_samples, num_test_samples)
        """
        num_clients = min(num_clients, len(possible_clients))
        self.selected_clients = self.rng.sample(possible_clients, num_clients)
        return [(len(c.train_data['y']), len(c.eval_data['y'])) for c in self.selected_clients]

    def train_model(self, 
                    num_epochs=1, 
                    batch_size=10, 
                    minibatch=None, 
                    clients=None, 
                    lr=None, 
                    malicious_devices=None, 
                    personalized=False,
                    personalized_models=None,
                    q=0,
                    qffl=False,
                    local_finetune=False,
                    attack="label_poison",
                    lmbda=0.05):

        """Trains self.model on given clients.
        
        Trains model on self.selected_clients if clients=None;
        each client's data is trained with the given number of epochs
        and batches.

        Args:
            clients: list of Client objects.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
            lr: learning rate to use
        Return:
            bytes_written: number of bytes written by each client to server 
                dictionary with client ids as keys and integer values.
            client computations: number of FLOPs computed by each client
                dictionary with client ids as keys and integer values.
            bytes_read: number of bytes read by each client from server
                dictionary with client ids as keys and integer values.
        """

        if clients is None:
            clients = self.selected_clients
        sys_metrics = {
            c.id: {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   LOCAL_COMPUTATIONS_KEY: 0} for c in clients}
        losses = []

        chosen_clients = clients

        for c in chosen_clients:
            if not personalized:
                self.model.send_to([c])  # reset client model
            elif c.id in personalized_models:
                c._model.optimizer.reset_w(personalized_models[c.id][1])
                c._model.size = personalized_models[c.id][0]
            if personalized and not local_finetune:
                c._model.optimizer.global_model = copy.deepcopy(self.model.model.optimizer.model)
                c._model.optimizer.reset_global_w(torch_to_numpy(self.model.model.optimizer.model.trainable_parameters()))
            c._model.optimizer.personalized = personalized and not local_finetune
            sys_metrics[c.id][BYTES_READ_KEY] += self.model.size
            if lmbda is not None:
                c._model.optimizer.lmbda = lmbda
            if lr is not None:
                c._model.optimizer.learning_rate = lr
            comp, num_samples, averaged_loss, update = c.train(num_epochs, batch_size, minibatch, lr, (c.id in malicious_devices), q, qffl, attack,lmbda)
            sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp
            losses.append(averaged_loss)

            if qffl:
                self.updates.append((np.exp(q*averaged_loss), update))
            else:
                self.updates.append((num_samples, update))
            sys_metrics[c.id][BYTES_WRITTEN_KEY] += self.model.size
            if personalized:
                personalized_models[c.id] = (c._model.optimizer.size(),np.copy(c._model.optimizer.w))

        avg_loss = np.nan if len(losses) == 0 else \
            np.average(losses, weights=[len(c.train_data['y']) for c in chosen_clients])
        return sys_metrics, avg_loss, losses, personalized_models
        
    def update_model(self, 
                     aggregation, 
                     losses=None,
                     clipping=False,
                     qffl=False,
                     k_aggregator=False,
                     k_aggr_ratio=1):

        num_comm_rounds, is_updated = self.model.update(self.updates, aggregation,clipping=clipping,k_aggregator=k_aggregator,k_aggr_ratio=k_aggr_ratio,losses=losses)
        self.total_num_comm_rounds += num_comm_rounds
        self.updates = []
        return self.total_num_comm_rounds, is_updated

    def test_model(self, 
                   clients_to_test=None, 
                   train_and_test=False, 
                   train_users=True,
                   malicious_devices=None,
                   personalized=False,
                   personalized_models=None):
        """Tests self.model on given clients.

        Tests model on self.selected_clients if clients_to_test=None.

        Args:
            clients_to_test: list of Client objects.
            train_and_test: If True, also measure metrics on training data
        """
        
        if clients_to_test is None:
            clients_to_test = self.selected_clients
        metrics = {}
        
        if not personalized:
            self.model.send_to(clients_to_test)

            for client in clients_to_test:
                c_metrics = client.test(self.model.cur_model, train_and_test, train_users=train_users, malicious=(client.id in malicious_devices))
                metrics[client.id] = c_metrics
                
        else:
            for client in clients_to_test:
                
                if client.id in personalized_models:
                    client._model.optimizer.reset_w(personalized_models[client.id][1])
                    client._model.size = personalized_models[client.id][0]
                    c_metrics = client.test(client.model, train_and_test, train_users=train_users, malicious=(client.id in malicious_devices))
                else:
                    self.model.send_to([client])
                    client._model.optimizer.global_model = self.model.model.optimizer.model
                    c_metrics = client.test(self.model.cur_model, train_and_test, train_users=train_users, malicious=(client.id in malicious_devices))
                metrics[client.id] = c_metrics

        return metrics

    def get_clients_info(self, clients=None):
        """Returns the ids, hierarchies, num_train_samples and num_test_samples for the given clients.

        Returns info about self.selected_clients if clients=None;

        Args:
            clients: list of Client objects.
        """
        if clients is None:
            clients = self.selected_clients
        ids = [c.id for c in clients]
        groups = {c.id: c.group for c in clients}
        num_train_samples = {c.id: c.num_train_samples for c in clients}
        num_test_samples = {c.id: c.num_test_samples for c in clients}

        return ids, groups, num_train_samples, num_test_samples

    def eval_losses_on_train_clients(self, clients=None):
        losses = []

        if clients is None:
            clients = self.selected_clients

        self.model.send_to(clients)

        for c in clients:
            c_dict = c.test(self.model.cur_model, False, train_users=True)
            loss = c_dict['train_loss']
            losses.append(loss)

        return losses

    def clients_weights(self, clients=None):
        if clients is None:
            clients = self.selected_clients
        res = []
        for c in clients:
            res.append(len(c.train_data['y']))
        return res