import warnings
import numpy as np
import copy


class Client:

    def __init__(self, client_id, group=None, train_data={'x': [], 'y': []}, eval_data={'x': [], 'y': []}, model=None, dataset='so'):
        self._model = model
        self.id = client_id  # integer
        self.group = group
        self.train_data = train_data
        self.eval_data = eval_data
        self.rng = model.rng  # use random number generator of the model


    def train(self, num_epochs=1, batch_size=10, minibatch=None, lr=None, malicious=False, q=0, qffl=False, attack="label_poison", lmbda=0.05):
        
        if minibatch is None:
            data = self.train_data
            comp, update, averaged_loss = self.model.train(data, num_epochs, batch_size, lr, malicious, q, qffl, attack, lmbda)
        else:
            frac = min(1.0, minibatch)
            num_data = max(1, int(frac * len(self.train_data['x'])))
            xs, ys = zip(*self.rng.sample(list(zip(self.train_data['x'], self.train_data['y'])), num_data))
            data = {'x': xs, 'y': ys}
            comp, update, averaged_loss = self.model.train(data, num_epochs, num_data, lr, malicious, q, qffl, attack, lmbda)
        num_train_samples = len(data['y'])
        return comp, num_train_samples, averaged_loss, update

    def test(self, model, train_and_test, train_users=True, malicious=False):
        
        if train_users:
            return model.test(self.train_data, train_users=train_users, malicious=malicious)
        else:
            return model.test(self.eval_data, train_users=train_users,malicious=malicious)
       

    def reinit_model(self):
        self._model.optimizer.initialize_w()

    @property
    def num_train_samples(self):
        try:
            return len(self.train_data['y'])
        except:
            return 0

    @property
    def num_test_samples(self):
        try:
        # print(self.train_data)
            return len(self.eval_data['y'])
        except:
            return 0

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
