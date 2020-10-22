import logging
from source.federated_devices import (ClientVirtualSequential,
                                      ClientVirtualModel,
                                      ServerSequential,
                                      ServerModel)
from source.fed_process import FedProcess
from source.constants import ROOT_LOGGER_STR


logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)


class VirtualFedProcess(FedProcess):

    def __init__(self, model_fn, num_clients, damping_factor=1,
                 fed_avg_init=False):
        super(VirtualFedProcess, self).__init__(model_fn, num_clients)
        self.damping_factor = damping_factor
        self.fed_avg_init = fed_avg_init

    def build(self, cards_train, hierarchical):
        if hierarchical:
            client_model_class = ClientVirtualModel
            server_model_class = ServerModel
        else:
            client_model_class = ClientVirtualSequential
            server_model_class = ServerSequential
        for indx in self.clients_indx:
            model = self.model_fn(client_model_class, cards_train[indx], cards_train[indx]/sum(cards_train))
            self.clients.append(model)
        self.server = self.model_fn(server_model_class, sum(cards_train), 1.)