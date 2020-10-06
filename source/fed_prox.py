import logging
from source.federated_devices import ClientSequential, ServerSequential
from source.fed_process import FedProcess
from source.constants import ROOT_LOGGER_STR
logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)


class FedProx(FedProcess):

    def __init__(self, model_fn, num_clients):
        super(FedProx, self).__init__(model_fn, num_clients)
        self.clients = None

    def build(self, *args, **kwargs):
        self.clients = [self.model_fn(ClientSequential, 1)
                        for _ in range(self.num_clients)]
        self.server = self.model_fn(ServerSequential, 1)
