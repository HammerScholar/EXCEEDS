import json


class Config:
    def __init__(self, args):
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.dataset = config["dataset"]
        self.ontology = config["ontology"]

        self.dist_emb_size = config["dist_emb_size"]
        self.lstm_hid_size = config["lstm_hid_size"]
        self.bert_hid_size = config["bert_hid_size"]
        self.dropout_rate = config["dropout_rate"]
        self.grid_channels = config["grid_channels"]
        self.grid_refine_layers = config["grid_refine_layers"]
        self.grid_refine_dropout = config["grid_refine_dropout"]
        self.grid_refine_kernel = config["grid_refine_kernel"]

        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]

        self.learning_rate = config["learning_rate"]
        self.weight_decay = config["weight_decay"]
        self.clip_grad_norm = config["clip_grad_norm"]
        self.bert_name = config["bert_name"]
        self.bert_learning_rate = config["bert_learning_rate"]
        self.warm_factor = config["warm_factor"]

        self.seed = config["seed"]

        for k, v in args.__dict__.items():
            if v is not None:
                self.__dict__[k] = v

    def __repr__(self):
        return "{}".format(self.__dict__.items())
