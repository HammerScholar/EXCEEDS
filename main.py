import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import random
import numpy as np
import prettytable as pt
import torch
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
import data_loader
import utils
from model import EXCEEDS


def multilabel_categorical_crossentropy(y_pred, y_true):
    """
    This multi-label loss is referenced from:
    https://kexue.fm/archives/7359
    """
    y_true = y_true.float()
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()



class Trainer(object):
    def __init__(self, model, vocab) -> None:
        self.model = model
        self.vocab = vocab
        
        bert_params = set(self.model.bert.parameters())
        other_params = list(set(self.model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': config.learning_rate,
             'weight_decay': config.weight_decay},
        ]

        self.optimizer = transformers.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                      num_warmup_steps=config.warm_factor * updates_total,
                                                                      num_training_steps=updates_total)

    def train(self, epoch, data_loader):
        self.model.train()
        loss_list = []
        
        for data_batch in tqdm(data_loader):
            data_batch = [data.cuda() for data in data_batch[:-1]]  # last one is golden events
            bert_inputs, pieces2word, dist_inputs, grid_labels, grid_mask2d, document_length = data_batch
            
            outputs = self.model(bert_inputs, pieces2word, dist_inputs, document_length)
            
            grid_mask2d = grid_mask2d.clone()

            # Skip NONE channel in loss
            loss = multilabel_categorical_crossentropy(
                outputs[grid_mask2d][..., 1:],
                grid_labels[grid_mask2d][..., 1:],
            )
                
            loss_list.append(loss.cpu().item())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.clip_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
        
        logger.info('Train - Epoch {} - Loss: {:4f}\n'.format(epoch, np.mean(loss_list)))
            
    
    def eval(self, epoch, data_loader):
        self.model.eval()
        
        pred_res = []
        gold_res = []
        
        with torch.no_grad():
            for data_batch in tqdm(data_loader):
                golden_events = data_batch[-1]
                data_batch = [data.cuda() for data in data_batch[:-1]]
                bert_inputs, pieces2word, dist_inputs, grid_labels, grid_mask2d, document_length = data_batch
                
                outputs = self.model(bert_inputs, pieces2word, dist_inputs, document_length)

                pred = torch.zeros_like(outputs, dtype=torch.bool)
                pred[..., 1:] = outputs[..., 1:] > 0
                predict_events = utils.decode(pred, self.vocab)

                gold_res.extend(golden_events)
                pred_res.extend(predict_events)
        
        trg_I_F1, trg_C_F1, arg_I_F1, arg_C_F1, ec_I_F1, ec_C_F1, \
        trg_I_p, trg_C_p, arg_I_p, arg_C_p, ec_I_p, ec_C_p, \
        trg_I_r, trg_C_r, arg_I_r, arg_C_r, ec_I_r, ec_C_r = utils.calculate_F1(gold_res, pred_res)
        
        logger.info('Dev - Epoch {}\n'.format(epoch))
        table = pt.PrettyTable(['Metrics', 'Precision', 'Recall', 'F1'])
        table.add_row(['trg_I', "{:3.4f}".format(trg_I_p), "{:3.4f}".format(trg_I_r), "{:3.4f}".format(trg_I_F1)])
        table.add_row(['trg_C', "{:3.4f}".format(trg_C_p), "{:3.4f}".format(trg_C_r), "{:3.4f}".format(trg_C_F1)])
        table.add_row(['arg_I', "{:3.4f}".format(arg_I_p), "{:3.4f}".format(arg_I_r), "{:3.4f}".format(arg_I_F1)])
        table.add_row(['arg_C', "{:3.4f}".format(arg_C_p), "{:3.4f}".format(arg_C_r), "{:3.4f}".format(arg_C_F1)])
        table.add_row(['ec_I', "{:3.4f}".format(ec_I_p), "{:3.4f}".format(ec_I_r), "{:3.4f}".format(ec_I_F1)])
        table.add_row(['ec_C', "{:3.4f}".format(ec_C_p), "{:3.4f}".format(ec_C_r), "{:3.4f}".format(ec_C_F1)])
        logger.info('\n' + str(table))
        
        return trg_C_F1, arg_C_F1
        
    
    def predict(self, data_loader, epoch=None):
        self.model.eval()

        pred_res = []
        gold_res = []
        output_res = []
        with torch.no_grad():
            for data_batch in tqdm(data_loader):
                golden_events = data_batch[-1]
                data_batch = [data.cuda() for data in data_batch[:-1]]
                bert_inputs, pieces2word, dist_inputs, grid_labels, grid_mask2d, document_length = data_batch

                outputs = self.model(bert_inputs, pieces2word, dist_inputs, document_length)

                pred = torch.zeros_like(outputs, dtype=torch.bool)
                pred[..., 1:] = outputs[..., 1:] > 0
                predict_events = utils.decode(pred, self.vocab)

                gold_res.extend(golden_events)
                pred_res.extend(predict_events)
                output_res.extend(predict_events)
                
        trg_I_F1, trg_C_F1, arg_I_F1, arg_C_F1, ec_I_F1, ec_C_F1, \
        trg_I_p, trg_C_p, arg_I_p, arg_C_p, ec_I_p, ec_C_p, \
        trg_I_r, trg_C_r, arg_I_r, arg_C_r, ec_I_r, ec_C_r = utils.calculate_F1(gold_res, pred_res)
        
        logger.info('Test\n')
        table = pt.PrettyTable(['Metrics', 'Precision', 'Recall', 'F1'])
        table.add_row(['trg_I', "{:3.4f}".format(trg_I_p), "{:3.4f}".format(trg_I_r), "{:3.4f}".format(trg_I_F1)])
        table.add_row(['trg_C', "{:3.4f}".format(trg_C_p), "{:3.4f}".format(trg_C_r), "{:3.4f}".format(trg_C_F1)])
        table.add_row(['arg_I', "{:3.4f}".format(arg_I_p), "{:3.4f}".format(arg_I_r), "{:3.4f}".format(arg_I_F1)])
        table.add_row(['arg_C', "{:3.4f}".format(arg_C_p), "{:3.4f}".format(arg_C_r), "{:3.4f}".format(arg_C_F1)])
        table.add_row(['ec_I', "{:3.4f}".format(ec_I_p), "{:3.4f}".format(ec_I_r), "{:3.4f}".format(ec_I_F1)])
        table.add_row(['ec_C', "{:3.4f}".format(ec_C_p), "{:3.4f}".format(ec_C_r), "{:3.4f}".format(ec_C_F1)])
        logger.info('\n' + str(table))
        
        with open(config.output_dir + '/test_predictions.json', 'w', encoding='utf-8') as f:
            json.dump(output_res, f, ensure_ascii=False)
        
        return arg_C_F1
        
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/scievents.json')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=52)
    
    args = parser.parse_args()
    
    config = Config(args)
    
    config, logger = utils.set_logger(config)
    logger.info(config)
    config.logger = logger

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    logger.info("Loading Data")
    datasets, ori_data, vocab = data_loader.load_data_bert(config)  # original data
    
    train_loader, dev_loader, test_loader = (
        DataLoader(dataset=dataset,
                   batch_size=config.batch_size,
                   collate_fn=data_loader.collate_fn,
                   shuffle= i==0,  # i==0 means training dataset
                   num_workers=4,
                   drop_last= i==0)
        for i, dataset in enumerate(datasets)
    )
    
    updates_total = len(datasets[0]) // config.batch_size * config.epochs
    
    logger.info("Building Model")
    model = EXCEEDS(config)
    model = model.cuda()
    
    
    trainer = Trainer(model, vocab)
    if config.ckpt == '':    
        best_f1 = 0
        best_tc_f1 = 0
        best_ac_f1 = 0
        best_epoch = 0
        for i in range(config.epochs):
            logger.info("Epoch: {}".format(i))
            trainer.train(i, train_loader)
            if i >= 5:  # a conservative training strategy, to prevent potential excessively long decoding time during early training stages.
                tc_dev_f1, ac_dev_f1 = trainer.eval(i, dev_loader)
                dev_f1 = 0.5 * tc_dev_f1 + 0.5 * ac_dev_f1
                if dev_f1 > best_f1:
                    best_f1 = dev_f1
                    best_tc_f1 = tc_dev_f1
                    best_ac_f1 = ac_dev_f1
                    best_epoch = i
                    trainer.save(config.output_dir + '/best_model.state')
                logger.info("Current Best Epoch: {}".format(best_epoch))
                logger.info("Current Best TC DEV F1: {:3.4f}".format(best_tc_f1))
                logger.info("Current Best AC DEV F1: {:3.4f}".format(best_ac_f1))
        logger.info("Best epoch: {}".format(best_epoch))
        logger.info("Best TC DEV F1: {:3.4f}".format(best_tc_f1))
        logger.info("Best AC DEV F1: {:3.4f}".format(best_ac_f1))
        logger.info(str(config.vocab.label2id))
        trainer.load(config.output_dir + '/best_model.state')
        trainer.predict(test_loader)
    else:
        logger.info('Loading checkpoint...')
        trainer.load(config.ckpt)
        logger.info('Loaded!')
        trainer.predict(test_loader)