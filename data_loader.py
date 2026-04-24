import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import torch
import numpy as np
import prettytable as pt
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
import utils

dis2idx = np.zeros((2000), dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9


class Vocabulary(object):
    def __init__(self) -> None:
        self.label2id = {'NONE': 0, 'HTL': 1, 'EAL': 2}
        self.id2label = {0: 'NONE', 1: 'HTL', 2: 'EAL'}
        self.label_freq = {'NONE': 0, 'HTL': 0, 'EAL': 0}
        self.ontology = {}
        
    def add_label(self, label: str) -> int:
        if label not in self.label2id:
            self.label2id[label] = len(self.label2id)
            self.id2label[self.label2id[label]] = label
            self.label_freq[label] = 0
        assert label == self.id2label[self.label2id[label]]
        return self.label2id[label]
        
    def load_ontology(self, path: str) -> None:
        with open(path) as fp:
            self.ontology = json.load(fp)
        for evt in self.ontology['event_types']:
            self.add_label(evt)
        for evt, args in self.ontology['event_types'].items():
            for arg in args:
                self.add_label(arg)
    
    def ontology_check(self, event_type: str, argument_type: str) -> bool:
        return False if argument_type not in self.ontology['event_types'][event_type].keys() else True
            
    def __len__(self) -> int:
        return len(self.label2id)
    
    def label_to_id(self, label: str) -> int:
        label = label.lower()
        return self.label2id[label]

    def id_to_label(self, i: int) -> str:
        return self.id2label[i]
    
    @property
    def label_num(self):
        return len(self.label2id)
    
    
def modify_grid(metrix: np.ndarray, vocab: Vocabulary, column: int, row: int, label: str) -> None:
    """modify the grid metrix with specific label

    Args:
        metrix (np.ndarray): [length, length, label_num]
        vocab (Vocabulary): the vocabulary that contains label ids
        column (int): the column index
        row (int): the row index
        label (str): the label to be modified
    """
    label_id = vocab.label2id[label]
    metrix[column, row, label_id] = 1
    vocab.label_freq[label] += 1

    
def encode_mention(metrix: np.ndarray, vocab: Vocabulary, idxs: list, label: str) -> None:
    for i in range(len(idxs)):
        if i == len(idxs) - 1:
            modify_grid(metrix, vocab, idxs[i], idxs[0], label)  # tail-head-link: specific argument or event type
        else:
            modify_grid(metrix, vocab, idxs[i], idxs[i+1], 'HTL')  # head-tail-link
        
        
def collate_fn(data):
    """data collate function, pad the data to the max length in the batch

    Args:
        data (list): 
            bert_inputs: torch.LongTensor (batch_size, pieces_num)
            pieces2word: torch.BoolTensor (batch_size, word_num, pieces_num)
            dist_inputs: torch.LongTensor (batch_size, word_num, word_num)
            grid_labels: torch.LongTensor (batch_size, word_num, word_num, label_num)
            grid_mask2d: torch.BoolTensor (batch_size, word_num, word_num)
            document_lengths: int (batch_size,)
            golden_events: list

    Returns:
        bert_inputs: torch.LongTensor (batch_size, max_pieces_num)
        pieces2word: torch.BoolTensor (batch_size, max_word_num, max_pieces_num)
        dist_inputs: torch.LongTensor (batch_size, max_word_num, max_word_num)
        grid_labels: torch.LongTensor (batch_size, max_word_num, max_word_num, label_num)
        grid_mask2d: torch.BoolTensor (batch_size, max_word_num, max_word_num)
        document_lengths: torch.LongTensor (batch_size,)
        golden_events: list
    """
    bert_inputs, pieces2word, dist_inputs, grid_labels, grid_mask2d, document_lengths, golden_events = map(list, zip(*data))
    
    max_tok = np.max(document_lengths)
    max_pie = np.max([x.shape[0] for x in bert_inputs])
    bert_inputs = pad_sequence(bert_inputs, True)
    batch_size = bert_inputs.size(0)
    
    def fill(data, new_data):
        """Two-dimensional data padding function. Make sure the remaining dimensions are the same."""
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = x
        return new_data

    sub_mat = torch.zeros((batch_size, max_tok, max_pie), dtype=torch.bool)
    pieces2word = fill(pieces2word, sub_mat)
    dist_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    dist_inputs = fill(dist_inputs, dist_mat)
    # if grid_labels[0].dim() == 3:
    #     labels_mat = torch.zeros((batch_size, max_tok, max_tok, grid_labels[0].size(-1)), dtype=torch.long)
    #     grid_labels = fill(grid_labels, labels_mat)
    # elif grid_labels[0].dim() == 2:
    #     labels_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    #     grid_labels = fill(grid_labels, labels_mat)
    # else:
    #     raise ValueError('grid dim should be 2 or 3')
    labels_mat = torch.zeros((batch_size, max_tok, max_tok, grid_labels[0].size(-1)), dtype=torch.long)
    grid_labels = fill(grid_labels, labels_mat)
    mask2d_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.bool)
    grid_mask2d = fill(grid_mask2d, mask2d_mat)
    document_lengths = torch.LongTensor(document_lengths)
    
    return bert_inputs, pieces2word, dist_inputs, grid_labels, grid_mask2d, document_lengths, golden_events

class EventDataset(Dataset):
    def __init__(self, bert_inputs, pieces2word, dist_inputs, grid_labels, grid_mask2d, document_lengths, golden_events) -> None:
        super(EventDataset, self).__init__()
        self.bert_inputs = bert_inputs
        self.pieces2word = pieces2word
        self.dist_inputs = dist_inputs
        self.grid_labels = grid_labels
        self.grid_mask2d = grid_mask2d
        self.document_lengths = document_lengths
        self.golden_events = golden_events
        
    def __getitem__(self, index: int) -> list:
        return torch.LongTensor(self.bert_inputs[index]), \
               torch.LongTensor(self.pieces2word[index]), \
               torch.LongTensor(self.dist_inputs[index]), \
               torch.LongTensor(self.grid_labels[index]), \
               torch.LongTensor(self.grid_mask2d[index]), \
               self.document_lengths[index], \
               self.golden_events[index]
    
    def __len__(self) -> int:
        return len(self.document_lengths)
    
def process_bert(data: list, tokenizer, vocab: Vocabulary) -> list:
    """Process data to several parts that will be used in model framework.

    Args:
        data (list): train/dev/test data
        tokenizer (_type_): a tokenizer
        vocab (Vocabulary): a vocabulary that contains label ids

    Returns:
        list: several parts that will be used in model framework and decoding
    """
    bert_inputs = []
    pieces2word = []
    dist_inputs = []
    grid_labels = []
    grid_mask2d = []
    document_lengths = []
    golden_events = []
    
    for instance in data:
        if len(instance['sentences']) == 0:
            continue
        
        _document = instance['document']
        _events = instance['events']
        
        tokens = [tokenizer.tokenize(word) for word in _document]
        pieces = [piece for pieces in tokens for piece in pieces]
        _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
        _bert_inputs = np.array([tokenizer.cls_token_id] + _bert_inputs + [tokenizer.sep_token_id])
        
        _length = len(_document)
        _pieces2word = np.zeros((_length, len(_bert_inputs)), dtype=bool)
        _dist_inputs = np.zeros((_length, _length), dtype=int)
        _grid_mask2d = np.ones((_length, _length), dtype=bool)
        
        ### create pieces2word matrix
        if tokenizer:
            start = 0
            for i, pieces in enumerate(tokens):
                if len(pieces) == 0:
                    continue
                pieces = list(range(start, start + len(pieces)))
                _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1  # set pieces --> word as 1, '+1' and '+2' because of the beginning CLS
                start += len(pieces)
                
        ### add distance information
        for k in range(_length):
            _dist_inputs[k, :] += k
            _dist_inputs[:, k] -= k

        for i in range(_length):
            for j in range(_length):
                if _dist_inputs[i, j] < 0:
                    _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9  # upper triangle is set to [10, 18]
                else:
                    _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]  # inferior triangle is set to [1, 9]
        _dist_inputs[_dist_inputs == 0] = 19  # diagonal is set to 19
        
        ### create grid labels 
        _grid_labels = np.zeros((_length, _length, vocab.label_num), dtype=int)
        for evt in _events:
            trg = evt['trigger']
                    
            encode_mention(_grid_labels, vocab, trg['offsets'], evt['event_type'])
            for arg in evt['arguments']:
                encode_mention(_grid_labels, vocab, arg['offsets'], arg['argument_type'])
                modify_grid(_grid_labels, vocab, trg['offsets'][0], arg['offsets'][0], 'EAL')  # e.g. T:[0,1,2] A:[2,3,4] EAL:[0,2]
                
        bert_inputs.append(_bert_inputs)
        pieces2word.append(_pieces2word)
        dist_inputs.append(_dist_inputs)
        grid_labels.append(_grid_labels)
        grid_mask2d.append(_grid_mask2d)
        document_lengths.append(_length)
        golden_events.append(_events)
        
    return bert_inputs, pieces2word, dist_inputs, grid_labels, grid_mask2d, document_lengths, golden_events
    
def load_data_bert(config) -> None:
    with open('./data/{}/train.json'.format(config.dataset), 'r', encoding='utf-8') as fp:
        train_data = json.load(fp)
    with open('./data/{}/dev.json'.format(config.dataset), 'r', encoding='utf-8') as fp:
        dev_data = json.load(fp)
    with open('./data/{}/test.json'.format(config.dataset), 'r', encoding='utf-8') as fp:
        test_data = json.load(fp)
        
    tokenizer = AutoTokenizer.from_pretrained(config.bert_name, cache_dir="./cache/")
    
    vocab = Vocabulary()
    vocab.load_ontology('./data/{}/ontology.json'.format(config.dataset))
    
    table = pt.PrettyTable(['', 'documents'])
    table.add_row(['train', len(train_data)])
    table.add_row(['dev', len(dev_data)])
    table.add_row(['test', len(test_data)])
    config.logger.info("\n{}".format(table))
    
    config.vocab = vocab
    
    train_dataset = EventDataset(*process_bert(train_data, tokenizer, vocab))
    dev_dataset = EventDataset(*process_bert(dev_data, tokenizer, vocab))
    test_dataset = EventDataset(*process_bert(test_data, tokenizer, vocab))
    
    config.label_num = vocab.label_num
    
    return (train_dataset, dev_dataset, test_dataset), (train_data, dev_data, test_data), vocab
    
    