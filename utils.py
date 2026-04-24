import json
import torch
import logging
import pickle
import time
import datetime
import os
import pprint


def set_logger(config):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
    output_dir = os.path.join(config.output_dir, timestamp)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_path = os.path.join(output_dir, "train.log")
    
    logger = logging.getLogger()
    if logger.handlers:
        logger.handlers.clear()
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s', 
                                  datefmt='[%Y-%m-%d %H:%M:%S]')
    
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"\n{pprint.pformat(vars(config), indent=4)}")
    
    config.output_dir = output_dir
    config.log_path = log_path

    return config, logger


def save_file(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_file(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def decode(batch_metrix, vocab) -> list:
    """Decode the word-word event grid metrix into event output. We apply DFS to find all possible events.
    
    HTL: head-tail-link, which represents the successive order between adjacent tokens in an event mention
    THL-type: tail-head-link (with a type), which represents the trigger or argument type of a mention
    EAL: event-argument-link, which represents the link between a trigger and its argument. 
        In this case, the head of a trigger mention links to the head of an argument mention. 

    Args:
        batch_metrix (torch.Tensor): the word-word event grid metrix, (batch_size, length, length, label_num)

    Returns:
        list: SciEvents's format event list decoded from given metrix, (batch_size,)
    """
    decode_events = []
    for metrix in batch_metrix:
        forward_dict = {}  # save HTL links
        head_dict = {}  # save THL-type links
        link_list = []  # save EAL links
        l = len(metrix[0])
        for i in range(l):
            for j in range(l):
                label_ids = metrix[i][j].nonzero().flatten()
                for label_id in label_ids:
                    label = vocab.id2label[label_id.item()]
                    if label == 'HTL':
                        if i == j:  # avoid self-loop
                            continue
                        forward_dict[i] = forward_dict.get(i, [])
                        if j not in forward_dict[i]:
                            forward_dict[i].append(j)
                    elif label == 'EAL':
                        link_list.append((i, j))
                    else:  # THL-type 
                        head_dict[j] = head_dict.get(j, [])
                        if i not in head_dict[j]:
                            head_dict[j].append(i)
                
        dots = []
        prune = []
        def find_dot(key: int, mention: list, tails: list) -> bool:
            good_key = False
            mention.append(key)
            if key in tails:  # a complete mention
                dots.append(mention.copy())
                good_key = True
            if key not in forward_dict:  # no further successive tokens 
                mention.pop()
                if not good_key:
                    prune.append(key)
                return good_key
            for k in forward_dict[key]:  # traverse HTL links
                if k not in mention and k not in prune:  # avoid loop and pruned paths
                    good_key = good_key or find_dot(k, mention, tails)
            mention.pop()
            if not good_key:
                prune.append(key)
            return good_key
        
        for head in head_dict:
            find_dot(head, [], head_dict[head])
            
        trgs = []
        args = []
        for dot in dots:
            label_ids = metrix[dot[-1]][dot[0]].nonzero().flatten()
            for label_id in label_ids:
                label = vocab.id2label[label_id.item()]
                if label in vocab.ontology['event_types'].keys():
                    trgs.append([dot, label])
                else:
                    args.append([dot, label])
            
        events = []
        for trg in trgs:
            events.append({
                "event_type": trg[1],
                "arguments": [],
                "trigger": {
                    "offsets": trg[0]
                }
            })
        for arg in args:
            for i, trg in enumerate(trgs):
                if (trg[0][0], arg[0][0]) in link_list and vocab.ontology_check(trg[1], arg[1]):
                    events[i]['arguments'].append({
                        "offsets": arg[0],
                        "argument_type": arg[1]
                    })
        decode_events.append(events)
    return decode_events


def safe_div(dividend: int, divisor: int) -> float:
    return dividend/divisor if divisor != 0 else 0.


def extract_nuggets(doc: list, target: str) -> list:
    """ 
    Args:
        doc (list): document events or event arguments
        target (str): 'trigger', 'argument'

    Returns:
        list: nuggets
    """
    nuggets = []  # {'tokens': list, 'offsets': list, 'type': str}
    if target == 'trigger':
        for evt in doc:
            temp = {'event_type': evt['event_type']}
            if 'tokens' in evt['trigger']:
                temp['tokens'] = evt['trigger']['tokens']
            if 'offsets' in evt['trigger']:
                temp['offsets'] = evt['trigger']['offsets']
            nuggets.append(temp)
    elif target == 'argument':
        for arg in doc:
            temp = {'argument_type': arg['argument_type']}
            if 'tokens' in arg:
                temp['tokens'] = arg['tokens']
            if 'offsets' in arg:
                temp['offsets'] = arg['offsets']
            nuggets.append(temp)
    return nuggets
            
            
        
def calculate_F1(gold_datas: list, predict_datas: list, expr_type: str = 'offset') -> list:
    """Each data is a list of SciEvents's format events, calculate F1 scores

    The SciEvents's format events is like the following:
    You can also refer to data_structure.py for more details.
    
    SciEvents's format events:
    [  # a certain document
        {  # a certain event
            "event_type": "PRP",
            "arguments": [
                {  # a certain argument
                    "text": "we",  # optional
                    "tokens": ["we"],  # optional, "text" and "tokens" need at least one of them when expr_type is 'token'
                    "offsets": [87],  # optional, necessary when expr_type is 'offset'
                    "argument_type": "Proposer"
                },
                {
                    "text": "r4c dataset",  # optional
                    "tokens": ["r4c", "dataset"],  # optional, "text" and "tokens" need at least one of them when expr_type is 'token'
                    "offsets": [93, 94],  # optional, necessary when expr_type is 'offset'
                    "argument_type": "Content"
                }
            ],
            "trigger": {
                "text": "propose",  # optional
                "tokens": ["propose"],  # optional, "text" and "tokens" need at least one of them when expr_type is 'token'
                "offsets": [10, 11],  # optional, necessary when expr_type is 'offset'
            }
        },
        ...
    ]
    
    !!! Important !!! Your inputs should be the list like: [SciEvents's format events, SciEvents's format events, ...]

    Args:
        gold_datas (list): each data is a list of SciEvents's format events
        predict_datas (list): each data is a list of SciEvents's format events
        expr_type (str, optional): how nuggets are evaluated. Defaults to 'offset'. You can also use 'token'.

    Returns:
        list: F1, precision and recall:
            trg_I_F1, trg_C_F1, arg_I_F1, arg_C_F1, ec_I_F1, ec_C_F1,
            trg_I_p, trg_C_p, arg_I_p, arg_C_p, ec_I_p, ec_C_p,
            trg_I_r, trg_C_r, arg_I_r, arg_C_r, ec_I_r, ec_C_r
    """
    assert len(gold_datas) == len(predict_datas)
    trg_I_tp = trg_I_fp = trg_I_fn = 0
    trg_C_tp = trg_C_fp = trg_C_fn = 0
    arg_I_tp = arg_I_fp = arg_I_fn = 0
    arg_C_tp = arg_C_fp = arg_C_fn = 0
    ec_I_tp = ec_I_fp = ec_I_fn = 0
    ec_C_tp = ec_C_fp = ec_C_fn = 0
    
    def nugget_expr(nugget: dict, expr=expr_type) -> str:
        if expr == 'offset':
            return ' '.join([str(offset) for offset in nugget['offsets']])
        elif expr == 'token':
            if 'text' in nugget:
                return nugget['text']
            elif 'tokens' in nugget:
                return ' '.join(nugget['tokens'])
            else:
                raise ValueError(f"Invalid nugget: {nugget}")
        else:
            raise ValueError(f"Invalid expression type: {expr}")
    
    for gold_events, predict_events in zip(gold_datas, predict_datas):
        gold_trgs = extract_nuggets(gold_events, 'trigger')
        gold_expr2trg = dict((nugget_expr(trg), {**trg, 'index': idx}) for idx, trg in enumerate(gold_trgs))
        predict_trgs = extract_nuggets(predict_events, 'trigger')
        predict_expr2trg = dict((nugget_expr(trg), {**trg, 'index': idx}) for idx, trg in enumerate(predict_trgs))
        
        for idx, trg in enumerate(gold_trgs):  # find TP and FN
            gold_args = extract_nuggets(gold_events[idx]['arguments'], 'argument')
            
            gold_subtrgs = []
            for arg in gold_args:
                if nugget_expr(arg) in gold_expr2trg:
                    gold_subtrgs.append(arg)
                    
            if nugget_expr(trg) in predict_expr2trg:  # trigger (A)
                trg_I_tp += 1
                
                trg2 = predict_expr2trg[nugget_expr(trg)]
                if trg['event_type'] == trg2['event_type']:  # event type (A+M)
                    trg_C_tp += 1
                    
                    idx2 = trg2['index']
                    predict_args = extract_nuggets(predict_events[idx2]['arguments'], 'argument')
                    predict_expr2arg = dict((nugget_expr(arg), {**arg, 'index': idx}) for idx, arg in enumerate(predict_args))
                    
                    for arg in gold_args:
                        if nugget_expr(arg) in predict_expr2arg:  # argument (A+M+B)
                            arg_I_tp += 1
                            if arg['argument_type'] == predict_expr2arg[nugget_expr(arg)]['argument_type']:  # argument type (A+M+B+BT)
                                arg_C_tp += 1
                            else:
                                arg_C_fn += 1
                        else:
                            arg_I_fn += 1
                            arg_C_fn += 1
                    
                    for subtrg in gold_subtrgs:
                        if nugget_expr(subtrg) in predict_expr2arg and nugget_expr(subtrg) in predict_expr2trg:  # argument and sub-trigger (A+M+C1(=C2)+N)
                            ec_I_tp += 1
                            if subtrg['argument_type'] == predict_expr2arg[nugget_expr(subtrg)]['argument_type']:  # argument type (A+M+C1(=C2)+N+CT)
                                ec_C_tp += 1
                            else:
                                ec_C_fn += 1
                        else:
                            ec_I_fn += 1
                            ec_C_fn += 1
                else:
                    trg_C_fn += 1
                    arg_I_fn += len(gold_args)
                    arg_C_fn += len(gold_args)
                    ec_I_fn += len(gold_subtrgs)
                    ec_C_fn += len(gold_subtrgs)
            else:
                trg_I_fn += 1
                trg_C_fn += 1
                arg_I_fn += len(gold_args)
                arg_C_fn += len(gold_args)
                ec_I_fn += len(gold_subtrgs)
                ec_C_fn += len(gold_subtrgs)
                
        for idx, trg in enumerate(predict_trgs):  # find FP
            predict_args = extract_nuggets(predict_events[idx]['arguments'], 'argument')
            predict_subtrgs = []
            for arg in predict_args:
                if nugget_expr(arg) in predict_expr2trg:
                    predict_subtrgs.append(arg)
            
            if nugget_expr(trg) in gold_expr2trg:  # trigger (A)
                trg2 = gold_expr2trg[nugget_expr(trg)]
                if trg['event_type'] == trg2['event_type']:  # event type (A+M)
                    idx2 = trg2['index']
                    gold_args = extract_nuggets(gold_events[idx2]['arguments'], 'argument')
                    gold_expr2arg = dict((nugget_expr(arg), {**arg, 'index': idx}) for idx, arg in enumerate(gold_args))
                    
                    for arg in predict_args:
                        if nugget_expr(arg) in gold_expr2arg:  # argument (A+M+B)
                            if arg['argument_type'] != gold_expr2arg[nugget_expr(arg)]['argument_type']:  # argument type (A+M+B+BT)
                                arg_C_fp += 1
                        else:
                            arg_I_fp += 1
                            arg_C_fp += 1
                    
                    for subtrg in predict_subtrgs:
                        if nugget_expr(subtrg) in gold_expr2arg and nugget_expr(subtrg) in gold_expr2trg:  # argument and sub-trigger (A+M+C1(=C2)+N)
                            if subtrg['argument_type'] != gold_expr2arg[nugget_expr(subtrg)]['argument_type']:  # argument type (A+M+C1(=C2)+N+CT)
                                ec_C_fp += 1
                        else:
                            ec_I_fp += 1
                            ec_C_fp += 1
                else:
                    trg_C_fp += 1
                    arg_I_fp += len(predict_args)
                    arg_C_fp += len(predict_args)
                    ec_I_fp += len(predict_subtrgs)
                    ec_C_fp += len(predict_subtrgs)
            else:
                trg_I_fp += 1
                trg_C_fp += 1
                arg_I_fp += len(predict_args)
                arg_C_fp += len(predict_args)
                ec_I_fp += len(predict_subtrgs)
                ec_C_fp += len(predict_subtrgs)
        
    trg_I_p = safe_div(trg_I_tp, trg_I_tp + trg_I_fp)
    trg_I_r = safe_div(trg_I_tp, trg_I_tp + trg_I_fn)
    trg_I_F1 = safe_div(2 * trg_I_p * trg_I_r, trg_I_p + trg_I_r) 
    trg_C_p = safe_div(trg_C_tp, trg_C_tp + trg_C_fp)
    trg_C_r = safe_div(trg_C_tp, trg_C_tp + trg_C_fn)
    trg_C_F1 = safe_div(2 * trg_C_p * trg_C_r, trg_C_p + trg_C_r)
    
    arg_I_p = safe_div(arg_I_tp, arg_I_tp + arg_I_fp)
    arg_I_r = safe_div(arg_I_tp, arg_I_tp + arg_I_fn)
    arg_I_F1 = safe_div(2 * arg_I_p * arg_I_r, arg_I_p + arg_I_r) 
    arg_C_p = safe_div(arg_C_tp, arg_C_tp + arg_C_fp)
    arg_C_r = safe_div(arg_C_tp, arg_C_tp + arg_C_fn)
    arg_C_F1 = safe_div(2 * arg_C_p * arg_C_r, arg_C_p + arg_C_r)
    
    ec_I_p = safe_div(ec_I_tp, ec_I_tp + ec_I_fp)
    ec_I_r = safe_div(ec_I_tp, ec_I_tp + ec_I_fn)
    ec_I_F1 = safe_div(2 * ec_I_p * ec_I_r, ec_I_p + ec_I_r)
    ec_C_p = safe_div(ec_C_tp, ec_C_tp + ec_C_fp)
    ec_C_r = safe_div(ec_C_tp, ec_C_tp + ec_C_fn)
    ec_C_F1 = safe_div(2 * ec_C_p * ec_C_r, ec_C_p + ec_C_r)
    
    return trg_I_F1, trg_C_F1, arg_I_F1, arg_C_F1, ec_I_F1, ec_C_F1, \
           trg_I_p, trg_C_p, arg_I_p, arg_C_p, ec_I_p, ec_C_p, \
           trg_I_r, trg_C_r, arg_I_r, arg_C_r, ec_I_r, ec_C_r
    
        
                
    
    