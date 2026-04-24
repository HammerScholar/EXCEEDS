"""Quick overall F1 for SciEvents-style predictions (baselines / model outputs).

Loads gold from ``data/SciEvents/test.json`` and evaluates ``--outputs`` files via
``utils.calculate_F1``. With multiple outputs, prints mean±std; single output
prints one run.

If you use id_check, please convert your output file into the following json form:
[
    {  # a certain document
        'doc_id': str,
        'events': SciEvents's format events
    },
    {  # a certain document
        'doc_id': str,
        'events': SciEvents's format events
    },
    ...
]
If you don't use id_check, please convert your output file into the following json form:
[
    SciEvents's format events,
    SciEvents's format events,
    ...
]

SciEvents's format events:
[  # a certain document
    {  # a certain event
        "event_type": "PRP",
        "arguments": [
            {  # a certain argument
                "text": "we",  # optional
                "tokens": ["we"],  # optional, "text" and "tokens" need at least one of them when args.format is 'token'
                "offsets": [87],  # optional, necessary when args.format is 'offset'
                "argument_type": "Proposer"
            },
            {
                "text": "r4c dataset",  # optional
                "tokens": ["r4c", "dataset"],  # optional, "text" and "tokens" need at least one of them when args.format is 'token'
                "offsets": [93, 94],  # optional, necessary when args.format is 'offset'
                "argument_type": "Content"
            }
        ],
        "trigger": {
            "text": "propose",  # optional
            "tokens": ["propose"],  # optional, "text" and "tokens" need at least one of them when args.format is 'token'
            "offsets": [10, 11],  # optional, necessary when args.format is 'offset'
        }
    },
    ...
]
"""

import argparse
import json

import numpy as np

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--outputs', type=str, nargs='+', default=['output.json'])
parser.add_argument('--id_check', action='store_true')
parser.add_argument('--format', default='offset', choices=['offset', 'token'])

args = parser.parse_args()

print(f"Recieved {len(args.outputs)} output files for evaluation.")

preds_pres = []
for output in args.outputs:
    with open(output) as fp:
        preds_pres.append(json.load(fp))
with open('data/SciEvents/test.json') as fp:
    golds_pre = json.load(fp)

scores = []
for i, preds_pre in enumerate(preds_pres):
    if args.id_check:
        assert len(golds_pre) == len(preds_pre), (
            "The number of documents in golds and preds are different. "
            f"Golds has {len(golds_pre)} documents, while preds has {len(preds_pre)} documents."
        )
        golds_id2doc, preds_id2doc = {}, {}
        for gold in golds_pre:
            golds_id2doc[gold['doc_id']] = gold['events']
        for pred in preds_pre:
            assert 'doc_id' in pred, (
                f"Please include doc_id in your {i + 1} predictions for id_check."
            )
            preds_id2doc[pred['doc_id']] = pred['events']
        preds, golds = [], []
        for doc_id in golds_id2doc.keys():
            if doc_id not in preds_id2doc:
                raise ValueError(
                    f"Document id {doc_id} not found in your {i + 1} predictions."
                )
            preds.append(preds_id2doc[doc_id])
            golds.append(golds_id2doc[doc_id])
    else:
        preds = preds_pre
        golds = [doc['events'] for doc in golds_pre]

    trg_I_F1, trg_C_F1, arg_I_F1, \
    arg_C_F1, ec_I_F1, ec_C_F1, \
    trg_I_p, trg_C_p, arg_I_p, \
    arg_C_p, ec_I_p, ec_C_p, \
    trg_I_r, trg_C_r, arg_I_r, \
    arg_C_r, ec_I_r, ec_C_r = utils.calculate_F1(golds, preds, args.format)

    scores.append((
        trg_I_F1, trg_C_F1, arg_I_F1, arg_C_F1, ec_I_F1, ec_C_F1,
        trg_I_p, trg_C_p, arg_I_p, arg_C_p, ec_I_p, ec_C_p,
        trg_I_r, trg_C_r, arg_I_r, arg_C_r, ec_I_r, ec_C_r
    ))

num_outputs = len(args.outputs)
if num_outputs > 1:
    avg_scores = np.mean(scores, axis=0)
    std_scores = np.std(scores, axis=0)
    print("Average Scores:")
    print("TRG_I: P={:3.2f}±{:3.2f}, R={:3.2f}±{:3.2f}, F1={:3.2f}±{:3.2f}".format(
        avg_scores[6] * 100, std_scores[6] * 100,
        avg_scores[12] * 100, std_scores[12] * 100,
        avg_scores[0] * 100, std_scores[0] * 100
    ))
    print("TRG_C: P={:3.2f}±{:3.2f}, R={:3.2f}±{:3.2f}, F1={:3.2f}±{:3.2f}".format(
        avg_scores[7] * 100, std_scores[7] * 100,
        avg_scores[13] * 100, std_scores[13] * 100,
        avg_scores[1] * 100, std_scores[1] * 100
    ))
    print("ARG_I: P={:3.2f}±{:3.2f}, R={:3.2f}±{:3.2f}, F1={:3.2f}±{:3.2f}".format(
        avg_scores[8] * 100, std_scores[8] * 100,
        avg_scores[14] * 100, std_scores[14] * 100,
        avg_scores[2] * 100, std_scores[2] * 100
    ))
    print("ARG_C: P={:3.2f}±{:3.2f}, R={:3.2f}±{:3.2f}, F1={:3.2f}±{:3.2f}".format(
        avg_scores[9] * 100, std_scores[9] * 100,
        avg_scores[15] * 100, std_scores[15] * 100,
        avg_scores[3] * 100, std_scores[3] * 100
    ))
    print("EC_C: P={:3.2f}±{:3.2f}, R={:3.2f}±{:3.2f}, F1={:3.2f}±{:3.2f}".format(
        avg_scores[11] * 100, std_scores[11] * 100,
        avg_scores[17] * 100, std_scores[17] * 100,
        avg_scores[5] * 100, std_scores[5] * 100
    ))
else:
    (trg_I_F1, trg_C_F1, arg_I_F1, arg_C_F1, ec_I_F1, ec_C_F1,
     trg_I_p, trg_C_p, arg_I_p, arg_C_p, ec_I_p, ec_C_p,
     trg_I_r, trg_C_r, arg_I_r, arg_C_r, ec_I_r, ec_C_r) = scores[0]
    print("Scores for the only output:")
    print("TRG_I: P={:3.2f}, R={:3.2f}, F1={:3.2f}".format(
        trg_I_p * 100, trg_I_r * 100, trg_I_F1 * 100
    ))
    print("TRG_C: P={:3.2f}, R={:3.2f}, F1={:3.2f}".format(
        trg_C_p * 100, trg_C_r * 100, trg_C_F1 * 100
    ))
    print("ARG_I: P={:3.2f}, R={:3.2f}, F1={:3.2f}".format(
        arg_I_p * 100, arg_I_r * 100, arg_I_F1 * 100
    ))
    print("ARG_C: P={:3.2f}, R={:3.2f}, F1={:3.2f}".format(
        arg_C_p * 100, arg_C_r * 100, arg_C_F1 * 100
    ))
    print("EC_C: P={:3.2f}, R={:3.2f}, F1={:3.2f}".format(
        ec_C_p * 100, ec_C_r * 100, ec_C_F1 * 100
    ))
