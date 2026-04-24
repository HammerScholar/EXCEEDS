"""Quick complex-subset F1 evaluation for SciEvents-style predictions.

Loads gold from ``data/SciEvents/test.json`` and one or more prediction JSON files.
Subsets (inconsecutive / overlap / reverse / subevent) are filtered then scored via
``utils.calculate_F1``. Branching uses the global ``args.format`` (``offset`` or
``token``) set by the CLI.

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
import warnings

import numpy as np

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--outputs', type=str, nargs='+', default=['output.json'])
parser.add_argument('--id_check', action='store_true')
parser.add_argument('--format', default='offset', choices=['offset', 'token'])

args = parser.parse_args()

print(f"Recieved {len(args.outputs)} output files for evaluation.")


def inconsecutive_filter(golds, preds, golds_docs=None):
    """Keep events that have at least one multi-span non-consecutive argument.

    For ``offset`` format, non-consecutive means adjacent offset indices differ by
    more than 1. For ``token`` format, gold may use offsets or document token
    sequence matching; predictions use document matching when available.

    Args:
        golds: List of documents; each document is a list of event dicts.
        preds: Same structure as ``golds``, aligned by document index (or by
            ``id_check`` preprocessing done by the caller).
        golds_docs: Optional list of full gold records (e.g. containing
            ``document`` tokens). Used only when ``args.format == 'token'``.

    Returns:
        tuple: ``(new_golds, new_preds)`` — filtered parallel structures.

    Side effects:
        May emit ``warnings.warn`` for token-format preds when ``document`` is
        missing. Reads global ``args.format``.
    """
    if args.format == 'offset':
        new_golds, new_preds = [], []
        for gold_doc, pred_doc in zip(golds, preds):
            new_gold_events, new_pred_events = [], []

            for gold_event in gold_doc:
                add_flag = False
                add_args = []
                for argument in gold_event['arguments']:
                    for i in range(1, len(argument['offsets'])):
                        if argument['offsets'][i] != argument['offsets'][i - 1] + 1:
                            add_flag = True
                            add_args.append(argument)
                            break
                if add_flag:
                    new_gold_event = {
                        'event_type': gold_event['event_type'],
                        'trigger': gold_event['trigger'],
                        'arguments': add_args
                    }
                    new_gold_events.append(new_gold_event)
            new_golds.append(new_gold_events)

            for pred_event in pred_doc:
                add_flag = False
                add_args = []
                for argument in pred_event['arguments']:
                    for i in range(1, len(argument['offsets'])):
                        if argument['offsets'][i] != argument['offsets'][i - 1] + 1:
                            add_flag = True
                            add_args.append(argument)
                            break
                if add_flag:
                    new_pred_event = {
                        'event_type': pred_event['event_type'],
                        'trigger': pred_event['trigger'],
                        'arguments': add_args
                    }
                    new_pred_events.append(new_pred_event)
            new_preds.append(new_pred_events)
    elif args.format == 'token':
        new_golds, new_preds = [], []
        for idx, (gold_doc, pred_doc) in enumerate(zip(golds, preds)):
            new_gold_events, new_pred_events = [], []

            # Same document order as preds; used for token-sequence checks.
            document = None
            if golds_docs is not None and idx < len(golds_docs):
                document = golds_docs[idx].get('document')

            for gold_event in gold_doc:
                add_flag = False
                add_args = []
                for argument in gold_event['arguments']:
                    tokens = argument['tokens']
                    if 'offsets' in argument and len(argument['offsets']) > 1:
                        for i in range(1, len(argument['offsets'])):
                            if argument['offsets'][i] != argument['offsets'][i - 1] + 1:
                                add_flag = True
                                add_args.append(argument)
                                break
                    elif document is not None and len(tokens) > 1:
                        found_consecutive = False
                        for start_pos in range(len(document) - len(tokens) + 1):
                            if document[start_pos:start_pos + len(tokens)] == tokens:
                                found_consecutive = True
                                break
                        if not found_consecutive:
                            add_flag = True
                            add_args.append(argument)
                if add_flag:
                    new_gold_event = {
                        'event_type': gold_event['event_type'],
                        'trigger': gold_event['trigger'],
                        'arguments': add_args
                    }
                    new_gold_events.append(new_gold_event)
            new_golds.append(new_gold_events)

            for pred_event in pred_doc:
                add_flag = False
                add_args = []
                for argument in pred_event['arguments']:
                    tokens = argument['tokens']
                    if document is not None and len(tokens) > 1:
                        found_consecutive = False
                        for start_pos in range(len(document) - len(tokens) + 1):
                            if document[start_pos:start_pos + len(tokens)] == tokens:
                                found_consecutive = True
                                break
                        if not found_consecutive:
                            add_flag = True
                            add_args.append(argument)
                    elif document is None:
                        warnings.warn(
                            "Document is not provided for token format "
                            "inconsecutive filtering."
                        )
                if add_flag:
                    new_pred_event = {
                        'event_type': pred_event['event_type'],
                        'trigger': pred_event['trigger'],
                        'arguments': add_args
                    }
                    new_pred_events.append(new_pred_event)
            new_preds.append(new_pred_events)
    else:
        raise NotImplementedError(
            "Only offset and token formats are supported for inconsecutive filtering."
        )

    return new_golds, new_preds


def overlap_filter(golds, preds, golds_docs=None):
    """Keep events involving span overlap with another trigger/argument in the doc.

    Gold filtering uses offset overlap (``offset``) or offset/token rules (``token``).
    Predictions additionally align to gold-kept triggers/arguments when spans match.

    Args:
        golds: List of per-document event lists.
        preds: List of per-document event lists, aligned with ``golds``.
        golds_docs: Full gold records for ``document`` tokens in ``token`` mode.

    Returns:
        tuple: ``(new_golds, new_preds)``.

    Side effects:
        Reads global ``args.format``.
    """
    def check_overlap(offsets1, offsets2):
        """Return True if two offset lists share at least one token index."""
        set1 = set(offsets1)
        set2 = set(offsets2)
        return len(set1 & set2) > 0

    def check_overlap_tokens(tokens1, tokens2, document=None):
        """Return True if occurrences of token sequences overlap in ``document``.

        If ``document`` is None, falls back to non-empty intersection of token sets
        (coarse).
        """
        if document is not None:
            pos1_ranges = []
            pos2_ranges = []

            for start_pos in range(len(document) - len(tokens1) + 1):
                if document[start_pos:start_pos + len(tokens1)] == tokens1:
                    pos1_ranges.append((start_pos, start_pos + len(tokens1) - 1))

            for start_pos in range(len(document) - len(tokens2) + 1):
                if document[start_pos:start_pos + len(tokens2)] == tokens2:
                    pos2_ranges.append((start_pos, start_pos + len(tokens2) - 1))

            for start1, end1 in pos1_ranges:
                for start2, end2 in pos2_ranges:
                    if not (end1 < start2 or end2 < start1):
                        return True
            return False
        else:
            set1 = set(tokens1)
            set2 = set(tokens2)
            return len(set1 & set2) > 0

    if args.format == 'offset':
        new_golds, new_preds = [], []
        for gold_doc, pred_doc in zip(golds, preds):
            new_gold_events, new_pred_events = [], []

            gold_nuggets_offsets = []
            for gold_event in gold_doc:
                gold_nuggets_offsets.append(gold_event['trigger']['offsets'])
                for argument in gold_event['arguments']:
                    gold_nuggets_offsets.append(argument['offsets'])

            gold_kept_triggers = set()
            gold_kept_arguments = set()

            for gold_event in gold_doc:
                trigger_offsets = gold_event['trigger']['offsets']
                trigger_overlaps = False
                argument_overlaps = []

                for ofst in gold_nuggets_offsets:
                    if ofst == trigger_offsets:
                        continue
                    if check_overlap(ofst, trigger_offsets):
                        trigger_overlaps = True
                        break

                for argument in gold_event['arguments']:
                    arg_offsets = argument['offsets']
                    arg_overlaps = False
                    for ofst in gold_nuggets_offsets:
                        if ofst == arg_offsets:
                            continue
                        if check_overlap(ofst, arg_offsets):
                            arg_overlaps = True
                            break
                    if arg_overlaps:
                        argument_overlaps.append(argument)
                        trigger_overlaps = True

                if trigger_overlaps or argument_overlaps:
                    gold_kept_triggers.add(tuple(trigger_offsets))
                    for arg in argument_overlaps:
                        gold_kept_arguments.add(tuple(arg['offsets']))
                    new_gold_event = {
                        'event_type': gold_event['event_type'],
                        'trigger': gold_event['trigger'],
                        'arguments': argument_overlaps
                    }
                    new_gold_events.append(new_gold_event)

            new_golds.append(new_gold_events)

            pred_nuggets_offsets = []
            for pred_event in pred_doc:
                pred_nuggets_offsets.append(pred_event['trigger']['offsets'])
                for argument in pred_event['arguments']:
                    pred_nuggets_offsets.append(argument['offsets'])

            for pred_event in pred_doc:
                trigger_offsets = pred_event['trigger']['offsets']
                trigger_overlaps = False
                trigger_matches_gold = False
                argument_overlaps = []

                for ofst in pred_nuggets_offsets:
                    if ofst == trigger_offsets:
                        continue
                    if check_overlap(ofst, trigger_offsets):
                        trigger_overlaps = True
                        break

                if tuple(trigger_offsets) in gold_kept_triggers:
                    trigger_matches_gold = True

                for argument in pred_event['arguments']:
                    arg_offsets = argument['offsets']
                    arg_overlaps = False
                    arg_matches_gold = False

                    for ofst in pred_nuggets_offsets:
                        if ofst == arg_offsets:
                            continue
                        if check_overlap(ofst, arg_offsets):
                            arg_overlaps = True
                            break

                    if tuple(arg_offsets) in gold_kept_arguments:
                        arg_matches_gold = True

                    if arg_overlaps or arg_matches_gold:
                        argument_overlaps.append(argument)
                        trigger_overlaps = True

                if trigger_overlaps or trigger_matches_gold or argument_overlaps:
                    new_pred_event = {
                        'event_type': pred_event['event_type'],
                        'trigger': pred_event['trigger'],
                        'arguments': argument_overlaps
                    }
                    new_pred_events.append(new_pred_event)

            new_preds.append(new_pred_events)
    elif args.format == 'token':
        new_golds, new_preds = [], []
        for idx, (gold_doc, pred_doc) in enumerate(zip(golds, preds)):
            new_gold_events, new_pred_events = [], []

            document = None
            if golds_docs is not None and idx < len(golds_docs):
                document = golds_docs[idx].get('document')

            gold_nuggets = []
            for gold_event in gold_doc:
                gold_nuggets.append((
                    'trigger',
                    gold_event['trigger']['tokens'],
                    gold_event['trigger'].get('offsets')
                ))
                for argument in gold_event['arguments']:
                    gold_nuggets.append((
                        'argument',
                        argument['tokens'],
                        argument.get('offsets')
                    ))

            gold_kept_triggers = set()
            gold_kept_arguments = set()

            for gold_event in gold_doc:
                trigger_tokens = gold_event['trigger']['tokens']
                trigger_offsets = gold_event['trigger'].get('offsets')
                trigger_overlaps = False
                argument_overlaps = []

                for _, toks, ofst in gold_nuggets:
                    if toks == trigger_tokens:
                        continue
                    if trigger_offsets is not None and ofst is not None:
                        if check_overlap(trigger_offsets, ofst):
                            trigger_overlaps = True
                            break
                    elif document is not None:
                        if check_overlap_tokens(trigger_tokens, toks, document):
                            trigger_overlaps = True
                            break

                for argument in gold_event['arguments']:
                    arg_tokens = argument['tokens']
                    arg_offsets = argument.get('offsets')
                    arg_overlaps = False
                    for _, toks, ofst in gold_nuggets:
                        if toks == arg_tokens:
                            continue
                        if arg_offsets is not None and ofst is not None:
                            if check_overlap(arg_offsets, ofst):
                                arg_overlaps = True
                                break
                        elif document is not None:
                            if check_overlap_tokens(arg_tokens, toks, document):
                                arg_overlaps = True
                                break
                    if arg_overlaps:
                        argument_overlaps.append(argument)
                        trigger_overlaps = True

                if trigger_overlaps or argument_overlaps:
                    gold_kept_triggers.add(tuple(trigger_tokens))
                    for arg in argument_overlaps:
                        gold_kept_arguments.add(tuple(arg['tokens']))
                    new_gold_event = {
                        'event_type': gold_event['event_type'],
                        'trigger': gold_event['trigger'],
                        'arguments': argument_overlaps
                    }
                    new_gold_events.append(new_gold_event)

            new_golds.append(new_gold_events)

            pred_nuggets = []
            for pred_event in pred_doc:
                pred_nuggets.append((
                    pred_event['trigger']['tokens'],
                    pred_event['trigger'].get('offsets')
                ))
                for argument in pred_event['arguments']:
                    pred_nuggets.append((argument['tokens'], argument.get('offsets')))

            for pred_event in pred_doc:
                trigger_tokens = pred_event['trigger']['tokens']
                trigger_overlaps = False
                trigger_matches_gold = False
                argument_overlaps = []

                for toks, _ofst in pred_nuggets:
                    if toks == trigger_tokens:
                        continue
                    if document is not None:
                        if check_overlap_tokens(trigger_tokens, toks, document):
                            trigger_overlaps = True
                            break

                if tuple(trigger_tokens) in gold_kept_triggers:
                    trigger_matches_gold = True

                for argument in pred_event['arguments']:
                    arg_tokens = argument['tokens']
                    arg_overlaps = False
                    arg_matches_gold = False

                    for toks, _ofst in pred_nuggets:
                        if toks == arg_tokens:
                            continue
                        if document is not None:
                            if check_overlap_tokens(arg_tokens, toks, document):
                                arg_overlaps = True
                                break

                    if tuple(arg_tokens) in gold_kept_arguments:
                        arg_matches_gold = True

                    if arg_overlaps or arg_matches_gold:
                        argument_overlaps.append(argument)
                        trigger_overlaps = True

                if trigger_overlaps or trigger_matches_gold or argument_overlaps:
                    new_pred_event = {
                        'event_type': pred_event['event_type'],
                        'trigger': pred_event['trigger'],
                        'arguments': argument_overlaps
                    }
                    new_pred_events.append(new_pred_event)

            new_preds.append(new_pred_events)
    else:
        raise NotImplementedError(
            "Only offset and token formats are supported for overlap filtering."
        )
    return new_golds, new_preds


def reverse_filter(golds, preds, golds_docs=None):
    """Keep events with at least one argument whose span order is 'reversed'.

    In ``offset`` mode, reversed means some ``offsets[i] < offsets[i-1]``. In
    ``token`` mode, gold uses offsets; predictions search ``document`` for a
    strictly left-to-right token chain.

    Args:
        golds: List of per-document event lists.
        preds: Aligned predictions.
        golds_docs: Gold records supplying ``document`` for token-mode preds.

    Returns:
        tuple: ``(new_golds, new_preds)``.

    Side effects:
        Reads global ``args.format``.
    """
    if args.format == 'offset':
        new_golds, new_preds = [], []
        for gold_doc, pred_doc in zip(golds, preds):
            new_gold_events, new_pred_events = [], []

            for gold_event in gold_doc:
                add_flag = False
                add_args = []
                for argument in gold_event['arguments']:
                    for i in range(1, len(argument['offsets'])):
                        if argument['offsets'][i] < argument['offsets'][i - 1]:
                            add_flag = True
                            add_args.append(argument)
                            break
                if add_flag:
                    new_gold_event = {
                        'event_type': gold_event['event_type'],
                        'trigger': gold_event['trigger'],
                        'arguments': add_args
                    }
                    new_gold_events.append(new_gold_event)
            new_golds.append(new_gold_events)

            for pred_event in pred_doc:
                add_flag = False
                add_args = []
                for argument in pred_event['arguments']:
                    for i in range(1, len(argument['offsets'])):
                        if argument['offsets'][i] < argument['offsets'][i - 1]:
                            add_flag = True
                            add_args.append(argument)
                            break
                if add_flag:
                    new_pred_event = {
                        'event_type': pred_event['event_type'],
                        'trigger': pred_event['trigger'],
                        'arguments': add_args
                    }
                    new_pred_events.append(new_pred_event)
            new_preds.append(new_pred_events)
    elif args.format == 'token':
        new_golds, new_preds = [], []
        for idx, (gold_doc, pred_doc) in enumerate(zip(golds, preds)):
            new_gold_events, new_pred_events = [], []

            document = None
            if golds_docs is not None and idx < len(golds_docs):
                document = golds_docs[idx].get('document')

            for gold_event in gold_doc:
                add_flag = False
                add_args = []
                for argument in gold_event['arguments']:
                    if 'offsets' in argument and len(argument['offsets']) > 1:
                        for i in range(1, len(argument['offsets'])):
                            if argument['offsets'][i] < argument['offsets'][i - 1]:
                                add_flag = True
                                add_args.append(argument)
                                break
                if add_flag:
                    new_gold_event = {
                        'event_type': gold_event['event_type'],
                        'trigger': gold_event['trigger'],
                        'arguments': add_args
                    }
                    new_gold_events.append(new_gold_event)
            new_golds.append(new_gold_events)

            for pred_event in pred_doc:
                add_flag = False
                add_args = []
                for argument in pred_event['arguments']:
                    tokens = argument['tokens']
                    if document is not None and len(tokens) > 1:
                        token_positions_list = []
                        for token in tokens:
                            positions = []
                            for pos, doc_token in enumerate(document):
                                if doc_token == token:
                                    positions.append(pos)
                            if not positions:
                                token_positions_list = []
                                break
                            token_positions_list.append(positions)

                        if len(token_positions_list) == len(tokens):
                            def can_form_sequential_order(pos_list, idx, last_pos):
                                """DFS: extend chain with indices strictly after ``last_pos``."""
                                if idx == len(pos_list):
                                    return True
                                for pos in pos_list[idx]:
                                    if pos > last_pos:
                                        if can_form_sequential_order(
                                                pos_list, idx + 1, pos):
                                            return True
                                return False

                            found_sequential = False
                            for first_pos in token_positions_list[0]:
                                if can_form_sequential_order(
                                        token_positions_list, 1, first_pos):
                                    found_sequential = True
                                    break

                            if not found_sequential:
                                add_flag = True
                                add_args.append(argument)
                if add_flag:
                    new_pred_event = {
                        'event_type': pred_event['event_type'],
                        'trigger': pred_event['trigger'],
                        'arguments': add_args
                    }
                    new_pred_events.append(new_pred_event)
            new_preds.append(new_pred_events)
    else:
        raise NotImplementedError(
            "Only offset and token formats are supported for reverse filtering."
        )

    return new_golds, new_preds


def subevent_filter(golds, preds):
    """Keep events where an argument span equals some event trigger span (same doc).

    Compared via ``tuple(offsets)`` or ``tuple(tokens)`` depending on
    ``args.format``.

    Args:
        golds: List of per-document event lists.
        preds: Aligned predictions.

    Returns:
        tuple: ``(new_golds, new_preds)``.

    Side effects:
        Reads global ``args.format``.
    """
    if args.format == 'offset':
        new_golds, new_preds = [], []
        for gold_doc, pred_doc in zip(golds, preds):
            new_gold_events, new_pred_events = [], []

            gold_triggers = []
            for gold_event in gold_doc:
                gold_triggers.append(tuple(gold_event['trigger']['offsets']))
            for gold_event in gold_doc:
                add_flag = False
                add_args = []
                for argument in gold_event['arguments']:
                    if tuple(argument['offsets']) in gold_triggers:
                        add_flag = True
                        add_args.append(argument)
                if add_flag:
                    new_gold_event = {
                        'event_type': gold_event['event_type'],
                        'trigger': gold_event['trigger'],
                        'arguments': add_args
                    }
                    new_gold_events.append(new_gold_event)
            new_golds.append(new_gold_events)

            pred_triggers = []
            for pred_event in pred_doc:
                pred_triggers.append(tuple(pred_event['trigger']['offsets']))
            for pred_event in pred_doc:
                add_flag = False
                add_args = []
                for argument in pred_event['arguments']:
                    if tuple(argument['offsets']) in pred_triggers:
                        add_flag = True
                        add_args.append(argument)
                if add_flag:
                    new_pred_event = {
                        'event_type': pred_event['event_type'],
                        'trigger': pred_event['trigger'],
                        'arguments': add_args
                    }
                    new_pred_events.append(new_pred_event)
            new_preds.append(new_pred_events)
    elif args.format == 'token':
        new_golds, new_preds = [], []
        for gold_doc, pred_doc in zip(golds, preds):
            new_gold_events, new_pred_events = [], []

            gold_triggers = []
            for gold_event in gold_doc:
                gold_triggers.append(tuple(gold_event['trigger']['tokens']))
            for gold_event in gold_doc:
                add_flag = False
                add_args = []
                for argument in gold_event['arguments']:
                    if tuple(argument['tokens']) in gold_triggers:
                        add_flag = True
                        add_args.append(argument)
                if add_flag:
                    new_gold_event = {
                        'event_type': gold_event['event_type'],
                        'trigger': gold_event['trigger'],
                        'arguments': add_args
                    }
                    new_gold_events.append(new_gold_event)
            new_golds.append(new_gold_events)

            pred_triggers = []
            for pred_event in pred_doc:
                pred_triggers.append(tuple(pred_event['trigger']['tokens']))
            for pred_event in pred_doc:
                add_flag = False
                add_args = []
                for argument in pred_event['arguments']:
                    if tuple(argument['tokens']) in pred_triggers:
                        add_flag = True
                        add_args.append(argument)
                if add_flag:
                    new_pred_event = {
                        'event_type': pred_event['event_type'],
                        'trigger': pred_event['trigger'],
                        'arguments': add_args
                    }
                    new_pred_events.append(new_pred_event)
            new_preds.append(new_pred_events)
    else:
        raise NotImplementedError(
            "Only offset and token formats are supported for subevent filtering."
        )
    return new_golds, new_preds


preds_pres = []
for output in args.outputs:
    with open(output) as fp:
        preds_pres.append(json.load(fp))
with open('data/SciEvents/test.json') as fp:
    golds_pre = json.load(fp)

scores = []
consecutive_scores = []
overlap_scores = []
reverse_scores = []
subevent_scores = []
for i, preds_pre in enumerate(preds_pres):
    if args.id_check:
        assert len(golds_pre) == len(preds_pre), (
            "The number of documents in golds and preds are different."
        )
        golds_id2doc, preds_id2doc = {}, {}
        golds_id2full_doc = {}
        for gold in golds_pre:
            golds_id2doc[gold['doc_id']] = gold['events']
            golds_id2full_doc[gold['doc_id']] = gold
        for pred in preds_pre:
            assert 'doc_id' in pred, (
                f"Please include doc_id in your {i + 1} predictions for id_check."
            )
            preds_id2doc[pred['doc_id']] = pred['events']
        preds, golds = [], []
        golds_docs = []
        for doc_id in golds_id2doc.keys():
            if doc_id not in preds_id2doc:
                raise ValueError(
                    f"Document id {doc_id} not found in your {i + 1} predictions."
                )
            preds.append(preds_id2doc[doc_id])
            golds.append(golds_id2doc[doc_id])
            golds_docs.append(golds_id2full_doc[doc_id])
    else:
        preds = preds_pre
        golds = [doc['events'] for doc in golds_pre]
        golds_docs = golds_pre

    inconsecutive_golds, inconsecutive_preds = inconsecutive_filter(
        golds, preds, golds_docs
    )
    overlap_golds, overlap_preds = overlap_filter(golds, preds, golds_docs)
    reverse_golds, reverse_preds = reverse_filter(golds, preds, golds_docs)
    subevent_golds, subevent_preds = subevent_filter(golds, preds)

    (trg_I_F1, trg_C_F1, arg_I_F1,
     arg_C_F1, ec_I_F1, ec_C_F1,
     trg_I_p, trg_C_p, arg_I_p,
     arg_C_p, ec_I_p, ec_C_p,
     trg_I_r, trg_C_r, arg_I_r,
     arg_C_r, ec_I_r, ec_C_r) = utils.calculate_F1(golds, preds, args.format)

    scores.append((
        trg_I_F1, trg_C_F1, arg_I_F1, arg_C_F1, ec_I_F1, ec_C_F1,
        trg_I_p, trg_C_p, arg_I_p, arg_C_p, ec_I_p, ec_C_p,
        trg_I_r, trg_C_r, arg_I_r, arg_C_r, ec_I_r, ec_C_r
    ))

    (c_trg_I_F1, c_trg_C_F1, c_arg_I_F1,
     c_arg_C_F1, c_ec_I_F1, c_ec_C_F1,
     c_trg_I_p, c_trg_C_p, c_arg_I_p,
     c_arg_C_p, c_ec_I_p, c_ec_C_p,
     c_trg_I_r, c_trg_C_r, c_arg_I_r,
     c_arg_C_r, c_ec_I_r, c_ec_C_r) = utils.calculate_F1(
        inconsecutive_golds, inconsecutive_preds, args.format
    )
    consecutive_scores.append((
        c_trg_I_F1, c_trg_C_F1, c_arg_I_F1, c_arg_C_F1, c_ec_I_F1, c_ec_C_F1,
        c_trg_I_p, c_trg_C_p, c_arg_I_p, c_arg_C_p, c_ec_I_p, c_ec_C_p,
        c_trg_I_r, c_trg_C_r, c_arg_I_r, c_arg_C_r, c_ec_I_r, c_ec_C_r
    ))

    (o_trg_I_F1, o_trg_C_F1, o_arg_I_F1,
     o_arg_C_F1, o_ec_I_F1, o_ec_C_F1,
     o_trg_I_p, o_trg_C_p, o_arg_I_p,
     o_arg_C_p, o_ec_I_p, o_ec_C_p,
     o_trg_I_r, o_trg_C_r, o_arg_I_r,
     o_arg_C_r, o_ec_I_r, o_ec_C_r) = utils.calculate_F1(
        overlap_golds, overlap_preds, args.format
    )
    overlap_scores.append((
        o_trg_I_F1, o_trg_C_F1, o_arg_I_F1, o_arg_C_F1, o_ec_I_F1, o_ec_C_F1,
        o_trg_I_p, o_trg_C_p, o_arg_I_p, o_arg_C_p, o_ec_I_p, o_ec_C_p,
        o_trg_I_r, o_trg_C_r, o_arg_I_r, o_arg_C_r, o_ec_I_r, o_ec_C_r
    ))

    (r_trg_I_F1, r_trg_C_F1, r_arg_I_F1,
     r_arg_C_F1, r_ec_I_F1, r_ec_C_F1,
     r_trg_I_p, r_trg_C_p, r_arg_I_p,
     r_arg_C_p, r_ec_I_p, r_ec_C_p,
     r_trg_I_r, r_trg_C_r, r_arg_I_r,
     r_arg_C_r, r_ec_I_r, r_ec_C_r) = utils.calculate_F1(
        reverse_golds, reverse_preds, args.format
    )
    reverse_scores.append((
        r_trg_I_F1, r_trg_C_F1, r_arg_I_F1, r_arg_C_F1, r_ec_I_F1, r_ec_C_F1,
        r_trg_I_p, r_trg_C_p, r_arg_I_p, r_arg_C_p, r_ec_I_p, r_ec_C_p,
        r_trg_I_r, r_trg_C_r, r_arg_I_r, r_arg_C_r, r_ec_I_r, r_ec_C_r
    ))

    (s_trg_I_F1, s_trg_C_F1, s_arg_I_F1,
     s_arg_C_F1, s_ec_I_F1, s_ec_C_F1,
     s_trg_I_p, s_trg_C_p, s_arg_I_p,
     s_arg_C_p, s_ec_I_p, s_ec_C_p,
     s_trg_I_r, s_trg_C_r, s_arg_I_r,
     s_arg_C_r, s_ec_I_r, s_ec_C_r) = utils.calculate_F1(
        subevent_golds, subevent_preds, args.format
    )
    subevent_scores.append((
        s_trg_I_F1, s_trg_C_F1, s_arg_I_F1, s_arg_C_F1, s_ec_I_F1, s_ec_C_F1,
        s_trg_I_p, s_trg_C_p, s_arg_I_p, s_arg_C_p, s_ec_I_p, s_ec_C_p,
        s_trg_I_r, s_trg_C_r, s_arg_I_r, s_arg_C_r, s_ec_I_r, s_ec_C_r
    ))

num_outputs = len(args.outputs)
if num_outputs > 1:
    consecutive_avg_scores = np.mean(consecutive_scores, axis=0)
    consecutive_std_scores = np.std(consecutive_scores, axis=0)
    overlap_avg_scores = np.mean(overlap_scores, axis=0)
    overlap_std_scores = np.std(overlap_scores, axis=0)
    reverse_avg_scores = np.mean(reverse_scores, axis=0)
    reverse_std_scores = np.std(reverse_scores, axis=0)
    subevent_avg_scores = np.mean(subevent_scores, axis=0)
    subevent_std_scores = np.std(subevent_scores, axis=0)
    
    print("Average Scores:")
    print("--------------------------------")
    print("Consecutive: AC F1={:3.2f}±{:3.2f}".format(consecutive_avg_scores[3] * 100, consecutive_std_scores[3] * 100))
    print("--------------------------------")
    print("Overlap: TC F1={:3.2f}±{:3.2f}".format(overlap_avg_scores[1] * 100, overlap_std_scores[1] * 100))
    print("Overlap: AC F1={:3.2f}±{:3.2f}".format(overlap_avg_scores[3] * 100, overlap_std_scores[3] * 100))
    print("--------------------------------")
    print("Reverse: AC F1={:3.2f}±{:3.2f}".format(reverse_avg_scores[3] * 100, reverse_std_scores[3] * 100))
    print("--------------------------------")
    print("Subevent: TC F1={:3.2f}±{:3.2f}".format(subevent_avg_scores[1] * 100, subevent_std_scores[1] * 100))
    print("Subevent: AC F1={:3.2f}±{:3.2f}".format(subevent_avg_scores[3] * 100, subevent_std_scores[3] * 100))
    print("Subevent: EC F1={:3.2f}±{:3.2f}".format(subevent_avg_scores[5] * 100, subevent_std_scores[5] * 100))
else:
    print("Scores for the only output:")
    print("--------------------------------")
    print("Consecutive: AC F1={:3.2f}".format(consecutive_scores[0][3] * 100))
    print("--------------------------------")
    print("Overlap: TC F1={:3.2f}".format(overlap_scores[0][1] * 100))
    print("Overlap: AC F1={:3.2f}".format(overlap_scores[0][3] * 100))
    print("--------------------------------")
    print("Reverse: AC F1={:3.2f}".format(reverse_scores[0][3] * 100))
    print("--------------------------------")
    print("Subevent: TC F1={:3.2f}".format(subevent_scores[0][1] * 100))
    print("Subevent: AC F1={:3.2f}".format(subevent_scores[0][3] * 100))
    print("Subevent: EC F1={:3.2f}".format(subevent_scores[0][5] * 100))
