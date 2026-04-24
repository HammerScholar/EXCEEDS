This file will introduce the format of data and ontology in SciEvents.

About the data `train.json`, `dev.json`, and `test.json`:

```
# A single document
{
	"venue": ACL,  # all ACL in current SciEvents
	"title": str,  # paper title
	"abstract": str,  # paper original abstract
	"doc_id": str,
	"publication_year": int,  # 2019-2022 in current SciEvents
	"sentences": [str, ...],  # split from abstract
	"events": SciEvents's format events,  # will be introduced following
	"document": [str, ...],  # token-level presentation of the abstract. it can be used to locate offset
}
```

```
# SciEvents's format events
[
    {  # a certain event
        "event_type": str,  # constrained by ontology
        "arguments": [
            {  # a certain argument
                "text": str,
                "nugget_type": str,  # constrained by ontology, not necessary
                "argument_type": str,  # constrained by ontology
                "tokens": [str, ...]  # include every single token. argument["text"].split(' ')
                "offsets": [int, ...]  # the same length as argument["tokens"]
            },
            ...
        ],
        "trigger": {
            "text": str,
            "tokens": [str, ...],  # include every single token. trigger["text"].split(' ')
            "offsets": [int, ...],  # the same length as trigger["tokens"]
        }
    },
    ...
]
```

About the ontology `ontology.json`:

```
{
	"nugget_types": [str, ...],  # In SciEvents, we add nugget types for all arguments (see constrains following). Those nugget types are not necessary for Event Extraction (EE) task. They can be used to enhance EE performance or other NLP tasks.
	"event_types": {
		"a certain event type": {
			"argument_1": contrained nugget types,
			"argument_2": contrained nugget types,
			...
		},
		...
	}
}
```

If you want to use EXCEEDS & SciEvents formats for your own datasets, nugget types are not necessary. Therefore, for compatibility purposes, your customized `ontology.json` can be like:

```
{
	"event_types": {
		"a certain event type": {
			"argument_1": [],  # empty list is ok
			"argument_2": [],
			...
		},
		...
	}
}
```

