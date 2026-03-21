# Notes

## special characters

As seen below some characters are stuck to words creating tokens with mutiple words. What to do with these?
```
{
    "id": "085e6abb-48c7-4388-9d70-901c4b173369", 
    "words": ["The", "paper", "describes", "an", "extension", "of", "word", "embedding", 
        "methods", "to", "also", "provide\r\nrepresentations", "for", "phrases", "and", 
        "concepts", "that", "correspond", "to", "words.", "", "The", "method\r\nworks", 
        "by", "fixing", "an", "identifier", "for", "groups", "of", "phrases,", "words", 
        "and", "the", "concept", "that\r\nall", "denote", "this", "concept,", "replace", 
        "the", "occurrences", "of", "the", "phrases", "and", "words", "by\r\nthis", 
        "identifier", "in", "the", "training", "corpus,", "creating", "a", "\"tagged\"", 
        "corpus,", "and", "then\r\nappending", "the", "tagged", "corpus", "to", "the", 
        "original", "corpus", "for", "training.", "", "The\r\nconcept/phrase/word", 
        "sets", "are", "taken", "from", "an", "ontology.", "", "Since", "the", "domain", 
        "of\r\napplication", "is", "biomedical,", "the", "related", "corpora", "and", 
        "ontologies", "are", "used.", "", "The\r\nresearchers", "also", "report", 
        "that", "they", "achieve", "competitive", "performance", "on", "concept", 
        "similarity", "and", "relatedness", "tasks,", "indicating", "the", 
        "effectiveness", "of", "their", "approach.", "Moreover,", "the", "method", 
        "showcases", "its", "advantage", "by", "requiring", "no", "human", "annotation", 
        "of", "the", "corpus,", "which", "reduces", "the", "time", "and", "cost", 
        "associated", "with", "manual", "efforts.", "The", "authors", "also", 
        "highlight", "the", "superior", "vocabulary", "coverage", "of", "their", 
        "embeddings,", "with", "more", "than", "3x", "coverage", "in", "terms", "of", 
        "vocabulary", "size", "compared", "to", "existing", "methods.", "This", 
        "aspect", "is", "crucial", "in", "ensuring", "a", "comprehensive", 
        "representation", "of", "concepts,", "phrases,", "and", "words", "in", "the", 
        "embedding", "space.", "By", "jointly", "embedding", "these", "different", 
        "linguistic", "units,", "the", "proposed", "method", "paves", "the", "way", 
        "for", "a", "more", "holistic", "understanding", "and", "analysis", "of", 
        "text", "data", "in", "the", "biomedical", "domain.", "Overall,", "the", 
        "paper", "contributes", "to", "the", "field", "of", "embedding", "ontology", 
        "concepts", "by", "presenting", "a", "novel", "weakly-supervised", "method", 
        "that", "captures", "the", "relationships", "between", "concepts,", "phrases,", 
        "and", "words", "in", "an", "unsupervised", "manner.", "The", "results", 
        "achieved", "in", "terms", "of", "performance,", "coverage,", "and", 
        "cost-effectiveness", "make", "this", "approach", "a", "valuable", "addition", 
        "to", "the", "existing", "literature.", "However,", "further", "experiments", 
        "and", "evaluations", "across", "different", "domains", "and", "corpora", 
        "could", "enhance", "the", "generalizability", "of", "the", "proposed", 
        "method", "and", "provide", "a", "more", "comprehensive", "assessment", "of", 
        "its", "capabilities.", "Nonetheless,", "the", "work", "presented", "in", 
        "this", "paper", "is", "a", "promising", "step", "towards", "jointly", 
        "embedding", "concepts,", "phrases,", "and", "words,", "and", "it", "opens", 
        "up", "avenues", "for", "future", "research", "in", "the", "field."
    ], 
    "labels": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ]
}
```

## Current output of context windows

Generating context windows from train_data gets us to 686461 windows, there are thus 686461 tokens currently in train_data

```
Window 1
{
    "id": "5df7a120-f5ff-4881-8624-058f8a2fee5a", 
    "target": "-", 
    "target_label": 0, 
    "words": ["<PAD>", "<PAD>", "-", "Strengths:\r\n*", "Outperforms"], 
    "labels": [-1, -1, 0, 0, 0]
}

Window 2
{
    "id": "5df7a120-f5ff-4881-8624-058f8a2fee5a", 
    "target": "Strengths:\r\n*", 
    "target_label": 0, 
    "words": ["<PAD>", "-", "Strengths:\r\n*", "Outperforms", "ALIGN"], 
    "labels": [-1, 0, 0, 0, 0]
}
```