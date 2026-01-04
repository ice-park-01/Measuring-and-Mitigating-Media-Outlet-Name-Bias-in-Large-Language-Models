# Custom Dataset & Prompts

This directory handles dataset management, prompt generation, and media outlet definitions.

## Files Description

- **`dataset.py`**: Functions to load and preprocess the dataset used for experiments (e.g., loading articles, formatting inputs).
- **`prompt.py`**: Logic for constructing prompts. It inserts media outlet names into templates to create the biased input conditions.
- **`media_list.py`**: Definitions of media outlet names and their categorized political leanings (e.g., Left, Center, Right).
- **`split_dataset.py`**: Utilities for splitting the dataset into training, validation, and test sets.
- **`analyze_dataset.py`**: Script to analyze statistics of the dataset itself (e.g., word counts, topic distribution).

## Adding a Custom Dataset

To use your own dataset, modify `dataset.py` to implement a loader that returns data in the expected format.

## Modifying Prompts

To change how the media bias is injected (e.g., changing the instruction phrasing), edit the templates in `prompt.py`.
