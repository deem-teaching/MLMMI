# Exercise 1 — Lightweight Model Management System

In this exercise, you will build a lightweight model registry for one ML task. The goal is to implement a simple model management system.

Use the [Titanic dataset](https://www.openml.org/search?type=data&sort=version&status=any&order=asc&exact_name=Titanic&id=40945) and create several dataset versions from the same original dataset by applying preprocessing steps such as handling missing values, encoding categorical features, selecting subsets of features, or using different train/test splits. The Titanic dataset contains both categorical and numerical features and includes missing values, so preprocessing is necessary.

You should store trained models together with metadata, track simple lineage between related models, and select the best model under a fixed budget constraint.

For inspiration, you can read the following papers:  
https://people.eecs.berkeley.edu/~matei/papers/2016/hilda_modeldb.pdf  
https://www.cidrdb.org/cidr2015/Papers/CIDR15_Paper18.pdf

**Special Note: Use of LLMs/Coding Agents are allowed but not recommended.**

## Expected Project Structure

```
1_Model_registry_and_selection/
├── requirements.txt
├── solution.ipynb            # main notebook orchestrating the pipeline (e.g. calling all scripts)
├── data/
│   ├── raw/                  # original dataset
│   └── ds_v*/                # versioned datasets (train.csv, test.csv, metadata.json)
├── models/
│   └── model_*/              # serialized model artifacts (e.g. .joblib)
├── registry/
│   └── model_*.json          # one metadata file per registered model
└── scripts/                  # reproducible Python scripts
```

## Tasks

### 1. Dataset store and versioning (3.0 pts)
- Create a dataset store with at least 3 dataset versions derived from the same original dataset.
- Each version must be reproducible through a script.
- For each dataset version, track information such as:
  - dataset version ID
  - source dataset
  - preprocessing / transformation steps
  - split information if relevant

### 2. Model registry (3.0 pts)
- Train and register at least 10 models.
- These may be:
  - different algorithms, and/or
  - different hyperparameter settings of the same algorithm
- For each model, track information such as:
  - model ID
  - path to the saved model
  - algorithm
  - hyperparameters
  - evaluation metrics, including accuracy
  - training or compute time
  - inference time (averaged over 10 predictions on a batch of 100 samples)
  - dataset version used
  - creation time

### 3. Lineage tracking (2.0 pts)
- Track simple lineage between models.
- Examples:
  - one model is derived from another by hyperparameter tuning
  - one model is retrained on a different dataset version
- Include at least 3 models with non-trivial lineage.

### 4. Model selection under budget constraints (2.0 pts)
- Implement a script that selects the best model under a fixed budget constraint.
- Example:
  - maximize accuracy subject to training time being below a threshold
- Clearly state:
  - the constraint
  - the selection rule
  - the selected model
