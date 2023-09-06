# feature-selection-utils
Feature Selection utils implemented for multivariate timeseries data

## What is 
Algorithms are implemented based on the description in [this paper](http://enggjournals.com/ijcse/doc/IJCSE11-03-05-051.pdf).
Current implementation assumes that the dataset is provided according to the shape [users, timepoints, features], assuming that there are the data for each user is matrix of `n` timepoints for `m` features.
Future implementations may include additional algorithms and will accomodate additional forms of input.

## Why do we need it? 
- It's hard to find implementations for feature selection algorithms that are multi-stage and do not fit into the current platform implementations (from scikit-learn to pytorch). 
- It's also hard to find good implementations for feature selection done on multivariate timeseries.

* This implementation deals with evalution of features in inference time, without retraining.

## Example of usage:
Plus-l-minus-r sequence
```python

...
# dataset: dataset
# labels: labels

# choose metric to optimize, select parameters
metric = 'recall'
l = 2
r = 1
print(f"Running LRS for metric: {metric} with m={m}, r={r}: ***************")
prediction_args = {'labels': labels, 'metric': metric}

# assuming our prediction method is self.predict_for_selected_metric
# Run LRS
evaluator = LRS(dataset, self.predict_for_selected_metric, l, r, prediction_args)
features_and_score_dict = evaluator.evaluate()

# print results
print(metric)
for features, score in features_and_score_dict.items():
    print(f"Features: {features}\tScore: {score:.2f}")
```
