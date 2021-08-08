# NLP_Metrics
A repository gathers utility module for evaluating NLP models.
- [CER](https://github.com/TeaKatz/NLP_Metrics/tree/main/src/nlp_metrics#CER)
- [WER](https://github.com/TeaKatz/NLP_Metrics/tree/main/src/nlp_metrics#WER)
- [BLEU](https://github.com/TeaKatz/NLP_Metrics/tree/main/src/nlp_metrics#BLEU)
- [Cosine_Similarity](https://github.com/TeaKatz/NLP_Metrics/tree/main/src/nlp_metrics#Cosine_Similarity)
- [Accuracy](https://github.com/TeaKatz/NLP_Metrics/tree/main/src/nlp_metrics#Accuracy)
- [Precision](https://github.com/TeaKatz/NLP_Metrics/tree/main/src/nlp_metrics#Precision)
- [Recall](https://github.com/TeaKatz/NLP_Metrics/tree/main/src/nlp_metrics#Recall)
- [F1](https://github.com/TeaKatz/NLP_Metrics/tree/main/src/nlp_metrics#F1)
- [MultiLabel_Accuracy](https://github.com/TeaKatz/NLP_Metrics/tree/main/src/nlp_metrics#MultiLabel_Accuracy)
- [MultiLabel_Precision](https://github.com/TeaKatz/NLP_Metrics/tree/main/src/nlp_metrics#MultiLabel_Precision)
- [MultiLabel_Recall](https://github.com/TeaKatz/NLP_Metrics/tree/main/src/nlp_metrics#MultiLabel_Recall)
- [MultiLabel_F1](https://github.com/TeaKatz/NLP_Metrics/tree/main/src/nlp_metrics#MultiLabel_F1)
- [MAE](https://github.com/TeaKatz/NLP_Metrics/tree/main/src/nlp_metrics#MAE)
- [MSE](https://github.com/TeaKatz/NLP_Metrics/tree/main/src/nlp_metrics#MSE)
- [Spearman](https://github.com/TeaKatz/NLP_Metrics/tree/main/src/nlp_metrics#Spearman)

## Requirements
- NumPy
- FastWER
- NLTK
- Scikit-learn

## Installation
    git clone https://github.com/TeaKatz/NLP_Metrics
    cd NLP_Metrics
    pip install --editable .

## Uninstallation
    pip uninstall nlp-metrics

## Example
    import numpy as np
    from nlp_metrics import Metrics

    metrics_func = Metrics(["Cosine_Similarity", "MAE"])

    vector1 = np.random.normal(size=[100, 200])
    vector2 = np.random.normal(size=[100, 200])

    metrics_dict = metrics_func(vector1, vector2)
    cosine_similarity = metrics_dict["Cosine_Similarity"]
    mae = metrics_dict["MAE"]