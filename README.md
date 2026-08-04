# Guidepost for reliable ML using the Wisconsin Breast Cancer dataset

## Good Machine Learning ≠ Good Science

First of all, a well-executed machine learning project alone does not guarantee scientific validity. The reliability of a model also depends on factors such as data quality and representativeness, study design, and the identification of potential sources of bias. Even technically correct machine learning can produce misleading conclusions if these aspects are neglected.

This distinction has been highlighted repeatedly in the scientific community, including in comments like these on PubPeer:
- [Advances in decision support for diagnosis and early management of acute leukaemia](https://pubpeer.com/publications/BEA6CA5480256F401C2C8A5B2670C9)
- [AI succeeds in diagnosing rare diseases](https://pubpeer.com/publications/36F46DAB87194CCE3B00FBF3B37583)

Ultimately, the sound application of any methodology (in this case, machine learning) is necessary for good science - but it is not sufficient on its own.

Furthermore, it should be noted that this workflow does not, in and of itself, provide a model that can be used in a real-world setting. Instead, it offers a framework for developing and evaluating machine learning models. Before such a model can be used in practice, further consideration is required. One example of this is a "reject" option, which allows uncertain cases to be referred to a human rather than forcing a prediction.

This is not a limitation of the workflow itself, but rather reflects the difference between evaluating models (in an academic setting) and their use in practice. In medical applications, it is preferable to refer uncertain cases to a physician rather than force a potentially incorrect prediction. In contrast, a rejection option is impractical in applications such as mass production, since manual checks are comparatively expensive.

## Content

**main.pdf** Explains the methodological choices, similar to
**main.md**, though the PDF is more detailed and polished.

**main.py** Implements the workflow:
- dataset inspection
- model selection
- model training
- model evaluation
- model comparison

## Run

Install dependencies:
```
numpy
pandas
scikit-learn
joblib
tqdm
altair
```

Then run and reproduce the analysis described in the document:
``` 
python main.py
```

## Dataset

Wisconsin Breast Cancer dataset (via [scikit-learn](https://scikit-learn.org/1.7/modules/generated/sklearn.datasets.load_breast_cancer.html)).

Introductory Paper: [Nuclear feature extraction for breast tumor diagnosis](https://www.semanticscholar.org/paper/Nuclear-feature-extraction-for-breast-tumor-Street-Wolberg/53f0fbb425bc14468eb3bf96b2e1d41ba8087f36).

## Credits

@numpy [Computing numerical arrays](https://github.com/numpy/numpy)

@pandas [Manipulating structured data](https://github.com/pandas-dev/pandas)

@scikit-learn [Modeling machine learning](https://github.com/scikit-learn/scikit-learn)

@joblib [Running parallel tasks](https://github.com/joblib/joblib)

@tqdm [Tracking loop progress](https://github.com/tqdm/tqdm)

@altair [Building declarative charts](https://github.com/vega/altair)
