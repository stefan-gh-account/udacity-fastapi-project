# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model uses a scikit learn random forest classifier with standard settings.

## Intended Use
The model predicts a person's salary based on a number other parameters like work class, education, race and sex.

## Training & Evaluation Data
Census data from https://archive.ics.uci.edu/ml/datasets/census+income has been used for both training and testing using a train-test-split of scikit learn.

## Metrics
Overall model performance:
* Precision: 0.7502
* Recall: 0.6213
* F1: 0.6797

## Ethical Considerations
The used dataset is highly biased. The following are rounded values:
* 90% native to the United-States
* 2/3 male, 1/3 female
* 85% white, 10% black, about 5% other races
* 40% husband with only 5% wife and the rest in different forms of non-marriage
* 70% private workers

Info: In the example payload given for API access you can easily change the prediction of income by just changing sex while leaving all other inputs equal. This might be realistic, but showcases ongoing income inequality.

## Caveats and Recommendations
Income prediction of white, male americans that are private workers should work fairly well. Especially when trying to extrapolate to other countries or races the model performance is hard to estimate. In my tests the precision and recall values of people from other native countries have sometimes been verified with single digit numbers of people, which might have been lucky draws.
