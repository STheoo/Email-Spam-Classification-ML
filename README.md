
# Technical Report for Spam Classification Project

A machine learning project to classify emails as spam or non-spam using NLP techniques. Includes data preprocessing, feature extraction, model training, and evaluation.

## Abstract

To initiate the analysis, the raw data will undergo preprocessing using established
techniques to ensure its suitability for algorithmic classification models. Subsequently, the data will be applied into various classifiers and sorted into a table. Simultaneously, the preprocessing approach will be refined to identify the most effective data format.
Furthermore, the parameters of each classifier will be systematically adjusted to optimise
their individual performance. By methodically comparing the results and referring to the
sorted table, the most promising classifier will be selected for further consideration and
analysis.

## Pre-Processing

The pre-processing steps follow a conventional procedure and the steps are as
followed:

- Change target column name to spam and values to 1s and 0s.
- Drop duplicate entries.
- Check for null entries.
- Change emails to lower case.
- Remove punctuation and stop words.
- Create a bag of words matrix using CountVectorizer
- Split to train and test data.

*Stemming the words has seemed to hinder the results and therefore has been
skipped in this process.

*Bag of words have performed better than TF-IDF as input.

## Model Development

Using the now pre-processed data it has been passed through 6 models with default parameters and sorted by accuracy and precision score.

Those models are:
- Support Vector Classifier
- K-Nearest Neighbour Classifier
- Multinomial Naive Bayes Classifier
- Decision Tree Classifier
- Bagging Classifier
- Gradient Boosting Classifier.

The next step is to tinker with the parameters to optimise the settings of each
individual classifier, but before that the results with default parameters have to be noted
down.

![image](https://github.com/user-attachments/assets/f30f2cf2-d2f3-49f5-8131-ea2b382bde2c)

![image](https://github.com/user-attachments/assets/03eca817-f24b-43e6-943a-c5bc8cfd17d7)

## Parameter Optimization

### Support Vector Machine:
- Gamma changed to scale.
- Class weight changed to balance.
- C increased to 3

### Multinomial Naive Bayes:
- Alpha increased to 1.1.

### Decision Tree:
- No changes. Default worked best.

### K-Nearest Neighbours:
- N changed to 8
- Weight changed to uniform.
- P changed to 2.

### Bagging Classifier:
- N-estimators changed to 70.

### Gradient Boosting:
- N-estimators changed to 250.

![image](https://github.com/user-attachments/assets/05aacc37-e6ce-4376-9368-53098a00ed1d)

![image](https://github.com/user-attachments/assets/5dc8dfba-bb9b-4080-9a79-60d92b14f40a)

## Final Model Evaluation

To more extensively evaluate our model and get a clearer picture about its performance confusion matrix and cross validation will be used. The confusion matrix will provide valuable insights about the model's performance, including accuracy, precision, recall, while cross validation further enhances our assessment by validating the model's performance across multiple subsets of the data.

![image](https://github.com/user-attachments/assets/13f898f7-472b-4d20-9db1-34c260514805)

Mean cross validation score between 5 subsets:

![image](https://github.com/user-attachments/assets/31ac3d84-eeb4-4f8d-adbf-e06619c32b5e)

## Conclusion

As seen in the confusion matrix the probability for false positives is very low and the model can accurately predict ham messages, but there could be further improvement to detect spam messages. Another concern is the time it takes to classify the emails, as it is the longest out of all classifiers and could be impractical with a larger dataset. The accuracy and precision are though satisfactory. The chosen model is the Gradient Boosting Decision Tree Classifier, an ensemble method. I would have liked to also test the Voting Classifier with the top 3 classifiers as Multinomial Naive Bayes also has shown promising results as well as others, but it would be out of this course's scope.


