# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Model is a supervised machine learning model trained to predict whether an individual's income is greater than $50,000 per year based on data from census.csv. Model was trained using the census.csv dataset and implemented as part of a machine learning deployment pipeline. model was trained using scikit-learn and is served through a FastAPI application. 

## Intended Use

 Model is intended for educational purposes as part of the Udacity Data Analytics nanodegree Its purpose is to confirm student compotency in machinelearning, specificlly constructing and managing machine learning models and pipelines. This means the model is very very bad at its job. Model predicts whether an individual’s salary is `>50K` or `<=50K` based on demographic and employment-related features. Wether it predicts them well is outside the scope of the project rubric.

## Training Data

The training data comes from the Census Income dataset provided in `census.csv`.Dataset is a big boy, and includes demographic and employment-related features such as age, workclass, education, marital status etc. census.csv is a fairly clean dataset, we like that in a csv. The target label is salary, indicating whether a person earns more than $50,000 annually.

## Evaluation Data

The evaluation data was taken from the same Census Income dataset. 20% of the dataset was reserved for testing, the remaining 80% was used for training. test data was processed using the same fitted encoder and label binarizer from the training pipeline.

## Metrics

The model was evaluated using precision, recall, and F1 score. These metrics were chosen because they provide a useful view of predictive performance.

The model achieved the following performance on the test set:
- Precision: `0.7427`
- Recall: `0.6467`
- F1 Score: `0.6914`

mediocre results, but that's ok we love the model anyway

## Ethical Considerations

 Model is cute but stupid, and uses demographic data including personal attributes such as sex, age and race. Model should not be used in real-world decision-making without bias testing, human oversight and basic common sense, as predictions from the model may reflect biases present in the training data. This should be taken into account when interpreting and reporting these results

## Caveats and Recommendations

Model was developed for instructional purposes and is not optimized for production use. 
Models quality is highly dependant on the quality of it's data
Model should not be exposed to water, excessive heat, or any code base making use of wingdings