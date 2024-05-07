# Typhoid Analysis in Uganda with a Comparative Study of Machine Learning Algorithms

## Abstract
Typhoid continues to be a public health issue in Uganda with over 5,000 incidences every year. Because of the inadequate health facilities, the rise of such a disease is quite alarming. It is essential for us to have proper analysis and prediction of the patterns and occurrences of Typhoid. This report presents a comparative analysis of different machine learning models for predicting Typhoid percentages among the population in the different districts of Uganda. For this particular regression problem, we evaluated the performance of five algorithms that is to say Linear Regression Model, Support Vector Regression model, Decision Tree, Random Forests and the Neural Networks Regression Model. Using a data set of Typhoid Incidences, Environmental factors and population our results show that the Neural Networks is the most optimal model. Each model was evaluated with the Mean squared error, the Root Mean squared error, the mean absolute error and the R2 score. 

## Keywords
Typhoid, Machine Learning, Regression, Linear Regression Model, SVR, Decision Tree, Random Forest, Neural Networks.

## Introduction
Typhoid fever, a waterborne disease caused by Salmonella Typhi, remains a significant public health concern in Uganda, with over 50,000 citizens attacked annually. The disease is often associated with poor sanitation, contaminated water, and inadequate hygiene. In Uganda, the disease poses a significant burden on the health care system, particularly in urban areas like Kampala, where population density and poor living conditions contribute to the fast spread of the disease.

This study aims to analyze and predict the percentage of people with typhoid in the different districts in Uganda using various regression models including the Linear Regression, Decision Tree, Random Forest, Support Vector Regression (SVR), and Neural Networks. The models' performance is evaluated using metrics which are Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R2) score.

Preliminary results indicate that population density has a great correlation with the target variable, and Kampala has the highest percentage of typhoid disease. The Neural Networks model has the best scores from the evaluation metrics. This study's findings will contribute to the development of effective predictive models for typhoid disease in Uganda, enabling health care professionals to carry out interventions and prevent the spread of the disease.

## Review of Related Literature
In [1] and [2], a typhoid diagnosis system was prompted to be used by anyone to assist in quick diagnosis of the Typhoid disease. The data set used was divided into five levels of severity of typhoid fever with the goal of providing treatment accordingly. In [3], deep learning and machine learning were employed to develop a Typhoid prediction model. Ten algorithms have were utilized and compared. From this comparison, the XGBoost classifier was the best performer with a 97.87% accuracy.

## Data set
The data set used has a dimension of 112 rows and 17 columns. It comprises of numerical values for most columns and string data under the column representing district names. This data was collected from the years 2011 to 2017 by the Uganda National Meteorological Authority and the Ministry of Health.
From the data, ten features were used with the target variable being the Typhoid percentage among the population of people in each district(\textbf{Typh\_per}). The features used are:
- Typh\_Inc representing the number of Typhoid incidences.
- PH\_Lands representaing the population that stays in the highlands.
- P\_Density representing the population density of each district.
- Urban\_level representing the proportion of the district that is urban.
- ARainfall representing the average rainfall amount in mm.
- 'Temp\_Min representing the minimum temperature in each district.
- Typh\_Rate representing the cases of Typhoid per unit population.
- Pn\_Floods representing the proportion of people affected by floods in a district.
- P\_male and P\_Female represent the gender proportions per district.

## Methodology and Algorithm architecture
### Linear Regression Model
The Linear Regression algorithm is a supervised Learning technique which focuses on finding a relationship between a dependent variable and one or more independent variables. For the case of our dataset, \textbf{Typh\_per} which is the typhoid percentage recorded is our dependent feature and for the independent variables, we selected those with a positive correlation. Through this model, we find the line of best fit which would minimize the difference between the predicted and actual values.

### Decision Tree Model
The Tree starts with a root node that represents input data. The algorithm recursively splits the data into subsets based on the most informative features, using a splitting criterion (e.g., information gain and entropy). Each internal node in the tree represents a decision made based on a feature and a threshold. The terminal nodes (leaves) represent the predicted values. New input data flows down the tree, making decisions at each node, until it reaches a leaf node, which outputs the predicted value. The tree grows by recursively partitioning the data into smaller subsets based on the most informative features. The tree can be pruned to avoid overfitting by removing branches that do not contribute much to the predictive power.

### Random Forest Model
Random Forest is a supervised machine learning algorithm that creates multiple decision trees and merges them together to get a more accurate and stable prediction. The Random Forest algorithm creates an ensemble of decision trees. Each tree is constructed using a random subset of the dataset to measure a random subset of features in each partition. In prediction, the algorithm aggregates the results of all trees, either by voting in the case of classification or by taking the average in the case of regression.

### SVR Model
Support Vector Regression (SVR) is a machine learning algorithm used for regression tasks. It's goal is to predict the percentage of typhoid fever (dependent variable, Y) based on various independent variables.
The input layer takes in the feature vector, which is a set of input variables that are used to predict the target variable. A radial basis function kernel is used to transform the input data into a higher-dimensional space, where the data can be linearly separated. A subset of the training data is selected as support vectors, which are used to define the hyperplane that separates the data. The hyperplane is the decision boundary that predicts the percentage of typhoid fever (Y) based on the input variables (X). The output layer generates the predicted target variable based on the hyperplane and the support vectors.

### Neural Networks
Neural networks are Machine learning models designed to mimic the functions and structure of the human brain. They are built to recognize patterns. Neural networks consist of interconnected nodes, or neurons, that collaborate and are organized in layers. Neural networks usually involve many processors operating in parallel and arranged in layers. The first layer receives the raw input information. Each successive layer receives the output from the layer preceding it rather than the raw input, the same way neurons further from the optic nerve receive signals from those closer to it. The last layer produces the output of the system.

## Experimental Setup and Results
### Data Pre-processing
The data set was fairly clean but just like most data sets, it needed to be prepared for proper utilization by the machine learning models. Firstly, the rows of the data set with missing values were dropped. This was only three rows. We replaced the zero values under the 'PH Lands' column with the mean of the values in that column since it was completely empty and did not seem to add any value to the data set. Furthermore, the misspelling on the column 'Urban\_leve' was corrected to 'Urban\_level'.

### Exploratory Data Analysis
In this step, a number of plots were generated in order to understand and analyze the trends in the data set. We started with analyzing the correlation of features with the target variable (\textbf{Typh\_per}) with the use of a heatmap.

From the figure above, the features Typh\_Inc and P\_Density have the highest correlation with the target variable representing the number of incidences in a district and the population density respectively. From the above diagram we can select the variables with a positive correlation to be the features that shall be used to make predictions of the target variable.

### Model Selection
Basing on the Regression problem and the models used, we evaluated our models using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R2) score OR Coefficient of Determination in order to select the model with the best scores.

The table below shows the scores for the Linear Regression model, the Decision Tree model, the Random Forest model, The Support Vector Regression model and the Neural Networks regression model. 

| Model              | MSE    | RSME   | MAE    | R2    |
|--------------------|--------|--------|--------|-------|
| Linear Regression  | 8.1e-2 | 2.8e-10| 2.4e-10| 1.0   |
| Decision Tree      | 0.17   | 0.42   | 0.15   | 0.72  |
| Random Forest      | 0.114  | 0.34   | 0.13   | 0.812 |
| SVR                | 0.23   | 0.48   | 0.25   | 0.63  |
| Neural Networks    | 0.111  | 0.33   | 0.22   | 0.817 |

As seen from the table, the Neural Networks Regression model has the most promising results. Despite the excellent results scored by the Linear Regression model, it was not selected as the most optimal model because the R2 score of 1.0 indicates overfitting therefore the model will perform poorly with unseen data.  

### Discussion of Results
As a result of the comparison of models, the Neural Networks Regression model was selected as the most optimal model with a Mean Square root of 0.111, Root mean square root of 0.33, Mean absolute error of 0.22, and an R2 score of 0.817. Hyperparameter tuning will be carried out on this model in order to use the best combination of parameter values to make predictions. After hyperparameter tuning, the model achieved a Mean Square root of 0.001, Root mean square root of 0.03, Mean absolute error of 0.02, and an R2 score of 0.998 which shows a better performance than before. The model was then tested with the actual test data and the final evaluation scores were a Mean Square root of 0.91, Root mean square root of 0.954, Mean absolute error of 0.21, and an R2 score of 0.97 which is a very impressive performance.

## Limitations of the Work
The main limitation of the work is based on the data set.
The meta data that was used to describe the data set was quite ambiguous due to the poor elaboration of column names. Additionally, since the modes of data collection were not specified, we are not certain of the data's level of integrity. Furthermore, the data was collected from the year 2011 to 2017 which is at least seven years ago. Therefore the relevancy of most findings from this data will not be the same as time goes on because of the dynamic nature of the factors that influence the target variable. 

## Conclusion
Conclusively, the relevancy of this work is of great significance due to the fact that Typhoid attacks over 50,000 Ugandans annually despite the inadequate medical facilities. With the use of machine learning, we have the capability to not only analyze but also predict trends in Typhoid cases in Uganda. Such an activity can allow Ugandans to act accordingly in order to control the levels of Typhoid attacks. We compared four regression models that are Linear Regression, the Decision Tree Model, Random Forests, Support Vector Regression model, and Neural Networks. From the comparison, the Neural Networks model has the best metric scores and therefore fits to be the best model for this regression problem.

## Future Works
To sustain the life of our work, we shall address the previously mentioned limitations with the use of up to date data which has a reliable degree of integrity. We plan to do this by finding data sources popular for having a standard quality of data sets such as the World Health Organization. We also plan to use other regression models to further investigate which model best fits this regression problem.

## References
- Oguntimilehin, A., Adetunmbi, A. O., & Abiola, O. B. (2013). A machine learning approach to clinical diagnosis of typhoid fever. A Machine Learning Approach to Clinical Diagnosis of Typhoid Fever, 2(4), 1-6.
- Oguntimilehin, A., Adetunmbi, A. O., & Olatunji, K. A. (2014). A machine learning based clinical decision support system for diagnosis and treatment of typhoid fever. A Machine Learning Based Clinical Decision Support System for Diagnosis and Treatment of Typhoid Fever, 4(6), 1-9.
- Bhuiyan, M. A., Rad, S. S., Johora, F. T., Islam, A., Hossain, M. I., & Khan, A. A. (2023, January). Prediction Of Typhoid Using Machine Learning and ANN Prior To Clinical Test. In 2023 International Conference on Computer Communication and Informatics (ICCCI) (pp. 1-7). IEEE.
