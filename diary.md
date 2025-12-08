## Week 10 (3-5/12/2025)

- Plotted class imbalance pie chart
- Renamed PAY_0 to PAY_1 to avoid the inconsistency in naming
- The dataset contains undocumented values in the Education and Marriage columns
  - Education: 0,5,6 are unlabelled. So, I group them into 4 (Others)
  - Marriage: 0 is unlabelled. So, I group it into 3 (Others)
- Plotted Tie-breaking strategy comparison bar chart
- Plotted tuning curve using seaborn
- Plotted final result comparison bar chart using seaborn
- added print_tree function to debug the tree growth process and confirm the 'Unified Splitting' mechanism is handling mixed data types correctly.
- added presentation file in the documents folder

## Week 10 (2/12/2025)

- Addressed the limitation of using a hardcoded value **(k=5)** in the k-NN algorithm.
- I implemented a search strategy to determine the optimal number of neighbors instead of using an most likely sweet spot.
- Efficiency Strategy: Since running the custom algorithm on the full dataset (30,000 samples) is slow, I implemented `resample` for the tuning phase. I reduced the data to a `3,000-sample` subset, using `stratify` to maintain the correct class balance **(22% default rate)**.
- Implementation: Created a manual `Grid Search` loop to test a list of k integers from `1 to 100`. The system iterates through each value, calculates accuracy, and stores the `best_k`.
- Final Output: Updated the main prediction loop to apply the found `best_k` to the `full X_test` set. This ensures the final results are based on optimized parameters.

## Week 10 (30-1/12/2025)

- Read chapter 6: Algorithm Chains and Pipelines from **Introduction to machine learning with Python: a guide for data scientists by Müller, A.C. and Guido, S.**.
- Learned:
  - **Data Leakage in Preprocessing**: Discovered a critical flaw in the **naive** approach of applying scaling (like MinMaxScaler) to the full dataset before splitting for Cross-Validation. This causes information from the test set (min/max values) to leak into the training process, leading to overly optimistic performance estimates.
  - **The Pipeline Solution**: Learned about the Pipeline class, which chains multiple processing steps (e.g., Scaler → Classifier) into a single object. This ensures that during Cross-Validation, the scaler is "fit" only on the training folds and then applied to the test folds, strictly preventing data leakage.
  - **Grid Searching Pipelines**: Studied how to tune parameters for the entire workflow at once. Using the syntax `step_name__parameter`, GridSearchCV can optimize both the model parameters (like k in k-NN) and preprocessing parameters (like polynomial degrees) simultaneously.

## Week 10 (29/11/2025)

- Read chapter 5: Model Evaluation and Improvement from **Introduction to machine learning with Python: a guide for data scientists by Müller, A.C. and Guido, S.**.
  - Learned:
    - **Cross-Validation**: Understood that a single train-test split can yield misleading results if the split is "lucky" or "unlucky". Learned that **Stratified k-Fold Cross-Validation** is essential for **classification** to ensure each fold maintains the same class proportions as the original dataset.
    - **Grid Search**: Studied how to systematically improve model generalization by **tuning parameters** (k in neighbors) using `GridSearchCV`, which tries all possible combinations of parameters.
    - **Evaluation Metrics**: Explored **Confusion Matrices** to visualize False Positives and False Negatives, and derived metrics like **Precision **(limiting false positives) and **Recall** (avoiding false negatives)

## Week 9 (27/11/2025)

- Learned that accuracy is an inadequate metric when one class is much more frequent than the other (e.g., a 9:1 imbalance), as a dummy classifier predicting the majority class can achieve high accuracy without learning anything.
- Selected the **Default of Credit Card Clients** dataset from the UCI Machine Learning Repository (30,000 samples, 23 features).
- I chose this dataset because it represents a imbalanced binary classification problem (22% Default vs 78% Non-Default), which poses a significant challenge for standard algorithms compared to simple datasets.
- I identified that the features had different scales (e.g, Credit Limit: 500,000 vs. Age: 30). Therefore, I implemented **Min-Max Normalization** to scale all features between 0-1, ensuring that the k-NN distance calculation would not be biased toward high-value features.
- Successfully tested the dataset on 1NN, kNN, and Decision Tree algorithms.
- Achieved accuracy ranging from 76-82%.

## Week 9 (24-25/11/2025)

- Implemented the recursive **create_tree()** method for **Tree Construction**
  - Uses **recursion** to build deeper nodes until a stopping condition is met (Max Depth or Purity).
  - Integrated logic to handle both numerical (binary split) and categorical attributes within the same structure.
- Implemented **predict()** and **traverse_tree()** to walk down the tree and classify new samples.
- Successfully tested the algorithm on the **Iris benchmark dataset**, achieving high accuracy score, verifying that the binary splitting logic works correctly for continuous data.

## Week 8 (22-23/11/2025)

- Implemented Gini Impurity as the measure of uniformity
- Implemented Entropy as the other measure of uniformity
- Successfully implemented the **find_best_split()** method
  - Loops through every feature and every unique value to find potential **thresholds**
  - Implemented **Boolean Masking** to split data into "Left" and "Right" groups based on the condition `value <= threshold`
  - Calculated **Weighted Gini** to compare splits and determine the lowest impurity
- Adopted a full **Object-Oriented Design** to meet project requirements

## Week 8 (20-21/11/2025)

- Learned about **CART algorithm**, Classification and Regression Trees, which can handle classification and regression tasks
- CART classification works by recursively splitting the data into two **binary groups**. At each step, it picks the best feature and threshold to split on, trying to make the resulting nodes as pure as possible
- CART use **Gini Impurity** as the splitting criterion. the lower the gini, the more pure the subset is
- Initially planned to use ID3 (from the textbooks), but realized it struggles with numerical attributes like the Iris dataset because it creates too many branches. So I switched to CART because its **Binary Split** approach handles the numerical attributes like Iris dataset and the categorical/string attributes like Animals dataset with the same logic. This saves me from writing two separate algorithms

## Week 8 (19/11/2025)

- Read **Machine Learning in Action** by Harrington,P chapter 3: Splitting datasets one feature at a time: decision trees.
- Finished section 3.1
- It introduces the fundamental concepts behind constructing decision trees, focusing on how to calculate **Shannon Entropy** (messiness), how to split a dataset into more organized subsets, how to recursively build the tree in Python.
- It also explains that the key to building an effective tree is identifying which feature provides the most **information gain**
- Section 3.2 will be about plotting the trees with Matplotlib

## Week 7 (15/11/2025)

- Read **Induction of Decision Trees** by Quinlan, J.R
- It introduced the **ID3 algorithm**, explaining the step-by-step process of building a Decision Tree
- Learned the precise theory behind the **different impurity measures** and provides the theoretical background for handling practical data issues like noise and missing values

## Week 7 (14/11/2025)

- Created a new issue to test and plot with datasets
- Explore the **UCI repository** for potential datasets
- However, as the **Iris dataset** was already implemented, I decided to finish it first before switching to a new dataset.
- Visualised the **1NN algorithm** results using the Iris dataset
  - Implemented a **side-by-side scatter plot** to compare **True Labels vs. Predicted Labels**
- Tested the **kNN algorithm** on the Iris dataset
- implemented a **bar plot** comparing the accuracy per species(Setosa, Versicolor, Virginica)

## Week 6 (07/11/2025)

- Implemented a new tie-breaking strategy
- It resolves a k-NN voting tie by selecting the label of the single neighbor that is closest to the test point among the tied classes.
- **Counter** code implements this because it counts the labels which are already sorted by distance, and most_common(1) picks the one it saw first when the vote counts are tied.

## Week 6 (03/11/2025)

- Implemented 1NN Conformal Predictor that calculate the alpha = d_diff/d_same (distance to nearest different-class neighbor / distance to nearest same-class neighbor)
- Use this conformity score function to calculate p-values and avearage false p-value
- Will function it with the actual dataset in week 7

## Week 5 (01/10/2025)

- Implemented kNN algorithm and tested with artificial simple dataset.
- Handled ties in kNN.
- Implemented tie-breaking by choosing randomly and alphabetically.

## Week 5 (27/10/2025)

- Load and test the 1NN algorithm with Iris dataset.
- Evulate the accuracy score of the dataset to examine the model's performance.
- Tested with new unlabeled data.

## Week 4 (24/10/2025)

- Implemented 1NN algorithm from scratch using simple artificial dataset.

## Week 4 (23/10/2025)

- Finshed reading Chapter 1 and 2 of **Machine Learning in Action** by Peter Harrington which explained kNN algorithm concept, impelmentation details and testing with datasets.
- Gained understanding of how to evaluate kNN using simple datasets and measure prediction accuracy.

## Week 3 (14/10/2025)

- Finished reading chapter 1,2 of **Introduction to machine learning with Python** by Müller, A.C. and Guido, S.
- Chapter 1 explains the basic concept of machine learning.
- Chapter 2 explains the difference between supervised and unsupervised learning, classification and regression algorithms like k-Nearest Neighbors, Linear Models, Support Vector Machines(SVM), and Decision Trees.
- Learned the foundations of ML and how algorithms are categorized, how to implement and evaluate basic models in scikit-learn

## Week 2 (9/10/2025)

- Added Timeline and Risk and Mitigation in **FYP Plan**
- Submitted the FYP Plan

## Week 2 (08/10/2025)

- Started writing Abstract part of **FYP Plan**
- Added the references

## Week 2 (06/10/2025)

- Gather more reading sources
- Started reading the **Machine Learning in Action by Harriington**

## Week 1 (03/09/2025)

- Studied the Week 2 Machine Learning lecture slides
- Continued reading the **Introduction to machine learing with Python** book

## Week 1 (29/09/2025)

- Read the introductory section (1 and half hour worth) of **Introduction to machine learning with Python** to have the better understanding of Nearest Neighbours
- Wrote a draft plan for tomorrow's first meeting with my supervisior
- Prepared questions to ask during the first meeting
- Added **Elements of Statistical Learning: Data Mining, Inference, and Prediction by Hastie, T., Tibshirani, R. and Friedman, J** to the reading list

## Week 1 (28/09/2025)

- Read the project description thoroughly
- Read the BSc Project Booklet
- Added **Introduction to machine learning with Python: a guide for data scientists by Müller, A.C. and Guido, S.** to the reading list
