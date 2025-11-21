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

- Finished reading chapter 1,2,3 of **Introduction to machine learning with Python** by Müller, A.C. and Guido, S.
- Chapter 1 explains the basic concept of machine learning.
- Chapter 2 explains the difference between supervised and unsupervised learning, classification and regression algorithms like k-Nearest Neighbors, Linear Models, Support Vector Machines(SVM), and Decision Trees.
- Chapter 3 explains algorithms such as k-Means clustering and Principal Component Analysis(PCA) for data visualisation and dimensionality reduction, and covers data preprocessing techniques.
- Learned the foundations of ML and how algorithms are categorized, how to implement and evaluate basic models in scikit-learn, and the importance of data preparation and model evaluation for accurate ML results.

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
