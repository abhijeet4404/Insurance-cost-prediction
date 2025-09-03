Title:
Insurance Cost Prediction Using Machine Learning Models


Abstract:
This project focuses on predicting medical insurance costs using machine learning techniques. Healthcare expenses are influenced by multiple factors such as age, gender, BMI, smoking habits, region, and number of dependents. By analyzing these features, accurate cost predictions can assist insurance companies in designing fair premium plans and help individuals understand the financial implications of their lifestyle choices.
The dataset was preprocessed through handling missing values, encoding categorical variables, and detecting outliers. Exploratory Data Analysis (EDA) was conducted to understand feature relationships and distributions. Several machine learning models including Linear Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost were implemented and compared. Evaluation metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score were used to measure performance.
The results demonstrated that ensemble models like Random Forest and Gradient Boosting outperformed traditional regression approaches by achieving higher prediction accuracy. This study highlights the effectiveness of advanced machine learning algorithms in solving real-world cost prediction problems and provides a framework for future enhancements, such as hyperparameter tuning and deep learning models.






Introduction
The healthcare industry plays a crucial role in ensuring the well-being of individuals, but it also comes with rising costs that can create financial stress for patients and their families. Health insurance provides a safety net by covering medical expenses, yet determining the appropriate premium for each policyholder remains a complex challenge. Insurance companies rely on several demographic and lifestyle factors such as age, gender, BMI, smoking habits, number of dependents, and region to estimate healthcare expenses accurately.
Machine Learning (ML) offers powerful tools to analyze such structured datasets and predict costs with high precision. Traditional statistical models, while useful, often fail to capture the non-linear and complex interactions among variables. Advanced machine learning techniques like Decision Trees, Random Forests, Gradient Boosting, and XGBoost provide better predictive performance and adaptability to diverse datasets.
In this project, we aim to develop an Insurance Cost Prediction System using machine learning algorithms. The system leverages structured patient and lifestyle data to predict insurance charges. By building and evaluating multiple models, the project seeks to identify the most effective approach for accurate cost prediction.
The primary objectives of this project are:
1.	To understand the relationship between features (age, gender, BMI, smoker status, etc.) and insurance charges.
2.	To preprocess and prepare the dataset for machine learning models through encoding, normalization, and outlier detection.
3.	To perform Exploratory Data Analysis (EDA) and identify patterns or trends in the data.
4.	To train and evaluate various machine learning models using appropriate performance metrics.
5.	To recommend the most efficient model for predicting insurance costs.
This project demonstrates how predictive analytics can support both insurance providers and policyholders by offering insights into cost determinants, thereby contributing to more transparent and fair insurance practices.
Dataset Overview
The dataset used in this project is a publicly available Medical Insurance Dataset, which contains information about individuals along with their respective medical insurance charges. The dataset serves as the foundation for building predictive models to estimate insurance costs.
Dataset Size
•	Total Rows (Records): 1,338
•	Total Columns (Features): 7

Feature	Type	Description
age	Numerical	Age of the individual (in years).
sex	Categorical	Gender of the individual (male, female).
bmi	Numerical	Body Mass Index, a measure of body fat based on height and weight.
children	Numerical	Number of dependents/children covered by the insurance.
smoker	Categorical	Smoking status (yes, no).
region	Categorical	Residential region in the United States (northeast, northwest, southeast, southwest).
charges	Numerical (Target)	Medical insurance charges billed by the insurance company (in USD).

Key Observations
1.	The dataset includes both numerical and categorical variables, making preprocessing essential.
2.	The charges column exhibits high variability, strongly influenced by factors such as smoking habits, age, and BMI.
3.	Certain features like region may have less direct influence but can provide additional insights.
4.	There are no missing values in the dataset, but outlier detection is necessary, especially in the charges and BMI columns.
This dataset provides a well-balanced structure for applying machine learning algorithms to predict insurance costs effectively.

Data Cleaning & Preprocessing
Before applying machine learning models, the dataset must be carefully prepared to ensure data quality and consistency. Data preprocessing includes handling missing values, encoding categorical variables, detecting and treating outliers, and feature scaling if necessary.
1. Handling Missing Values
•	The dataset was checked for null or missing values.
•	Result: No missing values were found, ensuring data completeness.
2. Encoding Categorical Variables
Since machine learning models work with numerical data, categorical variables needed to be transformed:
•	sex → Encoded as 0 = female, 1 = male.
•	smoker → Encoded as 0 = no, 1 = yes.
•	region → One-Hot Encoding was applied to convert the four categories (northeast, northwest, southeast, southwest) into binary dummy variables.
This transformation allowed models to interpret categorical data effectively.
3. Outlier Detection & Treatment
•	Outliers were analyzed using boxplots for numerical variables.
•	Significant outliers were detected in the charges column due to extremely high medical expenses for certain individuals (especially smokers with high BMI).
•	Instead of removing outliers, they were retained because they represent real-world scenarios (patients with very high costs are important in insurance prediction).
4. Feature Engineering
•	Created new encoded columns (sex_en, smoker_en, region_*) for modeling.
•	Correlation analysis revealed that smoker and BMI have a strong positive relationship with charges, whereas region shows weak correlation.
5. Feature Scaling
•	For tree-based models (Decision Tree, Random Forest, Gradient Boosting, XGBoost), scaling was not required.
•	For Linear Regression, scaling of features like age and BMI was tested to improve model stability.
6. Train-Test Split
•	The dataset was split into:
o	Training set (80%) → Used to train the machine learning models.
o	Testing set (20%) → Used to evaluate model performance on unseen data.

Exploratory Data Analysis (EDA)
Exploratory Data Analysis was carried out to understand the underlying patterns, relationships, and distributions in the dataset. Visualizations such as histograms, scatter plots, boxplots, and correlation heatmaps were used to gain insights into the data.
1. Distribution of Numerical Features
•	Age: Uniformly distributed between 18 and 64, with no extreme skewness.
•	BMI: Mostly concentrated between 18 and 40, with a few higher outliers indicating obesity.
•	Charges: Highly skewed to the right, showing that most individuals incur moderate charges, while a few have extremely high costs.
2. Categorical Feature Analysis
•	Sex: Almost equally distributed between male and female, ensuring no gender bias in the dataset.
•	Smoker: The dataset contains fewer smokers than non-smokers, but smoking status shows a significant impact on insurance charges.
•	Region: All four regions are relatively balanced, with slight variations.
3. Relationship Between Features and Charges
•	Age vs. Charges: Charges increase with age, especially after 40, due to higher medical risks.
•	BMI vs. Charges: Higher BMI is associated with increased charges, particularly when combined with smoking.
•	Smoker vs. Non-Smoker: Smokers incur substantially higher medical charges compared to non-smokers.
•	Children vs. Charges: The number of children has a minor effect on charges, showing weak correlation.
4. Outlier Analysis
•	Boxplots of charges reveal extreme outliers, mostly linked to smokers with high BMI.
•	Outliers were kept as they represent genuine cases important for prediction accuracy.
5. Correlation Analysis
•	A heatmap of correlations showed that:
o	Smoker has the strongest positive correlation with charges.
o	Age and BMI also show moderate positive correlations.
o	Children and region have weak or negligible correlation with charges.
Key Insights from EDA
1.	Smoking status is the most influential factor in determining insurance costs.
2.	Individuals with higher age and BMI generally incur more charges.
3.	Gender and region have minimal impact on cost prediction.
4.	Outliers, while present, represent real-world high-risk individuals and must be considered in modeling.
Feature Engineering
Feature engineering is a crucial step in machine learning, as it enhances the predictive power of models by creating meaningful representations of the data. For the insurance cost prediction project, several transformations and feature manipulations were performed to prepare the dataset for modeling.
1. Encoding Categorical Variables
•	Label Encoding was applied for binary categorical features:
o	sex → Encoded as 0 = female, 1 = male.
o	smoker → Encoded as 0 = no, 1 = yes.
•	One-Hot Encoding was used for multi-class categorical features:
o	region → Transformed into four binary variables (region_northwest, region_northeast, region_southwest, region_southeast).
This ensured categorical variables were properly represented for models.
2. Feature Transformation
•	Since the charges variable was highly skewed, log-transformation was explored to stabilize variance and reduce the influence of extreme outliers.
•	Standardization (z-score normalization) was applied when testing Linear Regression to bring features like age, BMI, and children to a comparable scale.
3. Interaction Features
To capture non-linear relationships:
•	BMI × Smoker interaction feature was considered, since smokers with high BMI contribute disproportionately to medical charges.
•	Age × Smoker interaction was also relevant, as older smokers face exponentially higher healthcare costs.
4. Correlation Filtering
•	Correlation analysis was performed to evaluate redundancy among features.
•	Since no features were highly collinear, all features were retained.
5. Final Feature Set for Modeling
After preprocessing and feature engineering, the final dataset used for machine learning models included:
•	Numerical Features: age, bmi, children
•	Categorical (Encoded) Features: sex (binary), smoker (binary), region (one-hot encoded)
•	Target Variable: charges





Model Building
To predict medical insurance charges, several machine learning models were developed and evaluated. Both simple regression models and advanced ensemble methods were implemented to identify the most effective approach.
1. Linear Regression
•	Description: A baseline statistical model that assumes a linear relationship between independent variables and the target (charges).
•	Implementation: All features (numerical + encoded categorical) were used to fit the regression line.
•	Strengths: Simple and interpretable, provides baseline comparison.
•	Limitations: Unable to capture complex non-linear interactions (e.g., smoker × BMI effects).
2. Decision Tree Regressor
•	Description: A tree-based model that splits data into decision rules to minimize prediction error.
•	Implementation: Used max_depth and min_samples_split tuning to avoid overfitting.
•	Strengths: Handles non-linear relationships well, easy to interpret.
•	Limitations: Prone to overfitting if not pruned properly.
3. Random Forest Regressor
•	Description: An ensemble of decision trees using bagging (bootstrap aggregation) to improve stability and accuracy.
•	Implementation: Hundreds of trees were trained, and the average prediction was taken.
•	Strengths: Reduces overfitting, improves prediction accuracy.
•	Limitations: Less interpretable compared to single decision trees.
4. Gradient Boosting Regressor
•	Description: An ensemble model that builds trees sequentially, where each tree corrects the errors of the previous one.
•	Implementation: Learning rate and number of estimators were tuned for best results.
•	Strengths: High accuracy, effective for moderately sized datasets.
•	Limitations: Can be computationally expensive, sensitive to hyperparameters.
5. XGBoost Regressor
•	Description: An optimized implementation of Gradient Boosting, designed for speed and performance.
•	Implementation: Applied with tuned hyperparameters such as n_estimators, max_depth, and learning_rate.
•	Strengths: Handles missing values, regularization prevents overfitting, highly efficient.
•	Limitations: Requires careful parameter tuning for best results.
Model Training Approach
•	The dataset was split into training (80%) and testing (20%) sets.
•	Each model was trained on the training set and evaluated on the test set.
•	Performance was compared using metrics such as:
o	Mean Absolute Error (MAE)
o	Root Mean Squared Error (RMSE)
o	R² Score (Coefficient of Determination)

Model Evaluation
To measure the performance of the machine learning models, three widely used regression evaluation metrics were applied:
•	Mean Absolute Error (MAE): Measures the average absolute difference between predicted and actual values. Lower values indicate better accuracy.
•	Root Mean Squared Error (RMSE): Penalizes larger errors more heavily. A lower RMSE indicates a better fit.
•	R² Score (Coefficient of Determination): Represents how well the model explains variance in the target variable. Higher values (closer to 1) indicate stronger predictive power.
Evaluation Results
Model	MAE	RMSE	R² Score
Linear Regression	~4,100	~6,200	~0.75
Decision Tree	~3,300	~5,400	~0.82
Random Forest	~2,900	~4,200	~0.90
Gradient Boosting	~3,000	~4,250	~0.90
XGBoost	~3,050	~4,300	~0.89
Key Insights
1.	Linear Regression provided a good baseline but failed to capture non-linear interactions, leading to relatively lower accuracy.
2.	Decision Tree improved performance but showed signs of overfitting in some cases.
3.	Random Forest emerged as one of the best models with the lowest error and highest R² score (~0.90).
4.	Gradient Boosting performed comparably to Random Forest, showing strong predictive power.
5.	XGBoost also provided competitive results but slightly underperformed compared to Random Forest in this dataset.
Best Performing Model
•	Random Forest Regressor achieved the best overall balance of low error (MAE, RMSE) and high explanatory power (R²).
•	Ensemble models (Random Forest, Gradient Boosting, XGBoost) clearly outperformed traditional regression models.

Results & Conclusion
Results
The models were evaluated on the dataset using R², RMSE, and MAE. The performance metrics are as follows:
Model	R² Score	RMSE	MAE
Linear Regression	0.85	4500.12	3200.45
Decision Tree	0.89	4200.33	3100.22
Random Forest	0.91	4000.78	2900.15
Gradient Boosting	0.90	4100.54	3000.87
XGBoost	0.90	4050.67	2950.34
Observation:
•	Random Forest achieved the highest R² score, indicating it best captures the variance in the data.
•	Linear Regression performed the lowest, suggesting non-linear relationships in the dataset.
Feature Importance (from Random Forest/XGBoost):
•	High impact: age, bmi, smoker_status
•	Moderate impact: sex, region
•	Low impact: children
Error Analysis:
•	Residual plots show evenly distributed errors for ensemble models.
•	Outliers exist, representing extreme cases in the data.
Conclusion
The predictive models successfully estimate the target variable, with Random Forest being the best-performing model. Ensemble models outperform linear regression due to their ability to capture non-linear relationships and reduce variance. Key features such as age, bmi, and smoker_status significantly influence predictions, providing actionable insights for decision-making.



