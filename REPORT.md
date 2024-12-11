# Report

> [!WARNING]
> This report is a work in progress and is subject to change throughout the development
> of the group project. If there are any errors or inaccuracies we have failed to 
> address, please change and report them accordingly.

This report is intended to provide a comprehensive overview of all sections of the group project. For
reference, the project is titled "Theft Over Open Data - Analysis and Visualization", and the following
sections are included within the report in their respective definitions (following the evaluation for 
this group project):
1. Data Exploration
2. Data Modelling
3. Predictive Model Building
4. Model Scoring and Evaluation
5. Model Deployment

The report is structured as follows:
1. `Introduction & Summary` - This section will provide an overview of the project, including the
   purpose, goals, and objectives of the project.
2. `Data Exploration`, `Data Modelling Process` - This section will provide a detailed analysis of 
    the data, including but not limited to the following:
    - Data sources used and exploration of the data.
    - Data cleaning and preprocessing.
    - Data transformation and feature engineering techniques.
    - Data modeling and feature selection.
3. `Predictive Model Building`, `Model Training and Evaluation`, `Model Deployment` - This section
   will provide a detailed analysis of the predictive model, including but not limited to the following:
    - Model selection and training.
    - Model evaluation and scoring.
    - Model deployment and deployment considerations.
4. `Bibliography` - This section will provide a list of references used in the report.
5. `Appendix` - This section will provide additional information and insights not covered in the
   report.
6. `Conclusion` - This section will provide a summary of the report and the findings.

## Summary

THis project, titled "Theft Over Open Data - Analysis and Visualization", aims to explore theft-related
incidents using publicly available datasets. The primary objective / goals are:
- Perform comprehensive data exploration with provided visualizations.
- Build predictive models that identify key patterns and trends in theft incidents.
- Provide actionable insights and recommendations for improving theft prevention and response.
- Develop a model that can be deployed with a frontend and backend.

### Executive Summary (Brief)

In this project, we utilized the `Toronto Theft Over` dataset from the Toronto Police Service Public Safety Data Portal 
to analyze and gain insights into theft trends in Toronto from 2014 to 2023. The dataset contained information regardin
the location where the offence occured based on `OCC_YEAR`, `OCC_MONTH`, `OCC_DAY`, and various other features. Our goal
with this project was to create a predictive model that could predict what type of crime would occur, given certain information
passed to the model as input.

## Columnar Analysis and Visualization

| Column Name       | Data Type | Column Description                             | Missing Values |
|-------------------|-----------|------------------------------------------------|----------------|
| OBJECTID          | Numeric   | Unique identifier for each record              | 0              |
| EVENT_UNIQUE_ID   | Object    | Unique identifier for each theft event         | 0              |
| REPORT_DATE       | Object    | Date and time when the theft was reported      | 0              |
| OCC_DATE          | Object    | Date and time when the theft occurred          | 0              |
| REPORT_YEAR       | Numeric   | Year when the theft was reported               | 0              |
| REPORT_MONTH      | Object    | Month when the theft was reported              | 0              |
| REPORT_DAY        | Numeric   | Day of the month when theft was reported       | 0              |
| REPORT_DOY        | Numeric   | Day of the year when theft was reported        | 0              |
| REPORT_DOW        | Object    | Day of the week when theft was reported        | 0              |
| REPORT_HOUR       | Numeric   | Hour when the theft was reported               | 0              |
| OCC_YEAR          | Numeric   | Year when the theft occurred                   | 4              |
| OCC_MONTH         | Object    | Month when the theft occurred                  | 4              |
| OCC_DAY           | Numeric   | Day of the month when theft occurred           | 4              |
| OCC_DOY           | Numeric   | Day of the year when theft occurred            | 4              |
| OCC_DOW           | Object    | Day of the week when theft occurred            | 4              |
| OCC_HOUR          | Numeric   | Hour when the theft occurred                   | 0              |
| DIVISION          | Object    | Police division that was notified of the theft | 0              |
| LOCATION_TYPE     | Object    | Type of location where theft occurred          | 0              |
| PREMISES_TYPE     | Object    | Type of premises where theft occurred          | 0              |
| UCR_CODE          | Numeric   | Uniform Crime Reporting code                   | 0              |
| UCR_EXT           | Numeric   | UCR extension code                             | 0              |
| OFFENCE           | Object    | Description of the theft offense (categorized) | 0              |
| MCI_CATEGORY      | Object    | Major Crime Indicator category                 | 0              |
| HOOD_158          | Object    | Hood 158 identifier (newer)                    | 0              |
| NEIGHBOURHOOD_158 | Object    | Neighbourhood name (158 areas)                 | 0              |
| HOOD_140          | Object    | Hood 140 identifier (older)                    | 0              |
| NEIGHBOURHOOD_140 | Object    | Neighbourhood name (140 areas)                 | 0              |
| LONG_WGS84        | Numeric   | Longitude coordinate (WGS84)                   | 0              |
| LAT_WGS84         | Numeric   | Latitude coordinate (WGS84)                    | 0              |
| x                 | Numeric   | X coordinate                                   | 0              |
| y                 | Numeric   | Y coordinate                                   | 0              |

## Types of Visualizations

Here are some of the visualizations you might see throughout this project, due note that these are most
commonly used in the data exploration stage of the project (which is stored in an .ipynb file):
1. **Bar Plots/Charts**: Illustrating the imbalances in the data, and exploration of relationships between potential features.
2. **Heatmaps**: Visualizing the relationship between various features, and their potential distributions / correlations.
3. **Scatter Plots**: Rough exploration of everything in the dataset.
4. and various other visualizations...

## Models and Model Evaluation

We used a variety of models to explore the data, with various techniques. For techniques used, we employed
SMOTE (Synthetic Minority Over-sampling Technique) to balance the data, and also SMOTENC (SMOTE for Numerical
and Categorical Variables) to balance the dataset.

For our models we used the following:
1. Logistic Regression (SMOTE, SMOTENC)
2. DecisionTreeClassifier (SMOTE, SMOTENC)
3. RandomForestClassifier (SMOTE, SMOTENC)

To see the results of our models, and the evaluation of the models, please refer to the `c309_r2_toodu_model.ipynb` notebook 
in the `server` folder for more information.

## Bibliography

1. Theft Over Open Data - https://data.torontopolice.on.ca/datasets/7530d9b637c340059ccb81a782481c04_0/explore?location=43.696813%2C-79.323493%2C10.36
2. Pandas - https://pandas.pydata.org/
3. Seaborn - https://seaborn.pydata.org/
4. Matplotlib - https://matplotlib.org/
5. Scikit-learn - https://scikit-learn.org/stable/
6. Numpy - https://numpy.org/