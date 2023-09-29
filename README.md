# Disaster Response Pipeline Project
### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Instructions](#instruction)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

The Disaster Response Pipeline project was built using Python 3.9.16

## Project Motivation<a name="motivation"></a>

The project is an analysis of the disaster data from [Appen](https://www.figure-eight.com/) (formally Figure 8) to build a model for an API that classifies disaster messages.

Your project will include a web app where an emergency worker can input a new message and get classification results in several categories.

## File Descriptions <a name="files"></a>
 The files structure and brief description can be found below:

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md


## Project Components

#### 1. ETL Pipeline
In a Python script `process_data.py`, write a data cleaning pipeline that:

- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

#### 2. ML Pipeline
In a Python script `train_classifier.py` , write a machine learning pipeline that:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

#### 3. Flask Web App
project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

#### Features

1. University Rank
2. Name of University
3. Location
4. No of student
5. No of student per staff
6. International Student
7. Female :Male Ratio
8. OverAll Score
9. Teaching Score
10. Research Score
11. Citations Score
12. Industry Income Score
13. International Outlook Score

Details of the dataset can be found [here](https://www.timeshighereducation.com/student/advice/world-university-rankings-explained)

## Results<a name="results"></a>

The main findings of the code can be found at the Medium blog available [here](https://medium.com/@kimjohnson9906/what-is-a-good-university-to-you-5f040a195b01).

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

The credit should be given to the collaborators for the data. You can find the Licensing for the data and other descriptive information at the Kaggle link available [here](https://www.kaggle.com/datasets/alitaqi000/world-university-rankings-2023) or the source [here][https://www.timeshighereducation.com/world-university-rankings/2023/world-ranking]
### Instructions <a name="instrution"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory

3. Run your the app: `python run.py`

4. Go to http://0.0.0.0:3001/
