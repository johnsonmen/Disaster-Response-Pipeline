import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sqlite3

def load_data(messages_filepath, categories_filepath):
    '''
    load data from 2 csv files
    input:
        messages_filepath: path to messages csv file
        categories_filepath: path to categories csv file
    output:
        df: merged dataframe
    '''
    # load
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge
    df = messages.merge(categories, on='id', how='outer')

    # split categories into separate category columns
    categories = df['categories'].str.split(';', expand=True)

    # rename columns of categories
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
        
    # convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
    
    # replace categories column in df with new category columns
    df.drop(columns=['categories'], axis=1, inplace=True)
    df = pd.concat([df,categories],axis=1)

    return df

def clean_data(df):
    # remove duplicates
    df.drop_duplicates(subset= 'id', inplace=True)

    return df

def save_data(df, database_filename):
    '''
    save data to a database
    input:
        df: dataframe
        database_filename: database name
    output:
        None
    '''
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
