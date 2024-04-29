import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import pycountry
import mplcursors
import seaborn as sns

def choose_chart(data):
    #verify if data is categorical and also numerical
    if dataset_is_categories_and_numeric_values(data):
        #verify if data is time series
        if dataset_is_time_series_data(data):
            #verify if data containe one set of numerical data or several
            if dataset_is_one_numiric(data):
                return area(data)
            else:
                return line(data)
        elif dataset_has_country_data(data):
            if dataset_is_one_numiric(data):
                return 'map with values'
        #verify if data containe one set of categorical data or several
        elif dataset_is_one_categorie(data):
            #verify if data containe one set of numerical values or several
            if dataset_is_one_numiric(data):
                #verify if categorical data have less then 6 values and have few similaire values
                if dataset_has_few_categories(data) and dataset_has_few_similaire_values(data):
                    return pie(data)
                #verify if categorical data have more then 6 values and have few similaire values
                elif not dataset_has_few_categories(data) and dataset_has_few_similaire_values(data):
                    return donut(data)
                else:
                    return bar(data)
            elif dataset_is_two_numiric(data):
                #verify if categorical data don't have any repeted values
                if dataset_has_no_duplicate_values(data) and dataset_has_less_then_4_categories(data):
                    return grouped_bar(data)
                else:
                    return 'box plot'
            elif dataset_is_three_numiric(data):
                if dataset_has_no_duplicate_values(data) and dataset_has_less_then_4_categories(data):
                    return grouped_bar(data)
                return bubble_one_cat(data)
            elif dataset_is_several_numiric(data):
                #verify if categorical data don't have any repeted values
                if dataset_has_no_duplicate_values(data):
                    #verify if categorical data have less then 4 values
                    if dataset_has_less_then_4_categories(data):
                        return 'radar chart'
                    heatmap(data)
                else:
                    return 'box plot'
        elif dataset_is_two_categorie(data):
            #verify if data containe one set of numerical values or several
            if dataset_is_one_numiric(data):
                #verify if data have categories and subcategories
                if dataset_has_sub_groups(data):
                    return 'treemap'
            elif dataset_is_three_numiric(data):
                return 'bubble chart'
        elif dataset_is_several_categorie(data):
            #verify if data containe one set of numerical values or several
            if dataset_is_one_numiric(data):
                #verify if data have categories and subcategories
                if dataset_has_sub_groups(data):
                    return 'sunburst chart'
    #verify if data containe only numerical data
    elif dataset_is_numeric(data) and not dataset_is_categorical(data):
        #verify if data containe one set of numerical values or several
        if dataset_is_one_numiric(data):
            return histogram(data,data.columns[0])
        elif dataset_is_two_numiric(data):
            #verify if data have many values more then 400 data 
            if dataset_has_many_point(data):
                return histogram(data,data.columns[0],data.columns[1])
            return scatter(data,data.columns[0],data.columns[1])
        elif dataset_is_three_numiric(data):
            return bubble(data)
    #verify if data containe only categorical data
    elif not dataset_is_numeric(data) and dataset_is_categorical(data):
        # return parallelCoordinates(data)
        return 'parallelCoordinates'
    return 'error'
    # return choose_chart(remove_least_important_column(data))
        
def histogram(data,xlabel="Value",ylabel="Frequency"):
    num_bins = int(np.sqrt(len(data)))
    plt.hist(data, bins=num_bins, edgecolor='black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    return 'histogram'
        
def scatter(data,xlabel="Value",ylabel="Frequency"):
    plt.scatter(data[xlabel], data[ylabel], alpha=0.5)
    plt.show()  
    return 'scatter plot'
        
def heatmap(data):
    categorie_column = data.select_dtypes(include=['object']).columns[0]
    data.set_index(categorie_column, inplace=True)
    sns.heatmap(data, cmap="Blues", annot=True, square=False,  linewidth = 1)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()  
    return 'scatter plot'

def bubble(data):
    numeric_columns = data.select_dtypes(include=['number']).columns
    x=data[numeric_columns[0]]
    y=data[numeric_columns[1]]
    sizes=data[numeric_columns[2]]
    colors = np.random.rand(len(x))
    plt.scatter(x, y, s=sizes, c=colors, alpha=0.5)

    plt.xlabel(numeric_columns[0])
    plt.ylabel(numeric_columns[1])
    plt.colorbar(label="Bubble sizes")

    plt.show()
    return 'bubble chart N'

def bubble_one_cat(data):
    numeric_columns = data.select_dtypes(include=['number']).columns
    categorie_column = data.select_dtypes(include=['object']).columns[0]

    labels=data[categorie_column]
    x=data[numeric_columns[0]]
    y=data[numeric_columns[1]]
    sizes=data[numeric_columns[2]]

    colors = np.random.rand(len(x))
    plt.scatter(x, y, s=sizes, c=colors, alpha=0.5)

    plt.xlabel(numeric_columns[0])
    plt.ylabel(numeric_columns[1])
    plt.colorbar(label="Bubble sizes")
    mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(labels[sel.target.index]))
    plt.show()
    return 'bubble chart'

def line(data):
    categorical_columns = data.select_dtypes(include=['object']).columns
    numeric_columns = data.select_dtypes(include=['number']).columns
    
    for cat_col in categorical_columns:
        for num_col in numeric_columns:
            plt.plot(data[cat_col], data[num_col], label=num_col)
    plt.xlabel(categorical_columns[0])
    if len(data[categorical_columns[0]]) > 8 or any(len(str(label)) > 7 for label in data[categorical_columns[0]]):
        plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.legend()
    plt.show()
    return 'line chart'

def area(data):
    numeric_columns = data.select_dtypes(include=['number']).columns
    categorical_columns = data.select_dtypes(include=['object']).columns
    plt.fill_between(data[categorical_columns[0]], data[numeric_columns[0]], color="skyblue", alpha=0.4)
    plt.plot(data[categorical_columns[0]],data[numeric_columns[0]], color="Slateblue", alpha=0.6, linewidth=2)
    plt.xlabel(categorical_columns[0])
    plt.ylabel(numeric_columns[0])
    if len(data[categorical_columns[0]]) > 8 or any(len(str(label)) > 7 for label in data[categorical_columns[0]]):
        plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.xlim(data[categorical_columns[0]].min(), data[categorical_columns[0]].max())  # Adjust as needed for x-axis
    plt.ylim(0, data[numeric_columns[0]].max()) 
    plt.show()
    return 'area chart'

def pie(data):
    categorical_columns = data.select_dtypes(include=['object']).columns
    numeric_columns = data.select_dtypes(include=['number']).columns
    num_data = data[numeric_columns[0]]
    labels = data[categorical_columns[0]]
    plt.pie(num_data,labels=labels,autopct='%1.1f%%')
    plt.show()  
    
    return 'pie chart'

def bar(data):
    categorical_columns = data.select_dtypes(include=['object']).columns
    numeric_columns = data.select_dtypes(include=['number']).columns
    num_data = data[numeric_columns[0]]
    labels = data[categorical_columns[0]]
    plt.bar(labels,num_data)
    plt.ylabel(numeric_columns[0])
    if len(labels) > 5 or any(len(str(label)) > 10 for label in labels):
        plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    return 'bar chart'

def grouped_bar(data):
    categorical_columns = data.select_dtypes(include=['object']).columns
    numeric_columns = data.select_dtypes(include=['number']).columns
    
    categories = data[categorical_columns[0]]
    num_data = data[numeric_columns]

    num_bars = len(numeric_columns)
    bar_width = 0.35
    index = np.arange(len(categories))
    opacity = 0.8
    
    for i, col in enumerate(numeric_columns):
        plt.bar(index + i * bar_width, num_data[col], bar_width, alpha=opacity, label=col)
    
    plt.xlabel(categorical_columns[0])
    plt.ylabel("Values")
    plt.xticks(index + bar_width, categories)
    
    if len(categories) > 5 or any(len(str(label)) > 10 for label in categories):
        plt.xticks(rotation=45, ha='right')
    
    plt.legend()
    plt.tight_layout()
    plt.show()


def donut(data):
    categorical_columns = data.select_dtypes(include=['object']).columns
    numeric_columns = data.select_dtypes(include=['number']).columns
    num_data = data[numeric_columns[0]]
    labels = data[categorical_columns[0]]
    plt.pie(num_data, labels=labels,
        autopct='%1.1f%%', pctdistance=0.85)
    centre_circle = plt.Circle((0, 0), 0.65, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.show()  
    return 'donut chart'

def dataset_has_sub_groups(data):
    unique_combinations = data.groupby(list(data.columns)).size().reset_index().rename(columns={0:'count'})
    if len(unique_combinations) > 1:
        return True
    else:
        return False

def dataset_has_no_duplicate_values(data):
    categorical_columns = data.select_dtypes(include=['object']).columns
    return len(set(data[categorical_columns[0]]))==len(data[categorical_columns[0]])


def IsTimeOrDate(input_str):
    formats = [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%m/%d/%Y %I:%M:%S %p",
        "%d/%m/%Y %I:%M:%S %p",
        "%m-%d-%Y %I:%M:%S %p",
        "%d-%m-%Y %I:%M:%S %p",
    ]

    for fmt in formats:
        try:
            datetime.strptime(input_str, fmt)
            return True
        except ValueError:
            pass
    return False

def dataset_is_time_series_data(data):
    for col in data.columns:
        is_date_or_time_column = True
        for value in data[col]:
            if not IsTimeOrDate(str(value)):
                is_date_or_time_column = False
                break
        if is_date_or_time_column:
            return True
    return False

def is_country(name):
    country_names = set(country.name for country in pycountry.countries)
    return name in country_names

def is_column_countries(column_values):
    for value in column_values:
        if not is_country(value):
            return False
    return True

def dataset_has_country_data(data):
    categorical_columns = [col for col in data.columns if data[col].dtype == 'object']
    for col in categorical_columns:
        if len(data[col]) == len(set(data[col])) and is_column_countries(data[col]):
            return True
    return False

def dataset_has_many_point(data):
    num_rows, num_columns = data.shape  
    return (num_rows > 400)

def dataset_is_categories_and_numeric_values(data):
    categorical_columns = data.select_dtypes(include=['object']).columns
    numeric_columns = data.select_dtypes(include=['number']).columns
    
    return len(categorical_columns) > 0 and len(numeric_columns) > 0

def dataset_is_one_numiric(data):
    numeric_columns = data.select_dtypes(include=['number']).columns
    
    return len(numeric_columns) == 1

def dataset_is_two_numiric(data):
    numeric_columns = data.select_dtypes(include=['number']).columns
    
    return len(numeric_columns) == 2

def dataset_is_three_numiric(data):
    numeric_columns = data.select_dtypes(include=['number']).columns
    
    return len(numeric_columns) == 3

def dataset_is_several_numiric(data):
    numeric_columns = data.select_dtypes(include=['number']).columns
    
    return len(numeric_columns) > 3

def dataset_has_more_then_one_numiric(data):
    numeric_columns = data.select_dtypes(include=['number']).columns
    
    return len(numeric_columns) > 1

def dataset_is_one_categorie(data):
    categorical_columns = data.select_dtypes(include=['object']).columns
    
    return len(categorical_columns) == 1

def dataset_is_two_categorie(data):
    categorical_columns = data.select_dtypes(include=['object']).columns
    
    return len(categorical_columns) == 2

def dataset_is_several_categorie(data):
    categorical_columns = data.select_dtypes(include=['object']).columns
    
    return len(categorical_columns) > 2

def dataset_is_numeric(data):
    numeric_columns = data.select_dtypes(include=['number']).columns
    
    return len(numeric_columns) > 0

def dataset_is_categorical(data):
    categorical_columns = data.select_dtypes(include=['object']).columns
    
    return len(categorical_columns) > 0

def dataset_has_one_value_per_categorie_group(data):
    categorical_columns = [col for col in data.columns if data[col].dtype == 'object']
    
    for cat_col in categorical_columns:
        if data[cat_col].duplicated().any():
            return False  
    
    return True


def dataset_has_few_categories(data):
    categorical_columns = [col for col in data.columns if data[col].dtype == 'object']
    threshold = 5
    for col in categorical_columns:
        unique_count = data[col].nunique()
        if unique_count > threshold:
            return False  
    
    return True
def dataset_has_less_then_4_categories(data):
    categorical_columns = [col for col in data.columns if data[col].dtype == 'object']
    threshold = 4
    for col in categorical_columns:
        unique_count = data[col].nunique()
        if unique_count > threshold:
            return False  
    
    return True

def dataset_has_few_similaire_values(data):
    numerical_columns = data.select_dtypes(include=['number']).columns
    
    numerical_column = numerical_columns[0]
    
    percentages = (data[numerical_column] / data[numerical_column].sum()) * 100
    
    similarity_count = 0

    for value in percentages:
        similar_values = percentages[(percentages >= value - 5) & (percentages <= value + 5)]
        if len(similar_values) >= 3:
            similarity_count += 1
    
    return similarity_count < 3

def dataset_cleaning(data):
    # Identify columns that might represent years
    potential_year_columns = []
    for col in data.columns:
        if 'year' in col.lower() or 'yr' in col.lower():
            potential_year_columns.append(col)
        elif data[col].dtype == 'int64' and data[col].min() >= 1900 and data[col].max() <= 2100:
            potential_year_columns.append(col)

    # Convert the identified year columns to categorical data
    for col in potential_year_columns:
        data[col] = data[col].astype(str)

    return data

def load_csv_dataset(file_path):
    # Load dataset from CSV file
    return dataset_cleaning(pd.read_csv(file_path))

def main(repo):
    dataset = load_csv_dataset(file_path=repo)
    selected_chart = choose_chart(dataset)
    print("Selected chart type:", selected_chart)

def launch_test(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            main(os.path.join(root, file))

if __name__ == "__main__":
    repo_path = str(os.getcwd()+"//Data")
    launch_test(repo_path)