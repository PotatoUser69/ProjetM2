import pandas as pd
import os

def choose_chart(data):
    if dataset_is_time_series_data(data):
        if dataset_is_one_time_series(data):
            return 'bar plot'
        elif dataset_is_several_time_series(data):
            return 'stacked area chart' 

    elif dataset_is_categories_and_numeric_values(data):
        if dataset_is_one_numiric(data) and dataset_is_one_categorie(data):
            if dataset_has_one_value_per_categorie_group(data):
                if dataset_has_few_categories(data) and dataset_has_few_similaire_values(data):
                    return 'pie chart'
                elif not dataset_has_few_categories(data) and dataset_has_few_similaire_values(data):
                    return 'doughnut chart'
                else:
                    return 'bar plot'
            elif not dataset_has_one_value_per_categorie_group(data):
                return 'density plot'
        elif dataset_is_one_categorie(data) and not dataset_is_one_numiric(data):
            if dataset_has_one_value_per_categorie_group(data):
                return 'lollipop'
            elif dataset_has_no_orderd_values(data):
                return '2D density'
            elif not dataset_has_no_orderd_values(data):
                return 'steam graph'
        elif dataset_is_one_numiric(data) and not dataset_is_one_categorie(data):
            if dataset_has_sub_groups(data):
                if dataset_has_one_value_per_sub_group(data):
                    return 'group bar plot'
                elif not dataset_has_one_value_per_sub_group(data):
                    return 'box plot'
            elif dataset_has_nested_leafs(data):
                if dataset_has_one_value_per_nested_group(data):
                    return 'sunburst'
                elif not dataset_has_one_value_per_nested_group(data):
                    return 'violin'
            elif dataset_has_adjacency(data):
                return 'arc'
    
    elif dataset_is_numeric(data):
        
        if dataset_is_one_numiric(data):
            return 'histogram'
        
        elif dataset_is_two_numiric(data):
            if not dataset_is_orderd(data):
                if dataset_has_few_points(data):
                    return 'scatter plot'
                elif not dataset_has_few_points(data):
                    return 'density plot'
            elif dataset_is_orderd(data):
                return 'area plot'
        
        elif dataset_is_three_numiric(data):
            if not dataset_is_orderd(data):
                    return 'bubble plot'
            elif dataset_is_orderd(data):
                return 'staked area plot'
        
        elif dataset_is_several_numiric(data):
            if not dataset_is_orderd(data):
                    return 'density plot'
            elif dataset_is_orderd(data):
                return 'line plot'
    
    elif dataset_is_networks_series(data):
        if dataset_is_nested_or_hierarchical(data):
            return 'DENDROGRAM'
        return 'network'


def dataset_has_one_value_per_sub_group(data):
    pass

def dataset_has_one_value_per_nested_group(data):
    pass

def dataset_has_sub_groups(data):
    pass

def dataset_has_nested_leafs(data):
    pass

def dataset_is_orderd(date):
    pass

def dataset_has_few_points(date):
    pass

def dataset_has_nested_lists(date):
    pass

def dataset_is_networks_series(data):
    pass

def dataset_is_nested_or_hierarchical(data):
    pass

def dataset_has_adjacency(data):
    pass


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

def dataset_is_one_categorie(data):
    categorical_columns = data.select_dtypes(include=['object']).columns
    
    return len(categorical_columns) == 1

def dataset_is_categorical(data):
    categorical_columns = data.select_dtypes(include=['object']).columns
    
    return len(categorical_columns) > 0

def dataset_is_numeric(data):
    numeric_columns = data.select_dtypes(include=['number']).columns
    
    return len(numeric_columns) > 0

def dataset_has_one_value_per_categorie_group(data):
    categorical_columns = [col for col in data.columns if data[col].dtype == 'object']
    
    for cat_col in categorical_columns:
        if data[cat_col].duplicated().any():
            return False  
    
    return True


def dataset_has_few_categories(data):
    categorical_columns = [col for col in data.columns if data[col].dtype == 'object']
    threshold = 6
    for col in categorical_columns:
        unique_count = data[col].nunique()
        if unique_count > threshold:
            return False  
    
    return True

def dataset_has_few_similaire_values(data):
    similarity_threshold = 2  
    
    for col in data.columns:
        value_counts = data[col].value_counts()
        
        if len(value_counts) > 1 and value_counts.iloc[0] >= value_counts.iloc[1:].sum() + similarity_threshold:
            return True  
    
    return False

def dataset_has_two_independent_lists(data):
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns

    if len(categorical_columns) < 2:
        return False
    return True

def dataset_has_no_orderd_values(data):
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numeric_columns:
        if not data[col].is_monotonic:
            return True  
    
    return False

def dataset_has_one_orderd_values(data):
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    ordered_count = 0
    for col in numeric_columns:
        if data[col].is_monotonic:
            return True
    return False

def dataset_is_time_series_data(data):
    if 'datetime_column' in data.columns:
        if data['datetime_column'].dtype == 'datetime64[ns]':
            if data['datetime_column'].is_monotonic_increasing or data['datetime_column'].is_monotonic_decreasing:
                return True
    return False

def dataset_is_one_time_series(data):
    datetime_columns = [col for col in data.columns if data[col].dtype == 'datetime64[ns]']
    if len(datetime_columns) == 1:  
        datetime_column = datetime_columns[0]
        if data[datetime_column].is_unique:
            if len(data.columns) == 1 or (len(data.columns) == 2 and 'other_column' in data.columns):  
                return True
    
    return False

def dataset_is_several_time_series(data):
    datetime_columns = [col for col in data.columns if data[col].dtype == 'datetime64[ns]']
    
    if len(datetime_columns) > 1:  
        
        unique_values = [data[col].is_unique for col in datetime_columns]
        if all(unique_values):
            return True
    
    return False


def dataset_cleaning(data):
    #change year data to str
    pass

def load_csv_dataset(file_path):
    # Load dataset from CSV file
    return pd.read_csv(file_path)

def main(repo):
    dataset = load_csv_dataset(file_path=repo)
    selected_chart = choose_chart(dataset)
    print("Selected chart type:", selected_chart)

def launch_test(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            print(file)
            main(os.path.join(root, file))

if __name__ == "__main__":
    repo_path = "C:\\Users\\totti\\VSCodeProjects\\Jupiter\\ProjetM2\\Data"
    launch_test(repo_path)