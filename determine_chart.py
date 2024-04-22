import pandas as pd
import os

def choose_chart(data):
    if dataset_is_time_series_data(data):
        if dataset_is_several_numiric(data):
            return 'line plot'
        if dataset_is_one_numiric(data):
            return 'area plot'
    elif dataset_is_networks_series(data):
        return 'network'
    elif dataset_is_categories_and_numeric_values(data):
        if dataset_is_one_categorie(data):
            if dataset_is_one_numiric(data):
                if dataset_has_few_categories(data) and dataset_has_few_similaire_values(data):
                    return 'pir chart'
                elif not dataset_has_few_categories(data) and dataset_has_few_similaire_values(data):
                    return 'dount chart'
                else:
                    return 'bar chart'
            elif dataset_is_two_numiric(data):
                if dataset_has_one_value_per_categorie_group(data):
                    return 'grouped bar plot'
                else:
                    return 'box plot'
            elif dataset_is_three_numiric(data):
                return 'bubble chart'
            elif dataset_is_several_numiric(data):
                if dataset_has_one_value_per_categorie_group(data):
                    if dataset_has_less_then_4_categories(data):
                        return 'radar chart'
                    return 'heatmap'
                else:
                    return 'box plot'
        elif dataset_is_two_categorie(data):
            #unfinished
            if dataset_is_one_numiric(data):
                if dataset_has_sub_groups(data):
                    return 'treemap'
            elif dataset_is_three_numiric(data):
                return 'bubble chart'
            else:
                return 'parallel coordinates plot'
        elif dataset_is_several_categorie(data):
            if dataset_is_one_numiric(data):
                if dataset_has_sub_groups(data):
                    return 'sunburst chart'
            else:
                return 'parallel coordinates plot'
    elif dataset_is_numeric(data):
        if dataset_is_one_numiric(data):
            return 'histogram'
        elif dataset_is_two_numiric(data):
            if dataset_has_many_point(data):
                return 'histogram'
            return 'scatter plot'
        elif dataset_is_three_numiric(data):
            return 'bubble chart'
    return 'unhendeld error'
        

def dataset_is_time_series_data(data):
    return False
    
def dataset_is_networks_series(data):
    return False

def dataset_has_sub_groups(data):
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
def dataset_has_less_then_4_categories(data):
    categorical_columns = [col for col in data.columns if data[col].dtype == 'object']
    threshold = 4
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