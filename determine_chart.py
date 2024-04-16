def choose_chart(data):
    if dataset_is_categories_and_numeric_values(data):
        if dataset_is_one_numiric(data) and dataset_is_one_categorie(data):
            if dataset_has_one_value_per_group(data):
                if dataset_has_few_categories(data):
                    return 'pie chart'
                elif dataset_has_several_categories(data):
                    return 'doughnut chart'
            elif dataset_has_several_value_per_group(data):
                if dataset_has_no_orderd_values(data):
                    pass
                elif dataset_has_orderd_values(data):
                    pass
                elif dataset_has_one_row_per_group(data):
                    pass
        elif dataset_is_one_categorie(data) and dataset_is_seventh_numiric(data):
            pass
        elif dataset_is_one_numiric(data) and dataset_is_several_categorie(data):
            pass

    elif dataset_is_categorical(data):
        if dataset_is_one_categorie(data):
            pass
        elif dataset_is_several_categorie(data):
            pass
    #condition finished
    elif dataset_is_numeric(data):
        #condition finished
        if dataset_is_one_numiric(data):
            return 'histogram'
        #condition finished
        elif dataset_is_two_numiric(data):
            if dataset_is_not_orderd(data):
                if dataset_has_few_points(data):
                    return 'scatter plot'
                elif dataset_has_many_points(data):
                    return 'density plot'
            elif dataset_is_orderd(data):
                return 'area plot'
        #condition finished
        elif dataset_is_three_numiric(data):
            if dataset_is_not_orderd(data):
                    return 'bubble plot'
            elif dataset_is_orderd(data):
                return 'staked area plot'
        #condition finished
        elif dataset_is_seventh_numiric(data):
            if dataset_is_not_orderd(data):
                    return 'density plot'
            elif dataset_is_orderd(data):
                return 'line plot'

    elif dataset_is_time_series_data(data):
        if dataset_is_one_time_series(data):
            return 'bar plot'
        elif dataset_is_several_time_series(data):
            return 'stacked area chart'
    
    elif dataset_is_networks_series(data):
        if dataset_is_nested_or_hierarchical(data):
            pass
        pass

def dataset_is_time_series_data(dataset):
    return any(isinstance(column_data, pd.Timestamp) for column_data in dataset.dtypes)

def dataset_is_categories_and_numeric_values(dataset):
    return all(pd.api.types.is_numeric_dtype(dataset[column]) or pd.api.types.is_categorical_dtype(dataset[column]) for column in dataset.columns)

def dataset_is_categorical_and_few_categories(dataset):
    return all(pd.api.types.is_categorical_dtype(dataset[column]) and len(dataset[column].unique()) <= 10 for column in dataset.columns)

def dataset_is_two_numeric_variables(dataset):
    numeric_columns = [column for column in dataset.columns if pd.api.types.is_numeric_dtype(dataset[column])]
    return len(numeric_columns) == 2

def dataset_is_numeric_values(dataset):
    return any(pd.api.types.is_numeric_dtype(dataset[column]) for column in dataset.columns)

def dataset_is_hierarchical(dataset):
    hierarchical_columns = ['parent', 'child']  # Example hierarchical columns
    return all(column in dataset.columns for column in hierarchical_columns)

def dataset_is_textual(dataset):
    return any(pd.api.types.is_string_dtype(dataset[column]) for column in dataset.columns)

def dataset_is_primary_measure(dataset):
    numeric_columns = [column for column in dataset.columns if pd.api.types.is_numeric_dtype(dataset[column])]
    return len(numeric_columns) == 1

def dataset_represents_funnel_data(dataset):
    numeric_columns = [column for column in dataset.columns if pd.api.types.is_numeric_dtype(dataset[column])]
    return all(dataset[numeric_columns[i]].max() > dataset[numeric_columns[i+1]].max() for i in range(len(numeric_columns)-1))

def dataset_is_flow_data(dataset):
    return all(pd.api.types.is_numeric_dtype(dataset[column]) for column in dataset.columns)

def dataset_is_relationship_data(dataset):
    return all(pd.api.types.is_categorical_dtype(dataset[column]) for column in dataset.columns)

def dataset_is_circular_data(dataset):
    pass  

def dataset_is_three_dimensional_data(dataset):
    return all(len(dataset[column].unique()) > 1 for column in dataset.columns)

    return all(pd.api.types.is_categorical_dtype(dataset[column]) for column in dataset.columns)

def dataset_is_spatial_data(dataset):
    return all(column in ['latitude', 'longitude'] for column in dataset.columns)

def dataset_is_two_time_points(dataset):
    time_columns = [column for column in dataset.columns if pd.api.types.is_datetime64_any_dtype(dataset[column])]
    return len(time_columns) == 2

def dataset_is_cumulative_changes(dataset):
    numeric_columns = [column for column in dataset.columns if pd.api.types.is_numeric_dtype(dataset[column])]
    return all(dataset[column].diff().fillna(0).ge(0).all() for column in numeric_columns)

def dataset_is_error_data(dataset):
    return all(pd.api.types.is_numeric_dtype(dataset[column]) for column in dataset.columns)

def dataset_is_three_component_data(dataset):
    return len(dataset.columns) == 3

def dataset_is_financial_data(dataset):
    return all(pd.api.types.is_numeric_dtype(dataset[column]) for column in dataset.columns)

def dataset_is_population_data(dataset):
    return all(pd.api.types.is_numeric_dtype(dataset[column]) for column in dataset.columns)

def dataset_is_multiple_variables(dataset):
    return len(dataset.columns) > 1
    
def dataset_is_matrix_like(dataset):
    return dataset.shape[0] == dataset.shape[1]

# def dataset_is_contour_data(dataset):
#     # Evaluate if the dataset is contour data
#     # For example, check if it contains continuous data over a 2D grid
#     return all(pd.api.types.is_numeric_dtype(dataset[column]) for column in dataset.columns)

# def dataset_is_parallel_set_data(dataset):
#     # Evaluate if the dataset is parallel set data
#     # For example, check if it contains categorical data with intersecting subsets
#     return all(pd.api.types.is_categorical_dtype(dataset[column]) for column in dataset.columns)

# def dataset_is_word_tree_data(dataset):
#     # Evaluate if the dataset is word tree data
#     # For example, check if it contains hierarchical textual data
#     return all(pd.api.types.is_string_dtype(dataset[column]) for column in dataset.columns)

# def dataset_represents_timeline_data(dataset):
#     # Evaluate if the dataset represents timeline data
#     # For example, check if it contains events with start and end times
#     time_columns = [column for column in dataset.columns if pd.api.types.is_datetime64_any_dtype(dataset[column])]
#     return len(time_columns) == 2

# def dataset_represents_network_data(dataset):
#     # Evaluate if the dataset represents network data
#     # For example, check if it represents nodes and edges
#     return 'source' in dataset.columns and 'target' in dataset.columns



def load_csv_dataset(file_path):
    # Load dataset from CSV file
    return pd.read_csv(file_path)

def __main__(self):
    dataset = load_csv_dataset(file_path='Data\\LocationGroup.csv')
    selected_chart = determine_chart(dataset)
    print("Selected chart type:", selected_chart)
