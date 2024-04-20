def choose_chart(data):
    #data can be anything
    if dataset_is_categories_and_numeric_values(data):
        if dataset_is_one_numiric(data) and dataset_is_one_categorie(data):
            #condition finished
            if dataset_has_one_value_per_group(data):
                if dataset_has_few_categories(data) and dataset_has_few_similaire_values(data):
                    #have few categories and few similaire values
                    return 'pie chart'
                elif not dataset_has_few_categories(data) and dataset_has_few_similaire_values(data):
                    #have severale categories and few similaire values
                    return 'doughnut chart'
                else:
                    #have similaire values
                    return 'bar plot'
            elif not dataset_has_one_value_per_group(data):
                return 'density plot'
        elif dataset_is_one_categorie(data) and not dataset_is_one_numiric(data):
            if dataset_has_no_orderd_values(data):
                return '2D density'
            elif dataset_has_one_orderd_values(data):
                return 'steam graph'
            elif dataset_has_one_row_per_group(data):
                return 'lollipop'
        elif dataset_is_one_numiric(data) and not dataset_is_one_categorie(data):
            if dataset_has_one_sub_group(data):
                return 'group bar plot'
            elif not dataset_has_one_sub_group(data):
                return 'box plot'
            elif dataset_has_one_nested_group(data):
                return 'sunburst'
            elif not dataset_has_one_nested_group(data):
                return 'violin'
            elif dataset_has_adjacency(data):
                return 'arc'
    #data can't be cat and num at the same time
    elif dataset_is_categorical(data):
        if dataset_is_one_categorie(data):
            return 'barplot'
        elif dataset_is_several_categorie(data):

            #condition finished
            if dataset_has_two_independent_lists(data):
                return 'venn diagram'
            elif dataset_has_sub_groups(data):
                #consider the other posibilitis as grouped bar plot and grouped scatter plot
                return 'groupedscatterplot'
            elif dataset_has_nested_lists(data):
                #consider the cas with multi layer nested like 4 or more layers
                return 'treemap'
            elif dataset_has_adjacency(data):
                return 'chord diagram'
    #condition finished
    #data is not cat at all can be num timeseries or network
    elif dataset_is_numeric(data):
        #condition finished
        if dataset_is_one_numiric(data):
            return 'histogram'
        #condition finished
        elif dataset_is_two_numiric(data):
            if not dataset_is_orderd(data):
                if dataset_has_few_points(data):
                    return 'scatter plot'
                elif not dataset_has_few_points(data):
                    return 'density plot'
            elif dataset_is_orderd(data):
                return 'area plot'
        #condition finished
        elif dataset_is_three_numiric(data):
            if not dataset_is_orderd(data):
                    return 'bubble plot'
            elif dataset_is_orderd(data):
                return 'staked area plot'
        #condition finished
        elif dataset_is_several_numiric(data):
            if not dataset_is_orderd(data):
                    return 'density plot'
            elif dataset_is_orderd(data):
                return 'line plot'
    #condition finished but can be improved
    #data is not cat or num at all
    elif dataset_is_time_series_data(data):
        if dataset_is_one_time_series(data):
            return 'bar plot'
        elif dataset_is_several_time_series(data):
            return 'stacked area chart'
    #data is eather netork or something else not detected
    elif dataset_is_networks_series(data):
        if dataset_is_nested_or_hierarchical(data):
            return 'DENDROGRAM'
        return 'network'

# Function to check if dataset has categorical and numeric values
def dataset_is_categories_and_numeric_values(data):
    # Check if the DataFrame contains both categorical and numeric columns
    categorical_columns = data.select_dtypes(include=['object']).columns
    numeric_columns = data.select_dtypes(include=['number']).columns
    
    return len(categorical_columns) > 0 and len(numeric_columns) > 0

# Function to check if dataset has only one numeric value
def dataset_is_one_numiric(data):
    # Check if the DataFrame contains only one numeric column
    numeric_columns = data.select_dtypes(include=['number']).columns
    
    return len(numeric_columns) == 1

# Function to check if dataset has only one categorical value
def dataset_is_one_categorie(data):
    # Check if the DataFrame contains only one categorical column
    categorical_columns = data.select_dtypes(include=['object']).columns
    
    return len(categorical_columns) == 1

# Function to check if dataset is categorical
def dataset_is_categorical(data):
    # Check if the DataFrame contains categorical columns
    categorical_columns = data.select_dtypes(include=['object']).columns
    
    return len(categorical_columns) > 0

# Function to check if dataset is numeric
def dataset_is_numeric(data):
    # Check if the DataFrame contains numeric columns
    numeric_columns = data.select_dtypes(include=['number']).columns
    
    return len(numeric_columns) > 0

# Function to check if dataset has one value per group
def dataset_has_one_value_per_group(data):
    # Implement this function based on your dataset structure and requirements
    categorical_columns = [col for col in data.columns if data[col].dtype == 'object']
    numeric_columns = [col for col in data.columns if data[col].dtype in ['int64', 'float64']]
    
    for cat_col in categorical_columns:
        num_cols_for_cat = [col for col in numeric_columns if data.groupby(cat_col)[col].nunique() == 1]
        if len(num_cols_for_cat) != 1:
            return False  # Return False if there is not exactly one numerical column per category
    
    return True

# Function to check if dataset has few categories
def dataset_has_few_categories(data):
    # Implement this function based on your dataset structure and requirements
    categorical_columns = [col for col in data.columns if data[col].dtype == 'object']
    
    for col in categorical_columns:
        unique_count = data[col].nunique()
        if unique_count > threshold:
            return False  # Return False if any column has more than the threshold number of unique values
    
    return True0

# Function to check if dataset has few similar values
def dataset_has_few_similaire_values(data):
    # Implement this function based on your dataset structure and requirements
    similarity_threshold = 2  # Two or fewer similar values
    
    # Check the number of similar values for each column
    for col in data.columns:
        # Count occurrences of each unique value
        value_counts = data[col].value_counts()
        
        # Check if the count of the most common value is greater than the count of all other unique values combined
        if len(value_counts) > 1 and value_counts.iloc[0] >= value_counts.iloc[1:].sum() + similarity_threshold:
            return True  # Return True if few similar values are found
    
    return False

# Function to check if dataset has no ordered values
def dataset_has_no_orderd_values(data):
    # Implement this function based on your dataset structure and requirements
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numeric_columns:
        # Check if the values in the column are unordered
        if not data[col].is_monotonic:
            return True  # Return True if any column has unordered values
    
    return False

# Function to check if dataset has one ordered value
def dataset_has_one_orderd_values(data):
    # Implement this function based on your dataset structure and requirements
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    ordered_count = 0
    
    for col in numeric_columns:
        # Check if the values in the column are ordered
        if data[col].is_monotonic:
            ordered_count += 1
            
            # If more than one column is ordered, return False
            if ordered_count > 1:
                return False
    
    # Return True if exactly one column is ordered
    return ordered_count == 1

# Function to check if dataset has one row per group
def dataset_has_one_row_per_group(data):
    # Implement this function based on your dataset structure and requirements
    pass

# Function to check if dataset has one sub-group
def dataset_has_one_sub_group(data):
    # Implement this function based on your dataset structure and requirements
    pass

# Function to check if dataset has one nested group
def dataset_has_one_nested_group(data):
    # Implement this function based on your dataset structure and requirements
    pass

# Function to check if dataset has adjacency
def dataset_has_adjacency(data):
    # Implement this function based on your dataset structure and requirements
    pass

# Function to check if dataset is time series data
def dataset_is_time_series_data(data):
    if 'datetime_column' in data.columns:
        # Check if the values in the datetime column are datetime objects
        if data['datetime_column'].dtype == 'datetime64[ns]':
            # Check if the datetime values are in sequential order
            if data['datetime_column'].is_monotonic_increasing or data['datetime_column'].is_monotonic_decreasing:
                return True  # Return True if all conditions are met
    return False

# Function to check if dataset is one time series
def dataset_is_one_time_series(data):
    # Implement this function based on your dataset structure and requirements
    datetime_columns = [col for col in data.columns if data[col].dtype == 'datetime64[ns]']
    
    if len(datetime_columns) == 1:  # Only one datetime column
        datetime_column = datetime_columns[0]
        
        # Check if the datetime column has unique values
        if data[datetime_column].is_unique:
            # Check if there are no other datetime columns
            if len(data.columns) == 1 or (len(data.columns) == 2 and 'other_column' in data.columns):  # Adjust 'other_column' as needed
                return True  # Return True if all conditions are met
    
    return False

# Function to check if dataset is several time series
def dataset_is_several_time_series(data):
    # Implement this function based on your dataset structure and requirements
    datetime_columns = [col for col in data.columns if data[col].dtype == 'datetime64[ns]']
    
    if len(datetime_columns) > 1:  # Multiple datetime columns
        # Check if each datetime column has unique values
        unique_values = [data[col].is_unique for col in datetime_columns]
        if all(unique_values):
            return True  # Return True if all conditions are met
    
    return False

# Function to check if dataset is network series
def dataset_is_networks_series(data):
    # Implement this function based on your dataset structure and requirements
    pass

# Function to check if dataset is nested or hierarchical
def dataset_is_nested_or_hierarchical(data):
    # Implement this function based on your dataset structure and requirements
    pass


def load_csv_dataset(file_path):
    # Load dataset from CSV file
    return pd.read_csv(file_path)

def __main__(self):
    dataset = load_csv_dataset(file_path='Data\\LocationGroup.csv')
    selected_chart = determine_chart(dataset)
    print("Selected chart type:", selected_chart)
