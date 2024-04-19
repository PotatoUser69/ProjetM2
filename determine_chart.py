def choose_chart(data):
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
            #condition finished
            if dataset_has_two_independent_lists(data):
                return 'venn diagram'
            elif dataset_has_sub_groups(data):
                #consider the other posibilitis as grouped bar plot and grouped scatter plot
                return 'test diagram'
            elif dataset_has_one_grp(data):
                return'test diagram'
            

            elif dataset_has_nested_lists(data):
                #consider the cas with multi layer nested like 4 or more layers
                return 'treemap'
            elif dataset_has_adjacency(data):
                return 'venn diagram'
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
    #condition finished but can be improved
    elif dataset_is_time_series_data(data):
        if dataset_is_one_time_series(data):
            return 'bar plot'
        elif dataset_is_several_time_series(data):
            return 'stacked area chart'
    
    elif dataset_is_networks_series(data):
        if dataset_is_nested_or_hierarchical(data):
            pass
        pass

def load_csv_dataset(file_path):
    # Load dataset from CSV file
    return pd.read_csv(file_path)

def __main__(self):
    dataset = load_csv_dataset(file_path='Data\\LocationGroup.csv')
    selected_chart = determine_chart(dataset)
    print("Selected chart type:", selected_chart)
