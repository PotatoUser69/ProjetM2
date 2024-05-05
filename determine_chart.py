import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import pycountry
import mplcursors
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from geonamescache import GeonamesCache

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

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
        elif dataset_has_country_data(data) and dataset_is_one_numiric(data) and dataset_is_one_categorie(data):
            return CountryMap(data)
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
                        return radar(data)
                    return heatmap(data)
                else:
                    return 'box plot'
        elif dataset_is_two_categorie(data):
            #verify if data containe one set of numerical values or several
            if dataset_is_one_numiric(data):
                #verify if data have categories and subcategories
                if dataset_has_sub_groups(data):
                    return sunburst(data)
            elif dataset_is_three_numiric(data):
                return 'bubble chart'
        elif dataset_is_three_categorie(data) and dataset_has_sub_groups(data) and dataset_is_one_numiric(data):
            return sunburst(data)
        elif dataset_is_several_categorie(data):
            #verify if data containe one set of numerical values or several
            if dataset_is_one_numiric(data):
                #verify if data have categories and subcategories
                if dataset_has_sub_groups(data):
                    return treemap(data)
    #verify if data containe only numerical data
    elif dataset_is_numeric(data) and not dataset_is_categorical(data):
        #verify if data containe one set of numerical values or several
        if dataset_is_one_numiric(data):
            return histogram(data)
        elif dataset_is_two_numiric(data):
            #verify if data have many values more then 400 data 
            if dataset_has_many_or_few_point(data):
                return histogram(data)
            return scatter(data)
        elif dataset_is_three_numiric(data):
            return bubble(data)
    #verify if data containe only categorical data
    elif not dataset_is_numeric(data) and dataset_is_categorical(data):
        return parallelCoordinates(data)
    return choose_chart(remove_least_important_column(data))

def histogram(data):
    plt.hist(data,edgecolor='black')
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
    return 'histogram'

def CountryMap(data):
    country_column = data.select_dtypes(include=['object']).columns[0]
    value_column = data.select_dtypes(include=['number']).columns[0]
    print(value_column, country_column)
    
    fig = px.choropleth(data, 
                        locations=country_column, 
                        locationmode="country names", 
                        color=value_column, 
                        hover_name=country_column,
                        title=f"Choropleth Map of {value_column} by Country")
    
    fig.show()
    return 'Map'

def parallelCoordinates(data):
    categories=list(data.select_dtypes(include=['object']).columns)
    cat_len=[len(set(data[i])) for i in  categories]
    dimensions=[]
    for i, cat in enumerate(categories):
        lab=list(set(data[cat]))
        values=[]
        list_val=list(range(len(lab)))
        for index in data[cat]:
            values.append(lab.index(index))
        dimensions.append(dict(range = [0,int(cat_len[i])],label = cat,tickvals = list_val, values = values,ticktext = lab))    
    fig = go.Figure(data=go.Parcoords(
            dimensions = dimensions
        )
    )

    fig.update_layout(
        plot_bgcolor = 'white',
        paper_bgcolor = 'white'
    )

    fig.show()
    return 'parallelCoordinates'
        
def box(data):
    data.boxplot(figsize = (5,5), rot = 90, fontsize= '8', grid = False)
    return 'box'
        
def scatter(data):
    [xlabel,ylabel]=data.columns
    plt.scatter(data[xlabel], data[ylabel], alpha=0.5)
    plt.show()  
    return 'scatter plot'
        
def treemap(data):
    path=list(data.select_dtypes(include=['object']).columns)
    values=data.select_dtypes(include=['number']).columns[0]
    fig = px.treemap(data, path=path, values=values,color=values,color_continuous_scale=['#6BAED6', '#08306B'])
    fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
    fig.show()

    return 'treemap'
        
def sunburst(data):
    path=list(data.select_dtypes(include=['object']).columns)
    values=data.select_dtypes(include=['number']).columns[0]
    fig = px.sunburst(data, path=path,  
                  values=values) 
    fig.show()

    return 'sunburst'
        
def heatmap(data):
    categorie_column = data.select_dtypes(include=['object']).columns[0]
    data.set_index(categorie_column, inplace=True)
    sns.heatmap(data, cmap="Blues", annot=True, square=False,  linewidth = 1)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()  
    return 'heatmap'

def radar(data):
    categorie_column = data.select_dtypes(include=['object']).columns[0]
    labels = list(data[categorie_column])
    categories = np.array(data.select_dtypes(include=['number']).columns)
    data.drop(categorie_column, axis=1, inplace=True)
    max_value = data.max().max()
    min_value = data.min().min()
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    for index, row in data.iterrows():
        values = list(row)
        values += values[:1]
        ax.plot(np.linspace(0, 2 * np.pi, len(categories) + 1), values, label=labels[index])

    ax.set_ylim(0, max_value)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(np.linspace(0, 2 * np.pi, len(categories) + 1)[:-1])
    ax.set_xticklabels(categories)

    plt.legend(bbox_to_anchor=(1.1, 1))
    plt.title('Radar Chart')
    plt.show()
    
    return 'radar'

def bubble(data):
    numeric_columns = data.select_dtypes(include=['number']).columns
    x = data[numeric_columns[0]]
    y = data[numeric_columns[1]]
    sizes = data[numeric_columns[2]]
    min_size = 20
    max_size = 200  
    sizes_scaled = np.interp(sizes, (sizes.min(), sizes.max()), (min_size, max_size))

    colors = np.random.rand(len(x))
    plt.scatter(x, y, s=sizes_scaled, c=colors, alpha=0.5)

    plt.xlabel(numeric_columns[0])
    plt.ylabel(numeric_columns[1])
    plt.colorbar(label="Bubble sizes")

    plt.show()
    return 'bubble chart'

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
    return "grouped bar"


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
def is_column_countries(column_values):
    gc = GeonamesCache()
    countries = gc.get_countries_by_names()
    country_list=['Czech Republic']
    country_name = set(country.name for country in pycountry.countries)
    country_names = set(countries.keys())
    for value in column_values:
        if not ((value in country_names) or (value in country_name) or (value in country_list)):
            print(value)
            return False
    return True

def dataset_has_country_data(data):
    categorical_columns = [col for col in data.columns if data[col].dtype == 'object']
    for col in categorical_columns:
        if len(data[col]) == len(set(data[col])) and is_column_countries(data[col]):
            return True
    return False

def dataset_has_many_or_few_point(data):
    num_rows, num_columns = data.shape  
    return (num_rows > 300 or num_rows < 50)

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

def dataset_is_three_categorie(data):
    categorical_columns = data.select_dtypes(include=['object']).columns
    
    return len(categorical_columns) == 3

def dataset_is_several_categorie(data):
    categorical_columns = data.select_dtypes(include=['object']).columns
    
    return len(categorical_columns) > 3

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

def get_feature_importance(data):
    targets = list(data.columns[:])

    column_trans = ColumnTransformer(transformers=
        [('num', MinMaxScaler(), selector(dtype_exclude="object")),
        ('cat', OrdinalEncoder(), selector(dtype_include="object"))],
        remainder='drop')

    # Create a random forest classifier for feature importance
    clf = RandomForestClassifier(random_state=42, n_jobs=6, class_weight='balanced')
    pipeline = Pipeline([('prep',column_trans),('clf', clf)])
    
    # Split the data into 30% test and 70% training
    X_train, X_test, y_train, y_test = train_test_split(data[targets], data[data.columns[-1]], test_size=0.3, random_state=0)
    
    pipeline.fit(X_train, y_train)

    feat_list = []

    total_importance = 0
    # Print the name and gini importance of each feature
    for feature in zip(targets, pipeline['clf'].feature_importances_):
        feat_list.append(feature)
        total_importance += feature[1]
            
    included_feats = []
    # Print the name and gini importance of each feature
    for feature in zip(targets, pipeline['clf'].feature_importances_):
        if feature[1] > .05:
            included_feats.append(feature[0])
            
    print('\n',"Cumulative Importance =", total_importance)

    # create DataFrame using data
    data_imp = pd.DataFrame(feat_list, columns =['FEATURE', 'IMPORTANCE']).sort_values(by='IMPORTANCE', ascending=False)
    # data_imp['CUMSUM'] = data_imp['IMPORTANCE'].cumsum()
    return data_imp

def get_least_significant_numerical_column(nums,data):
    for feature in nums:
        if feature in data['FEATURE'].values:
            return data.loc[data['FEATURE'] == feature, 'FEATURE'].iloc[0]

def remove_least_important_column(data):
    feature_importance= get_feature_importance(data)
    numeric_columns=data.select_dtypes(include=['number']).columns
    if (dataset_has_sub_groups(data) or dataset_is_numiric(data)) and not dataset_is_one_numiric(data):
        least_important=get_least_significant_numerical_column(numeric_columns,feature_importance)
        data = data.drop(columns=[least_important])
    print(data.columns)
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
    repo_path = str(os.getcwd()+"//Error")
    launch_test(repo_path)