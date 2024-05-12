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
import re

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

def perform_dimensionality_reduction(data):
    features=data[get_feature_importance(data,threshold=.09)[1]]
    return features

def choose_chart(data):
    #verify if data is categorical and also numerical
    if not dataset_has_sub_groups(data) and large_dataset(data):
        data=perform_dimensionality_reduction(data)
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
        elif dataset_is_two_categorie(data):
            #verify if data containe one set of numerical values or several
            if dataset_is_one_numiric(data):
                #verify if data have categories and subcategories
                if dataset_has_sub_groups(data):
                    return treemap(data)
        elif dataset_is_three_categorie(data) and dataset_has_sub_groups(data) and dataset_is_one_numiric(data):
            return treemap(data)
        elif dataset_is_several_categorie(data):
            #verify if data containe one set of numerical values or several
            if dataset_is_one_numiric(data):
                #verify if data have categories and subcategories
                if dataset_has_sub_groups(data):
                    return sunburst(data)
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
        else:
            return box(data)
    #verify if data containe only categorical data
    elif not dataset_is_numeric(data) and dataset_is_categorical(data):
        return parallelCoordinates(data)
    return choose_chart(remove_least_important_column(data))

def histogram(data,ssecondary=False):
    fig = px.histogram(data, x=data.columns[0], title="Histogram",
                       barmode='overlay', # to overlay bars
                       opacity=0.7, # to make bars transparent
                       barnorm='percent', # to normalize bars to percent
                       histnorm='probability density') # to normalize bars to probability density
    fig.update_xaxes(title="Value")
    fig.update_yaxes(title="Frequency")
    fig.update_traces(marker=dict(line=dict(color='black', width=1))) # to add black border to bars
    save_dir = 'charts'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define the file path for saving the plot
    file_path = 'histogram.html'

    fig.write_html(os.path.join(save_dir, file_path))   # Save the plot to an HTML file

    if ssecondary==True:
        return file_path
    return [file_path,'histogram.html']

def CountryMap(data,ssecondary=False):
    country_column = data.select_dtypes(include=['object']).columns[0]
    value_column = data.select_dtypes(include=['number']).columns[0]
    
    fig = px.choropleth(data, 
                        locations=country_column, 
                        locationmode="country names", 
                        color=value_column, 
                        hover_name=country_column,
                        title=f"Choropleth Map of {value_column} by Country")
    
    save_dir = 'charts'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define the file path for saving the plot
    file_path = 'CountryMap.html'

    fig.write_html(os.path.join(save_dir, file_path))   # Save the plot to an HTML file

    if ssecondary==True:
        return file_path
    return [file_path,bar(data,True)]

def parallelCoordinates(data,ssecondary=False):
    categories = list(data.select_dtypes(include=['object']).columns)
    cat_len = [len(set(data[i])) for i in  categories]
    dimensions = []
    
    for i, cat in enumerate(categories):
        lab = list(set(data[cat]))
        values = []
        list_val = list(range(len(lab)))
        
        for index in data[cat]:
            values.append(lab.index(index))
        
        dimensions.append(dict(range=[0,int(cat_len[i])], label=cat, tickvals=list_val, values=values, ticktext=lab))    
    
    fig = go.Figure(data=go.Parcoords(dimensions=dimensions))

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Define the directory for saving the plot
    save_dir = 'charts'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define the file path for saving the plot
    file_path = 'parallel_coordinates.html'
    fig.write_html(os.path.join(save_dir, file_path))  # Save the plot to an HTML file

    if ssecondary==True:
        return file_path
    return [file_path,'parallel_coordinates.html']

def treemap(data,ssecondary=False):
    path=list(data.select_dtypes(include=['object']).columns)
    values=data.select_dtypes(include=['number']).columns[0]
    fig = px.treemap(data, path=path, values=values,color=values,color_continuous_scale=['#6BAED6', '#08306B'])
    fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
    save_dir = 'charts'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define the file path for saving the plot
    file_path = 'treemap.html'
    fig.write_html(os.path.join(save_dir, file_path))  # Save the plot to an HTML file

    if ssecondary==True:
        return file_path
    return [file_path,sunburst(data,True)]

def sunburst(data,ssecondary=False):
    path=list(data.select_dtypes(include=['object']).columns)
    values=data.select_dtypes(include=['number']).columns[0]
    fig = px.sunburst(data, path=path,  
                  values=values) 
    save_dir = 'charts'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define the file path for saving the plot
    file_path = 'sunburst.html'
    fig.write_html(os.path.join(save_dir, file_path))  # Save the plot to an HTML file
    if ssecondary==True:
        return file_path
    return [file_path,treemap(data,True)]

def box(data,ssecondary=False):
    fig = px.box(data, title="Box Plot")
    fig.update_layout(xaxis_tickangle=-90,  # Rotate x-axis labels
                      font=dict(size=14),  # Font size
                      yaxis_title="Value")
    save_dir = 'charts'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define the file path for saving the plot
    file_path = 'box.html'
    fig.write_html(os.path.join(save_dir, file_path))  # Save the plot to an HTML file

    if ssecondary==True:
        return file_path
    return [file_path,'box.html']

def scatter(data,ssecondary=False):
    [xlabel, ylabel] = data.columns
    fig = px.scatter(data, x=xlabel, y=ylabel, title="Scatter Plot")
    fig.update_traces(marker=dict(size=5, opacity=0.5))  # Adjust marker size and opacity
    save_dir = 'charts'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define the file path for saving the plot
    file_path = 'scatter.html'
    fig.write_html(os.path.join(save_dir, file_path))  # Save the plot to an HTML file

    if ssecondary==True:
        return file_path
    return [file_path,histogram(data,True)]  

def heatmap(data,ssecondary=False):
    categorical_column = data.select_dtypes(include=['object']).columns[0]
    data.reset_index(inplace=True)  # Reset index so the categorical column can be accessed
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=data.columns,
        y=data[categorical_column],
        colorscale='Blues',  # Set the color scale
        hoverongaps=False
    ))
    fig.update_layout(
        title='Heatmap',
        xaxis=dict(title='Column'),
        yaxis=dict(title=categorical_column, autorange='reversed'),  # Reverse the y-axis
        height=600,  # Set the height of the figure
        width=800  # Set the width of the figure
    )
    
    save_dir = 'charts'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define the file path for saving the plot
    file_path = 'heatmap.html'
    fig.write_html(os.path.join(save_dir, file_path))  # Save the plot to an HTML file
    with open(f'charts/{file_path}', 'w') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))  # Exclude Plotly JS
        f.write('<style>')
        f.write('body {')  # Opening body tag for CSS
        f.write('display: flex;justify-content: center;')  # Example CSS property
        f.write('}')
        f.write('</style>') 
    if ssecondary==True:
        return file_path
    return [file_path,radar(data,True)]          

def radar(data,ssecondary=False):
    categorical_column = data.select_dtypes(include=['object']).columns[0]
    labels = list(data[categorical_column])
    categories = list(data.select_dtypes(include=['number']).columns)
    max_value = data.max()
    
    fig = go.Figure()
    
    for index, row in data.iterrows():
        values = list(row[1:])
        values += values[:] # Append the first value to the end to close the loop
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=labels[index]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, max_value]),
            angularaxis=dict(direction='clockwise')
        ),
        title='Radar Chart'
    )
    save_dir = 'charts'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define the file path for saving the plot
    file_path = 'radar.html'
    fig.write_html(os.path.join(save_dir, file_path))  # Save the plot to an HTML file

    if ssecondary==True:
        return file_path
    return [file_path,heatmap(data,True)]          
def bubble(data,ssecondary=False):
    numeric_columns = data.select_dtypes(include=['number']).columns
    x = data[numeric_columns[0]]
    y = data[numeric_columns[1]]
    sizes = data[numeric_columns[2]]
    min_size = 20
    max_size = 200  
    sizes_scaled = np.interp(sizes, (sizes.min(), sizes.max()), (min_size, max_size))

    # Create a scatter plot using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=sizes_scaled,
            color=np.random.rand(len(x)),  # Random colors
            opacity=0.5
        ),
        text=data.index,  # Use index as hover text
        hoverinfo='text'
    ))

    fig.update_layout(
        xaxis_title=numeric_columns[0],
        yaxis_title=numeric_columns[1],
        coloraxis_colorbar=dict(title='Bubble sizes')
    )  
    save_dir = 'charts'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define the file path for saving the plot
    file_path = 'bubble.html'
    fig.write_html(os.path.join(save_dir, file_path))  # Save the plot to an HTML file

    if ssecondary==True:
        return file_path
    return [file_path,'bubble.html']          
def bubble_one_cat(data,ssecondary=False):
    numeric_columns = data.select_dtypes(include=['number']).columns
    categorie_column = data.select_dtypes(include=['object']).columns[0]

    labels = data[categorie_column]
    x = data[numeric_columns[0]]
    y = data[numeric_columns[1]]
    sizes = data[numeric_columns[2]]

    # Create a scatter plot using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=sizes,
            color=np.random.rand(len(x)),  # Random colors
            opacity=0.5
        ),
        text=labels,  # Use category labels as hover text
        hoverinfo='text'
    ))

    fig.update_layout(
        xaxis_title=numeric_columns[0],
        yaxis_title=numeric_columns[1],
        coloraxis_colorbar=dict(title='Bubble sizes')
    )
    save_dir = 'charts'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define the file path for saving the plot
    file_path = 'bubble_one.html'
    fig.write_html(os.path.join(save_dir, file_path))  # Save the plot to an HTML file

    if ssecondary==True:
        return file_path
    return [file_path,bubble(data,True)]          
def line(data,ssecondary=False):
    categorical_columns = data.select_dtypes(include=['object']).columns
    numeric_columns = data.select_dtypes(include=['number']).columns

    fig = go.Figure()
    
    for num_col in numeric_columns:
        fig.add_trace(go.Scatter(
            x=data[categorical_columns[0]],
            y=data[num_col],
            mode='lines+markers',  # Display lines and markers
            name=num_col  # Use column name as trace name
        ))

    fig.update_layout(
        xaxis_title=categorical_columns[0],
        yaxis_title='Value',
        title='Line Chart',
        legend=dict(title='Numeric Columns')
    )  
    save_dir = 'charts'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define the file path for saving the plot
    file_path = 'line.html'
    fig.write_html(os.path.join(save_dir, file_path))  # Save the plot to an HTML file

    if ssecondary==True:
        return file_path
    return [file_path,area(data,True)]   

def area(data,ssecondary=False):
    numeric_columns = data.select_dtypes(include=['number']).columns
    categorical_columns = data.select_dtypes(include=['object']).columns

    fig = go.Figure()

    colors = px.colors.qualitative.Plotly

    # Iterate over all numeric columns
    for i, col in enumerate(numeric_columns):
        color = colors[i % len(colors)]  # Get color from palette
        r, g, b = re.findall('(..)', color[1:])
        rgba_color= f'rgba({int(r, 16)}, {int(g, 16)}, {int(b, 16)}, {0.4})'
        # Add the filled area trace
        fig.add_trace(go.Scatter(
            x=data[categorical_columns[0]],
            y=data[col],
            mode='lines',  # Display only lines for filled area
            fill='tozeroy',  # Fill area to zero y-axis
            fillcolor=rgba_color,  # Set the fill color
            line=dict(color=color),  # Set line color
            name=col  # Set the trace name
        ))

    fig.update_layout(
        xaxis_title=categorical_columns[0],
        yaxis_title='Value',
        title='Area Chart',
        legend=dict(title='Numeric Columns')
    )
    save_dir = 'charts'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define the file path for saving the plot
    file_path = 'area.html'
    fig.write_html(os.path.join(save_dir, file_path))  # Save the plot to an HTML file

    if ssecondary==True:
        return file_path
    return [file_path,line(data,True)]  
def pie(data,ssecondary=False):
    categorical_columns = data.select_dtypes(include=['object']).columns
    numeric_columns = data.select_dtypes(include=['number']).columns
    num_data = data[numeric_columns[0]]
    labels = data[categorical_columns[0]]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=num_data,
        textinfo='percent+label',  # Show percent and label
    )])

    fig.update_layout(
        title='Pie Chart',
    )  
    save_dir = 'charts'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define the file path for saving the plot
    file_path = 'pie.html'
    fig.write_html(os.path.join(save_dir, file_path))  # Save the plot to an HTML file

    if ssecondary==True:
        return file_path
    return [file_path,bar(data,True)]          
def bar(data,ssecondary=False):
    categorical_columns = data.select_dtypes(include=['object']).columns
    numeric_columns = data.select_dtypes(include=['number']).columns
    num_data = data[numeric_columns[0]]
    labels = data[categorical_columns[0]]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=labels,
        y=num_data,
    ))

    fig.update_layout(
        xaxis_title=categorical_columns[0],
        yaxis_title=numeric_columns[0],
        title='Bar Chart',
    )
    save_dir = 'charts'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define the file path for saving the plot
    file_path = 'bar.html'
    fig.write_html(os.path.join(save_dir, file_path))  # Save the plot to an HTML file

    if ssecondary==True:
        return file_path
    return [file_path,pie(data,True)] 

def grouped_bar(data,ssecondary=False):
    categorical_columns = data.select_dtypes(include=['object']).columns
    numeric_columns = data.select_dtypes(include=['number']).columns
    
    categories = data[categorical_columns[0]]
    num_data = data[numeric_columns]

    num_bars = len(numeric_columns)
    bar_width = 0.35
    index = list(range(len(categories)))
    opacity = 0.8
    fig = go.Figure()

    for i, col in enumerate(numeric_columns):
        x_values = [val + i * bar_width for val in index]  # Calculate x values for each bar group
        fig.add_trace(go.Bar(
            x=x_values, 
            y=num_data[col], 
            name=col
        ))
    
    x_tickvals = [val + bar_width * (num_bars - 1) / 2 for val in index]  # Calculate tick values for x-axis

    fig.update_layout(
        xaxis=dict(tickvals=x_tickvals, ticktext=categories),
        xaxis_title=categorical_columns[0],
        yaxis_title="Values",
        title="Grouped Bar Chart",
        barmode='group',
        legend_title="Numeric Columns"
    )
    save_dir = 'charts'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define the file path for saving the plot
    file_path = 'grouped.html'
    fig.write_html(os.path.join(save_dir, file_path))  # Save the plot to an HTML file

    if ssecondary==True:
        return file_path
    return [file_path,line(data,True)]    
def donut(data,ssecondary=False):
    categorical_columns = data.select_dtypes(include=['object']).columns
    numeric_columns = data.select_dtypes(include=['number']).columns
    num_data = data[numeric_columns[0]]
    labels = data[categorical_columns[0]]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=num_data,
        hole=0.6,  # Set the size of the hole for the donut chart
    )])

    fig.update_layout(
        title="Donut Chart",
    )
    save_dir = 'charts'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define the file path for saving the plot
    file_path = 'donut.html'
    fig.write_html(os.path.join(save_dir, file_path))  # Save the plot to an HTML file

    if ssecondary==True:
        return file_path
    return [file_path,bar(data,True)]          

def dataset_has_sub_groups(data):
    categorical_columns = data.select_dtypes(include=['object']).columns
    return len(categorical_columns)>1 and not dataset_has_no_duplicate_values(data)
def large_dataset(data):
    return len(data.columns)>4
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

def get_feature_importance(data,threshold=.05):
    data = data.dropna()
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
        if feature[1] > threshold:
            included_feats.append(feature[0])

    # create DataFrame using data
    data_imp = pd.DataFrame(feat_list, columns =['FEATURE', 'IMPORTANCE']).sort_values(by='IMPORTANCE', ascending=False)
    return [data_imp,included_feats]

def get_least_significant_numerical_column(nums,data):
    for feature in nums:
        if feature in data['FEATURE'].values:
            return data.loc[data['FEATURE'] == feature, 'FEATURE'].iloc[0]

def remove_least_important_column(data):
    feature_importance= get_feature_importance(data)[0]
    numeric_columns=data.select_dtypes(include=['number']).columns
    if (dataset_has_sub_groups(data) or dataset_is_numiric(data)) and not dataset_is_one_numiric(data):
        least_important=get_least_significant_numerical_column(numeric_columns,feature_importance)
        data = data.drop(columns=[least_important])
    else:
        least_important=get_least_significant_numerical_column(data.columns,feature_importance)
        data = data.drop(columns=[least_important])
    return data

def load_csv_dataset(file_path):
    # Load dataset from CSV file
    return dataset_cleaning(pd.read_csv(file_path))

def files():
    directory = str(os.getcwd()+"\\data")
    for root, dirs, files in os.walk(directory):
        return files
def main(repo,file):
    dataset = load_csv_dataset(file_path=repo)
    selected_chart = choose_chart(dataset)

def launch_test(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            main(os.path.join(root, file),file)

if __name__ == "__main__":
    repo_path = str(os.getcwd()+"//Error")
    launch_test(repo_path)