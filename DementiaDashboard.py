import pandas as pd
from dash import Dash, Input, Output, dcc, html, State, dash_table
import base64
import os
import io
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


# External stylesheets
external_stylesheets = [
    {
        "href": (
            "https://fonts.googleapis.com/css2?"
            "family=Lato:wght@400;700&display=swap"
        ),
        "rel": "stylesheet",
    },
]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Dementia Dashboard"


DATASET_DIR = "./"


preloaded_datasets = {
    filename: os.path.join(DATASET_DIR, filename)
    for filename in os.listdir(DATASET_DIR) if filename.endswith(".csv")
}

dropdown_options = [{"label": name, "value": path} for name, path in preloaded_datasets.items()]


# App loadout 
app.layout = html.Div(
    children=[

        # Header
        html.Div(
            children=[
                html.H1(
                    children="Dementia Data", className="header-title"
                ),
                html.P(
                    children=(
                        "Aggregate and analyze publicly available datasets for dementia research"
                    ),
                    className="header-description",
                ),

                # Choose your dataset 
                html.Div(
                    children=[
                        html.Div(
                            children="Dataset", className="choose-dataset-title"),

                    # Dropdown and upload dataset 
                        html.Div( 
                            children=[

                                dcc.Store(id="selected-file", storage_type="memory"),

                                dcc.Dropdown(
                                    id="file-dropdown",
                                    options=dropdown_options,
                                    value=[0] if dropdown_options else None,
                                    clearable=False,
                                    className="dropdown",
                                    style= {"width": "290px", "marginRight": "10px"},
                                ),
                                dcc.Upload(
                                    id= "upload-dataset", 
                                    children=html.Button("Upload csv file"),
                                    className="upload-button",
                                ),

                            ],

                            style={
                                "display": "flex", "justifyContent": "center"
                            }
                        ),
                    
                    ],
                    
                    style={
                    "textAlign": "left", 
                    "display": "inline-block",
                    }
                
                ),
            ],
            
            className="header",
        ),

        # Body
        html.Div( children=[

            # Preview of dataset 
            html.Div(id='file-content', className ="box"),

            # Sample distribution
            html.Div(
                children=[
                    html.H3(f"Sample Distribution"),
                    html.Label("Feature:"),
                    dcc.Dropdown(
                        id="sample-dist-dropdown",
                        placeholder="Select feature..."),
            
                    html.Div(id='sample-size-dist', className="wrapper")
                    ],
                className = "box"
            ),

            # Missing data distribution 
            html.Div(
                children=[
                    html.H3(f"Missing Data Distribution"),
                    html.Label("Graph format:"),
                    dcc.Dropdown(
                            id='missing-data-dropdown',
                            options=[
                                    {'label': 'Bar Plot', 'value': 'Bar Plot'},
                                    {'label': 'Heat Map', 'value': 'Heat Map'},
                                    ],
                                    value=None,
                                    clearable=False,
                                
                                ),

                    html.Div(id='missing-data', className="wrapper"),
                ],
                className = "box",
            ),

            # CLass imbalance
            html.Div(
                children=[
                    html.H3(f"Class Imbalance"),
                    html.Label("Target Variable:"), 
                    dcc.Dropdown(
                        id='class-imbalance-dropdown',
                        placeholder='Select target variable...'
                    ),

                    html.Div(id='class-imbalance', className="wrapper")
                ],
                className ="box"
            ),

            # Feature imbalance
            html.Div(
                children=[
                    html.H3(f"Feature/Demographic Imbalance"),
                    html.Label("Feature:"),
                    dcc.Dropdown(
                        id = 'feature-imbalance-dropdown',
                        placeholder = 'Select feature...'
                    ),

                    html.Div(id='feature-imbalance', className="wrapper"),
                ],
                className = "box"
            )
        ],

        className="body"
    
        )

    ],
)

def feature_distribution(df, feature):
    if feature not in df.columns:
        raise ValueError(f"{feature} not found in dataframe" )

    if df[feature].dtype in ["object", "category"]:
        value_counts = df[feature].value_counts().reset_index()
        value_counts.columns = [feature, "count"]
        fig = px.bar(value_counts, x=feature, y="count", title=f"Counts by {feature}" )

        return fig, value_counts
    
    else:
        fig = px.histogram(df, x=feature, nbins=10, title=f"Counts by {feature}")

        return fig, None

uploaded_datasets = {}

# Callback for dataset options and selected file 
@app.callback(
    [Output('selected-file', 'data'), Output("file-dropdown", "options")],
    [Input('file-dropdown', 'value'), Input("upload-dataset", "contents")],
    [State("upload-dataset", "filename"), State("file-dropdown", "options")],

)

def update_dataset_dropdown(selected_file, contents, filename, options):
    if contents is not None:

        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

        uploaded_datasets[filename] = df

        if not any(option['value'] == filename for option in options):
            new_option = {"label": filename, "value": filename}
            options.append(new_option)

            return selected_file, options
        
    return selected_file, options


# Callback for display some information about the selected file 
@app.callback(
    Output('file-content', 'children'),
    Input('selected-file', 'data')
)

def display_selected_file(file_name):

    if not file_name or not isinstance(file_name, str):
        return html.Div("Select a dataset", style={"textAlign": "center", "fontSize": "20px", "margin": "20px"})

    # Selected dataset
    if isinstance(file_name, list):
        file_name = file_name[0]

    if file_name in uploaded_datasets:
        df = uploaded_datasets[file_name]
    else:
        df = pd.read_csv(file_name)


    duplicate_rows = df[df.duplicated()]
    num_of_duplicate_rows = duplicate_rows.shape[0]

    missing_values = df.isnull().sum()
    columns_with_missing_values = missing_values[missing_values > 0]
    total_missing = missing_values.sum()

    return html.Div([
            html.H2(f"Dataset: {file_name}"),
            html.H3(f"{df.shape[0]} rows, {df.shape[1]} columns"),

            html.H4("Preview of data:"),
            dash_table.DataTable(
                data=df.head().to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '10px'},
                style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
                page_size=5
            ),

            html.H4(f"Duplicate rows: {num_of_duplicate_rows}"),
            html.H4(f"Number of missing values: {total_missing}"),
            html.H4(f"Columns with missing values:"),
            
            html.Div([
                dash_table.DataTable(
                    columns=[
                        {"name": "Column", "id": "Column"},
                        {"name": "# of Missing Values", "id": "Missing Values"}
                    ],
                    data=[{"Column": col, "Missing Values": missing_values[col]} for col in columns_with_missing_values.index],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '10px'},
                    style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
                ) if not columns_with_missing_values.empty else html.P("None")
            ]),

        ])


# Callback for sample distribution dropdown
@app.callback(
    [Output('sample-dist-dropdown', 'options'), Output('sample-dist-dropdown', 'value')],
    Input('selected-file', 'data')
)

def update_sample_dist_dropdown(file_name):
    if not file_name or not isinstance(file_name, str):
        return [], None

    if isinstance(file_name, list):
        file_name = file_name[0]

    if file_name in uploaded_datasets:
        df = uploaded_datasets[file_name]
    else:
        df = pd.read_csv(file_name)
    
    feature_options = [{"label" : col, "value": col} for col in df.columns]

    return feature_options, None


# Callback for showing sample distribution of selected feature 
@app.callback(
    Output('sample-size-dist', 'children'),
    [Input('selected-file', 'data'), Input('sample-dist-dropdown', 'value')]
)

def show_sample_size_dist(file_name, selected_feature):

    if not selected_feature:
        return html.Div()
    
    if not file_name:
        return html.Div("Select a dataset", style={"textAlign": "center", "fontSize": "20px", "margin": "20px"})

    if isinstance(file_name, list):
        file_name = file_name[0]

    if file_name in uploaded_datasets:
        df = uploaded_datasets[file_name]
    else:
        df = pd.read_csv(file_name)

    if selected_feature not in df.columns:
        return html.Div("Selected feature not found in dataset")

    
    try: 
        fig, table_data = feature_distribution(df, selected_feature)
    except ValueError as e:
        return html.Div(str(e))
    
    table = dash_table.DataTable(
            data=table_data.to_dict("records"),
            columns=[{"name": col, "id": col} for col in table_data.columns],
            style_table={"margin-top": '20px', 'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
    ) if table_data is not None else None

    
    return html.Div([
        dcc.Graph(figure=fig), table 

    ])

# Callback for showing missing data 
@app.callback(
    Output('missing-data', 'children'),
    [Input('selected-file', 'data'), Input('missing-data-dropdown', 'value')]
)

def show_missing_data(file_name, selected_graph):
    if not selected_graph:
        return html.Div()

    if not file_name:
            return html.Div("Select a dataset", style={"textAlign": "center", "fontSize": "20px", "margin": "20px"})

    if isinstance(file_name, list):
            file_name = file_name[0]

    if file_name in uploaded_datasets:
            df = uploaded_datasets[file_name]
    else:
            df = pd.read_csv(file_name)

    # If Bar Plot is selected to display missing data 
    if selected_graph == "Bar Plot":
        missing_percentage = (df.isnull().sum() / len(df)) * 100
        missing_df = missing_percentage[missing_percentage > 0].reset_index()
        missing_df.columns = ["Feature", "Missing Percentage"]

        if missing_df.empty:
            return html.Div("No missing data in this dataset")
        
        fig = px.bar(missing_df, x="Feature", y="Missing Percentage",
                    title = "Missing Data Percentage", 
                    labels={"Feature": "Column", "Missing Percentage": "% Missing"},
                    )
        
        fig.update_traces(
            text=missing_df['Missing Percentage'].round(2).astype(str) + '%', 
            textposition = 'outside'
        )

        return dcc.Graph(figure=fig)
    
    # If heatmap is selected to display missing data 
    elif selected_graph == "Heat Map":
        
        if df.isnull().sum().sum() == 0:
            return html.Div("No missing data in this dataset")

        fig = px.imshow(df.isnull(), color_continuous_scale="Viridis", title="Missing Data Heatmap")
    
        return dcc.Graph(figure=fig)

    return html.Div("Select a visualization type")

# Callback for showing if the dataset is imbalanced or not 
@app.callback(
    [Output('class-imbalance-dropdown', 'options'), Output('class-imbalance-dropdown', 'value')],
    Input('selected-file', 'data')
)

def update_class_imbalance_dropdown(file_name):
    if not file_name or not isinstance(file_name, str):
        return [], None

    if isinstance(file_name, list):
        file_name = file_name[0]

    if file_name in uploaded_datasets:
        df = uploaded_datasets[file_name]
    else:
        df = pd.read_csv(file_name)
    
    feature_options = [{"label" : col, "value": col} for col in df.columns]

    return feature_options, None

@app.callback(
    Output('class-imbalance', 'children'),
    [Input('selected-file', 'data'), Input('class-imbalance-dropdown', 'value')],
)

def show_data_imbalance(file_name, selected_feature):
    if not selected_feature:
        return html.Div()
    
    if not file_name:
        return html.Div("Select a dataset", style={"textAlign": "center", "fontSize": "20px", "margin": "20px"})

    if isinstance(file_name, list):
        file_name = file_name[0]

    if file_name in uploaded_datasets:
        df = uploaded_datasets[file_name]
    else:
        df = pd.read_csv(file_name)

    if selected_feature not in df.columns:
        return html.Div("Selected feature not found in dataset")

    value_counts = df[selected_feature].value_counts(normalize=True) * 100
    value_counts = value_counts.reset_index()
    value_counts.columns = [selected_feature, "Percentage"]

    if len(value_counts) < 2:
        return html.Div("This features does not contain enough categories")

    majority_class = value_counts.iloc[0]
    minority_class = value_counts.iloc[-1]
    minority_percent = minority_class["Percentage"]

    # Check the degree of imblance based on the percentage of data belonging to the minority class 
    if minority_percent < 1:
        imbalance_degree = "Extreme Imbalance"
    elif 1 <= minority_percent < 20:
        imbalance_degree = "Moderate Imbalance"
    elif 20 <= minority_percent < 40:
        imbalance_degree = "Mild Imbalance"
    else:
        imbalance_degree = "Balanced"
    
    # Show the distribution of catergorical values 
    if df[selected_feature].dtype in ['object', 'category', 'bool']:
        value_counts = df[selected_feature].value_counts().reset_index()
        value_counts.columns = [selected_feature, "count"]

        fig = px.bar(value_counts,
            x=selected_feature,y="count",
            labels={selected_feature: "Category", "count": "Count"},
            title=f"{selected_feature} Distribution")
    
    # Show the distribuion of numerical features 
    else:
        fig = px.histogram(df, x=selected_feature, nbins=10, 
                            title=f"{selected_feature} Distribution")
    
    imbalance_text = html.Div([
        html.H4(f"Label: {selected_feature}"),
        html.P(f"Minority Class: {minority_class[selected_feature]}"),
        html.P(f"Degree of imbalance: {imbalance_degree}")
    ])

    return html.Div([dcc.Graph(figure=fig), imbalance_text])

# Callback for showing feature imblanced - see if certain demographic groups are underrepresented
@app.callback(
    [Output('feature-imbalance-dropdown', 'options'), Output('feature-imbalance-dropdown', 'value')],
    Input('selected-file', 'data')
)

def update_feature_imbalance_dropdown(file_name):
    if not file_name or not isinstance(file_name, str):
        return [], None

    if isinstance(file_name, list):
        file_name = file_name[0]

    if file_name in uploaded_datasets:
        df = uploaded_datasets[file_name]
    else:
        df = pd.read_csv(file_name)
    
    feature_options = [{"label" : col, "value": col} for col in df.columns]

    return feature_options, None

@app.callback(
    Output('feature-imbalance', 'children'),
    [Input('selected-file', 'data'), Input('feature-imbalance-dropdown', 'value')],
)

def show_feature_imbalance(file_name, selected_feature):
    if not selected_feature:
        return html.Div()
    
    if not file_name:
        return html.Div("Select a dataset", style={"textAlign": "center", "fontSize": "20px", "margin": "20px"})

    if isinstance(file_name, list):
        file_name = file_name[0]

    if file_name in uploaded_datasets:
        df = uploaded_datasets[file_name]
    else:
        df = pd.read_csv(file_name)

    if selected_feature not in df.columns:
        return html.Div("Selected feature not found in dataset")

    # Show the distrubion of categorical features 
    if df[selected_feature].dtype in ['object', 'category', 'bool']:
        value_counts = df[selected_feature].value_counts().reset_index()
        value_counts.columns = [selected_feature, "count"]
        value_counts['Percentage'] = (value_counts['count'] / value_counts['count'].sum() * 100)

        fig = px.bar(value_counts,
            x=selected_feature,y="count",
            labels={selected_feature: selected_feature, "count": "Count"},
            title=f"{selected_feature} Distribution")
        
        max_percent = value_counts['Percentage'].max()
        min_percent = value_counts['Percentage'].min()
        imbalance_ratio = max_percent / min_percent if min_percent > 0 else float('inf')

        # Show the distrubion of numerical features
        fig.update_traces(
            text=value_counts['Percentage'].round(2).astype(str) + '%', 
            textposition = 'outside'
        )

           
        return html.Div([
                dcc.Graph(figure=fig),
                html.H4(f"Label: {selected_feature}"),
                html.P(f"Number of unique values: {df[selected_feature].nunique()}"),
                html.P(f"Most common value: {value_counts.iloc[0][selected_feature]} ({value_counts.iloc[0]['Percentage']}%)"),
                html.P(f"Least common value: {value_counts.iloc[-1][selected_feature]} ({value_counts.iloc[-1]['Percentage']}%)"),
                html.P(f"Imbalance ratio (max% / min%): {imbalance_ratio:.2f}")
        
    ])
    
    # Show the distribuion of numerical features 
    else:
        fig = px.histogram(df, x=selected_feature, nbins=10, 
                            title=f"{selected_feature} Distribution")

        skew = df[selected_feature].skew()

        return html.Div([
            dcc.Graph(figure=fig),
            html.H4(f"Label: {selected_feature}"),
            html.P(f"Mean: {df[selected_feature].mean():.2f}"),
            html.P(f"Median: {df[selected_feature].median():.2f}"),
            html.P(f"Standard Deviation: {df[selected_feature].std():.2f}"),
            html.P(f"Skewness: {skew:.2f}")
        ])


        return html.Div(dcc.Graph(figure=fig))

# Callback for bias detection 


# Run server 
if __name__ == "__main__":
    app.run_server(debug=True)
