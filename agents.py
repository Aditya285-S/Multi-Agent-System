from crewai import Crew, Task, Agent
import os
from langchain_groq import ChatGroq

# GROQ_API_KEY = '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'

llm = ChatGroq(
    groq_api_base = "https://api.groq.com/v1",
    groq_api_key=os.environ.get("GROQ_API_KEY"),
    model_name ="groq/mixtral-8x7b-32768",
    temperature = 0.1,
    max_tokens = 1000,
)


CodeGenerator_Agent = Agent(
    role='CodeGenerator',
    goal='Generate Python code to fetch data from the dataset based on the user query.',
    backstory=(
        "You are an expert at writing Python code to extract relevant information from datasets. "
        "The dataset file is '{file_name}', and all column names must be referenced exactly as provided in the dataset. "
        "The complete data must be loaded into the 'data' variable, and the final filtered results must always be stored "
        "in a variable named 'filtered_data'. Do not use any other variable names for the filtered results."
        "Always wrap the generated code in a ` ```python ` block followed by the code and ending with ` ``` `. "
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)



VisualizationGenerator_Agent = Agent(
    role="VisualizationGenerator",
    goal="Generate Python code for creating data visualizations specifically related to the user's query.",
    backstory=(
        "You are a data visualization expert specializing in Python. "
        "After the user provides a query and filtered data is retrieved, your task is to generate Python code for meaningful visualizations "
        "directly related to the query. Dynamically identify columns as numerical or categorical using the Pandas library. "
        "Ensure the code focuses only on the columns relevant to the query. "
        "Your task is to generate code that creates appropriate 4,5 meaningful visualizations using libraries like matplotlib, seaborn, or plotly. "
        "All visualizations must be created using the dataset's columns only, without creating or using additional columns. "
        "The code must include necessary imports and error handling to ensure it is executable and robust."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)




BasicDataAnalysis_Agent = Agent(
    role='DataAnalyzer',
    goal='Generate Python code for basic data analysis of a dataset',
    backstory=(
        "You are an expert Python coder specializing in data analysis using pandas and other essential libraries. "
        "Your task is to generate Python code that performs basic data analysis including: "
        "1. A descriptive summary of the data (mean, median, mode, std, etc.). "
        "2. Correlation matrix for numerical columns. "
        "3. Analysis of categorical columns (e.g., value counts, unique values). "
        "4. Clear distinction between numerical and categorical columns based on their data types. "
        "Ensure that your generated code is error-free, imports all necessary libraries, "
        "and handles potential issues such as missing values and incorrect data types."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)



# DataCleaningAgent = Agent(
#     role='DataCleaner',
#     goal='Generate Python code to clean and preprocess the provided dataset effectively.',
#     backstory=(
#         "You are a data cleaning and preprocessing expert. Your goal is to validate and process a given dataset "
#         "by dynamically analyzing each column's data type and content before categorizing it as numerical or categorical. "
#         "Use validation functions to ensure accurate separation of columns. "
#         "Perform actions such as filling or dropping missing values, removing duplicates, handling outliers, "
#         "and transforming categorical columns into numerical representations if needed. "
#         "**Do not create or use any additional column names that are not in the provided list `{columns}`.** "
#         "Store the cleaned and processed data in the variable `cleaned_data`. Always ensure error-free, executable code."
#     ),
#     verbose=True,
#     allow_delegation=False,
#     llm=llm,
# )



DataVisualizationAgent = Agent(
    role="VisualizationExpert",
    goal="Generate Python code to create robust visualizations for the provided dataset using its columns.",
    backstory=(
        "You are an expert in data visualization using Python. Your task is to generate Python code for meaningful visualizations "
        "based on the provided dataset. Dynamically and robustly classify columns as numerical or categorical using Pandas. "
        "Create visualizations such as histograms, bar plots, box plots, scatter plots. "
        "Your task is to generate code that creates appropriate 4,5 meaningful visualizations using libraries like matplotlib, seaborn, or plotly. "
        "All visualizations must use the dataset's columns without creating or using other columns. "
        "Don't create any functions and add print statements in generated code. "
        "The code must include error handling, appropriate library imports "
        "and generate plots with proper titles, labels, and legends."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)




code_generation_task = Task(
    description=(
        "Interpret the following natural language query: {query} "
        "and generate Python code that retrieves relevant data from a CSV file {file_name} using the pandas library. "
        "Use only provided column names {columns} for generating code to fetch the data and do not generate or use any other column names. "
        "Ensure that the entire dataset is loaded into a variable named 'data', and the final filtered data must always be stored "
        "in a variable named 'filtered_data'. Do not use any other variable names for the filtered data. "
        "The sample of the dataset is {values}. "
        "Always enclose the output code in ` ```python ` and ` ``` ` for consistency."
    ),
    expected_output=(
        "A block of executable Python code that retrieves, processes, and filters the specified data from the CSV file using pandas. "
        "The filtered data must always be stored in the 'filtered_data' variable, and the output must always be enclosed within "
        "` ```python ` and ` ``` `."
    ),
    agent=CodeGenerator_Agent,
)



visualization_generation_task = Task(
    description=(
        "Generate Python code to create visualizations for the filtered data stored in the variable name `data` based on the query `{query}`. "
        "Use the data stored in 'data' variable to make visualizations don't assume any other variables. "
        "Identify numerical and categorical columns dynamically using the Pandas library. "
        "Focus on creating 4, 5 various visualizations that are directly relevant to the query `{query}` and the provided dataset's columns `{columns}`. "
        "The generated code must use only the dataset's columns `{columns}` and must not create or use additional columns. "
        "Include necessary imports for Python visualization libraries and ensure the code is error-free and robust, "
        "handling missing data or unusual values gracefully. "
        "Wrap the output code in ` ```python ` and ` ``` ` for consistency."
    ),
    expected_output=(
        "A block of Python code that generates visualizations directly related to the query and dataset. "
        "The code must correctly distinguish between numerical and categorical columns, include all necessary imports, "
        "and provide meaningful visualizations with appropriate titles, labels, and legends. "
        "Ensure the output is always enclosed within ` ```python ` and ` ``` `."
    ),
    agent=VisualizationGenerator_Agent,
)



data_analysis_task = Task(
    description=(
        "Generate Python code for performing basic data analysis on a dataset with the following columns: {columns}. "
        "The dataset file is named {file_name}. The code must perform the following tasks: "
        "1. Provide a descriptive summary of the dataset, including statistical measures for numerical columns. "
        "2. Compute a correlation matrix for numerical columns only. "
        "3. Analyze categorical columns separately, providing value counts and unique values. "
        "4. Distinguish between numerical and categorical columns automatically based on data types. "
        "5. Handle potential issues such as missing values or incorrect data types gracefully. "
        "Ensure all required libraries (pandas, numpy, seaborn, matplotlib) are imported in the code, "
        "and store the final results in appropriately named variables (e.g., 'numerical_summary', 'correlation_matrix'). "
        "The generated code must be enclosed within ` ```python ` and ` ``` `."
    ),
    expected_output=(
        "A block of Python code that performs the requested data analysis and stores results in variables "
        "like 'numerical_summary', 'correlation_matrix', 'categorical_analysis'. "
        "The output must always be enclosed within ` ```python ` and ` ``` `."
    ),
    agent=BasicDataAnalysis_Agent,
)



# data_cleaning_task = Task(
#     description=(
#         "Generate Python code to clean and preprocess data from the provided CSV file `{file_name}`. "
#         "Dynamically classify columns `{columns}` into numerical and categorical types using validation functions. "
#         "**Only use the provided column names `{columns}` for all processing steps. Do not create or use additional columns.** "
#         "Handle data cleaning tasks using pandas libraries and their function such as: "
#         "1. Validating each column's datatype and values to classify them as numerical or categorical. "
#         "2. Removing or filling missing values based on context (e.g., forward-fill for time-series data, mean/mode imputation for other cases). "
#         "3. Removing duplicates and noisy data. "
#         "4. Handling outliers for numerical columns by capping or removing extreme values. "
#         "5. Transforming categorical columns into numerical representations using mapping or encoding. "
#         "Store the final cleaned and processed dataset in the variable `cleaned_data`. "
#         "Ensure all required libraries are imported and the code is error-free."
#         "Wrap all generated code in ` ```python ` and ` ``` ` for consistency."
#     ),
#     expected_output=(
#         "A block of executable Python code that dynamically validates and processes the dataset, "
#         "storing the cleaned and processed result in the `cleaned_data` variable. "
#         "All operations must use the provided column names and avoid assumptions or hardcoding. "
#         "The code must include robust error handling and validation steps. "
#         "The output must always be enclosed within ` ```python ` and ` ``` `."
#     ),
#     agent=DataCleaningAgent,
# )



visualization_task = Task(
    description=(
        "Generate Python code to create robust visualizations for the dataset provided in `{file_name}`. "
        "Use the dataset's columns `{columns}` to dynamically and robustly classify them as numerical or categorical. "
        "Ensure that all visualizations use only the dataset's columns `{columns}` and avoid creating or using additional columns. "
        # "Handle mixed or ambiguous data types gracefully. "
        "Your task is to generate code that creates appropriate 4,5 meaningful visualizations using libraries like matplotlib, seaborn, or plotly. "
        "The code must include necessary imports, error handling, and clear annotations (titles, labels, legends) for the plots. "
        "Ensure consistency by wrapping all generated code in ` ```python ` and ` ``` `."
    ),
    expected_output=(
        "A block of Python code that generates error-free and robust visualizations using the dataset's columns. "
        "The code must include necessary imports, robust error handling, and generate visualizations with meaningful annotations. "
        "Ensure that the output is always enclosed within ` ```python ` and ` ``` `."
    ),
    agent=DataVisualizationAgent,
)




crew1 = Crew(
    agents=[CodeGenerator_Agent],
    tasks=[code_generation_task],
    verbose=True,
)


crew2 = Crew(
    agents=[VisualizationGenerator_Agent],
    tasks=[visualization_generation_task],
    verbose=True,
)


crew3 = Crew(
    agents=[BasicDataAnalysis_Agent],
    tasks=[data_analysis_task],
    verbose=True,
)


# crew4 = Crew(
#     agents=[DataCleaningAgent],
#     tasks=[data_cleaning_task],
#     verbose=True,
# )


crew5 = Crew(
    agents=[DataVisualizationAgent],
    tasks=[visualization_task],
    verbose=True,
)