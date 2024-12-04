import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from agents import crew1,crew3,crew5



def extract_code_from_response(response: str) -> str:
    start_marker = "```python"
    end_marker = "```"
    
    start_index = response.find(start_marker)
    end_index = response.find(end_marker, start_index + len(start_marker))

    if start_index != -1 and end_index != -1:
        code = response[start_index + len(start_marker):end_index].strip()
        return code
    else:
        st.warning("Code block not found in the response.")
        return ""



def execute_data_retrieval_code(code: str, file):
    try:
        updated_code = code.replace('dataset.csv', file)
        st.code(updated_code, language='python')

        local_vars = {}
        exec(updated_code, globals(), local_vars)

        if 'filtered_data' in local_vars:
            return local_vars['filtered_data']
        else:
            st.error("filtered_data not found in the generated code's execution.")
            return None
    except Exception as e:
        st.error(f"Error executing the data r1etrieval code: {e}")
        return None



def execute_visualization_code(code: str, data):
    try:
        local_vars = {'data': data}
        st.code(code, language='python')
        
        exec(code, globals(), local_vars)
        figures = [plt.figure(i) for i in plt.get_fignums()]
        
        if figures:
            for fig in figures:
                st.pyplot(fig)
        else:
            st.error("No figures were generated.")
        
        plt.close('all')
    except Exception as e:
        st.error(f"Error executing the visualization code: {e}")



def automate_data_fetching(query: str, file):
    data = pd.read_csv(file)
    columns = data.columns.tolist()
    values = data.sample(10).to_dict(orient='list')

    inputs = {"query": query, "file_name": file.name, "columns": columns, 'values': values}
    data_result = crew1.kickoff(inputs=inputs)
    
    st.write("### Generated Code for Data Retrieval:")
    data_result = data_result.raw
    
    generated_code = extract_code_from_response(data_result)
    print(generated_code)
    filtered_data = execute_data_retrieval_code(generated_code, file=file.name)

    if filtered_data is not None and not filtered_data.empty:
        st.write("### Filtered Data:")
        st.dataframe(filtered_data)
        st.write(f"##### Shape: {filtered_data.shape[0]}")

    else:
        st.warning("\nNo data retrieved based on the query.")
        print("\nNo data retrieved based on the query.")



def generate_data_insights(query: str, file):
    st.write('## Basic Analysis')

    try:
        data = pd.read_csv(file)
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        return

    columns = ', '.join(data.columns.tolist())
    values = data.sample(10).to_dict(orient='list')

    inputs = {"query": query, "file_name": file.name, "columns": columns, "values": values}
    insights_result = crew3.kickoff(inputs=inputs)

    insights_result = insights_result.raw
    generated_insights_code = extract_code_from_response(insights_result)

    st.write("### Generated Code for Analysis:")
    st.code(generated_insights_code, language='python')

    try:
        local_vars = {'data': data}
        exec(generated_insights_code, globals(), local_vars)

        result_vars = {k: v for k, v in local_vars.items() if not callable(v) and not k.startswith("__") and k not in ['data', 'pd', 'np', 'plt', 'sns']}

        if result_vars:
            for var_name, var_value in result_vars.items():
                st.write(f"### {var_name}:")
                st.write(var_value)
                print(f"{var_name}: {var_value}")
        else:
            st.warning("No meaningful results were found in the executed code. Please verify the generated code.")

    except Exception as e:
        st.error(f"Error executing the insights code: {e}")
        print(f"Error executing the insights code: {e}")



def automate_data_visualizations(query: str, file):
    st.write("## Data Visualizations")

    try:
        data = pd.read_csv(file)
        st.dataframe(data)
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        return

    columns = ', '.join(data.columns.tolist())

    inputs = {"query": query, "file_name": file.name, "columns": columns}
    Visualizations_result = crew5.kickoff(inputs=inputs)

    st.write("### Generated Code for Visualizations:")
    Visualizations_result = Visualizations_result.raw

    generated_Visualizations_code = extract_code_from_response(Visualizations_result)
    execute_visualization_code(generated_Visualizations_code, data) 



def data_fetching():
    st.title("Automated Data Retrieval and Visualization App")

    query = st.text_input("Enter your query:")
    file = st.file_uploader("Upload the dataset file (CSV):", type=["csv"])

    if file and st.button("Run Query"):
        if not query:
            query = "Fetch complete data"
            automate_data_fetching(query=query, file=file)
        else:
            automate_data_fetching(query=query, file=file)



def data_visualization():
    st.title("Automated Data Visualizations")

    file = st.file_uploader("Upload the dataset file (CSV):", type=["csv"])

    if file and st.button("Visualizations"):
        query = "Generate code for visualizations that will represent data accurately."
        automate_data_visualizations(query=query, file=file)



def data_analysis():
    st.title("Automated Data Analysis")

    file = st.file_uploader("Upload the dataset file (CSV):", type=["csv"])
    query = ("Perform basic data analysis on the dataset. Provide descriptive statistics for numerical columns, a correlation matrix, and value counts for categorical columns. Handle missing values appropriately.")

    if file and st.button("Analysis"):
        generate_data_insights(query=query, file=file)



def main():
    st.sidebar.title('Select an Option')
    st.title('Welcome to the Automatic Data Science')

    option = st.sidebar.selectbox("Select Option", ["Data Analysis", "Data Fetching", "Data Visualizations"])

    if option == "Data Fetching":
        data_fetching()

    elif option == "Data Analysis":
        data_analysis()
    
    elif option == "Data Visualizations":
        data_visualization()


if __name__ == "__main__":
    main()
