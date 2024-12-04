# Multi-Agent System for Natural Language to Data Query and Visualization
## Overview
This project implements an AI-powered multi-agent system that transforms natural language queries into actionable data insights and visualizations. It leverages intelligent agents to generate Python code for querying datasets, creating visualizations, and performing analyses, enabling users to interact with their data in an intuitive and seamless way.

## Features
**Natural Language Query**: Users can input queries in plain English to fetch specific data or generate visualizations.

**Multi-Agent System**: Each agent specializes in a specific task:
- Data Retrieval Agent: Generates code to fetch and filter data based on user queries.
- Visualization Agent: Generates Python code to create static or interactive visualizations (e.g., bar charts, histograms).
- Analysis Agent: Performs data analysis and provides insights.

## Installation and Setup

Create .env file in the folder

```bash
GROQ_API_KEY=<your-api-key>
```

Install dependencies:
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

## Usage
Upload a CSV file with your dataset.
Input a natural language query (e.g., "Show the average salary for each job title").
View the retrieved data and generated visualizations.

## Dataset Format
The system supports datasets in CSV format. Ensure your dataset contains structured data with clearly defined column headers.


## Future Improvements
Enhance the accuracy of the Visualization Agent.
Add support for additional visualization types (e.g., geographic maps).
Incorporate advanced natural language processing models for better query understanding.
