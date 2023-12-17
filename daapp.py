import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Function to load and clean data
@st.cache_data
def load_and_clean_data(uploaded_file):
    # Load data
    data = pd.read_excel(uploaded_file)
    return data

# Function for Demographic Segmentation Analysis
@st.cache_data
def demographic_segmentation_analysis(data):
    st.subheader("Demographic Segmentation Analysis")
    
    # Age Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data['Age'], bins=30, kde=True, color="green")
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    st.pyplot(fig)

    # Marital Status Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Married/Single', data=data, palette="Set2")
    plt.title('Marital Status Distribution')
    plt.xlabel('Marital Status')
    plt.ylabel('Count')
    st.pyplot(fig)

    # House Ownership Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='House_Ownership', data=data, palette="Set1")
    plt.title('House Ownership Distribution')
    plt.xlabel('House Ownership')
    plt.ylabel('Count')
    st.pyplot(fig)

# Function for Professional Background Analysis
@st.cache_data
def professional_background_analysis(data):
    st.subheader("Professional Background Analysis")

    # Profession vs. Income
    profession_income = data.groupby('Profession')['Income'].mean().sort_values(ascending=False).reset_index()
    top_10_professions = profession_income.head(10)

    # Format income values
    top_10_professions['Income'] = top_10_professions['Income'] / 1e6  # Divide by 1e6 to convert to millions

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Income', y='Profession', data=top_10_professions, palette="Blues_d")
    plt.title('Top 10 Professions by Average Income')
    plt.xlabel('Average Income (Millions)')  # Update x-axis label
    plt.ylabel('Profession')

    # Format x-axis labels to show millions
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f'{x:.0f}M'))

    st.pyplot(fig)

# Function for Geographic Segmentation Analysis (Top 10 States by Income)
@st.cache_data
def top_10_states_by_income(data):
    st.subheader("Top 10 States by Average Income")

    # Group customers by state and calculate summary statistics
    state_summary = data.groupby('STATE').agg({
        'Income': 'mean',
        'Age': 'mean',
        # Add more relevant columns as needed
    }).reset_index()

    # Sort the states by average income in descending order
    state_summary = state_summary.sort_values(by='Income', ascending=False)

    # Select the top 10 states
    top_10_states = state_summary.head(10)

    # Create a bar chart to visualize average income by state
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Income', y='STATE', data=top_10_states, palette="Blues_d")
    plt.title('Top 10 States by Average Income')
    plt.xlabel('Average Income')
    plt.ylabel('State')
    
    def format_labels(x, pos):
        if x >= 1e12:
            return f'{x / 1e12:.0f}T'
        elif x >= 1e9:
            return f'{x / 1e9:.0f}B'
        elif x >= 1e6:
            return f'{x / 1e6:.0f}M'
        else:
            return f'{x:.0f}'

    ax.xaxis.set_major_formatter(mtick.FuncFormatter(format_labels))



    st.pyplot(fig)

# Function for Banking Sector Analysis
@st.cache_data
def analyze_banking_data(data):
    st.subheader("Data Cleaning")
    
    # Check for missing data and display the columns with missing values
    missing_data = data.isnull().sum()
    missing_columns = missing_data[missing_data > 0].index.tolist()
    
    if len(missing_columns) == 0:
        st.warning("No missing data found.")
    else:
        st.warning("Columns with missing data:")
        st.write(missing_data[missing_data > 0])
    
    # Check for duplicated rows and display them if any
    duplicated_rows = data[data.duplicated()]
    
    if duplicated_rows.shape[0] == 0:
        st.warning("No duplicated data found.")
    else:
        st.warning("Duplicated rows:")
        st.write(duplicated_rows)

    # Preprocess the data by handling missing values and duplicates
    # For example, you can use data.dropna() to remove rows with missing values and data.drop_duplicates() to remove duplicates.
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    
    # Data Preprocessing: Handling Missing Data
    st.warning("Preprocessing data...")

    st.success("Data has been cleaned.")

    # Calculate summary statistics for the preprocessed data
    st.subheader("Summary Statistics for Preprocessed Data:")
    summary_stats_preprocessed = data.describe()
    st.write(summary_stats_preprocessed)
    
    # Function to perform statistical tests
@st.cache_data
def perform_stat_tests(data):
    st.subheader("Statistical Tests:")
    # Perform statistical tests and display results here
    
# Function to display data info including duplicated data
def display_data_info(data):
    st.subheader("Data Info:")
    
    # Display current data types
    with st.expander("Current Data Types", expanded=False):
        st.write(data.dtypes)
        

# Function for Clustering Execution and Interpretation
@st.cache_data
def clustering_and_interpretation(data):
    st.subheader("Customer Profiling")

    # Select features for clustering, excluding non-numeric columns
    selected_features = ['Income', 'Age', 'Experience']
    X = data[selected_features]

    # One-hot encode the 'House_Ownership' column
    data_encoded = pd.get_dummies(data, columns=['House_Ownership'], prefix=['House_Ownership'])

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform K-means clustering (you can adjust the number of clusters)
    num_clusters = 3  # Change this to the desired number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    data_encoded['Cluster'] = kmeans.fit_predict(X_scaled)

    # Interpret clusters
    cluster_info = {}
    for cluster_num in range(num_clusters):
        cluster_data = data_encoded[data_encoded['Cluster'] == cluster_num]
        cluster_info[f'Cluster {cluster_num + 1}'] = {
            'Number of Customers': cluster_data.shape[0],
            'Income (mean)': cluster_data['Income'].mean(),
            'Age (mean)': cluster_data['Age'].mean(),
            'Experience (mean)': cluster_data['Experience'].mean(),
            'House Ownership Rate': cluster_data['House_Ownership_owned'].mean(),  # Adjust column name
        }

    # Display cluster information
    st.write(cluster_info)

    # Visualize the clusters (you can customize the visualization)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=data_encoded, x='Income', y='Age', hue='Cluster', palette='viridis', legend='full')
    plt.title('K-means Clustering')
    plt.xlabel('Income')
    plt.ylabel('Age')
    st.pyplot(fig)
    
    # Formulate hypotheses based on cluster analysis
    hypotheses = {
        "Cluster 1 (Purple)": [
            "Cluster 1 includes financially well-off and experienced individuals, possibly close to retirement.",
            "They tend to rent or have housing arrangements other than traditional homeownership.",
            "Financial products they may find useful: Retirement planning, wealth management, and investment services."
        ],
        
        "Cluster 2 (Green)": [
            "Cluster 2 represents younger professionals with moderate incomes.",
            "Many in this group prefer renting or shared housing.",
            "Financial products they may find useful: Personal loans, credit cards, and career-focused investments."
        ],
        
        "Cluster 3 (Yellow)": [
            "Cluster 3 comprises older individuals with moderate incomes, possibly nearing retirement.",
            "Similar to Cluster 1 and 2, they have a lower rate of homeownership.",
            "Financial products they may find useful: Retirement planning, healthcare, and services tailored for older individuals."
        ]
    }

    # Display hypotheses
    st.subheader("Hypotheses:")
    for cluster, bullet_points in hypotheses.items():
        st.write(f"**{cluster}:**")
        for point in bullet_points:
            st.write(f"- {point}")
            
# Function for creating and displaying a correlation heatmap
@st.cache_data
def create_correlation_heatmap(data, columns):
    st.subheader("Correlation Heatmap")

    # Select the columns for correlation analysis
    selected_data = data[columns]

    # Calculate the correlation matrix
    correlation_matrix = selected_data.corr()

    # Create a heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    plt.title('Correlation Heatmap')

    # Display the heatmap using st.pyplot()
    st.pyplot(fig)












# Streamlit app layout
def main():
    st.title("Data Analysis Web App")
    st.sidebar.title("Menu")

    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload your Excel file", type=["xlsx"])

    if uploaded_file is not None:
        data = load_and_clean_data(uploaded_file)
        
        if st.sidebar.button("Show Data Info"):
            display_data_info(data)

        if st.sidebar.button("Analyze Data"):
            analyze_banking_data(data)

        # Add the button for Demographic Segmentation Analysis
        if st.sidebar.button("Demographic Segmentation Analysis"):
            demographic_segmentation_analysis(data)
    
        if st.sidebar.button("Geographic Segmentation by State"):
            top_10_states_by_income(data)
            
        if st.sidebar.button("Correlation Heatmap"):
            selected_columns = ['Income', 'Age', 'Experience', 'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS']
            create_correlation_heatmap(data, selected_columns)
        
        # Add the button for Clustering Execution and Interpretation
        if st.sidebar.button("Customer Profiling"):
            clustering_and_interpretation(data)

if __name__ == "__main__":
    main()
