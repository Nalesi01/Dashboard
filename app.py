import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3
import numpy as np
import plotly.graph_objects as go
from pywaffle import Waffle
import plotly.express as px

# Load data
def load_data():
    uploaded_file = st.file_uploader("Upload your customer data CSV", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
    else:
        data = pd.read_csv("/Users/naledi/Downloads/Problem_2.csv")  # fallback to default
        st.info("Using default dataset. Upload a file to replace.")
    return data

# data = load_data()

# def load_data():
#     data = pd.read_csv("/Users/naledi/Downloads/Problem_2.csv")  # Update this path if needed
#     return data

data = load_data()

# List of product columns
product_cols = ['INSCOProduct', 'PPSIProduct', 'PROFMEDProduct', 'STIProduct']

# Create binary ownership flags for each product column (1 if owned, 0 if not)
for col in product_cols:
    data[col + '_own'] = data[col].notna().astype(int)

# Calculate the ownership count for each product
product_ownership_count = {
    col: data[data[col + '_own'] == 1].shape[0]  # Count customers who own the product
    for col in product_cols
}

# Total number of customers
total_customers = len(data)

# Calculate percentage of customers owning each product
product_ownership_percentage = {
    col.replace('Product', ''): (count / total_customers) * 100
    for col, count in product_ownership_count.items()
}

# Streamlit dashboard
st.title("📊 Product Ownership Insights Dashboard")

# 1. Percentage with Each Product
with st.expander("📈 % of Customers with Each Product", expanded=True):
    # Plot the percentage of ownership for each product
    plt.figure(figsize=(8, 6))  # Adjusted figure size without changing aspect ratio
    sns.set(style="whitegrid")  # Apply a grid style for better visibility

    # Create a bar plot with custom aesthetics using the "magma" color palette
    ax = sns.barplot(x=list(product_ownership_percentage.keys()), 
                     y=list(product_ownership_percentage.values()), 
                     palette="viridis")  # Magma color palette

    # Add title and labels with larger font
    ax.set_title('% of Customers with Each Product', fontsize=16)
    ax.set_xlabel('Product', fontsize=14)
    ax.set_ylabel(r'% customers with product', fontsize=14)
    
    # Rotate x-ticks for readability
    plt.xticks(rotation=0, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add gridlines for better visibility
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Display the plot
    st.pyplot(plt)

st.divider()

# # 2. Correlation Heatmap
# with st.expander("🔗 Correlation Between Products", expanded=False):
#     fig, ax = plt.subplots()
    
#     # Select and rename the columns (remove "_own")
#     corr_data = data[[col + '_own' for col in product_cols]].astype(int)
#     corr_data.columns = [col.replace("Product", "") for col in product_cols]  # e.g., INSCO, PPSI, etc.
    
#     sns.heatmap(corr_data.corr(), annot=True, cmap='coolwarm', ax=ax)
#     ax.set_title('Correlation Between Products')
#     st.pyplot(fig)



# 3. Heatmap by Demographics
# with st.expander("🧑‍🤝‍🧑 Product Uptake by Demographic Group", expanded=False):
#     group_by_option = st.selectbox("Group Heatmap By", ['Gender', 'Ethnicity', 'MemberTitle', 'OccupationGrouping'], key="group_heat")
#     heat_data = data.groupby(group_by_option)[[col + '_own' for col in product_cols]].mean().T * 100
#     heat_data.index = [col.replace("Product_own", "") for col in heat_data.index]
#     fig, ax = plt.subplots(figsize=(10, 4))
#     sns.heatmap(heat_data, annot=True, cmap='YlGnBu', ax=ax)
#     ax.set_title(f'Product Uptake by {group_by_option}')
#     st.pyplot(fig)

# st.divider()
# with st.expander("🧑‍🤝‍🧑 Product Uptake by Demographic Group", expanded=False):
#     group_by_option = st.selectbox("Group Heatmap By", ['Gender', 'Ethnicity', 'MemberTitle', 'OccupationGrouping'], key="group_heat")
    
#     # Compute mean uptake and convert to percentage integers
#     heat_data = data.groupby(group_by_option)[[col + '_own' for col in product_cols]].mean().T * 100
#     heat_data = heat_data.round(0).astype(int)  # Round and convert to integers
#     heat_data.index = [col.replace("Product_own", "") for col in heat_data.index]

#     # Plot
#     fig, ax = plt.subplots(figsize=(10, 4))
#     sns.heatmap(heat_data, annot=True, cmap='YlGnBu', fmt="d", ax=ax)
#     ax.set_title(f'Product Uptake by {group_by_option}')
#     st.pyplot(fig)

# st.divider()

with st.expander("📊 Customer Demographics", expanded=False):
    st.info("Explore customer demographics by age, gender, occupation, and ethnicity.")

    plot_option = st.selectbox(
        "Select a demographic view:",
        ("Age Distribution", "Gender Distribution", "Occupation Grouping Distribution", "Ethnicity Distribution")
    )

    if plot_option == "Age Distribution":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data['Age'], kde=True, ax=ax, color="skyblue")
        ax.set_title('Age Distribution of Customers')
        ax.set_xlabel('Age')
        ax.set_ylabel('Count')
        st.pyplot(fig)

    elif plot_option == "Gender Distribution":
            gender_counts = data['Gender'].value_counts()
            labels = gender_counts.index
            sizes = gender_counts.values
            colors = sns.color_palette("Set2", len(labels))

            fig, ax = plt.subplots(figsize=(6, 6))
            wedges, texts, autotexts = ax.pie(
                sizes,
                labels=labels,
                autopct='%1.1f%%',
                startangle=90,
                colors=colors,
                textprops={'color': "black"}
            )
            ax.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle.
            ax.set_title("Gender Distribution")
            st.pyplot(fig) 


    


    elif plot_option == "Occupation Grouping Distribution":
        # Calculate relative frequencies
        occupation_counts = data['OccupationGrouping'].value_counts(normalize=True)

        # Separate into major and minor occupations
        threshold = 0.026  # 4.9%
        major_occupations = occupation_counts[occupation_counts >= threshold]
        minor_total = occupation_counts[occupation_counts < threshold].sum()

        # Merge small slices under "Other"
        plot_data = major_occupations.copy()
        if minor_total > 0:
            plot_data['Other'] = minor_total

        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(
            plot_data,
            labels=None,  # Hide labels on chart
            autopct='%1.1f%%',
            startangle=140,
            wedgeprops=dict(width=0.4),  # Donut style
            colors=sns.color_palette("Set3", n_colors=len(plot_data))
        )
        ax.set_title('Occupation Grouping Distribution (Donut Chart)')
        ax.axis('equal')  # Equal aspect ratio for a circle
        ax.legend(wedges, plot_data.index, title="Occupations", loc="center left", bbox_to_anchor=(1, 0.5))

        st.pyplot(fig)


    elif plot_option == "Ethnicity Distribution":
        data['Ethnicity'] = data['Ethnicity'].replace('Will Not Disclose', 'None')

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x='Ethnicity', data=data, ax=ax, palette='muted')
        ax.set_title('Ethnicity Distribution')
        ax.set_xlabel('Ethnicity')
        ax.set_ylabel('Count')
        st.pyplot(fig)



with st.expander("🧑‍🤝‍🧑 Product Uptake by Demographic Group", expanded=False):
    group_by_option = st.selectbox("Group Heatmap By", ['Gender', 'Ethnicity', 'MemberTitle', 'OccupationGrouping'], key="group_heat")
    
    # Compute mean uptake and divide by 100 for percentage representation
    heat_data = data.groupby(group_by_option)[[col + '_own' for col in product_cols]].mean().T * 100
    heat_data = heat_data.round(0).astype(int)  # Convert to integers
    heat_data.index = [col.replace("Product_own", "") for col in heat_data.index]

    # Mask to remove annotations where value is 100
    annot_data = heat_data.applymap(lambda x: '' if x == 100 else str(x))  # Hide number if 100

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(heat_data, annot=annot_data, cmap='YlGnBu', fmt="", ax=ax)  # Don't use fmt here as we're controlling annotation directly
    ax.set_title(f'Product Uptake by {group_by_option}')
    
    # Add a subtle note to avoid cluttering the title
    st.markdown("Values represent product uptake percentages (as integers), with 100% values hidden.")
    st.pyplot(fig)

st.divider()





# # 4. Venn Diagram
# with st.expander("🟣 Product Overlap (Venn Diagram)", expanded=False):
#     st.info("Showing overlap of INSCO, PPSI, and PROFMED")
    
#     # Use the rows where each product is owned, based on non-null values
#     A = set(data[data['INSCOProduct'].notna()].index)
#     B = set(data[data['PPSIProduct'].notna()].index)
#     C = set(data[data['PROFMEDProduct'].notna()].index)

#     fig, ax = plt.subplots()
#     venn3([A, B, C], ('INSCO', 'PPSI', 'PROFMED'))
#     ax.set_title("Product Overlap")
#     st.pyplot(fig)

# st.divider()
# with st.expander("🟣 Product Overlap Heatmap", expanded=False):
#     st.info("Showing number of customers who own multiple products.\n\n"
#             "**Product Keys:**\n"
#             "- **INSCO** = Insurance Company Product\n"
#             "- **PPSI** = PPS Investment Product\n"
#             "- **PROFMED** = Professional Medical Aid\n"
#             "- **STI** = Short-Term Insurance Product")

#     A = set(data[data['INSCOProduct'].notna()].index)
#     B = set(data[data['PPSIProduct'].notna()].index)
#     C = set(data[data['PROFMEDProduct'].notna()].index)
#     D = set(data[data['STIProduct'].notna()].index)

#     overlap_data = {
#         'INSCO & PPSI': len(A & B),
#         'INSCO & PROFMED': len(A & C),
#         'INSCO & STI': len(A & D),
#         'PPSI & PROFMED': len(B & C),
#         'PPSI & STI': len(B & D),
#         'PROFMED & STI': len(C & D),
#         'INSCO & PPSI & PROFMED': len(A & B & C),
#         'INSCO & PPSI & STI': len(A & B & D),
#         'INSCO & PROFMED & STI': len(A & C & D),
#         'PPSI & PROFMED & STI': len(B & C & D),
#         'All Products (INSCO, PPSI, PROFMED, STI)': len(A & B & C & D)
#     }

#     overlap_df = pd.DataFrame(list(overlap_data.items()), columns=['Combination', 'Count'])

#     fig, ax = plt.subplots(figsize=(10, 4))
#     sns.heatmap(overlap_df.set_index('Combination').T, annot=True, cmap='YlGnBu', ax=ax, cbar=True, fmt="d")
#     ax.set_title('Product Overlap Counts')
#     st.pyplot(fig)


# Assuming 'data' is your DataFrame and already contains the necessary columns

with st.expander("🟣 Product Overlap Heatmap by Occupation", expanded=False):
    st.info("Showing number of customers who own multiple products by Occupation Group.\n\n"
            "**Product Keys:**\n"
            "- **INSCO** = Insurance Company Product\n"
            "- **PPSI** = PPS Investment Product\n"
            "- **PROFMED** = Professional Medical Aid\n"
            "- **STI** = Short-Term Insurance Product")

    # Define the product sets
    product_columns = {
        "INSCO": "INSCOProduct",
        "PPSI": "PPSIProduct",
        "PROFMED": "PROFMEDProduct",
        "STI": "STIProduct"
    }

    # Prepare a dictionary to store the overlap data by occupation
    overlap_data_by_occupation = {}

    # Loop through each occupation group
    for occupation in data["OccupationGrouping"].unique():
        # Filter data by occupation group
        occupation_data = data[data["OccupationGrouping"] == occupation]
        
        # Create product ownership sets for each product type
        A = set(occupation_data[occupation_data[product_columns["INSCO"]].notna()].index)
        B = set(occupation_data[occupation_data[product_columns["PPSI"]].notna()].index)
        C = set(occupation_data[occupation_data[product_columns["PROFMED"]].notna()].index)
        D = set(occupation_data[occupation_data[product_columns["STI"]].notna()].index)
        
        # Calculate overlaps for each product combination
        overlap_data = {
            'INSCO & PPSI': len(A & B),
            'INSCO & PROFMED': len(A & C),
            'INSCO & STI': len(A & D),
            'PPSI & PROFMED': len(B & C),
            'PPSI & STI': len(B & D),
            'PROFMED & STI': len(C & D),
            'INSCO & PPSI & PROFMED': len(A & B & C),
            'INSCO & PPSI & STI': len(A & B & D),
            'INSCO & PROFMED & STI': len(A & C & D),
            'PPSI & PROFMED & STI': len(B & C & D),
            'All Products (INSCO, PPSI, PROFMED, STI)': len(A & B & C & D)
        }
        
        # Store overlap data by occupation
        overlap_data_by_occupation[occupation] = overlap_data

    # Convert the dictionary of overlap data to a DataFrame
    overlap_df = pd.DataFrame(overlap_data_by_occupation).T

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(overlap_df, annot=True, cmap='YlGnBu', ax=ax, cbar=True, fmt="d")
    ax.set_title('Product Overlap Counts by Occupation')
    ax.set_xlabel('Product Combinations')
    ax.set_ylabel('Occupation Group')

    st.pyplot(fig)


# Assuming 'data' is your DataFrame
product_columns = ['INSCOProduct', 'PPSIProduct', 'PROFMEDProduct', 'STIProduct']

# Mapping old column names to new names
product_column_names = {
    'INSCOProduct': 'INSCO',
    'PPSIProduct': 'PPSI',
    'PROFMEDProduct': 'PROFMED',
    'STIProduct': 'STIP'
}

# Rename columns in the DataFrame
data.rename(columns=product_column_names, inplace=True)

# Create a binary indicator for each product (1 if owned, 0 if not owned)
for col in product_column_names.values():  # Use new column names
    if col in data.columns:  # Check if the column exists
        data[col] = data[col].notna().astype(int)

# If 'GrossPersonalIncome' is numerical, categorize it into income ranges (optional)
income_bins = [0, 25000, 50000, 100000, 150000, np.inf]
income_labels = ['0-25k', '25k-50k', '50k-100k', '100k-150k', '150k+']
data['IncomeCategory'] = pd.cut(data['GrossPersonalIncome'], bins=income_bins, labels=income_labels)

# Convert product_column_names.values() to a list before concatenating with ['IncomeCategory']
pivot_data = data[list(product_column_names.values()) + ['IncomeCategory']].groupby(list(product_column_names.values()) + ['IncomeCategory']).size().unstack(fill_value=0)

# Create the expander to allow for collapsible/expandable plot
with st.expander("📊 Product Ownership vs Income"):
    st.write("### Heatmap: Product Ownership vs Income")

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot_data, annot=True, fmt="d", cmap="YlOrRd", cbar=True, ax=ax, annot_kws={"ha": "center", "va": "center"})

    # Adjust y-tick labels to be centered
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='center', fontsize=10)

    # Explicitly adjust tick positions by changing padding and rotation
    ax.tick_params(axis='y', which='major', length=0)  # Remove the ticks on the y-axis
    ax.yaxis.set_tick_params(pad=15)  # Add padding to make the labels more readable
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='center', va='center', fontsize=12)

    # Rotate x-axis labels to prevent squishing and make them readable
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='right', fontsize=10)

    ax.set_title("Income vs Product Ownership Frequency")
    st.pyplot(fig)



# Define bins for income and age groups
income_bins = [0, 20000, 50000, float('inf')]  # Adjust based on your data
income_labels = ['Low', 'Middle', 'High']
data['IncomeGroup'] = pd.cut(data['GrossPersonalIncome'], bins=income_bins, labels=income_labels)

age_bins = [0, 25, 35, 45, 60, float('inf')]  # Adjust based on your data
age_labels = ['18-25', '26-35', '36-45', '46-60', '60+']
data['AgeGroup'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels)

# Create a new expander for "Demographic Insights: Income, Age & Gender"
with st.expander("📊 Demographic Insights: Income, Age & Gender", expanded=False):
    st.info("Explore the distribution of customers based on income, age, and gender.")

    # Set up the grid for subplots (2 rows, 1 column)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Loop through genders and create heatmaps for each
    for i, gender in enumerate(data['Gender'].unique()):
        # Filter data by gender
        gender_data = data[data['Gender'] == gender]

        # Create pivot table for Income vs Age counts by gender
        heatmap_data = pd.crosstab(gender_data['IncomeGroup'], gender_data['AgeGroup'])

        # Plot heatmap for each gender
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='OrRd', cbar=True, ax=axes[i], linewidths=0.5)
        axes[i].set_title(f'Income vs Age Group Distribution - {gender}', fontsize=12)
        axes[i].set_xlabel('Age Group', fontsize=10)
        axes[i].set_ylabel('Income Group', fontsize=10)

    # Adjust layout and display
    plt.tight_layout()
    st.pyplot(fig)






# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import streamlit as st

# # Load your dataset (adjust path accordingly)
# data = pd.read_csv('/Users/naledi/Downloads/Problem_2.csv')  # Replace with the actual file path or data source

# # Assuming your dataset has the following columns of interest:
# # INSCOProduct, PPSIProduct, PROFMEDProduct, STIProduct, Age, OccupationGrouping, Ethnicity, GrossPersonalIncome,
# # MemberOccupation, MemberTitle, Gender, InscoCountry, and other columns you're working with.

# with st.expander("🟣 Product Overlap Heatmap", expanded=False):
#     st.info("Showing number of customers who own multiple products.\n\n"
#             "**Product Keys:**\n"
#             "- **INSCO** = Insurance Company Product\n"
#             "- **PPSI** = PPSI Investment Product\n"
#             "- **PROFMED** = Professional Medical Aid\n"
#             "- **STI** = Short-Term Insurance Product")

#     # Create sets for each product to identify which customers own which product
#     A = set(data[data['INSCOProduct'].notna()].index)  # Customers with INSCOProduct
#     B = set(data[data['PPSIProduct'].notna()].index)  # Customers with PPSIProduct
#     C = set(data[data['PROFMEDProduct'].notna()].index)  # Customers with PROFMEDProduct
#     D = set(data[data['STIProduct'].notna()].index)  # Customers with STIProduct

#     # Overlap combinations: Calculate the number of customers in each overlap group
#     overlap_data = {
#         'INSCO & PPSI': len(A & B),
#         'INSCO & PROFMED': len(A & C),
#         'INSCO & STI': len(A & D),
#         'PPSI & PROFMED': len(B & C),
#         'PPSI & STI': len(B & D),
#         'PROFMED & STI': len(C & D),
#         'INSCO & PPSI & PROFMED': len(A & B & C),
#         'INSCO & PPSI & STI': len(A & B & D),
#         'INSCO & PROFMED & STI': len(A & C & D),
#         'PPSI & PROFMED & STI': len(B & C & D),
#         'All Products (INSCO, PPSI, PROFMED, STI)': len(A & B & C & D)
#     }

#     # Convert overlap data to DataFrame for easy visualization
#     overlap_df = pd.DataFrame(list(overlap_data.items()), columns=['Combination', 'Count'])

#     # Creating a matrix for the heatmap
#     overlap_matrix = {
#         'INSCO': [len(A & B), len(A & C), len(A & D), len(A & B & C), len(A & B & D), len(A & C & D), len(A & B & C & D)],
#         'PPSI': [len(A & B), len(B & C), len(B & D), len(A & B & C), len(A & B & D), len(B & C & D), len(A & B & C & D)],
#         'PROFMED': [len(A & C), len(B & C), len(C & D), len(A & B & C), len(A & C & D), len(B & C & D), len(A & B & C & D)],
#         'STI': [len(A & D), len(B & D), len(C & D), len(A & B & D), len(A & C & D), len(B & C & D), len(A & B & C & D)]
#     }

#     # Convert matrix to DataFrame
#     heatmap_df = pd.DataFrame(overlap_matrix, index=['INSCO & PPSI', 'INSCO & PROFMED', 'INSCO & STI', 
#                                                      'PPSI & PROFMED', 'PPSI & STI', 'PROFMED & STI', 
#                                                      'All Products'])

#     # Plot the heatmap using seaborn
#     fig, ax = plt.subplots(figsize=(10, 7))
#     sns.heatmap(heatmap_df, annot=True, cmap='YlGnBu', ax=ax, cbar=True, fmt="d")
#     ax.set_title('Product Overlap Heatmap')

#     # Display the plot in Streamlit
#     st.pyplot(fig)

# with st.expander("📊 Customer Demographics", expanded=False):
#     st.info("Demographics data based on age, gender, occupation, and ethnicity.")

#     # Create a count plot for age distribution
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.histplot(data['Age'], kde=True, ax=ax)
#     ax.set_title('Age Distribution of Customers')
#     st.pyplot(fig)

#     # Gender distribution plot
#     fig, ax = plt.subplots(figsize=(8, 5))
#     sns.countplot(x='Gender', data=data, ax=ax, palette='Set2')
#     ax.set_title('Gender Distribution')
#     st.pyplot(fig)

#     # Occupation group distribution plot
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.countplot(x='OccupationGrouping', data=data, ax=ax, palette='Set3')
#     ax.set_title('Occupation Grouping Distribution')
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
#     st.pyplot(fig)

#     # Ethnicity distribution plot
#     fig, ax = plt.subplots(figsize=(8, 5))
#     sns.countplot(x='Ethnicity', data=data, ax=ax, palette='muted')
#     ax.set_title('Ethnicity Distribution')
#     st.pyplot(fig)

# with st.expander("💰 Gross Personal Income", expanded=False):
#     st.info("Distribution of customers' gross personal income.")

#     # Gross personal income distribution
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.histplot(data['GrossPersonalIncome'], kde=True, ax=ax, color='green')
#     ax.set_title('Gross Personal Income Distribution')
#     st.pyplot(fig)

# with st.expander("📍 Insurance Product Country Distribution", expanded=False):
#     st.info("Geographical distribution of insurance product ownership.")

#     # Insurance product country distribution
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.countplot(x='InscoCountry', data=data, ax=ax, palette='coolwarm')
#     ax.set_title('Insurance Product Country Distribution')
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
#     st.pyplot(fig)

# with st.expander("📊 Customer Product Ownership", expanded=False):
#     st.info("Ownership of insurance, medical aid, and investment products.")

#     # Number of customers owning each product
#     product_ownership = data[['INSCOProduct', 'PPSIProduct', 'PROFMEDProduct', 'STIProduct']].notna().sum()
#     fig, ax = plt.subplots(figsize=(8, 6))
#     product_ownership.plot(kind='bar', color=['lightblue', 'lightgreen', 'lightcoral', 'lightskyblue'], ax=ax)
#     ax.set_title('Number of Customers Owning Each Product')
#     ax.set_ylabel('Count of Customers')
#     st.pyplot(fig)


# Define age bins and labels
age_bins = [0, 25, 35, 45, 60, float('inf')]
age_labels = ['18-25', '26-35', '36-45', '46-60', '60+']
data['AgeGroup'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels)

# Product columns
product_columns = ['INSCO', 'PPSI', 'PROFMED', 'STIP']

# Expander section
with st.expander("📈 Product Uptake by Age Group and Gender", expanded=False):
    # Dropdown for product selection
    selected_product = st.selectbox("Select a Product", product_columns)

    # Group by AgeGroup and Gender and sum the selected product
    grouped = data.groupby(['AgeGroup', 'Gender'])[selected_product].sum().unstack().fillna(0)

    # Plot
    st.write(f"### 📊 {selected_product} Uptake by Age Group and Gender")
    fig, ax = plt.subplots(figsize=(10, 6))
    grouped.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'])  # Blue/Orange for M/F
    ax.set_title(f'{selected_product} Ownership by Age Group and Gender')
    ax.set_ylabel('Number of Customers')
    ax.set_xlabel('Age Group')
    ax.legend(title='Gender')
    plt.xticks(rotation=0)
    plt.tight_layout()

    st.pyplot(fig)
    


# Function to format values with commas
def format_with_commas(value):
    return f"R{value:,}"


product_mapping = {
    "INSCO": "INSCOProduct_own",
    "PPSI": "PPSIProduct_own",
    "PROFMED": "PROFMEDProduct_own",
    "STI": "STIProduct_own"
}

with st.sidebar:
    st.header("Targeting Tool")
    selected_occupation = st.selectbox("Select Occupation Group", data["OccupationGrouping"].unique())
    income_range = st.slider("Gross Personal Income Range (In ZAR)", int(data["GrossPersonalIncome"].min()), int(data["GrossPersonalIncome"].max()), (10000, 50000))
    selected_product = st.selectbox("Select Product", list(product_mapping.keys()))  # Show full product names without "Product"

# Get the actual column name based on the selected product
selected_product_column = product_mapping[selected_product]

# Filter data based on the selected occupation and income range
filtered_data = data[
    (data["OccupationGrouping"] == selected_occupation) &
    (data["GrossPersonalIncome"].between(income_range[0], income_range[1]))
]

# Tab setup
tab1, tab2, tab3 = st.tabs(["Overview", "Visualizations", "Strategic Recommendations"])

# Overview Tab Content
with tab1:
    st.header("Overview")
    # st.write("### High-Level Insights")

    # Using markdown for better formatting and colors
    st.markdown(f"<h3 style='color:#4CAF50;'>Selected Occupation: {selected_occupation}</h3>", unsafe_allow_html=True)
    
    # Format the income range with commas
    formatted_income_range = f"R{income_range[0]:,} - R{income_range[1]:,}"
    st.markdown(f"<h3 style='color:#2196F3;'>Income Range: {formatted_income_range}</h3>", unsafe_allow_html=True)

    # Show a basic summary of the number of customers in the selected occupation and income range
    num_customers = filtered_data.shape[0]
    st.markdown(f"<h2 style='color:#009688;'>Number of Customers: {num_customers}</h2>", unsafe_allow_html=True)

    # You can also display a simple table with counts for each product, if you'd like
    product_summary = filtered_data[selected_product_column].value_counts().reset_index()
    product_summary.columns = [selected_product_column, "Customer Count"]
    product_summary[selected_product_column] = product_summary[selected_product_column].replace({0: "No", 1: "Yes"})

    # Adding a separator line for clarity
    st.markdown("<hr style='border: 1px solid #ddd;'>", unsafe_allow_html=True)

    # Table displaying the product ownership summary
    st.write("#### Product Ownership Summary:")
    st.table(product_summary)





# Visualizations Tab Content
with tab2:
    st.header("Visualizations")
    st.write("### Product Ownership Distribution")

    # Group by the product column and count the number of customers (i.e., non-null entries)
    product_count = filtered_data[selected_product_column].value_counts().reset_index()
    product_count.columns = [selected_product_column, "Customer Count"]

    # Replace 0 and 1 in the x-axis with "No" and "Yes"
    product_count[selected_product_column] = product_count[selected_product_column].replace({0: "No", 1: "Yes"})

    # Create a bar plot showing the number of customers per product ownership
    fig = px.bar(
        product_count, 
        x=selected_product_column, 
        y="Customer Count", 
        title=f"Number of Customers with {selected_product} in {selected_occupation} and Income Range R{income_range[0]} - R{income_range[1]}"
    )

    # Remove the index numbers on the x-axis and update the axis labels to be descriptive
    fig.update_layout(xaxis={'showticklabels': True})

    st.plotly_chart(fig)

# Strategic Recommendations Tab Content
with tab3:
    st.header("Strategic Recommendations")
    st.subheader("1. Targeted Insurance Bundles for High-Income Professionals")
    st.markdown("""
    **Insight:** Both female and male customers aged 46–60 represent the highest earning demographic.

    **Strategy:** Position high-value or premium insurance and investment products specifically for the 46–60 age group. Highlight benefits such as wealth protection, advanced healthcare coverage, and retirement planning. Tailor messaging to reflect their life stage—financial security, legacy planning, and peace of mind. Utilize personalized outreach via email, financial advisor consultations, and exclusive webinars to drive engagement and conversion.
    """)

    st.subheader("2. Upsell Campaign for Medical Professionals")
    st.markdown("""
    **Insight:** The majority of customers work in the medical field and typically hold only two products (INSCO and STI).

    **Strategy:** RLaunch an upsell campaign targeted at medical professionals, encouraging them to expand their coverage with complementary products such as PPSI or PROFMED. Position these as tailored solutions that align with their professional risks, income protection needs, or long-term financial planning. Use targeted messaging through email, in-app notifications, or whatsapp messaging to showcase the added value and peace of mind these additional products provide. Include limited-time bundle discounts or loyalty incentives to drive uptake.
    """)


################## Improved####################
# import pandas as pd
# import streamlit as st
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib_venn import venn3
# import numpy as np
# import plotly.graph_objects as go
# from pywaffle import Waffle
# import plotly.express as px

# # Load data
# def load_data():
#     data = pd.read_csv("/Users/naledi/Downloads/Problem_2.csv")  # Update this path if needed
#     return data

# data = load_data()

# # List of product columns
# product_cols = ['INSCOProduct', 'PPSIProduct', 'PROFMEDProduct', 'STIProduct']

# # Create binary ownership flags for each product column (1 if owned, 0 if not)
# for col in product_cols:
#     data[col + '_own'] = data[col].notna().astype(int)

# # Calculate the ownership count for each product
# product_ownership_count = {
#     col: data[data[col + '_own'] == 1].shape[0]
#     for col in product_cols
# }

# # Total number of customers
# total_customers = len(data)

# # Calculate percentage of customers owning each product
# product_ownership_percentage = {
#     col.replace('Product', ''): (count / total_customers) * 100
#     for col, count in product_ownership_count.items()
# }

# # Streamlit dashboard
# st.title("📊 Product Ownership Insights Dashboard")

# # 1. Percentage with Each Product
# with st.expander("📈 % of Customers with Each Product", expanded=True):
#     plt.figure(figsize=(8, 6))
#     sns.set(style="whitegrid")
#     ax = sns.barplot(x=list(product_ownership_percentage.keys()), 
#                      y=list(product_ownership_percentage.values()), 
#                      palette="viridis")
#     ax.set_title('% of Customers with Each Product', fontsize=16)
#     ax.set_xlabel('Product', fontsize=14)
#     ax.set_ylabel(r'% customers with product', fontsize=14)
#     plt.xticks(rotation=0, ha='right', fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.grid(True, axis='y', linestyle='--', alpha=0.7)
#     st.pyplot(plt)

# st.divider()

# # 2. Customer Demographics
# with st.expander("📊 Customer Demographics", expanded=False):
#     st.info("Explore customer demographics by age, gender, occupation, and ethnicity.")
#     plot_option = st.selectbox(
#         "Select a demographic view:",
#         ("Age Distribution", "Gender Distribution", "Occupation Grouping Distribution", "Ethnicity Distribution")
#     )

#     if plot_option == "Age Distribution":
#         fig, ax = plt.subplots(figsize=(10, 6))
#         sns.histplot(data['Age'], kde=True, ax=ax, color="skyblue")
#         ax.set_title('Age Distribution of Customers')
#         ax.set_xlabel('Age')
#         ax.set_ylabel('Count')
#         st.pyplot(fig)

#     elif plot_option == "Gender Distribution":
#         gender_counts = data['Gender'].value_counts()
#         labels = gender_counts.index
#         sizes = gender_counts.values
#         colors = sns.color_palette("Set2", len(labels))
#         fig, ax = plt.subplots(figsize=(6, 6))
#         wedges, texts, autotexts = ax.pie(
#             sizes,
#             labels=labels,
#             autopct='%1.1f%%',
#             startangle=90,
#             colors=colors,
#             textprops={'color': "black"}
#         )
#         ax.axis('equal')
#         ax.set_title("Gender Distribution")
#         st.pyplot(fig)

#     elif plot_option == "Occupation Grouping Distribution":
#         occupation_counts = data['OccupationGrouping'].value_counts(normalize=True)
#         threshold = 0.026
#         major_occupations = occupation_counts[occupation_counts >= threshold]
#         minor_total = occupation_counts[occupation_counts < threshold].sum()
#         plot_data = major_occupations.copy()
#         if minor_total > 0:
#             plot_data['Other'] = minor_total

#         fig, ax = plt.subplots(figsize=(8, 8))
#         wedges, texts, autotexts = ax.pie(
#             plot_data,
#             labels=None,
#             autopct='%1.1f%%',
#             startangle=140,
#             wedgeprops=dict(width=0.4),
#             colors=sns.color_palette("Set3", n_colors=len(plot_data))
#         )
#         ax.set_title('Occupation Grouping Distribution (Donut Chart)')
#         ax.axis('equal')
#         ax.legend(wedges, plot_data.index, title="Occupations", loc="center left", bbox_to_anchor=(1, 0.5))
#         st.pyplot(fig)

#     elif plot_option == "Ethnicity Distribution":
#         data['Ethnicity'] = data['Ethnicity'].replace('Will Not Disclose', 'None')
#         fig, ax = plt.subplots(figsize=(8, 5))
#         sns.countplot(x='Ethnicity', data=data, ax=ax, palette='muted')
#         ax.set_title('Ethnicity Distribution')
#         ax.set_xlabel('Ethnicity')
#         ax.set_ylabel('Count')
#         st.pyplot(fig)

# st.divider()

# # 3. Heatmap by Demographics
# with st.expander("🧑‍🤝‍🧑 Product Uptake by Demographic Group", expanded=False):
#     group_by_option = st.selectbox("Group Heatmap By", ['Gender', 'Ethnicity', 'MemberTitle', 'OccupationGrouping'], key="group_heat")
#     heat_data = data.groupby(group_by_option)[[col + '_own' for col in product_cols]].mean().T * 100
#     heat_data = heat_data.round(0).astype(int)
#     heat_data.index = [col.replace("Product_own", "") for col in heat_data.index]
#     annot_data = heat_data.applymap(lambda x: '' if x == 100 else str(x))
#     fig, ax = plt.subplots(figsize=(10, 4))
#     sns.heatmap(heat_data, annot=annot_data, cmap='YlGnBu', fmt="", ax=ax)
#     ax.set_title(f'Product Uptake by {group_by_option}')
#     st.markdown("Values represent product uptake percentages (as integers), with 100% values hidden.")
#     st.pyplot(fig)

# st.divider()

# # 4. Product Overlap Heatmap
# with st.expander("🟣 Product Overlap Heatmap", expanded=False):
#     st.info("Showing number of customers who own multiple products.\n\n"
#             "**Product Keys:**\n"
#             "- **INSCO** = Insurance Company Product\n"
#             "- **PPSI** = PPS Investment Product\n"
#             "- **PROFMED** = Professional Medical Aid\n"
#             "- **STI** = Short-Term Insurance Product")

#     A = set(data[data['INSCOProduct'].notna()].index)
#     B = set(data[data['PPSIProduct'].notna()].index)
#     C = set(data[data['PROFMEDProduct'].notna()].index)
#     D = set(data[data['STIProduct'].notna()].index)

#     overlap_data = {
#         'INSCO & PPSI': len(A & B),
#         'INSCO & PROFMED': len(A & C),
#         'INSCO & STI': len(A & D),
#         'PPSI & PROFMED': len(B & C),
#         'PPSI & STI': len(B & D),
#         'PROFMED & STI': len(C & D),
#         'INSCO & PPSI & PROFMED': len(A & B & C),
#         'INSCO & PPSI & STI': len(A & B & D),
#         'INSCO & PROFMED & STI': len(A & C & D),
#         'PPSI & PROFMED & STI': len(B & C & D),
#         'All Products (INSCO, PPSI, PROFMED, STI)': len(A & B & C & D)
#     }

#     overlap_df = pd.DataFrame(list(overlap_data.items()), columns=['Combination', 'Count'])

#     # Pivot table returns float, so we cast after pivot
#     pivot_df = overlap_df.pivot_table(index='Combination', values='Count')
#     pivot_df = pivot_df.astype(int)  # ✅ Force values to integer type

#     fig, ax = plt.subplots(figsize=(12, 6))
#     sns.heatmap(pivot_df, 
#                 annot=True, fmt='d', cmap='Purples', linewidths=0.5, ax=ax, cbar=False)
#     ax.set_title('Customer Overlap Between Products')
#     ax.set_xlabel("")
#     ax.set_ylabel("")
#     plt.xticks(rotation=0)
#     st.pyplot(fig)
