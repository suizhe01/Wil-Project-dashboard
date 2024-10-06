import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    initial_sidebar_state="collapsed"  # Sidebar will be collapsed on load
)

# Streamlit App
st.title("An adaptive learning platform to tailor the existing curriculum based on different students’ learning pace and style")
# st.title("Random Forest Classifier with Grid Search")

# Specify the file path directly from your local machine
file_path = 'SL_csv.csv'  # Replace with the correct path to your file
# st.write(f"Reading data from: {file_path}")

# Load the dataset
data = pd.read_csv(file_path)

st.write("# Uderstanding and analyse on data")

# Display the data types of each column
st.write("### Data Types:")
st.write(data.dtypes)

# Display the first 5 rows of the DataFrame (equivalent to df.head())
st.write("### First 5 rows of the DataFrame:")
st.write(data.head())

# Check for missing values and display the count for each column
st.write("### Missing Values in Each Column:")
st.write(data.isnull().sum())

# Display the value counts of the 'Learner' column
st.write("### Value Counts for target variable which is 'Learner':")
st.write(data['Learner'].value_counts())

# Non-Numeric Column Detection
st.write("### Non-Numeric Columns and Their Head Values")
non_numeric = []
for col in data.columns:
    if data[col].dtype != 'int64' and data[col].dtype != 'float64':  # Adjusted to detect non-numeric columns
        non_numeric.append(col)
        
        # Display the first 5 values of each non-numeric column
        st.write(f"#### {col} (First 5 values):")
        st.write(data[col].head())
        st.write('\n')  # Adds spacing between columns for readability


# Displaying the list of non-numeric columns
st.write("### List of Non-Numeric Columns:")
st.write(non_numeric)

st.write("### The minimum and maximum age of student")
# Display the maximum and minimum values of the 'Age' column
st.write("Maximum Age:", data['Age'].max())
st.write("Minimum Age:", data['Age'].min())

# Dictionary to store the highest age for each numerical column
highest_age_per_column = {}

# Loop through all numerical columns except 'Age'
for col in data.select_dtypes(include='number').columns:
    if col != 'Age':
        max_index = data[col].idxmax()  # Get the index of the maximum value
        highest_age = data.loc[max_index, 'Age']  # Get the age at that index
        highest_age_per_column[col] = highest_age

# Display the results using st.write()
st.write("### Age with the highest value in each numerical column:")

for col, age in highest_age_per_column.items():
    st.write(f"The age with the highest value in '{col}' is: {age}")


st.write("# Data Visualization")
# Visualization Section: Displaying bar plots
st.write("### Feature Distributions by Age")

# Loop through columns and display bar plots (excluding Age, Gender, and Learner)
for col in data.columns:
    if col != 'Age' and col != 'Gender' and col != 'Learner':
        st.write(f"#### Bar Plot for {col}")
        
        # Create the bar plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Age', y=col, data=data)
        plt.title(f"Bar Plot for {col}")
        
        # Display the plot in Streamlit
        st.pyplot(plt)
        plt.clf()  # Clear the plot for the next one

# Distribution of Age by Learning Style
st.write("### Distribution of Age by Learning Style")

plt.figure(figsize=(10, 6))
sns.boxplot(x='Learner', y='Age', data=data)
plt.title('Distribution of Age by Learning Style')

# Display the plot in Streamlit
st.pyplot(plt)

# Count the Learners by Gender
st.write("### Count of Learners by Gender")

plt.figure(figsize=(8, 5))
sns.countplot(x='Learner', hue='Gender', data=data)
plt.title('Count of Learners by Gender')

# Display the plot in Streamlit
st.pyplot(plt)


# Convert categorical variables into numeric values using one-hot encoding
df_numeric = pd.get_dummies(data)

# Calculate the correlation matrix
correlation_matrix = df_numeric.corr()

# Visualize the correlation matrix in a heatmap
st.write("### Correlation Matrix of the Dataset")

plt.figure(figsize=(24, 24))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of the Dataset')

# Display the plot in Streamlit
st.pyplot(plt)


st.title("Random Forest Classifier with Grid Search")
# Convert categorical 'Learner' type into numerical labels
label_encoder = LabelEncoder()
data['Learner'] = label_encoder.fit_transform(data['Learner'])

# Convert categorical variables to numerical
X = pd.get_dummies(data.drop(['Learner'], axis=1))

# Extract the target variable
y = data['Learner']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest model
rf_model = RandomForestClassifier(random_state=42)

st.sidebar.header("User Input Parameters")

# Parameters for GridSearchCV - Adjustable by the user
n_estimators = st.sidebar.slider("Number of estimators", min_value=50, max_value=200, value=100, step=1)
max_depth = st.sidebar.slider("Max depth", min_value=10, max_value=30, value=20, step=1)
min_samples_split = st.sidebar.slider("Min samples split", min_value=2, max_value=10, value=2, step=1)
min_samples_leaf = st.sidebar.slider("Min samples leaf", min_value=1, max_value=4, value=1, step=1)
bootstrap = st.sidebar.selectbox("Bootstrap", [True, False], index=0)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [n_estimators],
    'max_depth': [max_depth],
    'min_samples_split': [min_samples_split],
    'min_samples_leaf': [min_samples_leaf],
    'bootstrap': [bootstrap]
}

# Grid Search using 5-fold cross-validation
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

grid_search.fit(X_train, y_train)
            
# Best model after Grid Search
best_rf_model = grid_search.best_estimator_
            
# Predict using the best model
y_pred = best_rf_model.predict(X_test)
            
# Evaluating the best model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
best_params = grid_search.best_params_
            
# Displaying the results
st.write("### Best Parameters:", best_params)
st.write(f"### Accuracy: {accuracy:.4f}")
st.text("### Classification Report:\n")
st.text(report)
