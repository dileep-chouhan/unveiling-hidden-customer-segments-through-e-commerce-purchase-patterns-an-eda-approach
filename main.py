import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
# Number of customers
num_customers = 500
# Generate synthetic data
data = {
    'CustomerID': range(1, num_customers + 1),
    'Age': np.random.randint(18, 65, num_customers),
    'Gender': np.random.choice(['Male', 'Female'], num_customers),
    'TotalSpent': np.random.exponential(scale=500, size=num_customers), #Skewed distribution for spending
    'AvgOrderValue': np.random.uniform(50, 200, num_customers),
    'PurchaseFrequency': np.random.poisson(lam=5, size=num_customers) #Poisson distribution for frequency
}
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Feature Engineering ---
# No significant cleaning needed for synthetic data, but this section is crucial for real-world datasets.
# --- 3. Exploratory Data Analysis (EDA) ---
# Descriptive Statistics
print("Descriptive Statistics:")
print(df.describe())
# Customer Segmentation using KMeans Clustering (example)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# Scale numerical features for KMeans
scaler = StandardScaler()
numerical_features = ['Age', 'TotalSpent', 'AvgOrderValue', 'PurchaseFrequency']
df_scaled = df.copy()
df_scaled[numerical_features] = scaler.fit_transform(df[numerical_features])
# Determine optimal number of clusters (e.g., using the Elbow method - not implemented here for brevity)
# In a real-world scenario, techniques like the elbow method or silhouette analysis would be used.
n_clusters = 3 # Choosing 3 clusters as an example
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df_scaled['Cluster'] = kmeans.fit_predict(df_scaled[numerical_features])
# --- 4. Visualization ---
# Pairplot to visualize relationships between features and clusters
plt.figure(figsize=(12, 10))
sns.pairplot(df_scaled, hue='Cluster', vars=numerical_features, diag_kind='kde')
plt.suptitle('Pairplot of Customer Segments', y=1.02)
plt.savefig('customer_segments_pairplot.png')
print("Plot saved to customer_segments_pairplot.png")
# Bar plot of cluster sizes
plt.figure(figsize=(8, 6))
cluster_counts = df_scaled['Cluster'].value_counts()
sns.barplot(x=cluster_counts.index, y=cluster_counts.values)
plt.title('Distribution of Customers Across Clusters')
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.savefig('cluster_distribution.png')
print("Plot saved to cluster_distribution.png")
#Further analysis and visualization can be added based on specific business needs.  For example, you might analyze the average age, spending habits, etc. within each cluster.