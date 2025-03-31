import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ✅ Load Dataset
file_path = "customer_shopping_data.csv"
df = pd.read_csv(file_path)
print("\n✅ Dataset Loaded Successfully!\n")

# ✅ Preview Dataset
print("Dataset Preview:")
print(df.head())

# ✅ Check for Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# ✅ Data Preprocessing
print("\n🔹 Preprocessing Data...")

# Convert 'invoice_date' to datetime format
df['invoice_date'] = pd.to_datetime(df['invoice_date'], dayfirst=True)

# Feature Engineering: Extract month and year
df['month'] = df['invoice_date'].dt.month
df['year'] = df['invoice_date'].dt.year

# One-hot encoding for categorical variables
categorical_cols = ['gender', 'payment_method', 'shopping_mall']
numerical_cols = ['age', 'quantity', 'price']

# Creating pipelines for transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# Apply transformations
data_scaled = preprocessor.fit_transform(df)

# ✅ Elbow Method for Optimal Clusters
print("\n🔹 Plotting Elbow Method...")

# Elbow method to determine optimal clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# ✅ Apply KMeans Clustering
optimal_clusters = 5
print(f"\n🔹 Applying KMeans Clustering with {optimal_clusters} clusters...")

kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(data_scaled)
df['cluster'] = clusters

# ✅ PCA for Dimensionality Reduction (for visualization)
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)
df['pca1'] = data_pca[:, 0]
df['pca2'] = data_pca[:, 1]

# ✅ Cluster Visualization
plt.figure(figsize=(12, 6))
sns.scatterplot(x='pca1', y='pca2', hue='cluster', palette='viridis', data=df, s=100, alpha=0.8)
plt.title('Customer Segmentation Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# ✅ Customer Segmentation Insights
print("\n🔹 Cluster Insights:")
for cluster in range(optimal_clusters):
    cluster_data = df[df['cluster'] == cluster]
    print(f"\nCluster {cluster} Characteristics:")
    print(f"Number of Customers: {len(cluster_data)}")
    print(f"Average Spending: ${cluster_data['price'].mean():.2f}")
    print(f"Average Quantity Purchased: {cluster_data['quantity'].mean():.2f}")
    print(f"Top Shopping Malls: {cluster_data['shopping_mall'].value_counts().head(3).to_dict()}")

# ✅ Spending Patterns by Age Group and Gender
plt.figure(figsize=(12, 6))
sns.barplot(x='gender', y='price', hue='cluster', data=df, estimator=np.mean)
plt.title('Average Spending by Gender per Cluster')
plt.ylabel('Average Spending ($)')
plt.grid(True)
plt.show()

# ✅ Revenue Contribution by Payment Method
plt.figure(figsize=(12, 6))
payment_revenue = df.groupby('payment_method')['price'].sum().reset_index()
plt.pie(payment_revenue['price'], labels=payment_revenue['payment_method'], autopct='%1.1f%%', startangle=140, colors=sns.color_palette('Set2', len(payment_revenue)))
plt.title('Revenue Contribution by Payment Method')
plt.show()

# ✅ Monthly Sales Trends
monthly_sales = df.groupby(['year', 'month']).agg({'price': 'sum'}).reset_index()
monthly_sales['date'] = pd.to_datetime(monthly_sales[['year', 'month']].assign(day=1))

plt.figure(figsize=(12, 6))
plt.plot(monthly_sales['date'], monthly_sales['price'], marker='o')
plt.title('Monthly Sales Trend')
plt.xlabel('Date')
plt.ylabel('Total Sales ($)')
plt.grid(True)
plt.show()

# ✅ Sample Customers from Each Cluster
print("\n🔹 Sample Customers from Each Cluster:")
for cluster in range(optimal_clusters):
    print(f"\nCluster {cluster}:")
    print(df[df['cluster'] == cluster].sample(5)[['customer_id', 'age', 'price', 'quantity', 'shopping_mall']])

print("\n✅ Program Execution Completed Successfully!")
