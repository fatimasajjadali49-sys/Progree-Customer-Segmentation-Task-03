
# Customer Segmentation using K-Means Clustering

## Internship Task 03 – Machine Learning

**Project Title:** Customer Segmentation using K-Means  
**Internship Task:** Task 03  
**Intern:** Fatima Sajjad Ali  
**Programming Language:** Python  
**Development Environment:** Visual Studio Code

---

## 1. Project Overview

Customer Segmentation is an important application of Machine Learning that helps businesses understand different groups of customers based on their characteristics and purchasing behavior.

In this project, the K-Means Clustering algorithm is used to divide customers into different groups based on three selected features:

- Age
- Annual Income
- Spending Score

The project demonstrates an unsupervised machine learning approach in which the algorithm identifies groups in the dataset without using predefined labels.

The main purpose is to explore customer data, identify patterns, apply clustering, and visualize the resulting customer groups.

---

## 2. Project Objectives

The major objectives of this project are:

1. Generate and explore a customer dataset.
2. Understand the structure and features of customer data.
3. Perform data preprocessing.
4. Select relevant features for clustering.
5. Standardize numerical features using StandardScaler.
6. Determine a suitable number of clusters using the Elbow Method.
7. Implement the K-Means Clustering algorithm.
8. Apply Principal Component Analysis (PCA).
9. Visualize customer clusters using graphs.
10. Export the final customer segmentation results into a CSV file.

---

## 3. Technologies and Libraries

The following technologies and Python libraries were used:

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations and sample data generation |
| Matplotlib | Data visualization |
| Scikit-learn | Machine learning algorithms |
| VS Code | Development environment |

### Python Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
```

---

## 4. Dataset Information

For this project, a sample customer dataset was generated using Python and NumPy.

The dataset contains 200 customer records.

### Dataset Features

| Feature | Description |
|---|---|
| CustomerID | Unique identification number for each customer |
| Age | Age of the customer |
| Annual Income (k$) | Annual income of the customer in thousands |
| Spending Score (1-100) | Assigned spending score |

### Dataset Shape

- Total Records: 200
- Total Columns: 4
- Clustering Features: 3

The CustomerID column is used for identification and is not included in the clustering features.

---

## 5. Data Exploration

The dataset was explored to understand its structure and contents.

The following operations were performed:

- Displayed the first five records.
- Checked the dataset dimensions.
- Examined the data types.
- Reviewed the selected numerical features.

These steps helped understand the dataset before applying machine learning.

---

## 6. Feature Selection

Three numerical features were selected for clustering:

- Age
- Annual Income (k$)
- Spending Score (1-100)

These features were used to identify groups of customers with different characteristics.

CustomerID was excluded because it is only an identification field.

---

## 7. Data Preprocessing and Standardization

Before applying K-Means, the selected features were standardized using StandardScaler.

Standardization helps bring numerical features onto a comparable scale.

This is important because K-Means uses distances to assign observations to clusters.

### Standardization Code

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

The standardized data was then used for clustering.

---

## 8. Elbow Method

The Elbow Method was implemented to examine the appropriate number of clusters.

Different K-Means models were tested with cluster counts ranging from 1 to 10.

The Within-Cluster Sum of Squares (WCSS) was calculated for each model.

The results were plotted to create the Elbow Method graph.

This graph helps identify a point where increasing the number of clusters provides smaller improvements.

### Output

`elbow_method.png`

---

## 9. K-Means Clustering

The K-Means algorithm was applied to the standardized customer data.

The model was configured with:

- Number of Clusters: 4
- Random State: 42
- Number of Initializations: 10

The algorithm assigned each customer to one of four clusters.

### Implementation

```python
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)
```

The resulting cluster labels were added to the original dataset.

---

## 10. Principal Component Analysis (PCA)

Principal Component Analysis (PCA) was used to reduce the standardized dataset from three dimensions to two dimensions.

PCA makes it possible to represent the customer data in a 2D graph.

### PCA Implementation

```python
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

The transformed data was used to create a PCA-based cluster visualization.

### Output

`pca_clusters.png`

---

## 11. Data Visualization

Matplotlib was used to create visual representations of the results.

Three visualizations were generated:

### A. Elbow Method Graph

Displays WCSS values for different numbers of clusters.

File: `elbow_method.png`

### B. Customer Cluster Visualization

Displays customers based on Annual Income and Spending Score.

Different colors represent different cluster assignments.

File: `customer_clusters.png`

### C. PCA Cluster Visualization

Displays customer clusters using the first two principal components.

File: `pca_clusters.png`

---

## 12. Clustering Results

The K-Means model divided the 200 customer records into four clusters.

### Cluster Distribution

| Cluster | Number of Customers |
|---|---:|
| Cluster 0 | 55 |
| Cluster 1 | 50 |
| Cluster 2 | 39 |
| Cluster 3 | 56 |
| **Total** | **200** |

These values represent the number of records assigned to each cluster in this run.

---

## 13. Exporting Results

After completing the clustering process, the final dataset was exported into a CSV file.

The output file contains the original customer features along with the assigned cluster labels.

### Export Code

```python
df.to_csv("customer_segments.csv", index=False)
```

This file can be opened and analyzed using spreadsheet software or Python.

---

## 14. Project Files and Outputs

The project folder contains the following files:

| File Name | Description |
|---|---|
| customer_segmentation.py | Main Python source code |
| customer_segments.csv | Final customer segmentation dataset |
| elbow_method.png | Elbow Method graph |
| customer_clusters.png | Customer cluster graph |
| pca_clusters.png | PCA-based cluster graph |
| README.md | Project documentation |

---

## 15. Key Learning Outcomes

Through this project, I gained practical experience in:

- Working with customer datasets.
- Exploring and preparing numerical data.
- Applying feature standardization.
- Understanding unsupervised machine learning.
- Implementing K-Means Clustering.
- Using the Elbow Method.
- Understanding Principal Component Analysis.
- Creating visualizations with Matplotlib.
- Exporting machine learning results into CSV format.
- Organizing a Python machine learning project.

---

## 16. Conclusion

This project demonstrates the implementation of Customer Segmentation using the K-Means Clustering algorithm.

The dataset was generated and explored, numerical features were standardized, and the Elbow Method was used to examine different cluster counts.

K-Means was then applied with four clusters to group the customer records.

PCA was also implemented to reduce the data to two dimensions for visualization.

Overall, this project provided practical experience with data preprocessing, unsupervised machine learning, clustering techniques, and data visualization using Python.

---

## 17. Author

**Name:** Fatima Sajjad Ali  
**Project:** Customer Segmentation using K-Means Clustering  
**Task:** Machine Learning Internship – Task 03

---
