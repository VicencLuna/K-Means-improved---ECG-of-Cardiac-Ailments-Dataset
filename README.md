# CLUSTERING - K-Means-improved with a dbscan ---ECG-of-Cardiac-Ailments-Dataset

## README.md for a K-Means Clustering Project with ECG Data

This project applies the K-Means clustering algorithm to an ECG dataset to identify different groups or categories of ECG signals. The code uses Python and various popular data science libraries.

### Main Steps:

* **Data Preprocessing:**  
    * The dataset is loaded using the Pandas library and preprocessed.  
    * First, the "ID" column is removed.  
    * Next, the remaining data is standardized using `StandardScaler` to ensure all features are on the same scale, preventing features with larger variance from dominating the analysis.  
    * Highly correlated columns with an absolute correlation value greater than 0.90 are removed to avoid redundancy and multicollinearity. Additionally, the columns "Pseg," "NN50," "Tseg," "STseg," "hbpermin," and "SDRR" are manually removed.

* **Determining the Number of Clusters:**  
    * The elbow method is used to find the optimal number of clusters (k). This method analyzes the within-cluster sum of squares (WCSS) for different k values and looks for the point where the decrease in WCSS starts to slow down, forming an "elbow" in the plot.

* **Outlier Removal:**  
    * The IsolationForest algorithm with the parameter `contamination='auto'` is used to detect and remove outliers from the dataset. The 'auto' setting allows the algorithm to estimate the proportion of outliers automatically. This step improves clustering accuracy by eliminating data points that do not fit the general pattern.

* **K-Means Parameter Optimization:**  
    * A grid search (`GridSearchCV`) is used to find the best parameters for the K-Means algorithm, maximizing the silhouette score. Optimized parameters include 'n_clusters,' 'init,' 'n_init,' 'max_iter,' and 'algorithm.' The silhouette score measures a data point's similarity to its own cluster compared to other clusters, with a higher score indicating better clustering.

* **Applying K-Means:**  
    * The K-Means algorithm with optimized parameters is applied to the dataset without outliers to create the clusters.

* **Assigning Outliers to Clusters:**  
    * Outliers are assigned to the nearest cluster based on Euclidean distance to the cluster centroids.

* **Result Visualization:**  
    * Clusters and outliers are visualized using a scatter plot. For datasets with more than two dimensions, Principal Component Analysis (PCA) is used to reduce the dimensionality to two dimensions for visualization.

* **Model Evaluation:**  
    * Various metrics are used to evaluate clustering performance, including:  
        * **Inertia:** The sum of squared distances of each point to its cluster centroid. Lower values indicate better clustering.  
        * **Silhouette Score:** Measures a data point's similarity to its cluster compared to other clusters. Higher values indicate better clustering.  
        * **Davies-Bouldin Index:** Measures the ratio of within-cluster dispersion to between-cluster separation. Lower values indicate better clustering.

* **Cluster Analysis:**  
    * Descriptive statistics, such as mean and standard deviation for each feature within each cluster, are analyzed. Boxplots are generated to visualize feature distribution within each cluster.

### Project Files:

* `data.csv`: The file containing the ECG data.
* `result.csv`: The output file with clustered data, including the "ID" and "Category" columns indicating each data point's cluster.
* `matriz_correlacion.csv`: A CSV file containing the data correlation matrix.

### Libraries Used:

* NumPy  
* Pandas  
* Scikit-learn  
* Matplotlib  
* Seaborn  
* Statsmodels  

### Execution Instructions:

1. Ensure the necessary libraries are installed.
2. Download the ECG dataset (`data.csv`) and place it in the input directory.
3. Run the Python script.
4. Clustering results will be saved in the `result.csv` file in the output directory.

### Additional Notes:

* Model parameters, such as the number of clusters (k) and K-Means algorithm settings, can be adjusted based on project needs.
* The code includes comments explaining each step and the rationale behind decisions during clustering.
* The project can be expanded to include more detailed cluster analyses, such as identifying distinguishing features of each group and their interpretation in the context of ECG signals.
