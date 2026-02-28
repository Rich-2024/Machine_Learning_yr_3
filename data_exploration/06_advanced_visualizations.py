"""
Diabetes Prediction Project
File: 02_unsupervised_learning/01_all_unsupervised.py
Purpose: Implement and compare unsupervised learning algorithms
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# Create output directories
# -----------------------------
output_dir = '../outputs/figures/unsupervised'
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Set style
# -----------------------------
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# -----------------------------
# Load and prepare data
# -----------------------------
print("="*60)
print("UNSUPERVISED LEARNING - DIABETES PREDICTION")
print("="*60)

df = pd.read_csv('../diabetes_data_upload.csv')

# Create numeric version
df_numeric = df.copy()
df_numeric['Gender'] = df_numeric['Gender'].map({'Male': 1, 'Female': 0})
df_numeric['class'] = df_numeric['class'].map({'Positive': 1, 'Negative': 0})

symptom_cols = ['Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 
                'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 
                'Irritability', 'delayed healing', 'partial paresis', 
                'muscle stiffness', 'Alopecia', 'Obesity']

for col in symptom_cols:
    df_numeric[col] = df_numeric[col].map({'Yes': 1, 'No': 0})

# Feature matrix
X = df_numeric[symptom_cols].values
y_true = df_numeric['class'].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nDataset shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")
print(f"Class distribution: Positive: {sum(y_true)} ({sum(y_true)/len(y_true)*100:.1f}%), "
      f"Negative: {len(y_true)-sum(y_true)} ({(1-sum(y_true)/len(y_true))*100:.1f}%)")

# ============================================
# 1. K-MEANS CLUSTERING
# ============================================
print("\n" + "="*50)
print("1. K-MEANS CLUSTERING")
print("="*50)

# Optimal k using elbow and silhouette
inertias = []
silhouette_scores = []
K_range = range(2, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# Plot elbow & silhouette
fig, axes = plt.subplots(1,2, figsize=(14,5))
axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('k'); axes[0].set_ylabel('Inertia'); axes[0].set_title('Elbow Method')
axes[0].grid(True, alpha=0.3)

axes[1].plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('k'); axes[1].set_ylabel('Silhouette Score'); axes[1].set_title('Silhouette Score')
axes[1].grid(True, alpha=0.3)

plt.suptitle('K-Means: Optimal Number of Clusters', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir,'kmeans_elbow.png'), dpi=300)
plt.show()

# Choose k=3 (based on elbow/silhouette)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

print(f"\nK-Means with k={optimal_k}: Cluster sizes {np.bincount(kmeans_labels)}, "
      f"Silhouette {silhouette_score(X_scaled, kmeans_labels):.3f}, "
      f"Adjusted Rand {adjusted_rand_score(y_true, kmeans_labels):.3f}")

# ============================================
# 2. DBSCAN CLUSTERING
# ============================================
print("\n" + "="*50)
print("2. DBSCAN CLUSTERING")
print("="*50)

eps_values = np.arange(0.3,1.2,0.1)
dbscan_results = []
for eps in eps_values:
    dbscan = DBSCAN(eps=eps, min_samples=5)
    labels = dbscan.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = sum(labels == -1)
    if n_clusters > 1:
        mask = labels != -1
        sil_score = silhouette_score(X_scaled[mask], labels[mask]) if sum(mask)>1 else 0
    else: sil_score=0
    dbscan_results.append({'eps': eps,'n_clusters': n_clusters,'n_noise': n_noise,'silhouette': sil_score})

best_eps = 0.5
dbscan = DBSCAN(eps=best_eps, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)
n_clusters_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = sum(dbscan_labels==-1)
print(f"DBSCAN eps={best_eps}: Clusters={n_clusters_db}, Noise={n_noise}, Silhouette Score={silhouette_score(X_scaled[dbscan_labels!=-1], dbscan_labels[dbscan_labels!=-1]) if n_clusters_db>1 else 'N/A'}")

# ============================================
# 3. GAUSSIAN MIXTURE MODEL
# ============================================
print("\n" + "="*50)
print("3. GAUSSIAN MIXTURE MODEL")
print("="*50)

n_components_range = range(2,7)
bic_scores=[]
aic_scores=[]
for n in n_components_range:
    gmm = GaussianMixture(n_components=n, random_state=42, n_init=10)
    gmm.fit(X_scaled)
    bic_scores.append(gmm.bic(X_scaled))
    aic_scores.append(gmm.aic(X_scaled))

plt.figure(figsize=(10,6))
plt.plot(n_components_range, bic_scores,'bo-',label='BIC')
plt.plot(n_components_range, aic_scores,'ro-',label='AIC')
plt.xlabel('Components'); plt.ylabel('Score'); plt.title('GMM: BIC & AIC'); plt.grid(True, alpha=0.3)
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(output_dir,'gmm_bic_aic.png'), dpi=300)
plt.show()

optimal_n = n_components_range[np.argmin(bic_scores)]
gmm = GaussianMixture(n_components=optimal_n, random_state=42, n_init=10)
gmm_labels = gmm.fit_predict(X_scaled)
print(f"GMM {optimal_n} components: Silhouette {silhouette_score(X_scaled,gmm_labels):.3f}, Adjusted Rand {adjusted_rand_score(y_true,gmm_labels):.3f}")

# ============================================
# 4. HIERARCHICAL CLUSTERING
# ============================================
print("\n" + "="*50)
print("4. HIERARCHICAL CLUSTERING")
print("="*50)

linkage_matrix = linkage(X_scaled, method='ward')
plt.figure(figsize=(12,8))
dendrogram(linkage_matrix, truncate_mode='level', p=5, color_threshold=0, above_threshold_color='grey')
plt.axhline(y=8, color='red', linestyle='--', label='Cut at 8')
plt.xlabel('Sample'); plt.ylabel('Distance'); plt.title('Hierarchical Clustering Dendrogram')
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(output_dir,'dendrogram.png'), dpi=300)
plt.show()

hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
hierarchical_labels = hierarchical.fit_predict(X_scaled)
print(f"Hierarchical Clustering with {optimal_k} clusters: Silhouette {silhouette_score(X_scaled,hierarchical_labels):.3f}, Adjusted Rand {adjusted_rand_score(y_true,hierarchical_labels):.3f}")

# ============================================
# 5. Comparison Visualization (2D PCA)
# ============================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
fig, axes = plt.subplots(2,2,figsize=(16,14))
axes = axes.ravel()
clustering_results = [(kmeans_labels,'K-Means'),(dbscan_labels,f'DBSCAN eps={best_eps}'),
                      (gmm_labels,'GMM'),(hierarchical_labels,'Hierarchical')]

for idx,(labels,title) in enumerate(clustering_results):
    ax = axes[idx]
    if idx==1:
        noise_mask = labels==-1
        ax.scatter(X_pca[noise_mask,0],X_pca[noise_mask,1],c='black',marker='x',s=100,label='Noise')
        unique_labels = set(labels)-{-1}
        colors = plt.cm.viridis(np.linspace(0,1,len(unique_labels)))
        for i,label in enumerate(unique_labels):
            mask = labels==label
            ax.scatter(X_pca[mask,0],X_pca[mask,1],c=[colors[i]],label=f'Cluster {label}',s=60,alpha=0.6,edgecolor='black')
    else:
        scatter = ax.scatter(X_pca[:,0],X_pca[:,1],c=labels,cmap='viridis',s=60,alpha=0.6,edgecolor='black')
        plt.colorbar(scatter, ax=ax)
    ax.set_title(title); ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.grid(True,alpha=0.3)
    if idx==1: ax.legend(loc='best',fontsize=8)

plt.suptitle('Comparison of Unsupervised Algorithms', fontsize=16,fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir,'clustering_comparison.png'), dpi=300)
plt.show()

# ============================================
# 6. CLUSTER PROFILING & VISUALIZATION
# ============================================
df_with_clusters['diabetes'] = df['class']
cluster_profiles=[]
for i in range(optimal_k):
    cluster_data = df_with_clusters[df_with_clusters['cluster']==i]
    profile={'Cluster':i,'Size':len(cluster_data),
             'Diabetes Rate':(cluster_data['diabetes']=='Positive').mean()*100,
             'Avg Age':cluster_data['Age'].mean(),
             'Gender Ratio (Male)':(cluster_data['Gender']=='Male').mean()*100}
    for symptom in symptom_cols:
        profile[f'{symptom} (%)'] = (cluster_data[symptom]=='Yes').mean()*100
    cluster_profiles.append(profile)

profile_df = pd.DataFrame(cluster_profiles)
print("\nCluster Profiles:\n",profile_df.to_string())

# Visualization
fig, axes = plt.subplots(2,2,figsize=(16,12))
# 1. Diabetes Rate
ax=axes[0,0]; bars=ax.bar(range(optimal_k),profile_df['Diabetes Rate'],color=['lightcoral','lightblue','lightgreen'])
ax.set_xticks(range(optimal_k)); ax.set_title('Diabetes Rate by Cluster')
for bar,rate in zip(bars,profile_df['Diabetes Rate']): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f'{rate:.1f}%',ha='center',fontweight='bold')
# 2. Age distribution
ax=axes[0,1]
for i in range(optimal_k):
    cluster_data = df_with_clusters[df_with_clusters['cluster']==i]
    ax.hist(cluster_data['Age'],bins=15,alpha=0.5,label=f'Cluster {i}')
ax.set_title('Age Distribution by Cluster'); ax.legend(); ax.grid(True,alpha=0.3)
# 3. Symptom heatmap
ax=axes[1,0]; symptom_means=df_with_clusters.groupby('cluster')[symptom_cols].apply(lambda x:(x=='Yes').mean())
sns.heatmap(symptom_means,annot=True,fmt='.2f',cmap='YlOrRd',ax=ax,cbar_kws={'label':'Prevalence'}); ax.set_title('Symptom Prevalence by Cluster')
# 4. Cluster sizes pie
ax=axes[1,1]; sizes=profile_df['Size']; colors=plt.cm.Set3(np.linspace(0,1,optimal_k))
ax.pie(sizes, labels=[f'Cluster {i}' for i in range(optimal_k)], autopct='%1.1f%%', colors=colors, startangle=90)
ax.set_title('Cluster Size Distribution')

plt.suptitle('K-Means Cluster Analysis', fontsize=16,fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir,'cluster_profiles.png'), dpi=300)
plt.show()

# ============================================
# 7. Summary
# ============================================
comparison = pd.DataFrame({
    'Algorithm':['K-Means','DBSCAN','GMM','Hierarchical'],
    'Number of Clusters':[optimal_k,n_clusters_db,optimal_n,optimal_k],
    'Silhouette Score':[silhouette_score(X_scaled,kmeans_labels),
                        silhouette_score(X_scaled[dbscan_labels!=-1],dbscan_labels[dbscan_labels!=-1]) if n_clusters_db>1 else np.nan,
                        silhouette_score(X_scaled,gmm_labels),
                        silhouette_score(X_scaled,hierarchical_labels)],
    'Adjusted Rand Index':[adjusted_rand_score(y_true,kmeans_labels),
                           adjusted_rand_score(y_true[dbscan_labels!=-1],dbscan_labels[dbscan_labels!=-1]) if n_clusters_db>1 else np.nan,
                           adjusted_rand_score(y_true,gmm_labels),
                           adjusted_rand_score(y_true,hierarchical_labels)]
})
print("\nAlgorithm Comparison:\n",comparison.to_string())
print("\nUNSUPERVISED LEARNING COMPLETE")