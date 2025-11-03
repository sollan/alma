import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from packaging import version

if version.parse(mpl.__version__) < version.parse("3.6"): 
    mpl.style.use('seaborn')
else:
    mpl.style.use('seaborn-v0_8')

def find_files(folder):
    '''
    '''
    file_list = []
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        if file.endswith('.csv') and not file.startswith('averaged'):
            file_list.append(file_path)
    return file_list


def combine_files(file_lists, group_names, output_path, file_type='strides'):
    '''
    Merge result files from all groups into a single pd dataframe.
    For 'average' mode: averages all strides per file (one row per animal/video).
    For 'strides' mode: keeps all individual strides.
    '''
    dfs = []
    for group_id, file_list in enumerate(file_lists):
        for file in file_list:
            df = pd.read_csv(file, index_col=0)
            if 'cycle duration (no. frames)' in df.columns:
                df = df[df['cycle duration (no. frames)'] >= 0]
            if len(df) > 0:
                df['id'] = file
                df['group'] = group_names[group_id]
                dfs.append(df)
            else:
                # 100% dragging? no valid step cycles found
                pass

    if len(dfs) == 0:
        raise ValueError('No valid data files found. Please check that your CSV files contain valid gait cycle data.')

    # Concatenate all dataframes
    all_data = pd.concat(dfs, ignore_index=True)
    
    if file_type == 'strides':
        # Remove stride start/end frame columns (first 2 columns)
        combined_df = all_data.iloc[:,2:]
        combined_df.fillna(combined_df.mean(), inplace=True)
    elif file_type == 'average':
        # Remove stride start/end frame columns first, then group by id and group
        data_no_frames = all_data.iloc[:,2:]
        # Group by id and group, compute mean across all strides per file
        combined_df = data_no_frames.groupby(['id', 'group']).agg('mean').reset_index()
        
        # Check for remaining NaN values and report
        nan_counts = combined_df.isnull().sum()
        if nan_counts.sum() > 0:
            print(f"WARNING: NaN values found in averaged data:")
            print(nan_counts[nan_counts > 0])
            # Fill NaN values with column means
            numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
            combined_df[numeric_cols] = combined_df[numeric_cols].fillna(combined_df[numeric_cols].mean())
        
    file_name = f'concatenated_results_{file_type}.csv'
    combined_df.to_csv(os.path.join(output_path, file_name))
    print(f"Saved combined data to: {os.path.join(output_path, file_name)}")
    print(f"Combined data shape: {combined_df.shape}")
    print(f"Columns: {combined_df.columns.tolist()}")
    return combined_df


def random_forest(combined_df, output_path):

    # train data: remove id / group name columns
    x = combined_df.iloc[:, :-2]
    y = combined_df['group']
    test_size = 0.25
    random_state = 42
    n_estimators = 100
    n_variables = 44
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators)
    rf_classifier.fit(x_train, y_train)
    y_pred = rf_classifier.predict(x_test)
    accuracy = round(metrics.accuracy_score(y_test, y_pred), 3)
    
    plot_rf_crosstab(y_test, y_pred, accuracy, combined_df, output_path)
    
    feature_importance = get_feature_importance(x, rf_classifier)
    plot_rf_feature_importance(feature_importance, y, output_path, n_variables)
    save_feature_importance(feature_importance, y, output_path, n_variables)
    
    return accuracy


def plot_rf_crosstab(y_test, y_pred, accuracy, combined_df, output_path):

    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    sns.set(rc={'figure.figsize':(5,4)})
    sns.heatmap(confusion_matrix, cmap="rocket", annot=True)
    group_name_1 = combined_df['group'].unique()[0]
    group_name_2 = combined_df['group'].unique()[1]
    plot_name = f'{group_name_1}_vs_{group_name_2}_confusion_acc_{accuracy}.svg'
    plt.savefig(os.path.join(output_path, plot_name))
    

def get_feature_importance(x, rf_classifier):

    importance = [(x.columns[i], rf_classifier.feature_importances_[i]) for i in range(len(x.columns))]
    importance = sorted(importance, key=lambda x: x[1], reverse=True)
    importance = np.array(importance)
    return importance


def plot_rf_feature_importance(importance, y, output_path, n_var_to_plot=44):
    group_name_1 = y.unique()[0]
    group_name_2 = y.unique()[1]
    plt.figure(figsize=(20, 10))
    x_values = list(range(0, n_var_to_plot))
    plt.bar(x_values, importance[:n_var_to_plot,1].astype('float32'), orientation = 'vertical')
    plt.xticks(x_values, importance[:n_var_to_plot,0], rotation='vertical')
    plt.ylabel('Importance'); plt.xlabel('Parameter'); plt.title('Feature importance')
    plot_name = f'{group_name_1}_vs_{group_name_2}_feature_importance_top_{n_var_to_plot}.svg'
    plt.savefig(os.path.join(output_path, plot_name))


def save_feature_importance(importance, y, output_path, n_var_to_plot=44):
    group_name_1 = y.unique()[0]
    group_name_2 = y.unique()[1]
    importance_df = pd.DataFrame(importance[:n_var_to_plot])
    importance_df.columns = ['Parameter', 'Feature importance']
    importance_df.sort_values('Feature importance', ascending = False)
    file_name = f'{group_name_1}_vs_{group_name_2}_feature_importance_top_{n_var_to_plot}.csv'
    importance_df.to_csv(os.path.join(output_path, file_name))


def PCA(combined_df, output_path):
    """
    Performs Principal Component Analysis on averaged gait kinematic data.
    
    Workflow:
    1. Extract features (X) and group labels (y)
    2. Remove zero-variance features
    3. Standardize features (mean=0, std=1)
    4. Apply PCA transformation
    5. Generate visualization plots
    """
    print("\n" + "="*60)
    print("STARTING PCA ANALYSIS")
    print("="*60)
    
    print(f"\nCombined DF shape: {combined_df.shape}")
    print(f"Combined DF columns: {combined_df.columns.tolist()}")
    print(f"\nFirst few rows:\n{combined_df.head()}")
    
    # Extract features (skip 'id' and 'group' columns)
    x = combined_df.iloc[:, 2:]
    y = combined_df['group']
    
    print(f"\n--- DATA SUMMARY ---")
    print(f"Features (X) shape: {x.shape}")
    print(f"Number of samples (animals/videos): {len(x)}")
    print(f"Number of features (kinematic parameters): {x.shape[1]}")
    print(f"Groups: {y.unique().tolist()}")
    print(f"Samples per group: {y.value_counts().to_dict()}")
    
    # Validate minimum data requirements
    if x.shape[0] < 3:
        raise ValueError(f"Insufficient samples for meaningful PCA. Need at least 3 samples, got {x.shape[0]}. "
                        "Each group should have multiple animals/videos.")
    
    # Check for zero variance columns
    variance = x.var()
    zero_var_cols = variance[variance == 0].index.tolist()
    if zero_var_cols:
        print(f"\n--- WARNING: Removing {len(zero_var_cols)} zero-variance features ---")
        print(f"Features with no variation: {zero_var_cols[:5]}{'...' if len(zero_var_cols) > 5 else ''}")
        x = x.loc[:, variance > 0]
        print(f"Features after removal: {x.shape[1]}")
    
    # Check for very low variance columns
    low_var_threshold = 1e-10
    low_var_cols = variance[(variance > 0) & (variance < low_var_threshold)].index.tolist()
    if low_var_cols:
        print(f"\n--- WARNING: {len(low_var_cols)} features with very low variance ---")
        print("These features may not contribute meaningfully to the analysis.")
    
    # Determine number of components based on data
    n_components = min(5, x.shape[0] - 1, x.shape[1])
    print(f"\n--- PCA CONFIGURATION ---")
    print(f"Number of components to extract: {n_components}")
    
    if n_components < 2:
        raise ValueError(f"Insufficient data for PCA: need at least 2 samples and 2 features. "
                        f"Got {x.shape[0]} samples and {x.shape[1]} features.")

    # Standardize features (critical for PCA when features have different scales)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    print(f"✓ Data standardized (mean=0, std=1)")
    
    # Apply PCA
    pca = sklearn.decomposition.PCA(n_components=n_components)
    pc = pca.fit_transform(x_scaled)
    
    print(f"\n--- PCA RESULTS ---")
    print(f"Explained variance ratio per component:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%)")
    
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    print(f"\nCumulative explained variance:")
    for i, cum_var in enumerate(cumsum_var):
        print(f"  PC1-PC{i+1}: {cum_var:.4f} ({cum_var*100:.2f}%)")
    
    # Validate PCA results
    if pca.explained_variance_ratio_[0] > 0.99:
        print(f"\n⚠ WARNING: PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance!")
        print("This suggests:")
        print("  1. Data may be too uniform (little variation between samples)")
        print("  2. One feature dominates all others in scale")
        print("  3. Very few samples relative to features (overfitting)")
        print("  → Check 'concatenated_results_average.csv' for data quality")
    
    # Create DataFrame with principal components
    col_names = [f'PC{i+1}' for i in range(n_components)]
    pc_df = pd.DataFrame(pc, columns=col_names)
    pc_df['Cluster'] = list(y)
    
    # Save PC scores
    pc_output_file = os.path.join(output_path, 'PCA_scores.csv')
    pc_df.to_csv(pc_output_file, index=False)
    print(f"\n✓ Saved PC scores to: {pc_output_file}")

    # Generate plots
    print(f"\n--- GENERATING PLOTS ---")
    plot_pca_clusters(pca, pc_df, output_path)
    plot_pca_scree_plot(pca, output_path)
    
    print("\n" + "="*60)
    print("PCA ANALYSIS COMPLETE")
    print("="*60 + "\n")


def plot_pca_scree_plot(pca, output_path):

    pca_components = np.arange(pca.n_components_) + 1
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Individual variance explained
    ax1.plot(pca_components, pca.explained_variance_ratio_, 'ro-', linewidth=2, markersize=8)
    ax1.set_title('Scree Plot', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Principal Component', fontsize=10)
    ax1.set_ylabel('Proportion of Variance Explained', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(pca_components)
    
    # Add percentage labels on points
    for i, (comp, var) in enumerate(zip(pca_components, pca.explained_variance_ratio_)):
        ax1.text(comp, var, f'{var:.1%}', ha='center', va='bottom', fontsize=9)
    
    # Cumulative variance explained
    ax2.plot(pca_components, cumulative_variance, 'bo-', linewidth=2, markersize=8)
    ax2.axhline(y=0.9, color='r', linestyle='--', alpha=0.7, label='90% threshold')
    ax2.set_title('Cumulative Variance Explained', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Principal Component', fontsize=10)
    ax2.set_ylabel('Cumulative Proportion of Variance', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(pca_components)
    ax2.legend()
    
    # Add percentage labels
    for i, (comp, var) in enumerate(zip(pca_components, cumulative_variance)):
        ax2.text(comp, var, f'{var:.1%}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot_name = 'PCA_scree_plot.svg'
    plt.savefig(os.path.join(output_path, plot_name), bbox_inches='tight')
    plt.close()
    print(f"Saved scree plot to: {os.path.join(output_path, plot_name)}")


def plot_pca_clusters(pca, pc_df, output_path):

    explained_ratio_pc1 = round(pca.explained_variance_ratio_[0], 3)
    explained_ratio_pc2 = round(pca.explained_variance_ratio_[1], 3)
    
    # Create figure using lmplot (which returns a FacetGrid, not an axes object)
    g = sns.lmplot(x="PC1", 
                   y="PC2",
                   data=pc_df, 
                   fit_reg=False, 
                   hue='Cluster',
                   legend=True,
                   scatter_kws={"s": 80},
                   height=6,
                   aspect=1.2)
    
    # Update labels with explained variance
    g.set_axis_labels(f"PC1 (explained variance: {explained_ratio_pc1})", 
                      f"PC2 (explained variance: {explained_ratio_pc2})")
    g.fig.suptitle("PCA Cluster Analysis", y=1.02, fontsize=14, fontweight='bold')
    
    plot_name = 'PCA_clusters.svg'
    plt.savefig(os.path.join(output_path, plot_name), bbox_inches='tight')
    plt.close()
    print(f"Saved PCA cluster plot to: {os.path.join(output_path, plot_name)}")
