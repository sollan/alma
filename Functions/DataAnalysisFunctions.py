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

mpl.style.use('seaborn')


def find_files(folder):

    file_list = []
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        if file.endswith('.csv') and not file.startswith('averaged'):
            file_list.append(file_path)
    return file_list


def combine_files(file_lists, group_names, output_path, file_type='strides'):
    '''
    merge result files from all groups into a single pd dataframe
    '''
    dfs = []
    for group_id, file_list in enumerate(file_lists):
        for file in file_list:
            df = pd.read_csv(file, index_col=0)
            df = df[df['cycle duration (no. frames)'] >= 0]
            if len(df) > 0:
                df['id'] = file
                df['group'] = group_names[group_id]
                dfs.append(df)
            else:
                # 100% dragging? no valid step cycles found
                pass

        if file_type == 'strides':
            # remove stride start / end frame columns
            combined_df = pd.concat(dfs).iloc[:, 2:]
            combined_df.fillna(combined_df.mean(), inplace=True)
        elif file_type == 'average':
            combined_df = pd.concat(dfs).iloc[:, 2:].groupby(['id', 'group']).agg('mean').reset_index()

    file_name = f'concatenated_results_{file_type}.csv'
    combined_df.to_csv(os.path.join(output_path, file_name))
    return combined_df


def random_forest(combined_df, output_path):

    # train data: remove id / group name columns
    x = combined_df.iloc[:, :-2]
    y = combined_df['group']
    test_size = 0.25
    random_state = 42
    n_estimators = 100
    n_variables = 44
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state)
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

    confusion_matrix = pd.crosstab(
        y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    sns.set(rc={'figure.figsize': (5, 4)})
    sns.heatmap(confusion_matrix, cmap="rocket", annot=True)
    group_name_1 = combined_df['group'].unique()[0]
    group_name_2 = combined_df['group'].unique()[1]
    plot_name = f'{group_name_1}_vs_{group_name_2}_acc_{accuracy}.svg'
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
    plt.bar(
        x_values,
        importance[:n_var_to_plot, 1].astype('float32'),
        orientation='vertical')
    plt.xticks(
        x_values,
        importance[:n_var_to_plot, 0],
        rotation='vertical')
    plt.ylabel('Importance')
    plt.xlabel('Parameter')
    plt.title('Feature importance')
    plot_name = f'{group_name_1}_vs_{group_name_2}_feature_importance_top_{n_var_to_plot}.svg'
    plt.savefig(os.path.join(output_path, plot_name))


def save_feature_importance(importance, y, output_path, n_var_to_plot=44):
    group_name_1 = y.unique()[0]
    group_name_2 = y.unique()[1]
    importance_df = pd.DataFrame(importance[:n_var_to_plot])
    importance_df.columns = ['Parameter', 'Feature importance']
    importance_df.sort_values('Feature importance', ascending=False)
    file_name = f'{group_name_1}_vs_{group_name_2}_feature_importance_top_{n_var_to_plot}.csv'
    importance_df.to_csv(os.path.join(output_path, file_name))


def PCA(combined_df, output_path):
    # train data: remove id / group name columns
    x = combined_df.iloc[:, 2:]
    y = combined_df['group']

    pca = sklearn.decomposition.PCA(n_components=2)
    pc = pca.fit_transform(x)
    pc_df = pd.DataFrame(pc, columns=['PC1', 'PC2'])
    pc_df['Cluster'] = list(y)

    plot_pca_clusters(pca, pc_df, output_path)


def plot_pca_clusters(pca, pc_df, output_path):

    explained_ratio_pc1 = round(pca.explained_variance_ratio_[0], 3)
    explained_ratio_pc2 = round(pca.explained_variance_ratio_[1], 3)
    sns.lmplot(
        x="PC1",
        y="PC2",
        data=pc_df,
        fit_reg=False,
        hue='Cluster',  # color by cluster
        palette={'g', 'r', 'b'},
        legend=True,
        scatter_kws={"s": 80})
    plt.xlabel(f"PC1: explained variance {explained_ratio_pc1}")
    plt.ylabel(f"PC2: explained variance {explained_ratio_pc2}")
    plot_name = 'PCA_clusters.svg'
    plt.savefig(os.path.join(output_path, plot_name))
