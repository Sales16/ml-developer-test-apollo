import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score


def train_knn(df, distance_metric='euclidean', k_range=range(1, 16)):
    """
    Treina um modelo KNN usando validação cruzada e avalia seu desempenho.
    :param df: DataFrame com embeddings e rótulos.
    :param distance_metric: Métrica de distância ('euclidean' ou 'cosine').
    :param k_range: Range de valores de k a serem testados.
    :return: DataFrame com os resultados para cada k.
    """
    
    X = np.stack(df["embedding"].values)  
    y = df["syndrome_id"].values 
    
    results = []
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        accuracy = cross_val_score(knn, X, y, cv=cv, scoring='accuracy').mean()
        f1 = cross_val_score(knn, X, y, cv=cv, scoring='f1_weighted').mean()
        auc = cross_val_score(knn, X, y, cv=cv, scoring='roc_auc_ovr_weighted').mean()
        
        results.append([k, accuracy, f1, auc])
    
    results_df = pd.DataFrame(results, columns=["k", "Accuracy", "F1-Score", "AUC"])
    return results_df


if __name__ == "__main__":
    from data_processing import preprocess_data
    from data_loader import load_data
    
    file_path = "mini_gm_public_v0.1.p"
    data = load_data(file_path)
    df = preprocess_data(data)
    
    print("Treinando KNN com distância Euclidiana...")
    knn_results_euclidean = train_knn(df, distance_metric='euclidean')
    print(knn_results_euclidean)
    
    print("Treinando KNN com distância Cosseno...")
    knn_results_cosine = train_knn(df, distance_metric='cosine')
    print(knn_results_cosine)