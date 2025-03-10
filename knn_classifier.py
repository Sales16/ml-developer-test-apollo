import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score


def train_knn(df, distance_metric='euclidean', k_range=range(1, 16)):
    try:
        X = np.stack(df["embedding"].values)  
        y = df["syndrome_id"].values 
        
        results = []
        
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            
            accuracy = cross_val_score(knn, X, y, cv=cv, scoring='accuracy').mean()
            f1 = cross_val_score(knn, X, y, cv=cv, scoring='f1_weighted').mean()
            auc_score = cross_val_score(knn, X, y, cv=cv, scoring='roc_auc_ovr_weighted').mean()
            top_k = cross_val_score(knn, X, y, cv=cv, scoring='top_k_accuracy').mean()
            
            results.append([k, accuracy, f1, auc_score, top_k])
        
        results_df = pd.DataFrame(results, columns=["k", "Accuracy", "F1-Score", "AUC", "Top-k Accuracy"])
        return results_df
    except Exception as e:
        print(f"Erro no treinamento do KNN: {e}")
        return None