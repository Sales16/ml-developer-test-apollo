import matplotlib.pyplot as plt
import pandas as pd
from knn_classifier import train_knn
from data_loader import load_data
from data_processing import preprocess_data

def plot_knn_results(knn_results_euclidean, knn_results_cosine):
    """
    Plota gráficos comparando o desempenho do KNN com distância Euclidiana e Cosseno.
    """
    plt.figure(figsize=(12, 5))
    
    # Plot Accuracy
    plt.subplot(1, 3, 1)
    plt.plot(knn_results_euclidean["k"], knn_results_euclidean["Accuracy"], marker='o', label='Euclidean')
    plt.plot(knn_results_cosine["k"], knn_results_cosine["Accuracy"], marker='s', label='Cosine')
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("Comparação de Accuracy")
    plt.legend()
    
    # Plot F1-Score
    plt.subplot(1, 3, 2)
    plt.plot(knn_results_euclidean["k"], knn_results_euclidean["F1-Score"], marker='o', label='Euclidean')
    plt.plot(knn_results_cosine["k"], knn_results_cosine["F1-Score"], marker='s', label='Cosine')
    plt.xlabel("k")
    plt.ylabel("F1-Score")
    plt.title("Comparação de F1-Score")
    plt.legend()
    
    # Plot AUC
    plt.subplot(1, 3, 3)
    plt.plot(knn_results_euclidean["k"], knn_results_euclidean["AUC"], marker='o', label='Euclidean')
    plt.plot(knn_results_cosine["k"], knn_results_cosine["AUC"], marker='s', label='Cosine')
    plt.xlabel("k")
    plt.ylabel("AUC")
    plt.title("Comparação de AUC")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = "mini_gm_public_v0.1.p"
    data = load_data(file_path)
    df = preprocess_data(data)
    
    print("Treinando KNN...")
    knn_results_euclidean = train_knn(df, distance_metric='euclidean')
    knn_results_cosine = train_knn(df, distance_metric='cosine')
    
    print("Gerando gráficos...")
    plot_knn_results(knn_results_euclidean, knn_results_cosine)
