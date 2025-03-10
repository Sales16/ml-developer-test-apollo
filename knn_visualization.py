import matplotlib.pyplot as plt
import pandas as pd
from knn_classifier import train_knn

def plot_knn_results(knn_results_euclidean, knn_results_cosine):
    plt.figure(figsize=(16, 5))
    
    plt.subplot(1, 4, 1)
    plt.plot(knn_results_euclidean["k"], knn_results_euclidean["Accuracy"], marker='o', label='Euclidean')
    plt.plot(knn_results_cosine["k"], knn_results_cosine["Accuracy"], marker='s', label='Cosine')
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("Comparação de Accuracy")
    plt.legend()
    
    plt.subplot(1, 4, 2)
    plt.plot(knn_results_euclidean["k"], knn_results_euclidean["F1-Score"], marker='o', label='Euclidean')
    plt.plot(knn_results_cosine["k"], knn_results_cosine["F1-Score"], marker='s', label='Cosine')
    plt.xlabel("k")
    plt.ylabel("F1-Score")
    plt.title("Comparação de F1-Score")
    plt.legend()
    
    plt.subplot(1, 4, 3)
    plt.plot(knn_results_euclidean["k"], knn_results_euclidean["AUC"], marker='o', label='Euclidean')
    plt.plot(knn_results_cosine["k"], knn_results_cosine["AUC"], marker='s', label='Cosine')
    plt.xlabel("k")
    plt.ylabel("AUC")
    plt.title("Comparação de AUC")
    plt.legend()

    plt.subplot(1, 4, 4)
    plt.plot(knn_results_euclidean["k"], knn_results_euclidean["Top-k Accuracy"], marker='o', label='Euclidean')
    plt.plot(knn_results_cosine["k"], knn_results_cosine["Top-k Accuracy"], marker='s', label='Cosine')
    plt.xlabel("k")
    plt.ylabel("Top-k")
    plt.title("Comparação Top-k Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
