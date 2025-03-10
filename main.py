from data_loader import load_data
from data_processing import preprocess_data
from exploratory_analysis import exploratory_analysis
from visualization import visualize_tsne
from knn_classifier import train_knn
from knn_visualization import plot_knn_results

def main():
    """
    Script principal que executa todas as etapas do pipeline.
    """
    file_path = "mini_gm_public_v0.1.p"
    
    print("Carregando dados...")
    data = load_data(file_path)
    
    print("Processando dados...")
    df = preprocess_data(data)
    
    print("Executando análise exploratória...")
    exploratory_analysis(df)
    
    print("Gerando visualização com t-SNE...")
    visualize_tsne(df)
    
    print("Treinando KNN com distância Euclidiana...")
    knn_results_euclidean = train_knn(df, distance_metric='euclidean')
    print(knn_results_euclidean)
    
    print("Treinando KNN com distância Cosseno...")
    knn_results_cosine = train_knn(df, distance_metric='cosine')
    print(knn_results_cosine)

    print("Gerando gráficos...")
    plot_knn_results(knn_results_euclidean, knn_results_cosine)
    
    print("Pipeline concluído!")

if __name__ == "__main__":
    main()