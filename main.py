from data_processing import load_data, processing_data
from exploratory_analysis import exploratory_analysis
from visualization import visualize_tsne
from knn_classifier import train_knn
from knn_visualization import plot_knn_results

def gerar_linha(tamanho=50):
    print("=-" * tamanho)

def main():
    file = "mini_gm_public_v0.1.p"
    
    print("Carregando dados...")
    data = load_data(file)
    
    gerar_linha()

    print("Processando dados...")
    df = processing_data(data)

    gerar_linha()
    
    print("Executando análise exploratória...")
    exploratory_analysis(df)

    gerar_linha()
    
    print("Gerando visualização com t-SNE...")
    visualize_tsne(df)

    gerar_linha()
    
    print("Treinando KNN com distância Euclidiana...")
    knn_results_euclidean = train_knn(df, distance_metric='euclidean')
    print(knn_results_euclidean)

    gerar_linha()
    
    print("Treinando KNN com distância Cosseno...")
    knn_results_cosine = train_knn(df, distance_metric='cosine')
    print(knn_results_cosine)

    gerar_linha()

    print("Gerando gráficos...")
    plot_knn_results(knn_results_euclidean, knn_results_cosine)

    gerar_linha()
    
    print("Pipeline concluído!")

if __name__ == "__main__":
    main()