from tabulate import tabulate
from data_process import load_data, process_data
from exploratory_analysis import exploratory_analysis
from visualization import visualize_tsne
from knn_classifier import train_knn
from knn_visualization import plot_knn_results

def title(txt, tamanho=50):
    print(f"\n {txt.center(tamanho*2)} \n")
    print("=-" * tamanho)

def main():
    file = "mini_gm_public_v0.1.p"
    
    title("Carregando dados")
    data = load_data(file)

    title("Processando dados")
    df = process_data(data)

    title("Executando análise exploratória")
    exploratory_analysis(df)

    title("Gerando visualização com t-SNE")
    visualize_tsne(df)

    title("Treinando KNN com distância Euclidiana")
    knn_results_euclidean = train_knn(df, distance_metric='euclidean')
    print(tabulate(knn_results_euclidean, headers='keys', tablefmt='fancy_grid'))

    title("Treinando KNN com distância Cosseno")
    knn_results_cosine = train_knn(df, distance_metric='cosine')
    print(tabulate(knn_results_cosine, headers='keys', tablefmt='fancy_grid'))

    title("Gerando gráficos")
    plot_knn_results(knn_results_euclidean, knn_results_cosine)

    title("Finalizado, código feito por: @Sales16")

if __name__ == "__main__":
    main()