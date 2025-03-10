import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE

def visualize_tsne(df):
    """
    Realiza a redução de dimensionalidade com t-SNE e plota os embeddings.
    :param df: DataFrame contendo os embeddings.
    """
    embeddings_matrix = np.stack(df["embedding"].values)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(embeddings_matrix)
    print(tsne_results.shape)
    
    tsne_df = pd.DataFrame(tsne_results, columns=["Dim1", "Dim2"])
    tsne_df["syndrome_id"] = df["syndrome_id"].values
    
    plt.figure(figsize=(10, 7))
    print(tsne_df.head())
    sns.scatterplot(x="Dim1", y="Dim2", hue="syndrome_id", data=tsne_df, palette='tab10', alpha=0.7)
    plt.title("Visualização t-SNE dos Embeddings")
    plt.show()

if __name__ == "__main__":
    from data_processing import preprocess_data
    from data_loader import load_data
    
    file_path = "mini_gm_public_v0.1.p"
    data = load_data(file_path)
    df = preprocess_data(data)
    visualize_tsne(df)
