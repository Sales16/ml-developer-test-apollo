import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

def visualize_tsne(df):
    try:
        embeddings_matrix = np.stack(df["embedding"].values)

        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(embeddings_matrix)
        tsne_df = pd.DataFrame(tsne_results, columns=["Dim1", "Dim2"])
        tsne_df["syndrome_id"] = df["syndrome_id"].values

        plt.figure(figsize=(10, 7))
        sns.scatterplot(x="Dim1", y="Dim2", hue="syndrome_id", data=tsne_df, palette='tab10', alpha=0.7)
        plt.title("Visualização t-SNE dos Embeddings")
        plt.show()

        silhouette_avg = silhouette_score(tsne_results, df['syndrome_id'].values)
        print(f"Coeficiente de Silhueta (t-SNE): {silhouette_avg:.4f}")

    except Exception as e:
        print(f"Erro na visualização t-SNE: {e}")
