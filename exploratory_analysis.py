import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tabulate import tabulate

def exploratory_analysis(df):
    try:

        df_display = df.copy()
        df_display["embedding"] = df_display["embedding"].apply(lambda x: str(x[:5]) + " ...")
        print("\nResumo dos Dados:\n")
        print(tabulate(df_display.head(), headers='keys', tablefmt='fancy_grid'))

        summary_data = [
            ["Total de registros", len(df)],
            ["Síndromes únicas", df['syndrome_id'].nunique()],
            ["Sujeitos únicos", df['subject_id'].nunique()],
            ["Imagens únicas", df['image_id'].nunique()]
        ]
        print("\nEstatísticas Gerais:\n")
        print(tabulate(summary_data, headers=["Descrição", "Valor"], tablefmt="fancy_grid"))
        
        plt.figure(figsize=(12, 6))
        syndrome_counts = df['syndrome_id'].value_counts()
        sns.barplot(x=syndrome_counts.index, y=syndrome_counts.values, hue=syndrome_counts.index, palette='viridis', legend=False)
        plt.xlabel("ID da Síndrome")
        plt.ylabel("Número de Imagens")
        plt.title("Distribuição de Imagens por Síndrome")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
        
        embeddings_matrix = np.stack(df["embedding"].values)
        print("\nEstatísticas dos Embeddings:")
        print(f"Média dos primeiros valores: {np.mean(embeddings_matrix, axis=0)[:5]}")
        print(f"Variância dos primeiros valores: {np.var(embeddings_matrix, axis=0)[:5]}")

        plt.hist(np.linalg.norm(embeddings_matrix, axis=1), bins=30, alpha=0.7, color='blue')
        plt.title("Distribuição das Magnitudes dos Embeddings")
        plt.xlabel("Magnitude")
        plt.ylabel("Frequência")
        plt.show()
    
    except Exception as e:
        print(f"Erro na análise exploratória: {e}")
