import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def exploratory_analysis(df):
    """
    Realiza análise exploratória do dataset.
    :param df: DataFrame contendo os dados processados.
    """
    print("Resumo dos dados:")
    print(df.head())
    print(f"Total de registros: {len(df)}")
    print(f"Número de síndromes únicas: {df['syndrome_id'].nunique()}")
    print(f"Número de sujeitos únicos: {df['subject_id'].nunique()}")
    print(f"Número de imagens únicas: {df['image_id'].nunique()}")

    # Distribuição das síndromes
    plt.figure(figsize=(10, 5))
    syndrome_counts = df['syndrome_id'].value_counts()
    sns.barplot(x=syndrome_counts.index, y=syndrome_counts.values, hue=syndrome_counts.index, palette='viridis', legend=False)
    plt.xlabel("Síndrome ID")
    plt.ylabel("Número de Imagens")
    plt.title("Distribuição de Imagens por Síndrome")
    plt.xticks(rotation=45)
    plt.show()
    
    # Estatísticas dos embeddings
    embeddings_matrix = np.stack(df["embedding"].values)
    print("Estatísticas dos Embeddings:")
    print(f"Média: {np.mean(embeddings_matrix, axis=0)[:5]} ...")
    print(f"Variância: {np.var(embeddings_matrix, axis=0)[:5]} ...")

if __name__ == "__main__":
    from data_processing import preprocess_data
    from data_loader import load_data
    
    file_path = "mini_gm_public_v0.1.p" 
    data = load_data(file_path)
    df = preprocess_data(data)
    exploratory_analysis(df)
