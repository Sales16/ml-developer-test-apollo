# Teste Prático Desenvolvedor de Software Júnior 

Este repositório contém a implementação do teste prático para a posição de **Desenvolvedor de Software Júnior**. O objetivo do projeto é realizar a **classificação de imagens associadas a síndromes genéticas** utilizando embeddings previamente gerados. O pipeline inclui **processamento de dados, análise exploratória, treinamento de modelo KNN e avaliação dos resultados**.

## Estrutura do Projeto

```
ml-developer-test-apollo
│── README.md                    # Documentação do projeto
│── requirements.txt             # Dependências necessárias
│── main.py                      # Script principal
│── data_process.py              # Carregamento e processamento de dados
│── exploratory_analysis.py      # Análise exploratória e visualização
│── visualization.py             # Visualização de embeddings
│── knn_classifier.py            # Implementação do modelo KNN
│── knn_visualization.py         # Comparação gráfica dos resultados
│── mini_gm_public_v0.1.p        # Contém todos os dados necessários.
```

## Como Executar

**1 - Instale as dependências necessárias:**
```bash
pip install -r requirements.txt
```

**2 - Execute o script principal:**
```bash
python main.py
```
Isso processará os dados, treinará os modelos e exibirá os resultados.


## Etapas do Pipeline

### **1 - Script Principal (`main.py`)**
- Coordena todas as etapas do pipeline, desde o processamento de dados até a visualização dos resultados.

### **2 - Processamento de Dados (`data_process.py`)**
- Carrega o arquivo `.p` contendo os embeddings.
- Converte os dados para um DataFrame.
- Trata dados ausentes e gera estatísticas básicas.

### **3 - Análise Exploratória e Visualização (`exploratory_analysis.py`)**
- Analisa os embeddings e sua distribuição.
- Gera gráficos para entender a distribuição das síndromes.
  
### **4 - Visualização do gráfico T-SNE (`visualization.py`)**
- Cria visualizações detalhadas dos embeddings.
- Utiliza t-SNE para reduzir a dimensionalidade dos embeddings e visualizar a separação das classes.

### **5 - Treinamento do Modelo (`knn_classifier.py`)**
- Implementa o K-Nearest Neighbors (KNN) com duas métricas de distância:
  - Distância Euclidiana
  - Distância Cosseno
- Avaliação do modelo com validação cruzada (10-fold cross-validation)
- Métricas analisadas:
  - AUC (Área sob a curva ROC)
  - F1-Score
  - Top-k Accuracy
  - Accuracy

### **6 - Visualização dos Resultados (`knn_visualization.py`)**
- Gera visualizações gráficas dos resultados obtidos pelo modelo KNN.
- Ele compara os resultados das diferentes métricas de distância (Euclidiana e Cosseno) e mostra a performance do modelo com diferentes métricas.


## 📊 Resultados
Os testes mostraram que:
- A distância Cosseno apresentou um desempenho superior à Euclidiana.
- Os valores de Accuracy e F1-Score estabilizam entre k = 8 e 12.
- A métrica AUC indica que a separação das classes foi eficiente.
  