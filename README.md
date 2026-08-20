# ✈️ MBA em Redes Neurais Aplicadas a Negócios
## Projeto Prático: Precificação Dinâmica e Yield Management em e-Commerce de Companhias Aéreas usando Redes Neurais e Agentes de IA

---

### 📋 Visão Geral do Projeto
Este repositório contém a solução completa para o projeto de MBA com foco em precificação dinâmica algorítmica para companhias aéreas. O projeto combina:
1. **Rede Neural Feedforward (MLP em TensorFlow/Keras):** Modelagem preditiva da elasticidade-preço estocástica do passageiro e probabilidade de conversão.
2. **Agente de IA Prescritivo para Decisão de Negócios:** Otimizador autônomo de tarifas que maximiza a **Receita Esperada por Assento** $\\mathbb{E}[R(p)] = p \\times \\hat{P}(\\text{Compra} \\mid p)$ com guardrails corporativos de Yield Management.

---

### 📁 Estrutura dos Arquivos
- 
otebook_precificacao_dinamica_aerea.ipynb: Jupyter Notebook interativo contendo as 8 seções completas, código documentado, equações LaTeX e gráficos já pré-executados.
- pipeline_precificacao.py: Script Python modular pronto para execução em linha de comando, pipeline de CI/CD ou microsserviço de precificação.
- README.md: Este guia de documentação executiva.

---

### 🚀 Como Executar
1. **Via Jupyter Notebook / JupyterLab / VS Code:**
   Abra o arquivo 
otebook_precificacao_dinamica_aerea.ipynb e execute as células sequencialmente.

2. **Via Linha de Comando:**
   `ash
   python pipeline_precificacao.py
   `

---

### 📊 Seções do Projeto
- **SEÇÃO 1:** Configuração do Ambiente e Sementes Aleatórias (Reprodutibilidade).
- **SEÇÃO 2:** Geração do Dataset Sintético Realista (4.000 buscas com variáveis de mercado e formulação microeconômica de utilidade).
- **SEÇÃO 3:** Pré-processamento e Divisão dos Dados (80% treino / 20% teste com StandardScaler).
- **SEÇÃO 4:** Construção e Treinamento da Rede Neural Feedforward (MLP com Dropout e Early Stopping).
- **SEÇÃO 5:** Avaliação do Modelo Preditivo (Acurácia, ROC-AUC, Matriz de Confusão e função prever_probabilidade_venda).
- **SEÇÃO 6:** Construção da Arquitetura do Agente de IA para Decisão de Negócios (AgentePrecificacaoAerea).
- **SEÇÃO 7:** Simulação Prática de 3 Cenários Corporativos (Voo distante, Voo próximo corporativo, Guerra de preços).
- **SEÇÃO 8:** Conclusão e Resumo do ROI Executivo (Ganhos de Yield, RASM, Spoilage, Spill e Automação).
