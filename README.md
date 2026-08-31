# ✈️ MBA em Redes Neurais Aplicadas a Negócios
## Projeto Prático: Precificação Dinâmica e Yield Management em e-Commerce de Companhias Aéreas usando Redes Neurais e Agentes de IA

# Otimização de Precificação Dinâmica e Yield Management Aéreo com IA Neuro-Simbólica

> **Projeto de Conclusão de Disciplina:** Redes Neurais Aplicadas a Negócios  
> **Curso:** MBA em Inteligência Artificial e Analytics Aplicados a Negócios — FGV  
> **Autor:** Marcio Mizumoto Takahara (`marciomt87-ops`)  
> **Professor:** Marco Fidos  

---

## 📌 1. Visão Geral do Problema e Objetivo de Negócio

### O Problema de Negócio
No setor aéreo, o assento de uma aeronave é um bem **altamente perecível**: um assento vazio no momento da decolagem representa perda irreversível de receita (*spoilage*). Por outro lado, vender bilhetes antecipadamente por tarifas muito baixas causa diluição de margem e impede a captura de viajantes corporativos de última hora, dispostos a pagar tarifas premium (*spill*). 

A **precificação estática ou baseada em regras manuais rígidas** não acompanha as oscilações de mercado em tempo real, gerando perda constante de margem de contribuição e ineficiência operacional.

### A Solução Proposta (Arquitetura Neuro-Simbólica)
Esta Prova de Conceito (PoC) implementa uma solução de **dupla camada (Neuro-Simbólica)** integrada ao ecossistema do e-commerce:
1. **Camada Preditiva (Rede Neural MLP / Sensor de Elasticidade):** Estima continuamente a probabilidade calibrada de conversão de venda $P(\text{Compra} \mid \text{Preço}, \text{Contexto})$.
2. **Camada Prescritiva (Agente Decisor de Yield Management no Antigravity):** Um agente autônomo orienta a escolha do preço ótimo ($p^*$) para maximizar a receita esperada $E[\text{Receita}] = p \times P(\text{Compra} \mid p)$, aplicando travas rígidas de governança corporativa e regras operacionais de Yield Management em milissegundos.

### Métricas de Sucesso de Negócio e Impacto Projetado
* **Incremento no RASK (*Revenue per Available Seat Kilometer*):** Projeção de **+3% a +6%** (equivalente a uma receita incremental estimada de **R$ 45.000,00 por rota/mês** em uma malha simulada com 10 rotas ativas).
* **Otimização do *Load Factor* (Taxa de Ocupação):** Aumento de até **8%** na ocupação de voos em janelas de baixa demanda (*spoilage mitigation*).
* **Agilidade Operacional:** Redução do tempo de reação a oscilações da concorrência e picos de demanda de **horas para milissegundos (< 100ms)**.

---

## 📊 2. Comparativo Empírico: ML Tradicional vs. Redes Neurais

Para atender aos critérios de seleção de algoritmos e demonstrar a necessidade real do modelo profundo, treinou-se e avaliou-se um modelo **Baseline de Regressão Logística** em paralelo à **Rede Neural Perceptron Multicamadas (MLP)**, utilizando rigorosamente o mesmo *split* de dados (80% treino / 20% teste).

### Tabela Comparativa de Desempenho no Conjunto de Teste

| Modelo / Algoritmo | Acurácia | ROC-AUC | Log Loss | Complexidade Computacional | Justificativa de Negócio |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Regressão Logística (Baseline)** | 62,45% | 0,6612 | 0,6420 | Baixíssima (Infinita rapidez) | Não captura interações não lineares complexas entre ocupação, dias até a decolagem e concorrência. |
| **Rede Neural MLP (MLP-64-32)** | **66,88%** | **0,7181** | **0,5810** | Média (~20 épocas de treino) | **Vencedor:** Permite superfícies de decisão suaves para curvas de elasticidade-preço contínuas atrativas para o agente. |

> **Validação de Robustez Estatística (Múltiplas Seeds):**  
> Para garantir que o ganho do modelo profundo não resultou de sorte de amostragem (`seed=42`), a MLP foi submetida a execuções com 3 sementes aleatórias distintas (`10`, `42`, `100`), registrando um **AUC-ROC Médio de 0,7175 ± 0,0012**, confirmando a estabilidade do aprendizado.

---

## 💰 3. Viabilidade Econômica, Custos e Retorno (ROI)

A avaliação financeira do projeto considera uma estrutura típica de implantação em nuvem para e-commerce aéreo de médio porte:

### Tabela Financeira de Custos e Retorno Esperado (Estimativa Anual)

| Categoria Financeira | Descrição do Componente | Custo / Valor Estimado (R$) |
| :--- | :--- | :---: |
| **CAPEX (Custo de Construção)** | Engenharia de dados, desenvolvimento do modelo MLP, prompt engineering do Agente e testes em Shadow Mode | R$ 35.000,00 *(investimento único)* |
| **OPEX (Custo de Sustentação Anual)** | Infraestrutura Cloud (API REST FastAPI/Docker em AWS/GCP), pipelines MLOps e retraining semanal automatizado | R$ 30.000,00 *(R$ 2.500,00 / mês)* |
| **Custo Total do Projeto (Ano 1)** | **CAPEX + OPEX Anual** | **R$ 65.000,00** |
| **Retorno Anual Projetado (Benefício)** | Ganho incremental de +4,5% no RASK em 10 rotas ativas (R$ 45.000,00 / mês) | **R$ 540.000,00 / ano** |

### Cálculo de ROI e Payback
$$\text{ROI} = \frac{\text{Retorno Anual} - \text{Custo Total Anual}}{\text{Custo Total Anual}} = \frac{540.000 - 65.000}{65.000} \times 100 \approx \mathbf{730,77\%}$$

* **Payback Estimado:** **~1,4 meses** após o lançamento oficial em produção.
* **Justificativa de Troca:** O investimento em uma arquitetura de IA (+MLP + Agente) se paga no primeiro trimestre, superando amplamente os custos operacionais de uma tabela estática manual.

---

## 🏗️ 4. Arquitetura da Solução e Raciocínio do Agente

```text
[ Requisição do e-Commerce ]
            │
            ▼
┌──────────────────────────────────────────────┐
│  ETAPA 1: Percepção e Validação de Entrada   │
│  (Valida dias, ocupação, concorrência)       │
└────────────────────┬─────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────┐
│  ETAPA 2: Sensor Neural de Elasticidade      │
│  (Rede Neural MLP calcula P(Compra | Preço)) │
└────────────────────┬─────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────┐
│  ETAPA 3: Agente Prescritivo no Antigravity  │
│  (Calcula E[Receita] e aplica Guardrails)    │
└────────────────────┬─────────────────────────┘
                     │
                     ▼
[ Contrato de Saída JSON Auditável em < 100ms ]

Regras de Governança e Guardrails de Yield ManagementPiso Rígido: Nenhuma tarifa pode ser inferior ao custo marginal base de R$ 400,00.Teto Rígido: Nenhuma tarifa pode ultrapassar o teto comercial de R$ 2.000,00.Regra de Escassez (Yield Escalation): Se a ocupação for $\ge 80\%$ e faltarem $\le 7$ dias para o voo, o piso tarifário eleva-se automaticamente para $\max(\text{R\$ 850}, \text{preço\_concorrente} \times 0.98)$.Regra de Spoilage (Assento Vazio): Se a ocupação for $< 40\%$ e faltarem $\le 15$ dias, limita-se o teto tarifário para estimular o volume imediato de vendas.Regra de Surge de Demanda: Se as buscas na rota nas últimas 24h forem $\ge 800$, o piso da margem é ajustado em $+25\%$.📄


5. Contrato de Saída JSON (Exemplo de Cotação)
{
  "status": "SUCCESS",
  "decisao_tarifaria": {
    "preco_otimizado_brl": 1180.00,
    "probabilidade_conversao_pct": 50.52,
    "receita_esperada_brl": 596.10,
    "margem_sobre_custo_base_brl": 780.00,
    "piso_efetivo_brl": 1176.00,
    "teto_efetivo_brl": 2000.00
  },
  "contexto_mercado": {
    "dias_ate_voo": 3,
    "ocupacao_atual_pct": 85.0,
    "preco_concorrente_brl": 1200.00,
    "volume_buscas_24h": 900
  },
  "regras_acionadas": [
    "[YIELD-ESCASSEZ] Alta ocupação (85%) a 3 dias da partida: Piso tarifário elevado para R$ 1.176,00.",
    "[DEMAND-SURGE] Volume expressivo de buscas (900 buscas/24h): Margem mínima ajustada (+25%)."
  ],
  "parecer_executivo": "Demanda inelástica corporativa identificada. Piso tarifário elevado para proteger os últimos assentos, garantindo tarifa premium com probabilidade de conversão de 50.5%."
}


🔄 6. MLOps: Estratégia de Deploy, Monitoramento e Limiares

Implantação e RetreinoAmbiente de Implantação: API REST de alta performance desenvolvida com FastAPI, empacotada em contêiner Docker e integrada aos microsserviços de checkout do e-commerce.Ciclo de Retreino Automatizado: Retreinamento agendado semanalmente (batch retraining) utilizando as novas transações registradas no e-commerce.Limiares Numéricos de Alerta e Ação MLOpsMonitoramento de Data Drift (Teste Kolmogorov-Smirnov):Calculado diariamente sobre as variáveis de entrada (preco_concorrente e historico_buscas_24h). Se o teste indicar $KS > 0{,}15$ com $p\text{-valor} < 0{,}05$ por mais de 3 dias consecutivos, dispara-se um retreino emergencial fora do ciclo semanal.Monitoramento de Concept Drift (Desvio de Conversão Real):Avalia-se semanalmente o desvio absoluto entre a probabilidade de conversão prevista pela MLP ($\hat{P}$) e a taxa de conversão real observada nos voos. Se o desvio ultrapassar 5,0 pontos percentuais, a API é revertida temporariamente para o modelo baseline de segurança e aciona-se um alerta de auditoria manual para a equipe de Revenue Management.⚠️

7. Declaração de Limitações e Próximos Passos

Limitações do ProjetoMassa de Dados Sintética: O modelo foi treinado sobre um conjunto de dados sintético (4.000 amostras) gerado a partir de uma função logística pré-definida. Os resultados reportados servem para validar o pipeline de engenharia e a arquitetura do agente, devendo ser revalidados com dados históricos reais de reservas antes do lançamento definitivo.Simplificação de Variáveis Microeconômicas: Variáveis externas como câmbio, preço do combustível de aviação (QAV) e sazonalidade de feriados prolongados não foram incluídas neste MVP.Próximos Passos (Roadmap de Execução)Validação em Shadow Mode (Modo Espelho): Executar a API em paralelo ao sistema legado por 30 dias, sem interferir nos preços reais exibidos ao cliente, para comparar as conversões previstas contra as vendas reais.Testes A/B em Produção: Lançar a precificação dinâmica em 2 rotas de teste selecionadas (50% do tráfego orientado pela IA vs. 50% pelo modelo estático) para aferição exata do incremento de RASK.Migração para Feature Store: Implementar Feast ou Snowflake para garantir baixíssima latência na consulta de agregados temporais de pesquisas.



