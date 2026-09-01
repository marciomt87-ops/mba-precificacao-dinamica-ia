# ✈️ Precificação Dinâmica e Yield Management com Redes Neurais

## MBA — Redes Neurais Aplicadas a Negócios

Projeto aplicado de **Inteligência Artificial para precificação dinâmica de passagens aéreas**, utilizando Redes Neurais Artificiais, modelos estatísticos, otimização e regras de negócio de Yield Management.

> **Natureza do projeto:** Proof of Concept (PoC) acadêmica com dados sintéticos.
> Os resultados técnicos e financeiros apresentados neste projeto não representam resultados de produção.

---

## 1. Visão geral

A precificação dinâmica no setor aéreo busca determinar o preço mais adequado para cada oportunidade de venda considerando fatores como:

* proximidade da data do voo;
* ocupação atual da aeronave;
* preço praticado pela concorrência;
* volume de buscas;
* dia da semana;
* preço atualmente ofertado.

O problema pode ser representado como:

$$
P(\text{compra}) =
f(\text{contexto}, \text{preço})
$$

A partir da probabilidade estimada de compra, o sistema calcula a receita esperada:

$$
E[\text{Receita}] =
Preço \times P(\text{compra})
$$

O agente prescritivo então pesquisa diferentes preços possíveis e seleciona aquele que maximiza a receita esperada, respeitando limites de segurança e regras de negócio.

---

# 2. Objetivo

Desenvolver uma PoC de um sistema de **precificação dinâmica para companhias aéreas**, combinando:

1. geração de dados sintéticos;
2. preparação e normalização dos dados;
3. modelo baseline de Machine Learning;
4. Rede Neural MLP;
5. comparação empírica entre os modelos;
6. avaliação de robustez com múltiplas seeds;
7. regra determinística como alternativa sem IA;
8. cálculo de receita esperada;
9. agente prescritivo de precificação;
10. regras de Yield Management;
11. monitoramento de Data Drift e Concept Drift;
12. avaliação econômica por ROI e payback.

---

# 3. Problema de negócio

O assento de uma aeronave é um ativo **perecível**: depois da partida do voo, sua capacidade de geração de receita desaparece.

O sistema de precificação precisa equilibrar dois riscos:

### Spill

Venda antecipada por preço excessivamente baixo, reduzindo a capacidade de capturar maior disposição a pagar posteriormente.

### Spoilage

Manutenção de preços excessivamente altos, resultando em assentos vazios próximos à partida.

O objetivo da solução é encontrar um equilíbrio entre esses dois riscos.

---

# 4. Arquitetura da solução

```text
                    ┌─────────────────────┐
                    │ Dados de contexto   │
                    │                     │
                    │ • Ocupação          │
                    │ • Dias até voo      │
                    │ • Concorrência      │
                    │ • Buscas            │
                    │ • Dia da semana     │
                    │ • Preço             │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Pré-processamento   │
                    │                     │
                    │ Train/Test Split    │
                    │ StandardScaler      │
                    └──────────┬──────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
      ┌──────────────────┐          ┌──────────────────┐
      │ Baseline         │          │ Rede Neural MLP  │
      │                  │          │                  │
      │ Regressão        │          │ Dense + ReLU     │
      │ Logística        │          │ Dropout          │
      └────────┬─────────┘          └────────┬─────────┘
               │                             │
               └──────────────┬──────────────┘
                              ▼
                    ┌─────────────────────┐
                    │ Avaliação           │
                    │                     │
                    │ Accuracy            │
                    │ ROC-AUC             │
                    │ Log Loss            │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Agente Prescritivo  │
                    │                     │
                    │ P(compra)           │
                    │ ×                   │
                    │ preço                │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Receita Esperada    │
                    │                     │
                    │ + Guardrails        │
                    │ + Yield Management  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Preço recomendado   │
                    └─────────────────────┘
```

---

# 5. Dataset

O projeto utiliza um dataset **sintético com 4.000 observações**.

As variáveis utilizadas são:

| Variável               | Descrição                             |
| ---------------------- | ------------------------------------- |
| `dias_ate_decolagem`   | Dias restantes até o voo              |
| `taxa_ocupacao_atual`  | Ocupação atual da aeronave            |
| `preco_concorrente`    | Preço observado na concorrência       |
| `historico_buscas_24h` | Volume de buscas nas últimas 24 horas |
| `dia_da_semana`        | Dia da semana                         |
| `preco_ofertado`       | Preço apresentado ao cliente          |
| `venda_realizada`      | Variável alvo: 0 ou 1                 |

### Importante

Os dados são gerados artificialmente para fins acadêmicos.

Consequentemente:

* não representam comportamento real de passageiros;
* não substituem dados históricos de reservas;
* não permitem concluir que o modelo produzirá determinado aumento de receita em produção;
* servem para demonstrar a arquitetura e a metodologia da solução.

---

# 6. Baseline: Regressão Logística

Antes de utilizar uma Rede Neural, o projeto estabelece uma referência com **Regressão Logística**.

Essa comparação é importante porque um modelo mais complexo somente deve ser adotado se apresentar benefício suficiente para justificar seu custo adicional.

Os dois modelos são treinados e avaliados utilizando:

* o mesmo dataset;
* o mesmo train/test split;
* as mesmas variáveis;
* o mesmo pré-processamento.

As métricas utilizadas são:

* Accuracy;
* ROC-AUC;
* Log Loss.

---

# 7. Rede Neural MLP

A Rede Neural utilizada é uma arquitetura MLP (*Multilayer Perceptron*).

Estrutura:

```text
Input
  │
  ▼
Dense(64) + ReLU
  │
Dropout(20%)
  │
  ▼
Dense(32) + ReLU
  │
Dropout(20%)
  │
  ▼
Dense(16) + ReLU
  │
  ▼
Dense(1) + Sigmoid
  │
  ▼
Probabilidade de compra
```

A saída da rede representa:

$$
P(\text{compra} \mid contexto, preço)
$$

A MLP foi escolhida para testar a capacidade de representar relações não lineares entre preço, demanda, ocupação e proximidade do voo.

---

# 8. Comparação dos modelos

A avaliação não considera somente a performance da Rede Neural.

O projeto compara:

| Modelo               | Função                           |
| -------------------- | -------------------------------- |
| Regressão Logística  | Baseline                         |
| MLP                  | Modelo principal                 |
| Regra determinística | Alternativa sem Machine Learning |

A MLP deve ser considerada superior somente quando seu ganho for:

1. consistente;
2. estatisticamente razoável;
3. relevante para a decisão de negócio;
4. suficiente para justificar sua maior complexidade.

---

# 9. Robustez com múltiplas seeds

Para reduzir a dependência de uma única execução, o experimento é repetido utilizando:

```text
Seed 10
Seed 42
Seed 100
```

São calculadas:

* média;
* desvio padrão;
* Accuracy;
* ROC-AUC;
* Log Loss.

A utilização de múltiplas seeds reduz o risco de interpretar uma execução específica como evidência definitiva de superioridade do modelo.

---

# 10. Alternativa determinística

Além dos modelos de Machine Learning, o projeto implementa uma política baseada em regras.

Exemplos:

### Alta ocupação + voo próximo

Aumentar a proteção de preço.

### Baixa ocupação + voo próximo

Aplicar estratégia de redução para diminuir o risco de spoilage.

### Alta demanda

Considerar aumento do piso de preço.

### Defesa competitiva

Limitar o preço quando o concorrente apresenta uma tarifa muito inferior.

A regra determinística funciona como uma alternativa de menor complexidade e maior interpretabilidade.

---

# 11. Receita esperada

A previsão de Machine Learning é transformada em uma variável econômica.

Para cada preço candidato:

$$
E[R(p)] =
p \times P(\text{compra}|p,x)
$$

onde:

* \(p\) = preço;
* \(x\) = contexto da venda;
* \(P(\text{compra}|p,x)\) = probabilidade estimada pela MLP.

O sistema testa diferentes preços dentro de um intervalo permitido e seleciona:

$$
p^* = \arg\max_p E[R(p)]
$$

Dessa forma, o modelo deixa de ser apenas um classificador e passa a atuar como componente de um **agente prescritivo**.

---

# 12. Agente de precificação

O `AgentePrecificacaoAerea` combina:

```text
Rede Neural
     ↓
Probabilidade de compra
     ↓
Grid Search de preços
     ↓
Receita esperada
     ↓
Regras de Yield
     ↓
Guardrails
     ↓
Preço recomendado
```

O agente não pode simplesmente escolher qualquer preço.

São aplicados limites de segurança, incluindo:

* preço mínimo;
* preço máximo;
* proteção contra spoilage;
* proteção em cenários de alta ocupação;
* defesa competitiva;
* surge de demanda.

---

# 13. Cenários de teste

O pipeline inclui cenários representativos:

### Cenário A — Voo distante / baixa ocupação

Objetivo: reduzir risco de assentos vazios.

### Cenário B — Voo próximo / alta ocupação / alta demanda

Objetivo: capturar maior disposição a pagar.

### Cenário C — Guerra de preços

Objetivo: avaliar a resposta do agente diante de concorrência agressiva.

Para cada cenário, o sistema apresenta:

* preço ótimo;
* preço do concorrente;
* probabilidade estimada de venda;
* receita esperada;
* piso efetivo;
* teto efetivo;
* regras acionadas.

---

# 14. Yield Management

O sistema incorpora conceitos de Yield Management para controlar o comportamento do agente.

### Spoilage Protection

Quando a ocupação está baixa e a partida se aproxima, o sistema permite maior flexibilidade para reduzir preço.

### Escassez

Quando a ocupação está alta e faltam poucos dias para a partida, o sistema aumenta a proteção do inventário.

### Demand Surge

Quando há elevado volume de buscas, o sistema pode elevar o piso da tarifa.

### Defesa competitiva

Quando o preço concorrente é muito baixo, o sistema limita a faixa de preço para evitar perda excessiva de competitividade.

---

# 15. MLOps e monitoramento

Uma solução de IA para precificação não deve ser considerada encerrada após o treinamento.

O projeto propõe monitoramento contínuo.

## Data Drift

É monitorada a alteração na distribuição das variáveis de entrada.

Como critério operacional da PoC:

```text
KS > 0,15
e
p-value < 0,05
```

gera alerta de Data Drift.

O critério deve ser observado de forma persistente antes de um retreinamento emergencial.

---

## Concept Drift

O Concept Drift verifica se a relação entre previsão e comportamento observado mudou.

Critério utilizado na PoC:

```text
|probabilidade média prevista
 - conversão observada| > 5 p.p.
```

gera alerta.

Nesse caso, o sistema pode:

1. retirar temporariamente o modelo;
2. retornar ao baseline;
3. iniciar investigação;
4. realizar novo treinamento.

Os thresholds são parâmetros operacionais da PoC e devem ser recalibrados utilizando dados reais em produção.

---

# 16. Avaliação econômica

O projeto inclui um cenário financeiro para avaliar a viabilidade da solução.

### Premissas

| Indicador                  |      Valor |
| -------------------------- | ---------: |
| CAPEX                      |  R$ 35.000 |
| OPEX anual                 |  R$ 30.000 |
| Custo total Ano 1          |  R$ 65.000 |
| Benefício mensal projetado |  R$ 45.000 |
| Benefício anual projetado  | R$ 540.000 |

### ROI

$$
ROI =
\frac{Benefício - Custo}{Custo}
$$

Considerando as premissas acima:

$$
ROI_{Ano1}
=
\frac{540.000 - 65.000}{65.000}
$$

aproximadamente:

**730,8%**

### Payback

$$
Payback =
\frac{65.000}{45.000}
$$

aproximadamente:

**1,4 meses**

### Importante

Esses valores são **premissas de cenário**, utilizadas para demonstrar a metodologia de análise financeira.

Eles não foram observados empiricamente no dataset sintético e não devem ser interpretados como garantia de retorno em produção.

---

# 17. Limitações

O projeto possui limitações importantes.

### 1. Dados sintéticos

O dataset foi criado artificialmente e não representa comportamento real de passageiros.

### 2. Ausência de validação externa

Não existe, nesta etapa, validação em histórico real de vendas.

### 3. Receita simulada

A receita esperada é calculada a partir da probabilidade prevista pelo modelo.

Isso não demonstra causalidade econômica.

### 4. ROI projetado

Os valores de CAPEX, OPEX e benefício mensal são premissas de cenário.

### 5. Complexidade operacional

Uma operação aérea real envolve fatores adicionais, como:

* conexões;
* classes tarifárias;
* disponibilidade de inventário;
* cancelamentos;
* no-show;
* sazonalidade;
* feriados;
* eventos;
* restrições comerciais;
* custos operacionais;
* elasticidade de demanda real.

---

# 18. Roadmap para produção

A recomendação é não iniciar com alteração automática de preços.

## Fase 1 — Shadow Mode

Executar o modelo em paralelo ao sistema atual, sem alterar o preço apresentado ao cliente.

**Objetivo:** validar previsões.

---

## Fase 2 — A/B Test

Dividir o tráfego entre:

```text
Grupo Controle
    ↓
Precificação atual

Grupo Tratamento
    ↓
Precificação com IA
```

Avaliar:

* receita;
* RASK;
* conversão;
* load factor;
* margem;
* spill;
* spoilage.

---

## Fase 3 — Produção controlada

Expandir para mais rotas somente após comprovação de benefício.

---

## Fase 4 — MLOps

Implementar:

* monitoramento de Data Drift;
* monitoramento de Concept Drift;
* retreinamento;
* versionamento de modelos;
* auditoria das decisões;
* fallback para baseline.

---

# 19. Estrutura do projeto

```text
mba-precificacao-dinamica-ia/
│
├── notebook_precificacao_dinamica_aerea.ipynb
│
├── pipeline_precificacao.py
│
├── README.md
│
├── requirements.txt
│
└── data/
    └── README.md
```

### Notebook

`notebook_precificacao_dinamica_aerea.ipynb`

Contém a apresentação acadêmica, exploração dos dados, treinamento, avaliação, comparação de modelos, agente prescritivo, análise econômica e discussão das limitações.

### Pipeline

`pipeline_precificacao.py`

Implementa a execução estruturada do experimento e disponibiliza:

* geração dos dados;
* pré-processamento;
* baseline;
* MLP;
* múltiplas seeds;
* regras determinísticas;
* agente de precificação;
* monitoramento;
* ROI.

---

# 20. Instalação

Criar um ambiente virtual:

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Instalar as dependências:

```bash
pip install -r requirements.txt
```

---

# 21. Execução

Executar o pipeline completo:

```bash
python pipeline_precificacao.py
```

Executar com quantidade diferente de observações:

```bash
python pipeline_precificacao.py --amostras 5000
```

Executar com outra seed:

```bash
python pipeline_precificacao.py --seed 100
```

Para acelerar a execução e não realizar o experimento de robustez:

```bash
python pipeline_precificacao.py --sem-robustez
```

---

# 22. Principais resultados esperados

Ao executar o projeto, são produzidos resultados para:

### Modelagem

* Accuracy;
* ROC-AUC;
* Log Loss;
* matriz de confusão;
* relatório de classificação.

### Comparação

* Regressão Logística × MLP;
* MLP × regra determinística;
* múltiplas seeds;
* média e desvio padrão.

### Precificação

* preço ótimo;
* probabilidade de venda;
* receita esperada;
* regras acionadas;
* limites de preço.

### Gestão

* Data Drift;
* Concept Drift;
* ROI;
* payback.

---

# 23. Conclusão

O projeto demonstra uma arquitetura de IA aplicada à precificação dinâmica que combina:

**Predição → Otimização → Regras de negócio → Governança → Avaliação econômica**

A Rede Neural MLP é utilizada para estimar a probabilidade de compra, enquanto o agente prescritivo transforma essa previsão em uma decisão de preço.

A comparação com a Regressão Logística evita assumir que maior complexidade implica automaticamente maior valor.

A avaliação com múltiplas seeds aumenta a robustez experimental, enquanto a regra determinística fornece uma alternativa simples e auditável.

Finalmente, a inclusão de receita esperada, ROI, payback, Data Drift, Concept Drift e limitações aproxima a PoC de uma análise de viabilidade empresarial.

Entretanto, a conclusão definitiva sobre geração de valor depende de **dados reais, Shadow Mode e A/B Test**.

---

## Autor

**Marcio Mizumoto Takahara**

Projeto desenvolvido para o **MBA em Redes Neurais Aplicadas a Negócios**.

---

## Aviso acadêmico

Este projeto é uma **Proof of Concept acadêmica**.

Os dados são sintéticos e os valores financeiros são premissas de cenário. Nenhum resultado apresentado deve ser interpretado como evidência de desempenho ou retorno financeiro garantido em ambiente de produção.

