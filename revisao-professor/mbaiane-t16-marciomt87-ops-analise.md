# Análise — Marcio Mizumoto Takahara (`marciomt87-ops`)

**Projeto:** Precificação Dinâmica e Yield Management em e-Commerce de Companhias Aéreas usando Redes Neurais e Agentes de IA
**Repositório:** [marciomt87-ops/mba-precificacao-dinamica-ia](https://github.com/marciomt87-ops/mba-precificacao-dinamica-ia)
**Arquivos analisados:** `notebook_precificacao_dinamica_aerea.ipynb`, `pipeline_precificacao.py`, `README.md`, `requirements.txt`, `Trabalho Marcio M Takahara - Precificacao Aerea.pdf` (contém dois documentos: "Template 1: Prompt de Engenharia do Agente de Precificação e Yield Management" e "Arquitetura e Planejamento: Projeto Precificação Dinâmica & Yield Management Aéreo"), `Como-a-IA-Ajuda-a-Vender-Mais-Passagens-Aereas.pptx`

---

## 1. Abertura

Marcio, parabéns por concluir o projeto! Como Coordenador de Finanças na Polierg, com 14 anos de experiência na área, faz todo sentido você ter escolhido precificação dinâmica como tema — é essencialmente o problema de otimização de margem e receita que você já enxerga no dia a dia, só que aqui modelado como um agente autônomo que decide preço em tempo real. O cuidado de desenhar não só o modelo preditivo, mas também uma camada de governança corporativa (guardrails de piso/teto, regras de yield management) em cima dele, mostra uma preocupação com controle que é rara nos trabalhos da turma e bem alinhada ao seu perfil de Finanças.

## 2. Resumo do projeto

O projeto gera um dataset sintético de 4.000 buscas de passagens aéreas (a partir de uma fórmula logística conhecida definida pelo próprio autor) e treina uma Rede Neural Feedforward (MLP, 64→32 neurônios, ReLU, Dropout 0,20) para estimar a probabilidade de compra de um bilhete dado o preço ofertado e o contexto de mercado (dias até o voo, ocupação, preço do concorrente, buscas nas últimas 24h, dia da semana). Sobre essa rede é construído um "Agente de IA" prescritivo (`AgentePrecificacaoAerea`) que varre uma grade de preços, maximiza a receita esperada `E[Receita] = p × P̂(Compra|p)` e aplica regras táticas de Yield Management (escassez, spoilage, defesa competitiva, surge de demanda). O plano de projeto (PDF) menciona Regressão Logística e XGBoost/LightGBM como baseline e candidatos de comparação, mas nenhum dos dois é de fato treinado no notebook ou no pipeline — apenas a MLP é executada e avaliada.

## 3. Nota por critério

### Critérios de negócio (peso maior)

**1. Aderência ao negócio — 7,5/10**
1.1. Métrica de sucesso nomeada como receita ou custo: **5** — o plano de projeto define explicitamente "Aumento de 3% a 6% no RASK (*Revenue per Available Seat Kilometer*)" como métrica de sucesso (PDF, "Arquitetura e Planejamento", Seção 1), que é uma métrica de receita nomeada, não genérica.
1.2. Métrica quantificada: **5** — RASK +3% a +6%, Load Factor +8%, redução do tempo de reação "de horas para milissegundos", todos com número explícito (mesma seção do PDF).
1.3. Conexão entre métrica técnica e impacto de negócio: **2** — o AUC obtido (0,7181) e a acurácia (66,88%) nunca são traduzidos em RASK, receita ou R$. Pior: a conclusão do notebook (célula 21, Seção 8) afirma "+8% a +15% de incremento de receita", um número que não é derivado do resultado técnico do modelo e diverge do "+3% a +6% no RASK" citado no plano de projeto — os dois documentos não conversam entre si.

**2. Viabilidade econômica (ROI) — 1,25/10**
2.1. Custo de construção estimado: **1** — ausente em qualquer um dos três documentos (notebook, PDF, pptx).
2.2. Custo de sustentação estimado: **1** — o PDF menciona a *atividade* de retraining semanal e monitoramento de drift (Seção 7, "MLOps: Deploy e Monitoramento") mas nunca estima seu custo em R$ ou horas de engenharia.
2.3. Retorno esperado com número: **3** — há percentuais (RASK +3-6%, Load Factor +8%), mas nunca convertidos em R$ absoluto (ex: receita incremental projetada por rota/mês).
2.4. Comparação custo vs. retorno: **1** — nenhum payback, ROI% ou breakeven em lugar nenhum do material.

### Critérios técnicos (peso menor)

**3. Necessidade real de IA — 0/10**
3.1. Discute alternativa de automação/regra determinística: **1** — apesar do próprio agente já operar com regras de threshold (ocupação ≥ 80%, dias ≤ 7, etc. — célula 16), o projeto nunca discute explicitamente por que uma tabela de preços por faixa (regra pura, sem rede neural) não resolveria o problema com menor custo e complexidade antes de justificar o MLP.

**4. ML tradicional vs. Redes Neurais — 3,75/10**
4.1. Compara explicitamente contra ML tradicional (discussão): **4** — o PDF ("Arquitetura e Planejamento", Seção 4 "Seleção de Algoritmos") lista Regressão Logística como baseline e XGBoost/LightGBM como candidatos, com justificativa de que a MLP "permite extrair superfícies de decisão suaves para curvas de elasticidade-preço".
4.2. Baseline simples de fato executado e comparado no notebook: **1** — nem o notebook nem `pipeline_precificacao.py` treinam Regressão Logística, XGBoost ou qualquer modelo além da MLP. A promessa do plano nunca chega ao código.

**5. Aderência ao conteúdo do curso — 10/10**
5.1. Nomeia arquitetura vista em aula: **5** — MLP Feedforward com ReLU, Dropout e Adam (Aulas 2/3), nomeada explicitamente em `construir_rede_neural`/`construir_modelo_rede_neural` (célula 9 do notebook e `pipeline_precificacao.py`).
5.2. Arquitetura adequada ao tipo de dado: **5** — dados tabulares de contexto de mercado, MLP é escolha correta (não é problema de imagem, sequência ou texto).
5.3. *(não aplicável — problema não é de texto)*

**6. Aderência ao template de projeto — 8,75/10**
6.1. Cobre os 7 blocos do `templates_projetos_ia.md`: **5** — o documento "Arquitetura e Planejamento: Projeto Precificação Dinâmica & Yield Management Aéreo" (páginas 8-11 do PDF) cobre rigorosamente os 7 blocos: (1) Visão Geral e Objetivo, (2) Coleta e Preparação de Dados, (3) Estratégia de Bases e Separação, (4) Seleção de Algoritmos, (5) Estratégia de Treinamento e Otimização, (6) Testes, Validação e Métricas, (7) MLOps: Deploy e Monitoramento.
6.2. Profundidade do bloco 7 (MLOps): **4** — o bloco (páginas 10-11 do PDF) nomeia com precisão o que monitorar — "Data Drift: Monitoramento via teste Kolmogorov-Smirnov para identificar se o perfil de buscas ou os preços da concorrência mudaram significativamente" e "Concept Drift: Monitoramento do desvio entre a probabilidade prevista de conversão e a taxa real de vendas realizadas nos voos" — e define frequência de retreino ("Re-treinamento semanal automatizado do modelo preditivo"). Falta apenas o terceiro elemento: nenhum limiar numérico de KS ou de desvio de conversão é definido para disparar uma ação fora do ciclo semanal, nem um critério de quando o modelo deveria ser revisado manualmente ou aposentado.

**7. Correção técnica — 7,5/10**
7.1. Código executa sem erro: **5** — confirmado; `execution_count` sequencial de 1 a 13 no notebook, sem saídas de erro.
7.2. Split antes de pré-processamento: **5** — `StandardScaler().fit_transform(X_train_raw)` aplicado apenas ao treino, depois de `train_test_split` (célula 7), sem vazamento.
7.3. Métrica adequada à distribuição: **5** — dataset é aproximadamente balanceado (54% vs. 46%), e o autor usa acurácia + AUC-ROC + `classification_report` (célula 13), adequado.
7.4. Baseline avaliado no mesmo split/dados que o modelo principal: **1** — não há baseline algum implementado para ser avaliado no mesmo split; o item não pode ser atendido porque a comparação simplesmente não existe no código.

**8. Qualidade do código — 10/10**
8.1. Seeds fixadas: **5** — `SEED = 42` aplicado a `random`, `np.random`, `tf.random` e `PYTHONHASHSEED` (célula 2 do notebook e `configurar_ambiente()` no pipeline).
8.2. Dependências declaradas: **5** — `requirements.txt` no repositório, com versões mínimas (`tensorflow>=2.15.0`, `scikit-learn>=1.2.0` etc.).
8.3. Organização em funções/seções: **5** — código bem seccionado em funções reutilizáveis (`gerar_dataset_passagens_aereas`, `preparar_dados`, `construir_modelo_rede_neural`, `treinar_modelo`, `avaliar_modelo`, `prever_probabilidade_venda`, classe `AgentePrecificacaoAerea`), e o mesmo desenho é replicado com fidelidade em `pipeline_precificacao.py` como script pronto para CLI/CI-CD — sinal de maturidade de engenharia.

**9. Honestidade dos resultados — 0/10**
9.1. Múltiplas seeds/execuções ou justificativa: **1** — uma única execução com `seed=42`; AUC de 0,7181 nunca é confirmado com repetições.
9.2. Seção de limitações presente: **1** — ausente em todos os documentos. Em particular, o fato de o dataset ser 100% sintético (gerado por uma fórmula logística conhecida pelo próprio autor, célula 4) nunca é discutido como limitação — os resultados são apresentados na Seção 8 do notebook com tom de conclusão executiva definitiva, sem ressalva.

## 4. Pontos fortes

- **6.1 Cobertura dos 7 blocos do template** (5/5): o documento "Arquitetura e Planejamento: Projeto Precificação Dinâmica & Yield Management Aéreo" (páginas 8-11 do PDF) cobre rigorosamente os 7 blocos do `templates_projetos_ia.md`, incluindo uma estratégia de MLOps concreta (microsserviço FastAPI/Docker, teste Kolmogorov-Smirnov para data drift, retreino semanal automatizado).
- **7.2 Split antes de pré-processamento** (5/5): `StandardScaler().fit_transform(X_train_raw)` (célula 7) é ajustado somente após o `train_test_split`, sem vazamento de dados do teste para o treino.
- **8.1 Seeds fixadas** (5/5): `SEED = 42` aplicado consistentemente a `random`, `np.random`, `tf.random` e `PYTHONHASHSEED` (célula 2 e `configurar_ambiente()` em `pipeline_precificacao.py`).
- **8.3 Organização em funções/seções** (5/5): arquitetura conceitual em duas camadas bem desenhada — a MLP como "sensor" de elasticidade-preço encapsulada em funções reutilizáveis (`gerar_dataset_passagens_aereas`, `construir_modelo_rede_neural`, `treinar_modelo`, `avaliar_modelo`) e o `AgentePrecificacaoAerea` como camada prescritiva com regras de yield management — com o mesmo desenho replicado com fidelidade em `pipeline_precificacao.py` como script pronto para produção, raro na turma.

## 5. Pontos de melhoria

- **2.1/2.2/2.4 Custo de construção, sustentação e comparação com o retorno** (1/5): zero estimativa de custo em qualquer documento — os dois critérios de negócio que mais pesam na rubrica ficam sem qualquer número monetário, e por isso também não há payback, ROI% ou breakeven em lugar nenhum.
- **4.2 Baseline simples de fato executado** (1/5): o plano promete Regressão Logística e XGBoost/LightGBM (PDF, Seção 4), mas o código só executa a MLP — é a lacuna técnica mais grave do trabalho, porque não permite saber se a rede neural realmente agrega valor sobre um modelo mais simples e barato.
- **1.3 Conexão entre métrica técnica e impacto de negócio** (2/5): os números de impacto de negócio divergem entre documentos e não são derivados do resultado técnico real — "+3% a +6% no RASK" no plano de projeto vs. "+8% a +15% de incremento de receita" na conclusão do notebook (célula 21) — nenhum dos dois é conectado ao AUC=0,7181 obtido.
- **9.1/9.2 Múltiplas execuções e seção de limitações** (1/5): resultado obtido em execução única, sem seção de limitações que reconheça que o dataset é 100% sintético (gerado pela própria fórmula logística do autor) e que, portanto, os ganhos reportados não têm respaldo em dados reais de mercado.

## 6. Nota final

**5,0 / 10** — O projeto tem uma arquitetura conceitual sofisticada (rede neural + agente prescritivo com governança) e um plano de projeto tecnicamente bem escrito cobrindo os 7 blocos, mas a fundamentação econômica é quase inexistente (sem custo, sem ROI, sem conexão real entre AUC e receita) e a comparação contra ML tradicional prometida no plano nunca chega a ser executada no código — os dois fatores que mais pesam nesta rubrica puxam a nota para baixo, apesar da qualidade de engenharia do código ser um dos pontos altos da turma.

**Nível de maturidade: PoC/protótipo.** O bloco 7 (MLOps) do plano é mais concreto que a média da turma — nomeia o que monitorar e a frequência de retreino (6.2) — mas isso não é suficiente para um piloto controlado: não há qualquer estimativa de custo de construção ou sustentação (2.1/2.2), o retorno projetado diverge entre documentos e não deriva do resultado técnico (1.3), e todo o experimento roda sobre dados 100% sintéticos, sem qualquer validação com dado real de mercado. Antes de avançar a um piloto controlado, faltam a base econômica (custo vs. retorno) e uma primeira validação com dados reais.

## 7. Task list para evoluir o trabalho

**1. Aderência ao negócio**
- [ ] **1.3 Conexão entre métrica técnica e impacto de negócio (2/5):** o Bloco D é direto — "Métrica-Alvo de Negócio: a métrica que traduz o resultado estatístico em dinheiro ou eficiência operacional" — e o AUC=0,7181 nunca chega a essa tradução. O próprio curso tem um notebook do mesmo domínio: `08-precificacao-dinamica.ipynb` (células 16-19) calcula `receita = preço × demanda_prevista` em R$ de fato, comparando modelos — replique esse cálculo usando a probabilidade de compra prevista pela MLP para estimar a receita simulada por rota, e resolva a divergência entre "+3% a +6% no RASK" (PDF, Seção 1) e "+8% a +15% de incremento de receita" (notebook, célula 21, Seção 8), explicitando de onde vêm esses números. — ver `notebook_precificacao_dinamica_aerea.ipynb` (Seção 8, célula 21) e PDF "Arquitetura e Planejamento" (Seção 1)

**2. Viabilidade econômica (ROI)**
- [ ] **2.1 Custo de construção (1/5):** o Bloco C trata isso como regra prática de seleção de algoritmo — "custo computacional: treino e inferência têm custo, meça contra o orçamento disponível" e "tempo de treinamento — modelos complexos podem levar horas ou dias para treinar de novo". Nenhum dos três documentos estima o custo de coletar/tratar os dados de mercado (GDS, web scrapers), o tempo/infra de treino da MLP (célula 9) ou o custo de construir o microsserviço FastAPI/Docker descrito na Seção 7 do plano. — ver PDF "Arquitetura e Planejamento" (Seção 7, MLOps) e `pipeline_precificacao.py`
- [ ] **2.2 Custo de sustentação (1/5):** o Bloco B contrasta automação com IA numa linha de tabela (coluna "Manutenção": "Atualiza-se a regra manualmente" vs. "Retreina-se o modelo periodicamente") — e é exatamente esse custo recorrente que falta estimar aqui: o plano já define retreino semanal automatizado e monitoramento de Data/Concept Drift (Seção 7), mas nunca em R$ ou horas de engenharia por ciclo. — ver PDF "Arquitetura e Planejamento" (Seção 7, MLOps: Deploy e Monitoramento)
- [ ] **2.3 Retorno esperado com número (3/5):** o Bloco D recomenda formalizar com placeholders quando falta o dado exato — "utilize placeholders visíveis como [Inserir % de economia]... em vez de inventar números irreais". Os percentuais de RASK/Load Factor já existem (PDF, Seção 1); falta convertê-los em R$ absoluto por rota/mês, ex: `[Inserir receita média por rota] × Nº de rotas × 3-6%`, deixando a suposição explícita. — ver PDF "Arquitetura e Planejamento" (Seção 1, Métrica de Sucesso)
- [ ] **2.4 Comparação custo vs. retorno (1/5):** o Bloco C fecha esse raciocínio com "comece sempre por um baseline simples [...] um modelo mais complexo só se justifica se o ganho superar o custo". O curso não formaliza uma fórmula de ROI/payback — mas depois de estimar 2.1/2.2 (custo) e 2.3 (retorno em R$), aplique `ROI = (retorno - custo) / custo` ou um payback em meses para justificar a MLP + agente frente a uma tabela de preços simples. — ausente em todo o material

**3. Necessidade real de IA**
- [ ] **3.1 Discute alternativa de automação/regra determinística (1/5):** o Bloco B traz o teste direto: "a lógica pode virar regras fixas (se-então)? [...] Três "sim" seguidos = provavelmente um projeto de IA. Se alguma resposta for "não", automação simples resolve com menos custo e mais previsibilidade." O próprio agente já opera com regras de threshold (ocupação ≥ 80%, dias ≤ 7 — célula 16), o que é sinal de que uma tabela de preços por faixa (sem rede neural) merece ser testada como alternativa — como em `07-manutencao-preditiva.ipynb` (célula 26), onde "a regra simples de limiar teve desempenho comparável (levemente melhor) ao da LSTM". — nenhuma discussão encontrada no notebook, no pipeline ou no PDF

**4. ML tradicional vs. Redes Neurais**
- [ ] **4.1 Compara explicitamente contra ML tradicional — discussão (4/5):** a discussão já existe e é boa (PDF, Seção 4 "Seleção de Algoritmos" lista Regressão Logística e XGBoost/LightGBM com justificativa) — falta só transformar essa promessa em código para fechar com 5/5 (ver 4.2). — ver PDF "Arquitetura e Planejamento" (Seção 4)
- [ ] **4.2 Baseline simples de fato executado (1/5):** o Bloco C é claro — "comece sempre por um baseline simples. Se ele já resolve, um modelo mais complexo só se justifica se o ganho superar o custo" — mas nem `notebook_precificacao_dinamica_aerea.ipynb` nem `pipeline_precificacao.py` treinam a Regressão Logística ou o XGBoost/LightGBM prometidos no plano. Os notebooks `09-score-credito.ipynb` (células 13-17) e `01-deteccao-fraude.ipynb` (células 9-14) treinam e comparam Regressão Logística e MLP lado a lado no mesmo split — replique esse padrão: treine a Regressão Logística com os mesmos dados de `preparar_dados` e reporte AUC/acurácia ao lado da MLP. — ver `notebook_precificacao_dinamica_aerea.ipynb` (Seção 5, célula 13) e `pipeline_precificacao.py` (`avaliar_modelo`)

**6. Aderência ao template de projeto**
- [ ] **6.2 Profundidade do bloco 7/MLOps (4/5):** o plano já nomeia bem o que monitorar (Data Drift via teste Kolmogorov-Smirnov, Concept Drift via desvio entre conversão prevista e real) e a frequência de retreino ("semanal automatizado") — falta só o critério de quando revisar/aposentar o modelo: defina um limiar numérico (ex: estatística KS acima de um valor, ou desvio de conversão acima de X p.p. por N semanas seguidas) que dispare uma revisão manual ou um retreino fora do ciclo semanal padrão. — ver PDF "Arquitetura e Planejamento" (Seção 7, MLOps: Deploy e Monitoramento, página 11)

**7. Correção técnica**
- [ ] **7.4 Baseline avaliado no mesmo split (1/5):** consequência direta de 4.2 — sem baseline implementado, não há como comparar no mesmo split. Assim que a Regressão Logística for treinada (ver dica 4.2), avalie-a com o mesmo `X_test`/`y_test` usado para a MLP em `avaliar_modelo`, reportando as métricas lado a lado. — ver `pipeline_precificacao.py` (`avaliar_modelo`)

**9. Honestidade dos resultados**
- [ ] **9.1 Múltiplas seeds/execuções (1/5):** o Bloco A é direto — "cada comparação foi rodada com 3 sementes aleatórias diferentes (não uma vez só), para separar ganho real de sorte da rodada" — com exemplos como "CNN varia de 40% a 83% de acurácia dependendo da semente". O AUC de 0,7181 obtido aqui (`treinar_modelo`, célula 10, `seed=42`) nunca foi confirmado com outras sementes. Os notebooks `01-deteccao-fraude.ipynb` e `09-score-credito.ipynb` (células 22-25) repetem o treino com múltiplas seeds e reportam média ± desvio padrão — repita esse padrão para a MLP e, quando o baseline (4.2) estiver pronto, para ele também. — ver `notebook_precificacao_dinamica_aerea.ipynb` (célula 10, `treinar_modelo`)
- [ ] **9.2 Seção de limitações (1/5):** o curso não tem um trecho literal específico sobre "seção de limitações", mas o mesmo espírito de honestidade do Bloco A (separar ganho real de sorte da rodada) se aplica aqui: o dataset é 100% sintético, gerado por uma fórmula logística conhecida pelo próprio autor (`gerar_dataset_passagens_aereas`, célula 4), e isso nunca é declarado como limitação — a Seção 8 do notebook apresenta os resultados com tom de conclusão executiva definitiva. Adicione uma seção explícita reconhecendo que os resultados não substituem validação com dados reais de mercado. — ver `notebook_precificacao_dinamica_aerea.ipynb` (célula 4 e Seção 8, célula 21)

## 8. Tópicos para o aluno revisar

- **Validação de robustez / múltiplas execuções** (Bloco C — Design de Projetos de IA) — motivado por `notebook_precificacao_dinamica_aerea.ipynb` (célula 10): o AUC de 0,7181 foi obtido em uma única execução com `seed=42`; vale revisitar por que múltiplas execuções são necessárias antes de declarar um resultado como conclusivo.
- **ROI e viabilidade econômica de projetos de IA** (Bloco C / `templates_projetos_ia.md`, seção MLOps) — motivado pela ausência total de estimativa de custo no PDF "Arquitetura e Planejamento" (Seção 7): o template pede explicitamente custo de dados/treino/infra vs. retorno, e essa parte não aparece em nenhum documento do trabalho.
- **Comparação prática ML tradicional vs. Redes Neurais** (Aula 1/2 — O Neurônio e o Perceptron / Arquitetura e Camada Oculta) — motivado pelo fato de o plano citar Regressão Logística e XGBoost/LightGBM como candidatos (PDF, Seção 4) mas o notebook nunca os implementar; revisar por que a comparação empírica (não só a discussão teórica) é o que sustenta a escolha de uma rede neural sobre um modelo mais simples.
- **Estratégia de treinamento e otimização** (Aula 3 — Treinando Redes Neurais com Keras e PyTorch) — motivado pelo plano mencionar busca de hiperparâmetros via Optuna/Random Search (PDF, Seção 5) que nunca é executada no notebook (os hiperparâmetros de `construir_modelo_rede_neural`, célula 9, são fixados manualmente); revisar como conectar a estratégia de otimização planejada ao código de fato implementado.
