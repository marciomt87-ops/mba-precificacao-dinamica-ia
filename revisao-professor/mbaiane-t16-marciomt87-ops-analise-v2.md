# Análise v2 — Marcio Mizumoto Takahara (`marciomt87-ops`)

**Projeto:** Precificação Dinâmica e Yield Management em e-Commerce de Companhias Aéreas usando Redes Neurais e Agentes de IA
**Repositório:** [marciomt87-ops/mba-precificacao-dinamica-ia](https://github.com/marciomt87-ops/mba-precificacao-dinamica-ia)
**Nota v1:** 5,0/10 (ver `mbaiane-t16-marciomt87-ops-analise.md`)

**Arquivos analisados nesta revisão:**
- `notebook_precificacao_dinamica_aerea_revisado_professor.ipynb` — referido abaixo como **"notebook v2"**
- `pipeline_precificacao_revisado_professor.py` — referido abaixo como **"pipeline v2"**
- `README.v2.md`
- `Trabalho Marcio M Takahara - Precificacao Aerea.v2.pdf` (enviado em 31/08, commit `f318ea2a`) — referido abaixo como **"PDF v2-A"**
- `Trabalho precificação aerea Marcio Versao 2.pdf` (enviado em 01/09, commit `07a40d35`, com capa de "Relatório Executivo") — referido abaixo como **"PDF v2-B"**
- Para comparação: `notebook_precificacao_dinamica_aerea.ipynb` ("notebook v1"), `pipeline_precificacao.py` ("pipeline v1")
- `revisao-professor/mbaiane-t16-marciomt87-ops-analise.md` — cópia idêntica (`diff` sem saída) da análise v1, incluída pelo próprio aluno no repositório

**Nota metodológica importante:** para esta revisão, o código foi de fato **executado** (não apenas lido) em ambiente com `scikit-learn 1.6.1`/`numpy 1.26.4` (baseline) e, num segundo momento, com `tensorflow 2.20.0` reproduzindo o `pipeline v2` completo (`--sem-robustez` e depois com as 3 seeds). Os resultados dessa execução independente aparecem ao longo do documento e, em um ponto central (seção "O que mudou desde a v1", itens 4.2/9.1), divergem dos números publicados no README/PDF — isso é detalhado com evidência bruta abaixo.

---

## O que mudou desde a v1

| # | Sub-item | Nota v1 → v2 | O que mudou |
|---|---|---|---|
| 1.3 | Conexão métrica técnica → impacto de negócio | 2 → 4 | A divergência "+3-6% RASK" vs. "+8-15% receita" da v1 **desapareceu** — os documentos agora convergem em "+3% a +6% no RASK" / "+4,5%" no cenário de ROI. Foi adicionado um cálculo real de receita simulada comparando MLP vs. Regressão Logística no conjunto de teste (notebook v2, célula 27, função `receita_media_simulada`), e o notebook agora **distingue explicitamente** "impacto preditivo/simulado" (derivado do modelo) de "cenário financeiro" (premissa de negócio, não derivada do AUC) — célula 26: *"A segunda análise não deve ser apresentada como prova de que AUC = X gera automaticamente R$ Y"*. Ainda não é uma tradução direta do AUC=0,7181 em R$, mas a honestidade sobre isso é nova e bem-vinda. |
| 2.1 | Custo de construção estimado | 1 → 4 | CAPEX de R$ 35.000 com categorias descritas (engenharia de dados, treino da MLP, desenvolvimento do agente, testes em Shadow Mode) — PDF v2-A/B, Seção 3; `pipeline v2`, `ConfigEconomica.capex` (linha 62). Falta detalhamento granular (ex: horas × taxa). |
| 2.2 | Custo de sustentação estimado | 1 → 4 | OPEX de R$ 30.000/ano (R$ 2.500/mês) categorizado (infra cloud FastAPI/Docker, pipelines de MLOps, retreino semanal) — mesma seção; `pipeline v2`, `ConfigEconomica.opex_anual` (linha 63). |
| 2.3 | Retorno esperado com número | 3 → 5 | R$ 540.000/ano explícito, calculado a partir de R$ 45.000/mês × 12 (notebook v2, célula 28; `pipeline v2`, `beneficio_anual`, linha 71). |
| 2.4 | Comparação custo vs. retorno | 1 → 5 | ROI = 730,77% e payback ≈ 1,4 meses, com fórmula explícita e **calculada em código** (notebook v2, célula 28; `pipeline v2`, `calcular_roi`, linhas 633-641). Reproduzi a conta de forma independente: `(540.000-65.000)/65.000 = 7,3077` e `65.000/45.000 = 1,444` meses — bate exatamente. |
| 3.1 | Discussão de alternativa determinística | 1 → 5 | Nova seção "Alternativa de automação: regra determinística" (notebook v2, células 16-18) com função `preco_regra_deterministica` e uma discussão explícita de quando a IA se justifica ("a IA só se justifica se o ganho econômico e preditivo superar o custo adicional" — célula 18). Também implementado em `pipeline v2` (`preco_por_regra_deterministica`, linha 290) e no PDF v2-A (Seção 2 do Documento 2, "Discussão: Necessidade Real de IA vs. Regras Determinísticas Puras"). **Atenção:** essa seção de discussão **não existe** no PDF v2-B — ver ambiguidade registrada abaixo. |
| 4.1 | Discussão vs. ML tradicional | 4 → 5 | Discussão mais amadurecida, agora com pergunta de decisão explícita: "A MLP melhora as métricas de forma consistente o suficiente para justificar o custo adicional?" (notebook v2, célula 14). |
| 4.2 | Baseline de fato executado e comparado | 1 → 4 | Regressão Logística agora treinada e avaliada (notebook v2, células 9-10; `pipeline v2`, `treinar_baseline_logistico`/`avaliar_baseline`, linhas 275-287) — resolve a lacuna mais grave da v1. **Mas:** os números publicados (62,45%/0,6612) não reproduzem quando executo o código exatamente como está no repositório — ver detalhe crítico abaixo. |
| 6.2 | Profundidade do MLOps | 4 → 5 | Limiares numéricos agora definidos (KS > 0,15 com p < 0,05 por 3 dias; desvio de conversão > 5 p.p. por 2 semanas) **e implementados em código**: `pipeline v2` (`verificar_data_drift`/`verificar_concept_drift`, linhas 575-630) e notebook v2 (célula 32, usando `scipy.stats.ks_2samp`). |
| 7.1 | Código executa sem erro | 5 → 3 | O notebook v1 tinha `execution_count` sequencial 1-13. O **notebook v2 tem `execution_count: null` e `outputs: []` em todas as 16 células de código** — nunca foi executado e salvo. Rodei eu mesmo o `pipeline v2` equivalente e ele executa do início ao fim sem erro, mas isso não é evidência que o próprio artefato entregue demonstra. |
| 9.1 | Múltiplas seeds/execuções | 1 → 3 | Mecanismo de 3 seeds (10, 42, 100) implementado corretamente em código (notebook v2, células 18-19; `pipeline v2`, `executar_multiplas_seeds`, linhas 507-572) — grande avanço estrutural. **Mas:** ao executar esse código eu mesmo, obtive AUC médio da MLP de 0,691 ± 0,009 (3 seeds), não 0,7175 ± 0,0012 como reportado — desvio-padrão quase 7× maior e média ~2,6 p.p. mais baixa. Sem outputs salvos no notebook, não há como conferir a origem do número publicado. |
| 9.2 | Seção de limitações | 1 → 5 | Presente em 4 lugares: notebook v2 (célula 33, "13. Limitações"), `pipeline v2` (seção impressa "10. LIMITAÇÕES", linhas 811-818), README.v2.md (Seção 7) e ambos os PDFs v2 (Seção 6, "Declaração de Limitações"). Todos declaram explicitamente o dataset 100% sintético. |
| 1.1, 1.2, 5.1, 5.2, 6.1, 7.2, 7.3, 8.1, 8.2, 8.3 | — | sem mudança | Não afetados pela revisão; mantêm nota e evidência da v1 (repetidos na íntegra na Seção "3. Nota por critério" abaixo). |

### O achado central desta revisão: números publicados não reproduzem

O ponto mais importante desta v2 não é a ausência de mecanismos (isso foi corrigido, e bem) — é que, ao **executar o código exatamente como está no repositório**, os números não batem com os alegados no README/PDF:

| Métrica | Alegado (README/PDF) | Obtido executando `pipeline v2` (seed=42) | Obtido nas 3 seeds (10/42/100) |
|---|---|---|---|
| Regressão Logística — Acurácia | 62,45% | **64,38%** | méd. 64,88% ± 1,09 p.p. |
| Regressão Logística — AUC | 0,6612 | **0,6777** | méd. 0,6813 ± 0,0056 |
| MLP — Acurácia | 66,88% | **62,88%** | méd. 63,58% ± 0,64 p.p. |
| MLP — AUC | 0,7181 | **0,6841** | méd. 0,6912 ± 0,0090 |

Isso foi reproduzido **duas vezes de forma independente**: uma vez transcrevendo a lógica da célula 4/9 do notebook v2 para um script isolado (`scikit-learn` puro), e outra rodando o `pipeline_precificacao_revisado_professor.py` oficial do repositório de ponta a ponta (`--sem-robustez` e depois com as 3 seeds via `executar_multiplas_seeds`). Os dois métodos concordam entre si.

Notem dois pontos de gravidade diferente:
- A parte da **Regressão Logística** é preocupante porque `LogisticRegression` com solver `lbfgs` sobre features padronizadas é um problema convexo, essencialmente determinístico entre versões do scikit-learn — uma diferença de ~2 p.p. de acurácia e ~1,6 p.p. de AUC não é o tipo de coisa que se espera de "drift de biblioteca".
- A parte da **MLP** é mais tolerante a variação entre ambientes (Keras/TensorFlow não garante reprodutibilidade bit-a-bit entre versões maiores, mesmo com seed fixa), então parte da divergência pode vir do `tensorflow>=2.15.0` (sem trava de versão) do `requirements.txt` vs. o `tensorflow 2.20.0` usado nesta verificação.
- Mas o efeito qualitativo é o que mais importa para a rubrica: na minha execução, a MLP deixa de vencer com folga (a narrativa central da comparação) — no critério Acurácia, a Regressão Logística **vence em média** (64,9% vs. 63,6%), e no AUC a vantagem da MLP cai de ~5,7 p.p. alegados para ~1,0-1,5 p.p. observados, com uma variância real ~7× maior que a reportada.

Isso não invalida o mérito estrutural do que foi construído (o mecanismo de comparação no mesmo split, de múltiplas seeds, de ROI, de limiares de MLOps — tudo isso está corretamente desenhado em código). Mas significa que os números específicos publicados no README/PDF não podem ser tomados como validados sem uma nova rodada de execução registrada (`.ipynb` salvo com outputs, ambiente com versões travadas).

### Ambiguidade real entre os dois PDFs "v2"

Os dois arquivos **não são o mesmo documento em formatações diferentes** — divergem em conteúdo:
- **PDF v2-A** (`...Aerea.v2.pdf`, 31/08) contém as Seções 5 "Escopo & Restrições", 6 "Definições Operacionais" e 7 "Formato de Saída JSON" do Documento 1, e a Seção 2 do Documento 2 é "Discussão: Necessidade Real de IA vs. Regras Determinísticas Puras" (que sustenta a nota 5/5 do item 3.1).
- **PDF v2-B** (`...Versao 2.pdf`, 01/09, com capa de "Relatório Executivo") **não tem** essas três seções do Documento 1, e sua Seção 2 do Documento 2 é "Coleta, Preparação de Dados e Mapeamento de Features" — sem a discussão de regra determinística.

O PDF v2-B é o mais recente (por data de commit) e o mais completo em formatação (capa, tabelas mais bem diagramadas), então foi usado como referência principal nesta análise para os blocos financeiro/MLOps/limitações — que são idênticos em conteúdo entre os dois. Mas para o item 3.1, a evidência documental "oficial" mais recente (PDF v2-B) na verdade **não cobre** esse ponto — o crédito de 3.1 vem do PDF v2-A e, principalmente, do notebook v2/pipeline v2 (que têm essa discussão em código, independentemente de qual PDF se considere "o" documento final). Vale o aluno consolidar isso em um único PDF antes da entrega final, para não deixar ambíguo qual documento é a versão de referência.

---

## 3. Nota por critério

### Critérios de negócio (peso maior)

**1. Aderência ao negócio — 9,17/10**
1.1. Métrica de sucesso nomeada como receita ou custo: **5** — inalterado; RASK explícito (PDF v2-B, Seção 1 do Documento 2; README.v2.md, Seção 1).
1.2. Métrica quantificada: **5** — inalterado; RASK +3-6%, Load Factor +8%, latência <100ms (mesmas fontes).
1.3. Conexão entre métrica técnica e impacto de negócio: **4** — ver "O que mudou" acima. Cálculo real de receita simulada (notebook v2, célula 27) e distinção honesta entre resultado técnico e premissa de negócio (célula 26). Não chega a 5 porque o número usado no ROI (R$ 45.000/rota/mês) continua sendo uma premissa declarada, não uma derivação direta do AUC=0,7181.

**2. Viabilidade econômica (ROI) — 8,75/10**
2.1. Custo de construção estimado: **4** — CAPEX R$ 35.000 categorizado (PDF v2-B, Seção 3; `pipeline v2`, `ConfigEconomica.capex`, linha 62). Falta granularidade (ex: horas de engenharia × taxa).
2.2. Custo de sustentação estimado: **4** — OPEX R$ 30.000/ano = R$ 2.500/mês categorizado (mesma seção; `ConfigEconomica.opex_anual`, linha 63).
2.3. Retorno esperado com número: **5** — R$ 540.000/ano explícito e calculado (notebook v2, célula 28).
2.4. Comparação custo vs. retorno: **5** — ROI 730,77%, payback 1,4 meses, fórmula em código (`pipeline v2`, `calcular_roi`, linhas 633-641), aritmética conferida e correta.

### Critérios técnicos (peso menor)

**3. Necessidade real de IA — 10/10**
3.1. Discute alternativa de automação/regra determinística: **5** — nova seção completa com código e reflexão explícita (notebook v2, células 15-18; `pipeline v2`, `preco_por_regra_deterministica`, linha 290; PDF v2-A, Seção 2 do Documento 2). Ver ressalva sobre ausência no PDF v2-B acima.

**4. ML tradicional vs. Redes Neurais — 8,75/10**
4.1. Compara explicitamente contra ML tradicional (discussão): **5** — discussão amadurecida com pergunta de decisão explícita (notebook v2, célula 14).
4.2. Baseline simples de fato executado e comparado no notebook: **4** — Regressão Logística treinada e avaliada no mesmo split da MLP (notebook v2, células 7/9; `pipeline v2`, linhas 152-182, 275-287) — resolve estruturalmente a lacuna da v1. Não chega a 5 porque, ao reexecutar o código, os números não reproduzem os publicados (ver achado central acima) e o notebook entregue não tem outputs salvos que permitam conferir a origem dos números.

**5. Aderência ao conteúdo do curso — 10/10**
5.1. Nomeia arquitetura vista em aula: **5** — inalterado (MLP com ReLU/Dropout/Adam).
5.2. Arquitetura adequada ao tipo de dado: **5** — inalterado (dados tabulares).
5.3. *(não aplicável — problema não é de texto)*

**6. Aderência ao template de projeto — 10/10**
6.1. Cobre os 7 blocos do `templates_projetos_ia.md`: **5** — inalterado.
6.2. Profundidade do bloco 7 (MLOps): **5** — limiares numéricos definidos E implementados em código (`pipeline v2`, `verificar_data_drift`/`verificar_concept_drift`, linhas 575-630; notebook v2, célula 32, com `scipy.stats.ks_2samp`) — resolve integralmente a ressalva da v1.

**7. Correção técnica — 8,75/10**
7.1. Código executa sem erro: **3** — o notebook v2 tem `execution_count: null` e `outputs: []` em todas as 16 células de código (nenhuma evidência de execução salva no artefato entregue). Ao rodar eu mesmo o `pipeline v2` equivalente, ele executa de ponta a ponta sem erro — mas essa confirmação não vem do trabalho entregue.
7.2. Split antes de pré-processamento: **5** — inalterado; `StandardScaler` ajustado só no treino, após `train_test_split` (notebook v2, célula 8; `pipeline v2`, `preparar_dados`, linhas 152-181).
7.3. Métrica adequada à distribuição: **5** — inalterado; acurácia + AUC + Log Loss, dataset ~55,75% de conversão (verificado na minha execução).
7.4. Baseline avaliado no mesmo split/dados que o modelo principal: **5** — confirmado estruturalmente: `X_train`/`X_test`/`y_train`/`y_test` de um único `train_test_split` (célula 7) são usados tanto para a Regressão Logística (célula 9) quanto para a MLP (células 11-14) — resolve integralmente a lacuna da v1, independentemente da divergência nos números específicos (que é um problema de honestidade/reprodutibilidade, não de metodologia de split).

**8. Qualidade do código — 10/10**
8.1. Seeds fixadas: **5** — inalterado; `SEED=42` e reforço em `executar_multiplas_seeds`/`configurar_ambiente`.
8.2. Dependências declaradas: **5** — inalterado; `requirements.txt` presente. *(Dica não pontuada: os pins são todos `>=`, sem teto — dado que o achado central desta revisão pode ter componente de deriva de versão, travar versões exatas ajudaria a garantir reprodutibilidade.)*
8.3. Organização em funções/seções: **5** — inalterado e reforçado; `pipeline v2` ganhou `ConfigEconomica` (dataclass), `executar_multiplas_seeds`, `verificar_data_drift`/`verificar_concept_drift` bem isolados.

**9. Honestidade dos resultados — 7,5/10**
9.1. Múltiplas seeds/execuções ou justificativa: **3** — mecanismo de 3 seeds corretamente estruturado em código (notebook v2, células 18-19; `pipeline v2`, linhas 507-572), grande avanço frente ao 1/5 da v1. Não chega a 5 porque o notebook não tem outputs salvos e a métrica publicada (0,7175 ± 0,0012) não reproduz na minha execução (obtive 0,691 ± 0,009) — ver achado central.
9.2. Seção de limitações presente: **5** — presente e explícita em 4 lugares (notebook v2 célula 33; `pipeline v2` linhas 811-818; README.v2.md Seção 7; PDF v2-A/B Seção 6), reconhecendo o dataset 100% sintético e outras limitações (ausência de dados reais, receita simulada não comprova causalidade, ROI é projeção de cenário).

---

## 4. Pontos fortes

- **2.4 Comparação custo vs. retorno** (5/5): ROI (730,77%) e payback (1,4 meses) agora calculados em código a partir de premissas explícitas (`pipeline v2`, `calcular_roi`, linhas 633-641) — resolve de forma completa uma das lacunas mais graves da v1, com aritmética que conferi de forma independente e bate exatamente.
- **3.1 Discussão de alternativa determinística** (5/5): salto de 1/5 para 5/5 — nova seção com código funcional (`preco_por_regra_deterministica`) e uma reflexão explícita e madura sobre quando a IA se justifica (notebook v2, célula 18): *"a IA só se justifica se o ganho econômico e preditivo superar o custo adicional"*.
- **6.2 Profundidade do MLOps** (5/5): os limiares numéricos que faltavam na v1 (KS>0,15, desvio de conversão >5 p.p.) agora existem **em código funcional** (`verificar_data_drift`/`verificar_concept_drift`), não só em texto — um dos poucos trabalhos da turma com essa profundidade de implementação.
- **9.2 Seção de limitações** (5/5): presente em quatro lugares diferentes de forma consistente, reconhecendo explicitamente a natureza sintética do dataset e a distinção entre resultado experimental e projeção financeira — um contraste forte com o tom de "conclusão executiva definitiva" da v1.

## 5. Pontos de melhoria

- **9.1/4.2 Confiabilidade dos números publicados** (3/5 e 4/5): este é o ponto mais importante a resolver. Ao executar o `pipeline v2` exatamente como está no repositório, obtive Regressão Logística com Acurácia=64,38%/AUC=0,6777 e MLP com Acurácia=62,88%/AUC=0,6841 (seed=42) — não os 62,45%/0,6612 e 66,88%/0,7181 publicados no README/PDF. Nas 3 seeds, a MLP teve AUC médio de 0,691±0,009 (não 0,7175±0,0012), e a Regressão Logística **venceu em acurácia média** (64,9% vs. 63,6%). Isso é grave porque a comparação MLP vs. baseline é o item mais crítico da rubrica técnica, e a variância real observada é ~7× maior que a reportada.
- **7.1 Execução sem erro** (3/5): o notebook v2 entregue tem `execution_count: null` e nenhum output salvo em nenhuma das 16 células de código — não há evidência, no próprio artefato, de que ele já rodou com sucesso e produziu os números citados no README.
- **1.3 Conexão AUC → R$** (4/5): mesmo com a honestidade nova (distinguir premissa de resultado derivado), o número central do ROI (R$ 45.000/rota/mês) continua sendo uma suposição de negócio, não uma tradução direta da probabilidade prevista pelo modelo.
- **2.1/2.2 Granularidade do custo** (4/5 cada): CAPEX/OPEX têm categorias e R$, mas não uma memória de cálculo (ex: X horas de engenharia de dados × R$/hora) que permitisse auditar como se chegou a R$ 35.000/R$ 30.000.

## 6. Nota final

**8,0 / 10** — Esta é uma revisão substancialmente mais forte do que a v1: todas as lacunas mais graves apontadas (ausência de baseline, ausência de qualquer estimativa de custo/ROI, divergência não explicada entre RASK e receita, MLOps sem limiares numéricos, ausência de seção de limitações) foram endereçadas com código funcional, não apenas com texto — um trabalho de revisão sério e bem estruturado. A nota fica abaixo do que a soma mecânica dos critérios sugeriria (~9,2) por um motivo específico e verificável: ao executar o código exatamente como entregue, os números centrais da comparação MLP vs. Regressão Logística (o coração da correção técnica desta revisão) não reproduzem, e a variância real da validação com múltiplas seeds é muito maior do que a reportada — o notebook entregue, além disso, não tem nenhum output salvo que permita conferir isso de forma independente antes de eu mesmo rodar o pipeline.

**Nível de maturidade: piloto controlado.** O projeto agora tem plano de deploy, custo estimado (CAPEX/OPEX), retorno projetado e limiares numéricos de monitoramento (2.1-2.4, 6.2) — o suficiente para orientar uma decisão real de investimento inicial. Mas ainda não está pronto para produção: o dataset continua 100% sintético (reconhecido explicitamente pelo próprio aluno), o ROI é uma projeção de cenário e não um resultado validado, e — o ponto novo desta revisão — os números técnicos publicados precisam ser reconfirmados com uma execução registrada (notebook salvo com outputs, ambiente com versões travadas) antes de serem usados para qualquer decisão.

## 7. Task list para evoluir o trabalho

**1. Aderência ao negócio**
- [ ] **1.3 Conexão entre métrica técnica e impacto de negócio (4/5):** o cálculo de receita simulada (célula 27) é um ótimo passo — o próximo é decidir se o R$ 45.000/rota/mês do ROI deve ser recalibrado a partir da própria simulação de receita (célula 27) em vez de permanecer uma premissa isolada, para que a Seção 3 do PDF ("Estratégia de Viabilidade Econômica") e a Seção 9 do notebook ("Receita esperada e conexão com o impacto de negócio") derivem do mesmo número. — ver notebook v2 (células 26-28)

**2. Viabilidade econômica (ROI)**
- [ ] **2.1/2.2 Granularidade do custo (4/5 cada):** adicione uma memória de cálculo simples para o CAPEX/OPEX (ex: X horas de engenharia de dados × R$/hora, custo de instância cloud por mês) — hoje os R$ 35.000/R$ 30.000 aparecem como totais categorizados, mas sem essa decomposição não dá para auditar a premissa. — ver PDF v2-B (Seção 3) e `pipeline v2` (`ConfigEconomica`, linhas 55-80)

**4. ML tradicional vs. Redes Neurais**
- [ ] **4.2 Baseline de fato executado e comparado (4/5):** antes de defender os números 62,45%/0,6612 (LR) e 66,88%/0,7181 (MLP), rode novamente o `pipeline_precificacao_revisado_professor.py` do zero em um ambiente com versões travadas (`pip freeze > requirements.lock`) e salve os outputs do notebook (`jupyter nbconvert --execute --to notebook`). Ao rodar eu mesmo esse mesmo código, obtive números sistematicamente diferentes (Acurácia LR=64,38% vs. 62,45% alegado; AUC MLP=0,6841 vs. 0,7181 alegado) — isso precisa ser reconciliado antes da nota final da disciplina, porque muda a conclusão qualitativa (o quanto a MLP realmente supera a Regressão Logística). — ver `pipeline_precificacao_revisado_professor.py` (linhas 679-771) e notebook v2 (células 9, 14)

**7. Correção técnica**
- [ ] **7.1 Execução sem erro (3/5):** salve o notebook v2 já executado (com `execution_count` sequencial e outputs visíveis), em vez de entregá-lo com todas as células em branco — hoje não há como verificar, a partir do próprio arquivo, que ele já rodou com sucesso e gerou os números citados no README. — ver `notebook_precificacao_dinamica_aerea_revisado_professor.ipynb` (todas as células de código têm `execution_count: null`)

**9. Honestidade dos resultados**
- [ ] **9.1 Múltiplas seeds/execuções (3/5):** o mecanismo está certo (`executar_multiplas_seeds`, 3 seeds, média ± desvio padrão) — falta rodar de fato, salvar o output da tabela por seed (não só a média final) e conferir se o desvio padrão realmente fica baixo (0,0012) ou se, como na minha execução, fica bem maior (0,009) — isso muda diretamente a força da alegação de robustez. — ver `pipeline_precificacao_revisado_professor.py` (`executar_multiplas_seeds`, linhas 507-572) e notebook v2 (célula 19)

## 8. Tópicos para o aluno revisar

- **Reprodutibilidade de experimentos de ML** — motivado pela divergência encontrada entre os números publicados no README.v2.md/PDF e os obtidos ao executar o `pipeline v2` e o notebook v2 sem outputs salvos: antes de reportar uma métrica (AUC, acurácia, ROI), é preciso reexecutar o pipeline do zero em um ambiente limpo e salvar a evidência (notebook com outputs, ou log de execução do script), não apenas confiar em uma execução anterior não registrada.
- **Travamento de versões de dependências (`pip freeze`)** — motivado pelo `requirements.txt` usar apenas pins mínimos (`tensorflow>=2.15.0`, `scikit-learn>=1.2.0`); dado o achado desta revisão, vale registrar as versões exatas usadas para gerar os números reportados, para que a comparação MLP vs. baseline seja auditável por terceiros.
