# -*- coding: utf-8 -*-
"""
Projeto de MBA: Redes Neurais Aplicadas a Negócios
Tema: Precificação Dinâmica e Yield Management em e-Commerce de Companhias Aéreas
usando Redes Neurais e Agentes de IA

Versão revisada conforme a avaliação do professor.

Principais melhorias:
- baseline com Regressão Logística no mesmo split da MLP;
- alternativa de regra determinística;
- múltiplas seeds e média +/- desvio padrão;
- cálculo de receita esperada em R$;
- ROI/payback com premissas explicitamente parametrizadas;
- monitoramento de Data Drift e Concept Drift;
- limitações e distinção entre resultado experimental e projeção econômica.
"""

import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import callbacks, layers


SEED = 42

FEATURE_COLS = [
    "dias_ate_decolagem",
    "taxa_ocupacao_atual",
    "preco_concorrente",
    "historico_buscas_24h",
    "dia_da_semana",
    "preco_ofertado",
]
TARGET_COL = "venda_realizada"


@dataclass
class ConfigEconomica:
    """Premissas financeiras do cenário de viabilidade.

    Estes valores NÃO são resultados do dataset sintético.
    Devem ser substituídos por dados reais antes de uma decisão de investimento.
    """
    capex: float = 35_000.0
    opex_anual: float = 30_000.0
    beneficio_mensal: float = 45_000.0

    @property
    def custo_ano_1(self) -> float:
        return self.capex + self.opex_anual

    @property
    def beneficio_anual(self) -> float:
        return self.beneficio_mensal * 12

    @property
    def roi_ano_1(self) -> float:
        return (self.beneficio_anual - self.custo_ano_1) / self.custo_ano_1

    @property
    def payback_meses(self) -> float:
        return self.custo_ano_1 / self.beneficio_mensal


def configurar_ambiente(seed: int = SEED) -> None:
    """Fixa seeds para tornar as execuções mais reprodutíveis."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def gerar_dataset_passagens_aereas(
    n_amostras: int = 4000,
    seed: int = SEED,
) -> pd.DataFrame:
    """Gera o dataset sintético utilizado na PoC."""
    rng = np.random.default_rng(seed)

    dias_ate_decolagem = rng.integers(1, 91, size=n_amostras)
    taxa_ocupacao_atual = rng.uniform(0.10, 0.95, size=n_amostras)
    preco_concorrente = rng.uniform(300, 1500, size=n_amostras)
    historico_buscas_24h = rng.integers(50, 1001, size=n_amostras)
    dia_da_semana = rng.integers(0, 7, size=n_amostras)

    ruido_preco = rng.normal(0, 120, size=n_amostras)
    fator_urgencia = np.where(
        dias_ate_decolagem < 7,
        1.35,
        np.where(dias_ate_decolagem < 21, 1.12, 0.92),
    )

    preco_ofertado = np.clip(
        preco_concorrente * fator_urgencia + ruido_preco,
        250,
        1800,
    )

    razao_preco = preco_ofertado / preco_concorrente
    fator_proximidade = 1 / (np.log1p(dias_ate_decolagem) + 0.3)
    fator_demanda = (historico_buscas_24h - 50) / 950
    fator_dia_nobre = (
        np.isin(dia_da_semana, [3, 4, 6]).astype(float) * 0.40
    )

    logit = (
        3.0
        - 4.20 * razao_preco
        + 2.30 * fator_proximidade
        + 1.10 * fator_demanda
        + fator_dia_nobre
        - 0.60 * (taxa_ocupacao_atual - 0.50)
        + rng.normal(0, 0.40, size=n_amostras)
    )

    probabilidade_compra = 1 / (1 + np.exp(-logit))
    venda_realizada = (
        rng.uniform(0, 1, size=n_amostras) < probabilidade_compra
    ).astype(int)

    return pd.DataFrame(
        {
            "dias_ate_decolagem": dias_ate_decolagem,
            "taxa_ocupacao_atual": np.round(taxa_ocupacao_atual, 3),
            "preco_concorrente": np.round(preco_concorrente, 2),
            "historico_buscas_24h": historico_buscas_24h,
            "dia_da_semana": dia_da_semana,
            "preco_ofertado": np.round(preco_ofertado, 2),
            "venda_realizada": venda_realizada,
        }
    )


def preparar_dados(
    df: pd.DataFrame,
    test_size: float = 0.20,
    seed: int = SEED,
):
    """Split antes do scaler para evitar data leakage."""
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        X_train_raw,
        X_test_raw,
    )


def construir_modelo_rede_neural(input_dim: int, seed: int = SEED):
    """MLP Feedforward usada como modelo principal."""
    tf.keras.utils.set_random_seed(seed)

    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.20),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.20),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="RedeNeural_Precificacao_Aerea",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def treinar_modelo(
    model,
    X_train,
    y_train,
    epochs: int = 100,
    batch_size: int = 64,
    val_split: float = 0.20,
):
    early_stop = callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=12,
        restore_best_weights=True,
        verbose=0,
    )

    return model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=[early_stop],
        verbose=0,
    )


def calcular_metricas(
    y_true,
    probabilidades,
    threshold: float = 0.50,
) -> Dict[str, float]:
    pred = (probabilidades >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, probabilidades)),
        "log_loss": float(log_loss(y_true, probabilidades)),
    }


def avaliar_modelo(model, X_test, y_test) -> Dict:
    """Avalia a MLP no conjunto de teste."""
    y_pred_prob = model.predict(X_test, verbose=0).ravel()
    metricas = calcular_metricas(y_test, y_pred_prob)

    y_pred = (y_pred_prob >= 0.50).astype(int)

    resultado = {
        **metricas,
        "matriz_confusao": confusion_matrix(y_test, y_pred),
        "relatorio": classification_report(
            y_test,
            y_pred,
            digits=4,
        ),
        "y_pred_prob": y_pred_prob,
        "y_pred": y_pred,
    }

    return resultado


def treinar_baseline_logistico(X_train, y_train, seed: int = SEED):
    """Baseline simples e barato, avaliado no mesmo split da MLP."""
    model = LogisticRegression(
        max_iter=2000,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    return model


def avaliar_baseline(model, X_test, y_test) -> Dict:
    probabilidades = model.predict_proba(X_test)[:, 1]
    return calcular_metricas(y_test, probabilidades)


def preco_por_regra_deterministica(row: pd.Series) -> float:
    """Alternativa sem IA para comparação econômica."""
    preco_base = float(row["preco_concorrente"])

    if (
        row["taxa_ocupacao_atual"] >= 0.80
        and row["dias_ate_decolagem"] <= 7
    ):
        preco = max(850.0, preco_base * 0.98)
    elif (
        row["taxa_ocupacao_atual"] < 0.40
        and row["dias_ate_decolagem"] <= 15
    ):
        preco = min(900.0, preco_base * 0.90)
    elif row["historico_buscas_24h"] >= 800:
        preco = preco_base * 1.10
    else:
        preco = preco_base

    return float(np.clip(preco, 400, 2000))


def aplicar_regra_deterministica(df: pd.DataFrame) -> pd.DataFrame:
    resultado = df.copy()
    resultado["preco_regra_deterministica"] = resultado.apply(
        preco_por_regra_deterministica,
        axis=1,
    )
    return resultado


def estimar_receita_por_preco(
    modelo,
    scaler,
    contexto: Dict,
    precos: np.ndarray,
) -> pd.DataFrame:
    """Calcula E[Receita] = preço x P(compra | preço, contexto)."""
    cenarios = []

    for preco in precos:
        cenario = contexto.copy()
        cenario["preco_ofertado"] = float(preco)
        cenarios.append([cenario[col] for col in FEATURE_COLS])

    X_cenarios = scaler.transform(
        pd.DataFrame(cenarios, columns=FEATURE_COLS)
    )
    probabilidades = modelo.predict(
        X_cenarios,
        verbose=0,
    ).ravel()

    resultado = pd.DataFrame(
        {
            "preco": precos,
            "probabilidade_compra": probabilidades,
        }
    )
    resultado["receita_esperada"] = (
        resultado["preco"] * resultado["probabilidade_compra"]
    )

    return resultado


class AgentePrecificacaoAerea:
    """Camada prescritiva: MLP + otimização + guardrails de Yield."""

    def __init__(
        self,
        model,
        scaler,
        custo_base_minimo: float = 400.0,
        preco_teto: float = 2000.0,
        step_preco: float = 10.0,
    ):
        self.model = model
        self.scaler = scaler
        self.custo_base_minimo = custo_base_minimo
        self.preco_teto = preco_teto
        self.step_preco = step_preco

    def otimizar_tarifa(
        self,
        dias: int,
        ocupacao: float,
        preco_concorrente: float,
        buscas: int,
        dia_semana: int = 2,
    ) -> Dict:
        preco_piso = self.custo_base_minimo
        preco_teto = self.preco_teto
        regras = []

        # Yield: escassez
        if ocupacao >= 0.80 and dias <= 7:
            preco_piso = max(
                preco_piso,
                preco_concorrente * 0.98,
                850.0,
            )
            regras.append("YIELD-ESCASSEZ")

        # Yield: proteção contra spoilage
        if ocupacao < 0.40 and dias <= 15:
            preco_teto = min(
                preco_teto,
                preco_concorrente * 1.05,
                950.0,
            )
            regras.append("SPOILAGE-PROTECTION")

        # Yield: defesa competitiva
        if preco_concorrente < 450.0 and ocupacao < 0.70:
            preco_teto = min(
                preco_teto,
                preco_concorrente * 1.15,
            )
            regras.append("DEFESA-COMPETITIVA")

        # Yield: surge de demanda
        if buscas >= 800:
            preco_piso = max(
                preco_piso,
                self.custo_base_minimo * 1.25,
            )
            regras.append("DEMAND-SURGE")

        if preco_piso > preco_teto:
            # Guardrail de segurança: nunca retorna intervalo inválido.
            preco_piso = preco_teto

        precos = np.arange(
            preco_piso,
            preco_teto + self.step_preco,
            self.step_preco,
        )

        contexto = {
            "dias_ate_decolagem": dias,
            "taxa_ocupacao_atual": ocupacao,
            "preco_concorrente": preco_concorrente,
            "historico_buscas_24h": buscas,
            "dia_da_semana": dia_semana,
            "preco_ofertado": preco_piso,
        }

        curva = estimar_receita_por_preco(
            self.model,
            self.scaler,
            contexto,
            precos,
        )

        melhor = curva.loc[
            curva["receita_esperada"].idxmax()
        ]

        if not regras:
            regras.append("OTIMIZACAO-PADRAO")

        curva["viavel_regras"] = True

        return {
            "preco_otimizado": float(melhor["preco"]),
            "probabilidade_venda": float(
                melhor["probabilidade_compra"]
            ),
            "receita_esperada": float(
                melhor["receita_esperada"]
            ),
            "preco_concorrente": float(preco_concorrente),
            "preco_piso_efetivo": float(preco_piso),
            "preco_teto_efetivo": float(preco_teto),
            "regras_acionadas": regras,
            "curva_decisao": curva,
        }

    def emitir_parecer_executivo(
        self,
        decisao: Dict,
        nome_cenario: str = "Cenário",
    ) -> None:
        print("=" * 75)
        print(f"DECISÃO DO AGENTE DE IA: {nome_cenario.upper()}")
        print("=" * 75)
        print(
            f"Tarifa ótima: R$ "
            f"{decisao['preco_otimizado']:,.2f}"
        )
        print(
            f"Tarifa concorrente: R$ "
            f"{decisao['preco_concorrente']:,.2f}"
        )
        print(
            f"Probabilidade estimada de venda: "
            f"{decisao['probabilidade_venda']:.2%}"
        )
        print(
            f"Receita esperada por assento: R$ "
            f"{decisao['receita_esperada']:,.2f}"
        )
        print(
            f"Piso efetivo: R$ "
            f"{decisao['preco_piso_efetivo']:,.2f}"
        )
        print(
            f"Teto efetivo: R$ "
            f"{decisao['preco_teto_efetivo']:,.2f}"
        )
        print("Regras acionadas:")
        for regra in decisao["regras_acionadas"]:
            print(f" - {regra}")
        print("=" * 75)


def executar_multiplas_seeds(
    df: pd.DataFrame,
    seeds: Tuple[int, ...] = (10, 42, 100),
) -> pd.DataFrame:
    """Repete baseline e MLP para reportar média +/- desvio padrão."""
    resultados = []

    for seed in seeds:
        configurar_ambiente(seed)

        (
            X_train,
            X_test,
            y_train,
            y_test,
            scaler,
            _,
            _,
        ) = preparar_dados(df, seed=seed)

        baseline = treinar_baseline_logistico(
            X_train,
            y_train,
            seed,
        )
        metricas_baseline = avaliar_baseline(
            baseline,
            X_test,
            y_test,
        )

        resultados.append(
            {
                "seed": seed,
                "modelo": "Regressão Logística",
                **metricas_baseline,
            }
        )

        model = construir_modelo_rede_neural(
            X_train.shape[1],
            seed=seed,
        )
        treinar_modelo(
            model,
            X_train,
            y_train,
        )

        metricas_mlp = avaliar_modelo(
            model,
            X_test,
            y_test,
        )

        resultados.append(
            {
                "seed": seed,
                "modelo": "MLP",
                "accuracy": metricas_mlp["accuracy"],
                "roc_auc": metricas_mlp["roc_auc"],
                "log_loss": metricas_mlp["log_loss"],
            }
        )

    return pd.DataFrame(resultados)


def verificar_data_drift(
    referencia: np.ndarray,
    atual: np.ndarray,
    ks_limite: float = 0.15,
    p_limite: float = 0.05,
) -> Dict[str, float]:
    """Data Drift: KS > 0.15 e p < 0.05 gera alerta.

    Implementação sem depender de scipy diretamente.
    """
    referencia = np.sort(np.asarray(referencia))
    atual = np.sort(np.asarray(atual))

    todos = np.sort(np.concatenate([referencia, atual]))
    cdf_ref = np.searchsorted(referencia, todos, side="right") / len(referencia)
    cdf_atual = np.searchsorted(atual, todos, side="right") / len(atual)

    ks = float(np.max(np.abs(cdf_ref - cdf_atual)))

    # Aproximação conservadora do p-valor assintótico do KS.
    n_eff = len(referencia) * len(atual) / (len(referencia) + len(atual))
    lambda_ks = (np.sqrt(n_eff) + 0.12 + 0.11 / np.sqrt(n_eff)) * ks

    p_valor = 2 * sum(
        (-1) ** (k - 1) * np.exp(-2 * (k * lambda_ks) ** 2)
        for k in range(1, 101)
    )
    p_valor = float(np.clip(p_valor, 0.0, 1.0))

    alerta = ks > ks_limite and p_valor < p_limite

    return {
        "ks": ks,
        "p_valor": p_valor,
        "alerta": bool(alerta),
    }


def verificar_concept_drift(
    probabilidade_prevista: np.ndarray,
    conversao_real: np.ndarray,
    limite_pp: float = 0.05,
) -> Dict[str, float]:
    """Concept Drift: diferença absoluta > 5 p.p. gera alerta."""
    desvio = float(
        abs(
            np.mean(probabilidade_prevista)
            - np.mean(conversao_real)
        )
    )

    return {
        "desvio_absoluto": desvio,
        "limite_pp": limite_pp,
        "alerta": bool(desvio > limite_pp),
    }


def calcular_roi(config: ConfigEconomica) -> Dict[str, float]:
    return {
        "capex": config.capex,
        "opex_anual": config.opex_anual,
        "custo_ano_1": config.custo_ano_1,
        "beneficio_anual": config.beneficio_anual,
        "roi_ano_1": config.roi_ano_1,
        "payback_meses": config.payback_meses,
    }


def executar_cenarios(agente: AgentePrecificacaoAerea) -> Dict:
    cenarios = {
        "A - Voo distante / baixa ocupação": {
            "dias": 60,
            "ocupacao": 0.20,
            "preco_concorrente": 600.0,
            "buscas": 200,
            "dia_semana": 2,
        },
        "B - Voo próximo / alta ocupação / alta demanda": {
            "dias": 3,
            "ocupacao": 0.85,
            "preco_concorrente": 1200.0,
            "buscas": 900,
            "dia_semana": 4,
        },
        "C - Guerra de preços": {
            "dias": 5,
            "ocupacao": 0.55,
            "preco_concorrente": 380.0,
            "buscas": 600,
            "dia_semana": 1,
        },
    }

    resultados = {}

    for nome, params in cenarios.items():
        decisao = agente.otimizar_tarifa(**params)
        agente.emitir_parecer_executivo(decisao, nome)
        resultados[nome] = decisao

    return resultados


def executar_pipeline(
    n_amostras: int = 4000,
    seed: int = SEED,
    executar_robustez: bool = True,
) -> Dict:
    """Executa o pipeline principal e retorna os artefatos."""
    configurar_ambiente(seed)

    print("\n=== 1. GERAÇÃO DO DATASET ===")
    df = gerar_dataset_passagens_aereas(
        n_amostras=n_amostras,
        seed=seed,
    )
    print(f"Observações: {len(df)}")
    print(
        f"Taxa de conversão: "
        f"{df[TARGET_COL].mean():.2%}"
    )

    print("\n=== 2. SPLIT E PRÉ-PROCESSAMENTO ===")
    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        X_train_raw,
        X_test_raw,
    ) = preparar_dados(df, seed=seed)

    print("\n=== 3. BASELINE: REGRESSÃO LOGÍSTICA ===")
    baseline = treinar_baseline_logistico(
        X_train,
        y_train,
        seed,
    )
    metricas_baseline = avaliar_baseline(
        baseline,
        X_test,
        y_test,
    )
    print(
        "AUC={:.4f} | Accuracy={:.4f} | Log Loss={:.4f}".format(
            metricas_baseline["roc_auc"],
            metricas_baseline["accuracy"],
            metricas_baseline["log_loss"],
        )
    )

    print("\n=== 4. REDE NEURAL MLP ===")
    model = construir_modelo_rede_neural(
        len(FEATURE_COLS),
        seed=seed,
    )
    history = treinar_modelo(
        model,
        X_train,
        y_train,
    )
    metricas_mlp = avaliar_modelo(
        model,
        X_test,
        y_test,
    )
    print(
        "AUC={:.4f} | Accuracy={:.4f} | Log Loss={:.4f}".format(
            metricas_mlp["roc_auc"],
            metricas_mlp["accuracy"],
            metricas_mlp["log_loss"],
        )
    )

    print("\n=== 5. COMPARAÇÃO NO MESMO SPLIT ===")
    comparacao = pd.DataFrame(
        [
            {
                "modelo": "Regressão Logística",
                **metricas_baseline,
            },
            {
                "modelo": "MLP",
                "accuracy": metricas_mlp["accuracy"],
                "roc_auc": metricas_mlp["roc_auc"],
                "log_loss": metricas_mlp["log_loss"],
            },
        ]
    )
    print(comparacao.to_string(index=False))

    print("\n=== 6. ALTERNATIVA DETERMINÍSTICA ===")
    df_regras = aplicar_regra_deterministica(df)
    print(
        df_regras[
            ["preco_concorrente", "preco_regra_deterministica"]
        ].head()
    )

    robustez = None
    if executar_robustez:
        print("\n=== 7. ROBUSTEZ: 3 SEEDS ===")
        robustez = executar_multiplas_seeds(
            df,
            seeds=(10, 42, 100),
        )
        print(robustez.to_string(index=False))
        print("\nMédia +/- desvio padrão:")
        resumo = (
            robustez.groupby("modelo")[
                ["accuracy", "roc_auc", "log_loss"]
            ]
            .agg(["mean", "std"])
        )
        print(resumo)

    print("\n=== 8. AGENTE PRESCRITIVO ===")
    agente = AgentePrecificacaoAerea(
        model,
        scaler,
        custo_base_minimo=400.0,
        preco_teto=2000.0,
    )
    resultados_cenarios = executar_cenarios(agente)

    print("\n=== 9. ROI / PAYBACK — CENÁRIO DE PREMISSAS ===")
    config_economica = ConfigEconomica()
    roi = calcular_roi(config_economica)
    print(pd.Series(roi).to_string())
    print(
        "\nObservação: ROI e payback são projeções de cenário, "
        "não resultados derivados do dataset sintético."
    )

    print("\n=== 10. LIMITAÇÕES ===")
    print(
        "- Dataset 100% sintético.\n"
        "- Sem validação externa em dados reais.\n"
        "- Receita simulada não comprova causalidade.\n"
        "- ROI usa premissas financeiras parametrizadas.\n"
        "- Produção exige Shadow Mode e A/B Test."
    )

    return {
        "dataset": df,
        "baseline": baseline,
        "modelo_mlp": model,
        "history": history,
        "scaler": scaler,
        "X_test_raw": X_test_raw,
        "metricas_baseline": metricas_baseline,
        "metricas_mlp": metricas_mlp,
        "comparacao": comparacao,
        "df_regras": df_regras,
        "robustez": robustez,
        "agente": agente,
        "cenarios": resultados_cenarios,
        "roi": roi,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline de precificação dinâmica com MLP."
    )
    parser.add_argument(
        "--amostras",
        type=int,
        default=4000,
        help="Número de observações sintéticas.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Seed principal.",
    )
    parser.add_argument(
        "--sem-robustez",
        action="store_true",
        help="Não executar as 3 seeds para acelerar a execução.",
    )

    args = parser.parse_args()

    executar_pipeline(
        n_amostras=args.amostras,
        seed=args.seed,
        executar_robustez=not args.sem_robustez,
    )


if __name__ == "__main__":
    main()
