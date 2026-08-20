# -*- coding: utf-8 -*-
"""
Projeto de MBA: Redes Neurais Aplicadas a Negócios
Tema: Precificação Dinâmica e Yield Management em e-Commerce de Companhias Aéreas
      usando Redes Neurais e Agentes de IA
"""

import os
import sys
import random
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

def configurar_ambiente(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Ambiente configurado com sucesso (seed={seed}).")

def gerar_dataset_passagens_aereas(n_amostras=4000, seed=42):
    np.random.seed(seed)
    dias_ate_decolagem = np.random.randint(1, 91, size=n_amostras)
    taxa_ocupacao_atual = np.random.uniform(0.10, 0.95, size=n_amostras)
    preco_concorrente = np.random.uniform(300, 1500, size=n_amostras)
    historico_buscas_24h = np.random.randint(50, 1001, size=n_amostras)
    dia_da_semana = np.random.randint(0, 7, size=n_amostras)
    
    # Preço ofertado baseado na urgência, concorrência e ruído de mercado
    ruido_preco = np.random.normal(loc=0.0, scale=120.0, size=n_amostras)
    fator_urgencia_base = np.where(dias_ate_decolagem < 7, 1.35, np.where(dias_ate_decolagem < 21, 1.12, 0.92))
    preco_ofertado = (preco_concorrente * fator_urgencia_base + ruido_preco)
    preco_ofertado = np.clip(preco_ofertado, 250.0, 1800.0)
    
    # Modelagem microeconômica da probabilidade de conversão
    razao_preco = preco_ofertado / preco_concorrente
    fator_proximidade = 1.0 / (np.log1p(dias_ate_decolagem) + 0.3)
    fator_demanda = (historico_buscas_24h - 50) / 950.0
    fator_dia_nobre = np.isin(dia_da_semana, [3, 4, 6]).astype(float) * 0.40
    
    logit = (
        3.0
        - 4.20 * razao_preco
        + 2.30 * fator_proximidade
        + 1.10 * fator_demanda
        + 0.40 * fator_dia_nobre
        - 0.60 * (taxa_ocupacao_atual - 0.50)
        + np.random.normal(loc=0, scale=0.40, size=n_amostras)
    )
    
    probabilidade_compra = 1.0 / (1.0 + np.exp(-logit))
    venda_realizada = (np.random.uniform(0, 1, size=n_amostras) < probabilidade_compra).astype(int)
    
    df = pd.DataFrame({
        'dias_ate_decolagem': dias_ate_decolagem,
        'taxa_ocupacao_atual': np.round(taxa_ocupacao_atual, 3),
        'preco_concorrente': np.round(preco_concorrente, 2),
        'historico_buscas_24h': historico_buscas_24h,
        'dia_da_semana': dia_da_semana,
        'preco_ofertado': np.round(preco_ofertado, 2),
        'venda_realizada': venda_realizada
    })
    return df

def preparar_dados(df, test_size=0.20, seed=42):
    feature_cols = [
        'dias_ate_decolagem',
        'taxa_ocupacao_atual',
        'preco_concorrente',
        'historico_buscas_24h',
        'dia_da_semana',
        'preco_ofertado'
    ]
    target_col = 'venda_realizada'
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    return X_train, X_test, y_train, y_test, scaler, feature_cols, X_test_raw

def construir_modelo_rede_neural(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu', name='camada_oculta_1'),
        layers.Dropout(0.20, name='dropout_1'),
        layers.Dense(32, activation='relu', name='camada_oculta_2'),
        layers.Dropout(0.20, name='dropout_2'),
        layers.Dense(1, activation='sigmoid', name='saida_probabilidade')
    ], name='RedeNeural_Precificacao_Aerea')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    return model

def treinar_modelo(model, X_train, y_train, epochs=25, batch_size=32, val_split=0.2):
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=6,
        restore_best_weights=True,
        verbose=0
    )
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=[early_stop],
        verbose=1
    )
    return history

def avaliar_modelo(model, X_test, y_test):
    y_pred_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_pred_prob >= 0.50).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    print("=== RESULTADOS DA AVALIACAO DE TESTE ===")
    print(f"Acuracia Global: {acc * 100:.2f}%")
    print(f"Score ROC-AUC:   {roc_auc:.4f}
")
    print("Relatorio Detalhado de Classificacao:")
    print(report)
    
    return {
        'acuracia': acc,
        'auc': roc_auc,
        'matriz_confusao': cm,
        'relatorio': report,
        'y_pred_prob': y_pred_prob,
        'y_pred': y_pred,
        'fpr': fpr,
        'tpr': tpr
    }

def prever_probabilidade_venda(model, scaler, dias, ocupacao, preco_concorrente, buscas, preco_proposto, dia_semana=2):
    amostra = pd.DataFrame([{
        'dias_ate_decolagem': dias,
        'taxa_ocupacao_atual': ocupacao,
        'preco_concorrente': preco_concorrente,
        'historico_buscas_24h': buscas,
        'dia_da_semana': dia_semana,
        'preco_ofertado': preco_proposto
    }])
    amostra_scaled = scaler.transform(amostra)
    probabilidade = float(model.predict(amostra_scaled, verbose=0)[0][0])
    return probabilidade

class AgentePrecificacaoAerea:
    def __init__(self, model, scaler, custo_base_minimo=400.0, preco_teto=2000.0, step_preco=10.0):
        self.model = model
        self.scaler = scaler
        self.custo_base_minimo = custo_base_minimo
        self.preco_teto = preco_teto
        self.step_preco = step_preco
        
    def otimizar_tarifa(self, dias, ocupacao, preco_concorrente, buscas, dia_semana=2):
        precos_candidatos = np.arange(self.custo_base_minimo, self.preco_teto + self.step_preco, self.step_preco)
        
        df_grid = pd.DataFrame({
            'dias_ate_decolagem': np.full_like(precos_candidatos, dias),
            'taxa_ocupacao_atual': np.full_like(precos_candidatos, ocupacao),
            'preco_concorrente': np.full_like(precos_candidatos, preco_concorrente),
            'historico_buscas_24h': np.full_like(precos_candidatos, buscas),
            'dia_da_semana': np.full_like(precos_candidatos, dia_semana),
            'preco_ofertado': precos_candidatos
        })
        
        grid_scaled = self.scaler.transform(df_grid)
        probabilidades = self.model.predict(grid_scaled, verbose=0).ravel()
        receitas_esperadas = precos_candidatos * probabilidades
        
        regras_acionadas = []
        preco_piso_efetivo = self.custo_base_minimo
        preco_teto_efetivo = self.preco_teto
        
        if ocupacao >= 0.80 and dias <= 7:
            preco_piso_efetivo = max(preco_piso_efetivo, preco_concorrente * 0.98, 850.0)
            regras_acionadas.append(
                f"[YIELD-ESCASSEZ] Alta ocupacao ({ocupacao*100:.0f}%) a {dias} dias da partida: "
                f"Piso tarifario elevado para R$ {preco_piso_efetivo:.2f}."
            )
        elif ocupacao < 0.40 and dias <= 15:
            preco_teto_efetivo = min(preco_teto_efetivo, preco_concorrente * 1.05, 950.0)
            regras_acionadas.append(
                f"[SPOILAGE-PROTECTION] Baixa ocupacao ({ocupacao*100:.0f}%) a {dias} dias da partida: "
                f"Tarifa teto ajustada para R$ {preco_teto_efetivo:.2f} para acelerar conversao."
            )
            
        if preco_concorrente < 450.0 and ocupacao < 0.70:
            preco_teto_efetivo = min(preco_teto_efetivo, preco_concorrente * 1.15)
            regras_acionadas.append(
                f"[DEFESA-COMPETITIVA] Concorrente agressivo (R$ {preco_concorrente:.2f}). "
                f"Tarifa teto limitada a R$ {preco_teto_efetivo:.2f} para defender share."
            )
            
        if buscas >= 800:
            preco_piso_efetivo = max(preco_piso_efetivo, self.custo_base_minimo * 1.25)
            regras_acionadas.append(
                f"[DEMAND-SURGE] Volume expressivo de buscas ({buscas} buscas/24h): "
                f"Margem minima ajustada (+25%) para R$ {preco_piso_efetivo:.2f}."
            )
            
        if not regras_acionadas:
            regras_acionadas.append("[OTIMIZACAO-PADRAO] Maximizacao pura de receita sem restricoes de excecao.")

        mascara_viavel = (precos_candidatos >= preco_piso_efetivo) & (precos_candidatos <= preco_teto_efetivo)
        
        if np.any(mascara_viavel):
            indices_viaveis = np.where(mascara_viavel)[0]
            melhor_idx_local = np.argmax(receitas_esperadas[indices_viaveis])
            melhor_idx = indices_viaveis[melhor_idx_local]
        else:
            melhor_idx = np.argmax(receitas_esperadas)
            
        preco_otimo = precos_candidatos[melhor_idx]
        prob_otima = probabilidades[melhor_idx]
        receita_otima = receitas_esperadas[melhor_idx]
        
        df_curva = pd.DataFrame({
            'preco': precos_candidatos,
            'probabilidade_compra': probabilidades,
            'receita_esperada': receitas_esperadas,
            'viavel_regras': mascara_viavel
        })
        
        return {
            'preco_otimizado': preco_otimo,
            'probabilidade_venda': prob_otima,
            'receita_esperada': receita_otima,
            'preco_concorrente': preco_concorrente,
            'preco_piso_efetivo': preco_piso_efetivo,
            'preco_teto_efetivo': preco_teto_efetivo,
            'regras_acionadas': regras_acionadas,
            'curva_decisao': df_curva
        }
        
    def emitir_parecer_executivo(self, decisao, nome_cenario="Cenário"):
        print("=" * 75)
        print(f" DECISÃO DO AGENTE DE IA: {nome_cenario.upper()}")
        print("=" * 75)
        print(" Parâmetros e Resultados da Otimização:")
        print(f"   • Tarifa Ótima Recomendada:   R$ {decisao['preco_otimizado']:,.2f}")
        print(f"   • Tarifa do Concorrente:      R$ {decisao['preco_concorrente']:,.2f}")
        print(f"   • Margem sobre Custo Base:    +R$ {decisao['preco_otimizado'] - self.custo_base_minimo:,.2f}")
        print(" Previsão da Rede Neural (Deep Learning): ")
        print(f"   • Probabilidade Estimada de Venda: {decisao['probabilidade_venda'] * 100:.2f}%")
        print(f"   • Receita Esperada por Assento:    R$ {decisao['receita_esperada']:,.2f}")
        print(" Regras Estratégicas e Yield Management Aplicadas:")
        for r in decisao['regras_acionadas']:
            print(f"   {r}")
        print("=" * 75 + "\n")

def executar_cenarios(agente):
    cenarios = {
        'Cenário A: Voo distante (60 dias) com baixa ocupação (20%)': {
            'dias': 60,
            'ocupacao': 0.20,
            'preco_concorrente': 600.0,
            'buscas': 200,
            'dia_semana': 2
        },
        'Cenário B: Voo próximo (3 dias) com alta ocupação (85%) e alta demanda corporativa': {
            'dias': 3,
            'ocupacao': 0.85,
            'preco_concorrente': 1200.0,
            'buscas': 900,
            'dia_semana': 4
        },
        'Cenário C: Guerra de preços com concorrente direto a 5 dias do voo': {
            'dias': 5,
            'ocupacao': 0.55,
            'preco_concorrente': 380.0,
            'buscas': 600,
            'dia_semana': 1
        }
    }
    
    resultados = {}
    for nome, params in cenarios.items():
        decisao = agente.otimizar_tarifa(**params)
        agente.emitir_parecer_executivo(decisao, nome_cenario=nome)
        resultados[nome] = decisao
    return resultados

if __name__ == '__main__':
    configurar_ambiente(42)
    df = gerar_dataset_passagens_aereas(4000)
    X_train, X_test, y_train, y_test, scaler, feature_cols, X_test_raw = preparar_dados(df)
    modelo = construir_modelo_rede_neural(len(feature_cols))
    treinar_modelo(modelo, X_train, y_train, epochs=25, batch_size=32)
    avaliar_modelo(modelo, X_test, y_test)
    agente = AgentePrecificacaoAerea(modelo, scaler, custo_base_minimo=400.0, preco_teto=2000.0)
    executar_cenarios(agente)
