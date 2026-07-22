# ============
# IMPORTS
# ============
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
import yaml
import logging
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')


# ============
# CONSTANTES GLOBAIS
# ============
RADIO_VARIABLES = ['226Ra', '232Th', '40K', 'Raeq', 'Theq', 'Keq', 'IG', 'IA', 'IB']



#conditions = ['MinMax', 'Log1ScalerMinMax', 'Log1MinMax', 'TransformerQuantileMinMax']

# ============
# CONFIGURAÇÃO YAML
# ============
CONFIG_YAML = """
# config.yaml - Arquivo de configuração para análise GAN
project:
  name: "GAN_Synthetic_Data_Analysis"
  version: "1.0.0"

paths:
  folder_project: "/home/nbc/Documentos/test_py/GAN_normalization/"
  folder_datas: "Datas/"
  folder_metrics: "metrics_results/"
  folder_synth: "synthetic_data/"
  folder_data_denorm: "data_denorm/"
  folder_sts_results: "sts_results/"

data:
  numerical_columns:
    - '226Ra'
    - '232Th'
    - '40K'
    - 'Raeq'
    - 'Theq'
    - 'Keq'
    - 'IG'
    - 'IA'
    - 'IB'
  conditions: [1, 2, 3, 4]
  runs: [0, 1, 2, 3, 4, 5, 6]
  epochs: [400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000, 4400, 4500]
  target_epochs: [400, 2000, 4500]
  n_samples: 109
  n_features: 9

analysis:
  metrics:
    univariate: true
    multivariate: true
    structure: true
    coverage: true
  plotting:
    save_figures: true
    figure_format: "png"
    dpi: 300
    show_plots: true

logging:
  level: "INFO"
  format: "%(asctime)s - %(levelname)s - %(message)s"
"""

# ============
# FUNÇÃO DE CARREGAMENTO DE CONFIGURAÇÃO (CORRIGIDA)
# ============

def load_config(config_file='config.yaml'):
    """Carrega configuração de arquivo YAML ou usa default"""
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                content = f.read().strip()
                if not content:
                    print(f"Arquivo {config_file} está vazio. Usando configuração padrão.")
                    with open(config_file, 'w') as f:
                        f.write(CONFIG_YAML)
                    return yaml.safe_load(CONFIG_YAML)
                return yaml.safe_load(content) or yaml.safe_load(CONFIG_YAML)
        else:
            # Salva configuração padrão
            with open(config_file, 'w') as f:
                f.write(CONFIG_YAML)
            print(f"Arquivo de configuração criado: {config_file}")
            return yaml.safe_load(CONFIG_YAML)
    except yaml.YAMLError as e:
        print(f"Erro no formato YAML: {e}. Usando configuração padrão.")
        with open(config_file, 'w') as f:
            f.write(CONFIG_YAML)
        return yaml.safe_load(CONFIG_YAML)
    except Exception as e:
        print(f"Erro ao carregar config: {e}")
        return yaml.safe_load(CONFIG_YAML)
    

def load_all_data(folder_root, conditions, runs, epochs):
    """Carrega dados estruturados de métricas e amostras"""
    all_data = {}

    for cond in conditions:
        cond_key = f"cond_{cond}"
        cond_path = os.path.join(folder_root, f"cond_{cond}")
        
        if not os.path.exists(cond_path):
            logging.warning(f"Pasta não encontrada: {cond_path}")
            continue

        all_data[cond_key] = {"metrics": {}, "samples": {}}

        for run in runs:
            metrics_path = os.path.join(cond_path, "metrics", f"metrics_final_run{run}.csv")
            samples_path = os.path.join(cond_path, "samples_denorm")

            # --- Leitura das métricas ---
            if os.path.exists(metrics_path):
                df_metrics = pd.read_csv(metrics_path)
                expected_cols = ["epoch", "time_epoch_s", "d_loss_mean", "g_loss_mean",
                                 "gp_mean", "mmd", "emd_mean", "n_eval"]

                missing_cols = [c for c in expected_cols if c not in df_metrics.columns]
                if missing_cols:
                    logging.warning(f"Colunas ausentes em {metrics_path}: {missing_cols}")
                all_data[cond_key]["metrics"][f"run_{run}"] = df_metrics
                logging.info(f"  ✓ Métricas carregadas: cond_{cond}, run_{run}")
            else:
                logging.warning(f"Arquivo de métricas não encontrado: {metrics_path}")

            # --- Leitura das amostras ---
            samples_dict = {}
            for epoch in epochs:
                sample_file = os.path.join(samples_path, f"samples_denorm_{epoch}_{run}.csv")
                if os.path.exists(sample_file):
                    samples_dict[epoch] = pd.read_csv(sample_file)
                    logging.debug(f"  ✓ Amostra carregada: cond_{cond}, run_{run}, época {epoch}")
                else:
                    logging.debug(f"Arquivo de amostra ausente: {sample_file}")
            all_data[cond_key]["samples"][f"run_{run}"] = samples_dict

    return all_data    

def debug_data_structure(data_all, df_real, target_epochs=[400, 2000, 4500]):
    """Debug da estrutura de dados"""
    print("\n" + "="*60)
    print("DEBUG: ESTRUTURA DOS DADOS")
    print("="*60)
    
    print(f"\nDados reais shape: {df_real.shape}")
    print(f"Colunas reais: {df_real.columns.tolist()}")
    
    for cond_key in data_all:
        print(f"\n{cond_key}:")
        samples_found = 0
        total_runs = len(data_all[cond_key]["samples"])
        
        for run_key in data_all[cond_key]["samples"]:
            samples_dict = data_all[cond_key]["samples"][run_key]
            if samples_dict:
                # Verificar uma amostra
                sample_epoch = list(samples_dict.keys())[0]
                sample_df = samples_dict[sample_epoch]
                print(f"  {run_key}: {len(samples_dict)} épocas")
                print(f"Exemplo época {sample_epoch}: shape={sample_df.shape}")
                print(f" Colunas: {sample_df.columns.tolist()[:5]}...")
                samples_found += 1
            else:
                print(f"{run_key}: SEM AMOSTRAS")
        
        print(f" Total: {samples_found}/{total_runs} runs com amostras")
    
    print("="*60)




def compare_final_metrics(data_all, conditions, runs):
    results = []

    for cond in conditions:
        cond_key = f"cond_{cond}"
        for run in runs:
            df = data_all[cond_key]["metrics"].get(f"run_{run}")
            if df is not None:
                last_row = df.iloc[-1]
                results.append({
                    "cond": cond,
                    "run": run,
                    "d_loss_mean": last_row["d_loss_mean"],
                    "g_loss_mean": last_row["g_loss_mean"],
                    "gp_mean": last_row["gp_mean"],
                    "mmd": last_row["mmd"],
                    "emd_mean": last_row["emd_mean"]
                })

    df_final = pd.DataFrame(results)
    return df_final


def plot_simple_kde(df, columns, figsize=(18, 12)):
    """
    Versão simplificada com apenas os gráficos KDE.
    """
    n_cols = 3
    n_rows = (len(columns) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(columns):
        if i < len(axes):
            data = df[col].dropna()

            sns.kdeplot(data=data, ax=axes[i], color='royalblue',
                       linewidth=2, fill=True, alpha=0.5)

            # Linhas de média e mediana
            axes[i].axvline(data.mean(), color='red', linestyle='--', alpha=0.8, label='Média')
            axes[i].axvline(data.median(), color='green', linestyle='--', alpha=0.8, label='Mediana')

            axes[i].set_title(f'{col}\nAssimetria: {data.skew():.2f} | Curtose: {data.kurtosis():.2f}')
            axes[i].set_xlabel('Valor')
            axes[i].set_ylabel('Densidade')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

    # Remove eixos extras
    for j in range(len(columns), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle('Distribuições das Variáveis com KDE', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_temporal_analysis(data_all, metrics=None, output_dir='article_plots'):
    """
    Análise temporal das métricas internas do WGAN.
    Salva um dicionário com todas as métricas e gera figura com 3 subplots.
    
    Args:
        data_all: Dicionário com dados carregados
        metrics: Lista de métricas a serem plotadas (padrão: g_loss_mean, d_loss_mean, emd_mean)
        output_dir: Diretório para salvar as figuras
    
    Returns:
        fig: Figura matplotlib
        stats_df: DataFrame com estatísticas resumidas
        all_metrics_dict: Dicionário com todas as métricas por condição e época
    """

    print("Entrando na função PLOT_TEMPORAL_ANALYSIS....")

    if metrics is None:
        metrics = ['g_loss_mean', 'd_loss_mean', 'emd_mean']
    
    print("\n" + "=" * 80)
    print("INTERNAL DYNAMICS AND TRAINING STABILITY")
    print("=" * 80)

    conditions = list(data_all.keys())
    n_metrics = len(metrics)

    # Criar figura com 3 subplots horizontais
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    if n_metrics == 1:
        axes = [axes]

    # Cores para cada condição
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Dicionário de nomes das métricas
    metric_names = {
        'd_loss_mean': 'Discriminator Loss',
        'g_loss_mean': 'Generator Loss',
        'gp_mean': 'Gradient Penalty',
        'mmd': 'Maximum Mean Discrepancy (MMD)',
        'emd_mean': 'Earth Mover\'s Distance (EMD)'
    }

    # Nomes das condições
    cond_names = {
        'cond_1': 'C1',
        'cond_2': 'C2',
        'cond_3': 'C3',
        'cond_4': 'C4'
    }

    # ============================================================
    # DICIONÁRIO PARA ARMAZENAR TODAS AS MÉTRICAS
    # ============================================================
    all_metrics_dict = {}
    
    # Lista de todas as métricas disponíveis
    all_metrics_list = ['d_loss_mean', 'g_loss_mean', 'gp_mean', 'mmd', 'emd_mean']
    
    for cond in conditions:
        all_metrics_dict[cond] = {}
        for metric in all_metrics_list:
            all_metrics_dict[cond][metric] = {}
            
            for run_key, run_data in data_all[cond]['metrics'].items():
                if metric in run_data.columns:
                    valid_data = run_data[metric].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(valid_data) > 0:
                        run_name = run_key.replace('run_', 'Run ')
                        all_metrics_dict[cond][metric][run_name] = valid_data.values.tolist()
    
    # Salvar dicionário em arquivo
    import json
    dict_path = os.path.join(output_dir, 'temporal_metrics_dict.json')
    
    # Converter arrays numpy para listas para serialização JSON
    dict_serializable = {}
    for cond in all_metrics_dict:
        dict_serializable[cond] = {}
        for metric in all_metrics_dict[cond]:
            dict_serializable[cond][metric] = {}
            for run_name, values in all_metrics_dict[cond][metric].items():
                dict_serializable[cond][metric][run_name] = [float(v) for v in values]
    
    with open(dict_path, 'w') as f:
        json.dump(dict_serializable, f, indent=2, default=str)
    print(f" Dicionário de métricas salvo em: {dict_path}")
    
    # ============================================================
    # SALVAR TAMBÉM EM CSV PARA FÁCIL ACESSO
    # ============================================================
    metrics_csv_data = []
    for cond in conditions:
        for run_key, run_data in data_all[cond]['metrics'].items():
            run_name = run_key.replace('run_', 'Run ')
            for metric in all_metrics_list:
                if metric in run_data.columns:
                    valid_data = run_data[metric].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(valid_data) > 0:
                        for epoch, value in zip(run_data['epoch'].values[:len(valid_data)], valid_data.values):
                            metrics_csv_data.append({
                                'Condition': cond_names.get(cond, cond),
                                'Run': run_name,
                                'Metric': metric,
                                'Epoch': int(epoch),
                                'Value': float(value)
                            })
    
    df_metrics = pd.DataFrame(metrics_csv_data)
    csv_path = os.path.join(output_dir, 'temporal_metrics_all.csv')
    df_metrics.to_csv(csv_path, index=False)
    print(f"Métricas completas salvas em: {csv_path}")

    # ============================================================
    # PLOTAR FIGURA
    # ============================================================
    stats_data = []

    for j, metric in enumerate(metrics):
        if j < len(axes):
            ax = axes[j]
            has_valid_data = False

            for idx, cond in enumerate(conditions):
                all_runs_data = []
                epochs_ref = None

                for run_key, run_data in data_all[cond]['metrics'].items():
                    if metric in run_data.columns:
                        valid_data = run_data[metric].replace([np.inf, -np.inf], np.nan).dropna()

                        if len(valid_data) > 0:
                            if epochs_ref is None:
                                epochs_ref = run_data['epoch'].values
                            all_runs_data.append(valid_data.values)

                if all_runs_data and len(all_runs_data) > 0:
                    min_length = min(len(arr) for arr in all_runs_data)
                    truncated_data = [arr[:min_length] for arr in all_runs_data]

                    if epochs_ref is not None:
                        epochs_plot = epochs_ref[:min_length]

                    mean_metric = np.mean(truncated_data, axis=0)
                    std_metric = np.std(truncated_data, axis=0)

                    if metric in ['emd_mean']:
                        epsilon = 1e-10
                        mean_metric_plot = mean_metric + epsilon
                        std_metric_plot = std_metric + epsilon

                        ax.plot(epochs_plot, mean_metric_plot,
                               label=cond_names.get(cond, cond),
                               linewidth=2.5,
                               color=colors[idx % len(colors)])

                        ax.fill_between(epochs_plot,
                                      mean_metric_plot - std_metric_plot,
                                      mean_metric_plot + std_metric_plot,
                                      alpha=0.3,
                                      color=colors[idx % len(colors)])

                        ax.set_yscale('log')
                        ax.set_ylabel(f'{metric_names.get(metric, metric)} (log scale)', fontsize=12, fontweight='bold')

                    else:
                        ax.plot(epochs_plot, mean_metric,
                               label=cond_names.get(cond, cond),
                               linewidth=2.5,
                               color=colors[idx % len(colors)])

                        ax.fill_between(epochs_plot,
                                      mean_metric - std_metric,
                                      mean_metric + std_metric,
                                      alpha=0.3,
                                      color=colors[idx % len(colors)])

                        ax.set_ylabel(metric_names.get(metric, metric), fontsize=12, fontweight='bold')

                    has_valid_data = True

            if has_valid_data:
                title = metric_names.get(metric, metric)
                #ax.set_title(f'({chr(97+j)}) {title}', fontsize=13, fontweight='bold', pad=15)
                ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
                
                #ax.legend(fontsize=9, framealpha=0.9, loc='best')
                ax.grid(True, alpha=0.2, linestyle='--')
                ax.tick_params(axis='both', labelsize=10)

                if metric in ['emd_mean']:
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0e}'))
                    y_min, y_max = ax.get_ylim()
                    if y_min > 0:
                        ax.set_ylim(y_min * 0.8, y_max * 1.2)
            else:
                ax.text(0.5, 0.5, f'No valid data\nfor {metric}',
                       transform=ax.transAxes, ha='center', va='center', fontsize=12)
                ax.set_title(f'{metric_names.get(metric, metric)} - Insufficient Data',
                           fontsize=13, fontweight='bold', pad=15)
                ax.set_xlabel('Epoch', fontsize=12)
                ax.set_ylabel(metric_names.get(metric, metric), fontsize=12)


    # Criar legenda global na parte inferior
    handles, labels = [], []
    for idx, cond in enumerate(conditions):
        # Criar um proxy artist para cada condição
        proxy = plt.Line2D([0], [0], color=colors[idx % len(colors)], linewidth=2.5, label=cond_names.get(cond, cond))
        handles.append(proxy)
        labels.append(cond_names.get(cond, cond))

    # Adicionar legenda na parte inferior
    fig.legend(handles, labels, 
            loc='lower center', 
            bbox_to_anchor=(0.5, -0.02), #0.5, 0.0
            ncol=len(conditions),
            fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True)
    #plt.suptitle('Figure E: Internal Dynamics and Training Stability',
    #            fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Salvar figura
    fig_path = os.path.join(output_dir, 'figureE_temporal_dynamics.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f" Figura E salva: {fig_path}")
    
    # ============================================================
    # ANÁLISE ESTATÍSTICA
    # ============================================================
    print("\n" + "-" * 80)
    print("TEMPORAL STATISTICS SUMMARY")
    print("-" * 80)

    for metric in metrics:
        print(f"\n{metric.upper()}:")
        print("-" * 40)

        for cond in conditions:
            final_values = []
            initial_values = []

            for run_data in data_all[cond]['metrics'].values():
                if metric in run_data.columns:
                    valid_data = run_data[metric].replace([np.inf, -np.inf], np.nan).dropna()

                    if len(valid_data) > 0:
                        final_values.append(valid_data.iloc[-1])
                        initial_values.append(valid_data.iloc[0])

            if final_values and initial_values:
                mean_final = np.mean(final_values)
                std_final = np.std(final_values)
                mean_initial = np.mean(initial_values)
                
                if mean_initial != 0:
                    reduction_pct = ((mean_initial - mean_final) / abs(mean_initial)) * 100
                else:
                    reduction_pct = 0

                if reduction_pct > 10:
                    trend = "✓ Significant improvement"
                elif reduction_pct > 0:
                    trend = "✓ Moderate improvement"
                elif reduction_pct < -10:
                    trend = "✗ Significant degradation"
                elif reduction_pct < 0:
                    trend = "✗ Moderate degradation"
                else:
                    trend = "→ Stable"

                stats_data.append({
                    'Condition': cond_names.get(cond, cond),
                    'Metric': metric,
                    'Initial': mean_initial,
                    'Final': mean_final,
                    'Std_Final': std_final,
                    'Reduction_%': reduction_pct,
                    'Trend': trend
                })

                if metric in ['emd_mean'] and mean_final < 0.001:
                    print(f"  {cond_names.get(cond, cond)}: {mean_final:.2e} ± {std_final:.2e} "
                          f"(Reduction: {reduction_pct:+.1f}%) {trend}")
                else:
                    print(f"  {cond_names.get(cond, cond)}: {mean_final:.6f} ± {std_final:.6f} "
                          f"(Reduction: {reduction_pct:+.1f}%) {trend}")
            else:
                print(f"  {cond_names.get(cond, cond)}: No valid data")
    
    # Criar DataFrame com estatísticas
    stats_df = pd.DataFrame(stats_data)
    stats_path = os.path.join(output_dir, 'temporal_statistics.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"\n  Estatísticas salvas em: {stats_path}")
    
    plt.close(fig)
    
    return fig, stats_df, all_metrics_dict



def print_summary_statistics(data_all, target_epochs):
    """
    Calcula e exibe estatísticas resumidas para cada condição
    """
    print("\n ESTATÍSTICAS RESUMIDAS POR CONDIÇÃO:")
    print("-" * 60)

    metrics = ['mmd', 'emd_mean', 'g_loss_mean', 'd_loss_mean']

    for epoch in target_epochs:
        print(f"\n ÉPOCA {epoch}:")
        print("-" * 40)

        for metric in metrics:
            print(f"\n{metric}:")
            for cond in data_all.keys():
                values = []
                for run_data in data_all[cond]['metrics'].values():
                    if metric in run_data.columns:
                        epoch_idx = (run_data['epoch'] - epoch).abs().idxmin()
                        values.append(run_data.loc[epoch_idx, metric])

                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    iqr = np.percentile(values, 75) - np.percentile(values, 25)
                    print(f"  {cond}: {mean_val:.6f} ± {std_val:.6f} (IQR: {iqr:.6f})")


#
def compute_summary_statistics(results_dict, metric_name):
    """Calcula média, desvio padrão e IQR para uma métrica"""
    values = []
    for cond_key in results_dict:
        for run_key in results_dict[cond_key]:
            val = results_dict[cond_key][run_key].get(metric_name)
            if val is not None and not np.isnan(val) and not np.isinf(val):
                values.append(val)
    
    if values:
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'q1': np.percentile(values, 25),
            'q3': np.percentile(values, 75),
            'iqr': np.percentile(values, 75) - np.percentile(values, 25),
            'count': len(values)
        }
    return None

def generate_summary_report(data_all, target_epochs):
    """Gera um relatório resumido com estatísticas por época"""
    print("\n" + "="*80)
    print("RESUMO ESTATÍSTICO POR ÉPOCA")
    print("="*80)
    
    metrics = ['MMD', 'Frechet_Distance', 'Energy_Distance']
    
    for epoch in target_epochs:
        print(f"\n{'='*60}")
        print(f"ÉPOCA {epoch}")
        print(f"{'='*60}")
        
        # Coletar todos os resultados para esta época
        results_by_cond = {}
        
        for cond in [1, 2, 3, 4]:
            cond_key = f"cond_{cond}"
            results_by_cond[cond_key] = {}
            
            for run in range(7):
                run_key = f"run_{run}"
                # Simular a obtenção dos resultados (usando os dados do relatório)
                # Na prática, você deve usar os dados reais
                pass
        
        # Para cada métrica, mostrar estatísticas
        for metric in metrics:
            all_values = []
            for cond in [1, 2, 3, 4]:
                cond_key = f"cond_{cond}"
                # Aqui você deve extrair os valores do seu dicionário de resultados
                # Por enquanto, vamos usar os dados do relatório manualmente
                pass
            
            if all_values:
                print(f"\n{metric}:")
                print(f"  Média: {np.mean(all_values):.6f}")
                print(f"  Desvio: {np.std(all_values):.6f}")
                print(f"  IQR: {np.percentile(all_values, 75) - np.percentile(all_values, 25):.6f}")
                print(f"  Min: {np.min(all_values):.6f}")
                print(f"  Max: {np.max(all_values):.6f}")

###################   Class Metrics:


class UnivariateAnalysis:
    """Classe para análise univariada de similaridade entre distribuições"""

    def __init__(self, real_data, synthetic_data_dict, conditions, runs, epochs):
        self.real_data = real_data
        self.synthetic_data_dict = synthetic_data_dict
        self.conditions = conditions
        self.runs = runs
        self.epochs = epochs
        self.variables = real_data.columns.tolist()

    def get_synthetic_samples(self, cond_key, run_key, epoch):
        """Obtém amostras sintéticas - CORRIGIDO"""
        try:
            # DEBUG: Verificar estrutura
            if cond_key not in self.synthetic_data_dict:
                # print(f" {cond_key} não encontrado em synthetic_data_dict")
                return None

            if "samples" not in self.synthetic_data_dict[cond_key]:
                # print(f" 'samples' não encontrado em {cond_key}")
                return None

            if run_key not in self.synthetic_data_dict[cond_key]["samples"]:
                # print(f" {run_key} não encontrado em {cond_key}['samples']")
                return None

            samples_dict = self.synthetic_data_dict[cond_key]["samples"][run_key]

            # Procurar a época (pode ser string ou int)
            epoch_found = None
            for available_epoch in samples_dict.keys():
                if str(available_epoch) == str(epoch):
                    epoch_found = available_epoch
                    break

            if epoch_found is None:
                # print(f" Época {epoch} não encontrada em {cond_key}_{run_key}. Disponíveis: {list(samples_dict.keys())}")
                return None

            samples = samples_dict[epoch_found]
            if samples is not None and not samples.empty:
                # print(f" Dados encontrados: {cond_key}_{run_key}_{epoch} - Shape: {samples.shape}")
                return samples
            else:
                # print(f"Dados vazios: {cond_key}_{run_key}_{epoch}")
                return None

        except Exception as e:
            print(f" Erro ao obter amostras para {cond_key}_{run_key}_{epoch}: {e}")
            return None

    def compute_univariate_metrics(self, real_series, synthetic_series):
        """Calcula métricas univariadas de dissimilaridade"""
        try:
            # Remover NaN values e garantir que há dados suficientes
            real_clean = real_series[~np.isnan(real_series)]
            synth_clean = synthetic_series[~np.isnan(synthetic_series)]

            if len(real_clean) < 10 or len(synth_clean) < 10:
                return {
                    'EMD': np.nan,
                    'KL_Divergence': np.nan,
                    'JS_Divergence': np.nan,
                    'KS_Statistic': np.nan
                }

            # Earth Mover's Distance
            emd = stats.wasserstein_distance(real_clean, synth_clean)

            # Preparar histogramas com bins consistentes
            all_data = np.concatenate([real_clean, synth_clean])
            hist_real, bin_edges = np.histogram(real_clean, bins=50,
                                              range=(np.percentile(all_data, 1),
                                                     np.percentile(all_data, 99)),
                                              density=True)
            hist_synth, _ = np.histogram(synth_clean, bins=bin_edges, density=True)

            # Suavização para evitar zeros
            eps = 1e-10
            hist_real = np.clip(hist_real + eps, eps, None)
            hist_synth = np.clip(hist_synth + eps, eps, None)

            # Normalizar para distribuições de probabilidade
            hist_real = hist_real / np.sum(hist_real)
            hist_synth = hist_synth / np.sum(hist_synth)

            # KL Divergence
            kl_div = stats.entropy(hist_real, hist_synth)

            # Jensen-Shannon Divergence
            js_div = jensenshannon(hist_real, hist_synth)

            # Kolmogorov-Smirnov Statistic
            ks_stat, _ = stats.ks_2samp(real_clean, synth_clean)

            return {
                'EMD': emd,
                'KL_Divergence': kl_div,
                'JS_Divergence': js_div,
                'KS_Statistic': ks_stat
            }
        except Exception as e:
            print(f"Erro em compute_univariate_metrics: {e}")
            return {
                'EMD': np.nan,
                'KL_Divergence': np.nan,
                'JS_Divergence': np.nan,
                'KS_Statistic': np.nan
            }

    def analyze_univariate_similarity(self, epoch):
        """Executa análise univariada completa"""
        results = {}

        # print(f" Analisando similaridade univariada (época: {epoch})...")

        for condition in self.conditions:
            cond_key = f"cond_{condition}"
            condition_results = {}
            valid_runs = 0

            for run in self.runs:
                run_key = f"run_{run}"

                # Obter dados sintéticos
                synthetic_data = self.get_synthetic_samples(cond_key, run_key, epoch)
                if synthetic_data is None:
                    continue

                run_results = {}
                valid_vars = 0

                for var in self.variables:
                    if var in synthetic_data.columns and var in self.real_data.columns:
                        try:
                            metrics = self.compute_univariate_metrics(
                                self.real_data[var].values,
                                synthetic_data[var].values
                            )
                            run_results[var] = metrics
                            valid_vars += 1
                        except Exception as e:
                            # print(f"Erro processando {var} em {cond_key}_{run_key}: {e}")
                            run_results[var] = {
                                'EMD': np.nan, 'KL_Divergence': np.nan,
                                'JS_Divergence': np.nan, 'KS_Statistic': np.nan
                            }

                if run_results and valid_vars > 0:
                    condition_results[run_key] = run_results
                    valid_runs += 1

            if condition_results:
                results[cond_key] = condition_results
                # print(f"{cond_key}: {valid_runs} runs válidas")
            # else:
                # print(f" {cond_key}: nenhum run válido")

        return results

    def plot_univariate_analysis(self, univariate_results, epoch):
        """Plota resultados da análise univariada"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(20, 15))
            metrics = ['EMD', 'KL_Divergence', 'JS_Divergence', 'KS_Statistic']
            metric_names = ['Earth Mover Distance', 'KL Divergence',
                           'JS Divergence', 'KS Statistic']

            # Agregar dados para plotting
            plot_data = []

            for condition in self.conditions:
                cond_key = f"cond_{condition}"
                if cond_key in univariate_results:
                    for run in self.runs:
                        run_key = f"run_{run}"
                        if run_key in univariate_results[cond_key]:
                            for var in self.variables:
                                if var in univariate_results[cond_key][run_key]:
                                    for metric in metrics:
                                        value = univariate_results[cond_key][run_key][var][metric]
                                        if not np.isnan(value):
                                            plot_data.append({
                                                'Condition': f'Cond {condition}',
                                                'Variable': var,
                                                'Metric': metric,
                                                'Value': value,
                                                'Run': run
                                            })

            if not plot_data:
                print(f" Nenhum dado disponível para plotagem univariada (época {epoch})")
                for ax in axes.flatten():
                    ax.text(0.5, 0.5, 'Sem dados disponíveis',
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                plt.suptitle(f'ANÁLISE UNIVARIADA - ÉPOCA {epoch}\nSEM DADOS DISPONÍVEIS',
                            fontsize=16, fontweight='bold', y=0.98)
                return fig

            plot_df = pd.DataFrame(plot_data)

            for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
                ax = axes.flatten()[idx]

                # Preparar dados para o heatmap
                metric_data = plot_df[plot_df['Metric'] == metric]
                if metric_data.empty:
                    ax.text(0.5, 0.5, 'Sem dados', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{metric_name}', fontsize=14, fontweight='bold')
                    continue

                # Calcular média por condição e variável
                heatmap_data = metric_data.pivot_table(
                    values='Value', index='Variable', columns='Condition', aggfunc='mean'
                )

                # Ordenar para melhor visualização
                if not heatmap_data.empty:
                    heatmap_data = heatmap_data.reindex(sorted(heatmap_data.columns), axis=1)
                    heatmap_data = heatmap_data.sort_index()

                    # Plotar heatmap
                    vmin = heatmap_data.min().min()
                    vmax = heatmap_data.max().max()

                    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
                               ax=ax, cbar_kws={'label': metric_name},
                               vmin=vmin, vmax=vmax)
                    ax.set_title(f'{metric_name}\n(Menor = Melhor)', fontsize=14, fontweight='bold')
                else:
                    ax.text(0.5, 0.5, 'Sem dados', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{metric_name}', fontsize=14, fontweight='bold')

                ax.set_xlabel('Condição de Normalização')
                ax.set_ylabel('Variável')

            plt.suptitle(f'ANÁLISE UNIVARIADA - SIMILARIDADE ENTRE DISTRIBUIÇÕES\n'
                        f'Época: {epoch} | Comparação Marginal Variável a Variável',
                        fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Erro em plot_univariate_analysis: {e}")
            import traceback
            traceback.print_exc()
            return plt.figure()




class MultivariateAnalysis:
    """Classe para análise multivariada de similaridade entre distribuições"""

    def __init__(self, real_data, synthetic_data_dict, conditions, runs, epochs):
        self.real_data = real_data
        self.synthetic_data_dict = synthetic_data_dict
        self.conditions = conditions
        self.runs = runs
        self.epochs = epochs
        self.variables = real_data.columns.tolist()

    def get_synthetic_samples(self, cond_key, run_key, epoch):
        """Obtém amostras sintéticas - MESMA CORREÇÃO"""
        try:
            if cond_key not in self.synthetic_data_dict:
                return None

            if "samples" not in self.synthetic_data_dict[cond_key]:
                return None

            if run_key not in self.synthetic_data_dict[cond_key]["samples"]:
                return None

            samples_dict = self.synthetic_data_dict[cond_key]["samples"][run_key]

            # Procurar a época (pode ser string ou int)
            epoch_found = None
            for available_epoch in samples_dict.keys():
                if str(available_epoch) == str(epoch):
                    epoch_found = available_epoch
                    break

            if epoch_found is None:
                return None

            samples = samples_dict[epoch_found]
            if samples is not None and not samples.empty:
                return samples
            else:
                return None

        except Exception as e:
            print(f" Erro ao obter amostras para {cond_key}_{run_key}_{epoch}: {e}")
            return None

    def compute_mmd(self, X, Y, sigma=None):
        """Calcula Maximum Mean Discrepancy com kernel RBF - otimizado"""
        try:
            # Remover NaN values e garantir tamanho mínimo
            X_mask = ~np.isnan(X).any(axis=1)
            Y_mask = ~np.isnan(Y).any(axis=1)
            X_clean = X[X_mask]
            Y_clean = Y[Y_mask]

            if len(X_clean) < 20 or len(Y_clean) < 20:
                return np.nan

            # Amostrar para eficiência computacional
            n_samples = min(500, len(X_clean), len(Y_clean))
            X_sampled = X_clean[np.random.choice(len(X_clean), n_samples, replace=False)]
            Y_sampled = Y_clean[np.random.choice(len(Y_clean), n_samples, replace=False)]

            if sigma is None:
                # Bandwidth heuristic usando distâncias pareadas
                XX = np.sum(X_sampled**2, axis=1)
                YY = np.sum(Y_sampled**2, axis=1)
                XY = np.dot(X_sampled, Y_sampled.T)
                distances = XX[:, None] + YY[None, :] - 2 * XY
                sigma = np.sqrt(np.median(distances[distances > 0]))

            if sigma <= 0:
                sigma = 1.0

            gamma = 1.0 / (2 * sigma**2)

            # Kernel matrices
            K_XX = np.exp(-gamma * np.sum((X_sampled[:, None] - X_sampled[None, :])**2, axis=2))
            K_YY = np.exp(-gamma * np.sum((Y_sampled[:, None] - Y_sampled[None, :])**2, axis=2))
            K_XY = np.exp(-gamma * np.sum((X_sampled[:, None] - Y_sampled[None, :])**2, axis=2))

            mmd = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
            return np.sqrt(max(mmd, 0))
        except Exception as e:
            print(f"Erro em compute_mmd: {e}")
            return np.nan

    def compute_frechet_distance(self, real_data, synthetic_data):
        """Calcula Fréchet Distance adaptada para dados tabulares - CORRIGIDO"""
        try:
            # Limpeza de dados
            real_mask = ~np.isnan(real_data).any(axis=1)
            synth_mask = ~np.isnan(synthetic_data).any(axis=1)
            real_clean = real_data[real_mask]
            synth_clean = synthetic_data[synth_mask]

            if len(real_clean) < 10 or len(synth_clean) < 10:
                return np.nan

            # Estatísticas
            mu_real = np.mean(real_clean, axis=0)
            mu_synth = np.mean(synth_clean, axis=0)

            # Matrizes de covariância com regularização
            sigma_real = np.cov(real_clean, rowvar=False)
            sigma_synth = np.cov(synth_clean, rowvar=False)

            # Verificar dimensões
            if sigma_real.shape[0] != sigma_synth.shape[0]:
                return np.nan

            # Regularização para garantir matrizes positivas definidas
            eps = 1e-6
            sigma_real += eps * np.eye(sigma_real.shape[0])
            sigma_synth += eps * np.eye(sigma_synth.shape[0])

            diff_mu = mu_real - mu_synth

            try:
                # Calcular a raiz quadrada da matriz usando decomposição espectral
                # Para garantir estabilidade numérica
                eigenvals, eigenvecs = np.linalg.eigh(sigma_real @ sigma_synth)
                # Garantir que os autovalores sejam não-negativos
                eigenvals = np.maximum(eigenvals, 0)
                sqrt_matrix = eigenvecs @ np.diag(np.sqrt(eigenvals)) @ eigenvecs.T
                cov_mean = sqrt_matrix
            except:
                # Fallback: usar a média das covariâncias
                cov_mean = (sigma_real + sigma_synth) / 2

            # Calcular a distância
            fd = np.sum(diff_mu**2) + np.trace(sigma_real + sigma_synth - 2 * cov_mean)
            
            # Garantir que não seja negativo devido a erros numéricos
            return max(fd, 0)
        except Exception as e:
            print(f"Erro em compute_frechet_distance: {e}")
            return np.nan


    def compute_energy_distance(self, real_data, synthetic_data):
        """Calcula Energy Distance multivariada otimizada"""
        try:
            # Limpeza de dados
            real_mask = ~np.isnan(real_data).any(axis=1)
            synth_mask = ~np.isnan(synthetic_data).any(axis=1)
            real_clean = real_data[real_mask]
            synth_clean = synthetic_data[synth_mask]

            n_real = len(real_clean)
            n_synth = len(synth_clean)

            if n_real < 10 or n_synth < 10:
                return np.nan

            # Amostragem balanceada
            n_samples = min(300, n_real, n_synth)
            real_sampled = real_clean[np.random.choice(n_real, n_samples, replace=False)]
            synth_sampled = synth_clean[np.random.choice(n_synth, n_samples, replace=False)]

            # Distâncias intra-conjunto (otimizado)
            real_diff = real_sampled[:, None] - real_sampled[None, :]
            synth_diff = synth_sampled[:, None] - synth_sampled[None, :]
            real_synth_diff = real_sampled[:, None] - synth_sampled[None, :]

            dist_real_real = np.mean(np.linalg.norm(real_diff, axis=2))
            dist_synth_synth = np.mean(np.linalg.norm(synth_diff, axis=2))
            dist_real_synth = np.mean(np.linalg.norm(real_synth_diff, axis=2))

            energy_dist = 2 * dist_real_synth - dist_real_real - dist_synth_synth
            return max(energy_dist, 0)
        except Exception as e:
            print(f"Erro em compute_energy_distance: {e}")
            return np.nan

    #
    def analyze_multivariate_similarity(self, epoch):
        """Executa análise multivariada completa - VERSÃO FUNCIONAL"""
        results = {}
        
        variables = self.variables  # Colunas dos dados reais
        
        for condition in self.conditions:
            cond_key = f"cond_{condition}"
            condition_results = {}
            
            for run in self.runs:
                run_key = f"run_{run}"
                
                # Obter dados sintéticos
                synthetic_data = self.get_synthetic_samples(cond_key, run_key, epoch)
                if synthetic_data is None:
                    continue
                
                # Verificar colunas comuns
                synth_cols = synthetic_data.columns.tolist()
                common_vars = [var for var in variables if var in synth_cols]
                
                if len(common_vars) < 2:
                    continue
                
                # Preparar dados
                real_subset = np.array(self.real_data[common_vars].values, dtype=np.float64)
                synth_subset = np.array(synthetic_data[common_vars].values, dtype=np.float64)
                
                # Limpar dados
                real_subset = real_subset[~np.isnan(real_subset).any(axis=1)]
                synth_subset = synth_subset[~np.isnan(synth_subset).any(axis=1)]
                real_subset = real_subset[~np.isinf(real_subset).any(axis=1)]
                synth_subset = synth_subset[~np.isinf(synth_subset).any(axis=1)]
                
                if len(real_subset) < 20 or len(synth_subset) < 20:
                    continue
                
                # Calcular métricas
                mmd = self.compute_mmd(real_subset, synth_subset)
                fd = self.compute_frechet_distance(real_subset, synth_subset)
                ed = self.compute_energy_distance(real_subset, synth_subset)
                
                # Verificar validade
                mmd_valid = mmd is not None and not np.isnan(mmd) and not np.isinf(mmd)
                fd_valid = fd is not None and not np.isnan(fd) and not np.isinf(fd)
                ed_valid = ed is not None and not np.isnan(ed) and not np.isinf(ed)
                
                if mmd_valid or fd_valid or ed_valid:
                    condition_results[run_key] = {
                        'MMD': float(mmd) if mmd_valid else None,
                        'Frechet_Distance': float(fd) if fd_valid else None,
                        'Energy_Distance': float(ed) if ed_valid else None
                    }
            
            if condition_results:
                results[cond_key] = condition_results
        
        return results



    def plot_multivariate_analysis(self, multivariate_results, epoch):
        """Plota resultados da análise multivariada - CORRIGIDO"""
        try:
            # Preparar dados
            plot_data = []
            metrics = ['MMD', 'Frechet_Distance', 'Energy_Distance']
            metric_names = ['Maximum Mean Discrepancy', 'Fréchet Distance', 'Energy Distance']

            for condition in self.conditions:
                cond_key = f"cond_{condition}"
                if cond_key in multivariate_results:
                    for run in self.runs:
                        run_key = f"run_{run}"
                        if run_key in multivariate_results[cond_key]:
                            for metric in metrics:
                                value = multivariate_results[cond_key][run_key].get(metric)
                                if value is not None and not np.isnan(value):
                                    plot_data.append({
                                        'Condition': f'Cond {condition}',
                                        'Metric': metric,
                                        'Value': value,
                                        'Run': run
                                    })

            if not plot_data:
                print(" Nenhum dado disponível para plotagem multivariada")
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                for ax in axes:
                    ax.text(0.5, 0.5, 'Sem dados disponíveis',
                        ha='center', va='center', transform=ax.transAxes, fontsize=12)
                plt.suptitle(f'ANÁLISE MULTIVARIADA - ÉPOCA {epoch}\nSEM DADOS DISPONÍVEIS',
                            fontsize=16, fontweight='bold', y=1.02)
                return fig
            


            plot_df = pd.DataFrame(plot_data)

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
                ax = axes[idx]
                metric_data = plot_df[plot_df['Metric'] == metric]

                if not metric_data.empty:
                    # Ordenar condições
                    condition_order = sorted(metric_data['Condition'].unique())
                    metric_data['Condition'] = pd.Categorical(metric_data['Condition'],
                                                            categories=condition_order)

                    sns.boxplot(data=metric_data, x='Condition', y='Value',
                               ax=ax, palette='viridis', order=condition_order)
                    sns.stripplot(data=metric_data, x='Condition', y='Value',
                                 ax=ax, color='black', alpha=0.6, order=condition_order)

                    ax.set_title(f'{metric_name}\n(Menor = Melhor)', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Valor da Métrica')

                    # CORREÇÃO: Verificação segura para escala log
                    value_min = metric_data['Value'].min()
                    value_max = metric_data['Value'].max()

                    # Evitar divisão por zero e verificar se os valores são positivos
                    if value_min > 0 and value_max > 0:
                        ratio = value_max / value_min
                        if ratio > 100:  # Apenas usar log se a variação for muito grande
                            ax.set_yscale('log')
                            ax.set_ylabel('Valor da Métrica (escala log)')

                else:
                    ax.text(0.5, 0.5, 'Sem dados', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold')

                ax.set_xlabel('Condição de Normalização')
                ax.tick_params(axis='x', rotation=45)

            plt.suptitle(f'ANÁLISE MULTIVARIADA - SIMILARIDADE ENTRE DISTRIBUIÇÕES\n'
                        f'Época: {epoch} | Comparação no Espaço Conjunto das Variáveis',
                        fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Erro em plot_multivariate_analysis: {e}")
            import traceback
            traceback.print_exc()
            return plt.figure()

class CorrelationStructureAnalysis:
    """Classe para análise de correlação e preservação de estrutura interna"""

    def __init__(self, real_data, synthetic_data_dict, conditions, runs, epochs):
        self.real_data = real_data
        self.synthetic_data_dict = synthetic_data_dict
        self.conditions = conditions
        self.runs = runs
        self.epochs = epochs
        self.variables = real_data.columns.tolist()

    def get_synthetic_samples(self, cond_key, run_key, epoch):
        """Obtém amostras sintéticas - mesma correção das outras classes"""
        try:
            if cond_key not in self.synthetic_data_dict:
                return None

            if "samples" not in self.synthetic_data_dict[cond_key]:
                return None

            if run_key not in self.synthetic_data_dict[cond_key]["samples"]:
                return None

            samples_dict = self.synthetic_data_dict[cond_key]["samples"][run_key]

            # Procurar a época (pode ser string ou int)
            epoch_found = None
            for available_epoch in samples_dict.keys():
                if str(available_epoch) == str(epoch):
                    epoch_found = available_epoch
                    break

            if epoch_found is None:
                return None

            samples = samples_dict[epoch_found]
            if samples is not None and not samples.empty:
                return samples
            else:
                return None

        except Exception as e:
            print(f" Erro ao obter amostras para {cond_key}_{run_key}_{epoch}: {e}")
            return None

    #
    def compute_correlation_preservation(self, real_data, synthetic_data):
        """Analisa preservação de correlações e dependências - CORRIGIDO"""
        try:
            # Converter para numpy arrays e forçar tipo float
            real_data = np.array(real_data, dtype=float)
            synthetic_data = np.array(synthetic_data, dtype=float)
            
            # Verificar se há dados válidos
            if real_data.shape[1] < 2 or synthetic_data.shape[1] < 2:
                return {
                    'Frobenius_Difference': np.nan,
                    'MACE': np.nan,
                    'Spearman_Difference': np.nan
                }
            
            # Correlação de Pearson
            corr_real = np.corrcoef(real_data.T)
            corr_synth = np.corrcoef(synthetic_data.T)
            
            # Frobenius Norm Difference
            frobenius_diff = np.linalg.norm(corr_real - corr_synth, 'fro')
            
            # Mean Absolute Correlation Error
            mace = np.mean(np.abs(corr_real - corr_synth))
            
            # Correlação de Spearman - com tratamento de erro
            try:
                spear_real, _ = stats.spearmanr(real_data)
                spear_synth, _ = stats.spearmanr(synthetic_data)
                
                if isinstance(spear_real, float):
                    spear_real = np.array([[1.0, spear_real], [spear_real, 1.0]])
                    spear_synth = np.array([[1.0, spear_synth], [spear_synth, 1.0]])
                
                spear_diff = np.mean(np.abs(spear_real - spear_synth))
            except:
                spear_diff = np.nan
            
            return {
                'Frobenius_Difference': frobenius_diff,
                'MACE': mace,
                'Spearman_Difference': spear_diff
            }
        except Exception as e:
            print(f"Erro em compute_correlation_preservation: {e}")
            return {
                'Frobenius_Difference': np.nan,
                'MACE': np.nan,
                'Spearman_Difference': np.nan
            }

    def compute_mutual_information_preservation(self, real_data, synthetic_data):
        """Calcula preservação de informação mútua entre pares de variáveis"""
        try:
            n_vars = real_data.shape[1]
            mi_real = np.zeros((n_vars, n_vars))
            mi_synth = np.zeros((n_vars, n_vars))

            for i in range(n_vars):
                for j in range(i+1, n_vars):
                    # Discretizar para cálculo de MI
                    real_i_discrete = pd.cut(real_data[:, i], bins=20, labels=False)
                    real_j_discrete = pd.cut(real_data[:, j], bins=20, labels=False)
                    synth_i_discrete = pd.cut(synthetic_data[:, i], bins=20, labels=False)
                    synth_j_discrete = pd.cut(synthetic_data[:, j], bins=20, labels=False)

                    mi_real[i,j] = mutual_info_score(real_i_discrete, real_j_discrete)
                    mi_synth[i,j] = mutual_info_score(synth_i_discrete, synth_j_discrete)

            mi_diff = np.mean(np.abs(mi_real - mi_synth))
            return mi_diff
        except Exception as e:
            print(f"Erro em compute_mutual_information_preservation: {e}")
            return np.nan

    def analyze_structure_preservation(self, epoch):
        """Analisa preservação completa da estrutura interna para uma época específica"""
        results = {}

        for condition in self.conditions:
            cond_key = f"cond_{condition}"
            condition_results = {}
            valid_runs = 0

            for run in self.runs:
                run_key = f"run_{run}"

                synthetic_data = self.get_synthetic_samples(cond_key, run_key, epoch)
                if synthetic_data is None:
                    continue

                common_vars = list(set(self.variables) & set(synthetic_data.columns))
                if len(common_vars) < 2:
                    continue

                real_subset = self.real_data[common_vars].values
                synth_subset = synthetic_data[common_vars].values

                # Métricas de correlação
                corr_metrics = self.compute_correlation_preservation(real_subset, synth_subset)

                # Informação mútua
                mi_diff = self.compute_mutual_information_preservation(real_subset, synth_subset)

                condition_results[run_key] = {
                    **corr_metrics,
                    'MI_Difference': mi_diff
                }
                valid_runs += 1

            if condition_results:
                results[cond_key] = condition_results

        return results

    def plot_structure_preservation(self, structure_results, epoch):
        """Plota análise de preservação de estrutura para uma época específica"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            metrics = ['Frobenius_Difference', 'MACE', 'Spearman_Difference', 'MI_Difference']
            metric_names = ['Diferença de Frobenius', 'MACE',
                           'Diferença de Spearman', 'Diferença de MI']

            # Preparar dados
            plot_data = []
            for condition in self.conditions:
                cond_key = f"cond_{condition}"
                if cond_key in structure_results:
                    for run in self.runs:
                        run_key = f"run_{run}"
                        if run_key in structure_results[cond_key]:
                            for metric, metric_name in zip(metrics, metric_names):
                                value = structure_results[cond_key][run_key][metric]
                                if not np.isnan(value):
                                    plot_data.append({
                                        'Condition': f'Cond {condition}',
                                        'Metric': metric_name,
                                        'Value': value,
                                        'Run': run
                                    })

            if not plot_data:
                print(f" Nenhum dado disponível para plotagem de estrutura (época {epoch})")
                for ax in axes.flatten():
                    ax.text(0.5, 0.5, 'Sem dados disponíveis',
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                plt.suptitle(f'PRESERVAÇÃO DE ESTRUTURA - ÉPOCA {epoch}\nSEM DADOS DISPONÍVEIS',
                            fontsize=16, fontweight='bold', y=0.98)
                return fig

            plot_df = pd.DataFrame(plot_data)

            for idx, metric_name in enumerate(metric_names):
                ax = axes.flatten()[idx]
                metric_data = plot_df[plot_df['Metric'] == metric_name]

                if not metric_data.empty:
                    # Ordenar condições
                    condition_order = sorted(metric_data['Condition'].unique())
                    metric_data['Condition'] = pd.Categorical(metric_data['Condition'],
                                                            categories=condition_order)

                    sns.boxplot(data=metric_data, x='Condition', y='Value',
                               ax=ax, palette='coolwarm', order=condition_order)
                    sns.stripplot(data=metric_data, x='Condition', y='Value',
                                 ax=ax, color='black', alpha=0.6, order=condition_order)

                    ax.set_title(f'{metric_name}\n(Menor = Melhor)', fontsize=12, fontweight='bold')
                else:
                    ax.text(0.5, 0.5, 'Sem dados', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold')

                ax.set_xlabel('Condição de Normalização')
                ax.set_ylabel('Valor da Métrica')
                ax.tick_params(axis='x', rotation=45)

            plt.suptitle(f'PRESERVAÇÃO DA ESTRUTURA INTERNA\nÉpoca: {epoch} | Correlações e Dependências entre Variáveis',
                        fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Erro em plot_structure_preservation: {e}")
            import traceback
            traceback.print_exc()
            return plt.figure()




def execute_complete_analysis(data_all, real_data, conditions, runs, target_epochs):
    """
    Executa análise completa para todas as épocas alvo
    """
    # DEBUG: Mostrar estrutura dos dados
    debug_data_structure(data_all, real_data, target_epochs)
    
    # Verificar se há amostras carregadas
    has_samples = False
    for cond_key in data_all:
        for run_key in data_all[cond_key]["samples"]:
            if data_all[cond_key]["samples"][run_key]:
                has_samples = True
                break
        if has_samples:
            break
    
    if not has_samples:
        print("\n" + "="*60)
        print(" ATENÇÃO: NENHUMA AMOSTRA SINTÉTICA CARREGADA!")
        print("="*60)
        print("As análises univariada, multivariada e de cobertura dependem de amostras.")
        print("Verifique se os arquivos estão no local correto:")
        print("  sts_results/cond_*/samples_denorm/samples_denorm_*_*.csv")
        print("="*60)
    
    print("\n" + "="*80)
    print("INICIANDO ANÁLISE COMPLETA DOS DADOS SINTÉTICOS")
    print("="*80)
    
    results = {
        'results': {},
        'figures': {}
    }
    
    for epoch in target_epochs:
        print(f"\n{'='*60}")
        print(f"ANALISANDO ÉPOCA: {epoch}")
        print(f"{'='*60}")
        
        epoch_results = {}
        epoch_figures = []
        
        # 1. ANÁLISE UNIVARIADA
        print("\n1. ANÁLISE UNIVARIADA - Similaridade Distribuição por Variável")
        print("-" * 50)
        
        try:
            univariate_analyzer = UnivariateAnalysis(
                real_data, data_all, conditions, runs, [epoch]
            )
            uni_results = univariate_analyzer.analyze_univariate_similarity(epoch)
            epoch_results['univariate'] = uni_results
            
            fig_uni = univariate_analyzer.plot_univariate_analysis(uni_results, epoch)
            epoch_figures.append(('univariate', fig_uni))
            
            total_runs = sum(len(runs) for runs in uni_results.values()) if uni_results else 0
            print(f" Univariada: {len(uni_results)} condições, {total_runs} runs analisadas")
        except Exception as e:
            print(f" Erro na análise univariada: {e}")
            epoch_results['univariate'] = {}
            fig_uni = plt.figure()
            fig_uni.text(0.5, 0.5, f'Erro na análise univariada:\n{str(e)}', 
                        ha='center', va='center', fontsize=12)
            epoch_figures.append(('univariate_error', fig_uni))
        
        # 2. ANÁLISE MULTIVARIADA


        print("\n2. ANÁLISE MULTIVARIADA - Similaridade no Espaço Conjunto")
        print("-" * 50)

        try:
            # Criar o analyzer
            multivariate_analyzer = MultivariateAnalysis(
                real_data, data_all, conditions, runs, [epoch]
            )
            
            # Chamar a análise
            multi_results = multivariate_analyzer.analyze_multivariate_similarity(epoch)
            
            print(f"  DEBUG: multi_results = {multi_results}")
            print(f"  DEBUG: {len(multi_results)} condições retornadas")
            
            epoch_results['multivariate'] = multi_results
            
            # Verificar se há resultados antes de plotar
            if multi_results:
                fig_multi = multivariate_analyzer.plot_multivariate_analysis(multi_results, epoch)
                epoch_figures.append(('multivariate', fig_multi))
                total_runs = sum(len(runs) for runs in multi_results.values()) if multi_results else 0
                print(f"Multivariada: {len(multi_results)} condições, {total_runs} runs analisadas")
            else:
                print(f"Nenhum resultado para plotar")
                fig_multi = plt.figure()
                fig_multi.text(0.5, 0.5, 'Nenhum dado disponível para multivariada', 
                            ha='center', va='center', fontsize=12)
                epoch_figures.append(('multivariate_no_data', fig_multi))
                
        except Exception as e:
            print(f"  ✗ Erro na análise multivariada: {e}")
            import traceback
            traceback.print_exc()
            epoch_results['multivariate'] = {}
            fig_multi = plt.figure()
            fig_multi.text(0.5, 0.5, f'Erro na análise multivariada:\n{str(e)}', 
                        ha='center', va='center', fontsize=12)
            epoch_figures.append(('multivariate_error', fig_multi))


        
        # 3. ANÁLISE DE ESTRUTURA
        print("\n3. ANÁLISE DE ESTRUTURA - Correlações e Dependências")
        print("-" * 50)
        
        try:
            structure_analyzer = CorrelationStructureAnalysis(
                real_data, data_all, conditions, runs, [epoch]
            )
            struct_results = structure_analyzer.analyze_structure_preservation(epoch)
            epoch_results['structure'] = struct_results
            
            fig_struct = structure_analyzer.plot_structure_preservation(struct_results, epoch)
            epoch_figures.append(('structure', fig_struct))
            
            total_runs = sum(len(runs) for runs in struct_results.values()) if struct_results else 0
            print(f" Estrutura: {len(struct_results)} condições, {total_runs} runs analisadas")
        except Exception as e:
            print(f"Erro na análise de estrutura: {e}")
            epoch_results['structure'] = {}
            fig_struct = plt.figure()
            fig_struct.text(0.5, 0.5, f'Erro na análise de estrutura:\n{str(e)}', 
                           ha='center', va='center', fontsize=12)
            epoch_figures.append(('structure_error', fig_struct))
        
        
        print(f"\n{'='*40}")
        print(f"  RESUMO ÉPOCA {epoch}:")
        print(f"  • Univariada: {len(epoch_results.get('univariate', {}))} condições")
        print(f"  • Multivariada: {len(epoch_results.get('multivariate', {}))} condições")
        print(f"  • Estrutura: {len(epoch_results.get('structure', {}))} condições")
        print(f"{'='*40}")
        print(f"\n  Época {epoch} completa!")
    
    print("\n" + "="*80)
    print("ANÁLISE COMPLETA FINALIZADA!")
    print("="*80)
    
    return results




#######  =====   Class  PLOTS

class ArticlePlots:
    """
    Classe para gerar todos os plots para artigo científico de alto impacto.
    
    Subseções:
    a) Univariate Statistical Fidelity: EMD, KLD, JSD, KS statistic
    b) Multivariate Similarity: Correlation Matrices, MMD, Energy Distance
    c) Preservation of Internal Structure: Frobenius, MACE, Spearman, MI
    d) Visual Validation: KDE, PCA Projection, Bivariate relationships
    """
    #
    def __init__(self, data_all, df_real, conditions, runs, target_epochs, output_dir='article_plots'):
        """
        Inicializa a classe com os dados.
        """
        self.data_all = data_all
        self.df_real = df_real
        self.conditions = conditions
        self.runs = runs
        self.target_epochs = target_epochs
        self.output_dir = output_dir
        
        # Cores para condições
        self.colors = {
            1: '#1f77b4',
            2: '#ff7f0e',
            3: '#2ca02c',
            4: '#d62728'
        }
        
        # Nomes das condições
        self.cond_names = {
            1: 'C1',
            2: 'C2',
            3: 'C3',   
            4: 'C4'
        }
      
        # Criar diretório de saída
        os.makedirs(output_dir, exist_ok=True)

        # ============================================================
        # CORREÇÃO: Inicializar os analyzers
        # ============================================================
        # Assumindo que as classes estão no mesmo arquivo
        self.univariate_analyzer = UnivariateAnalysis(
            df_real, data_all, conditions, runs, target_epochs
        )
        self.multivariate_analyzer = MultivariateAnalysis(
            df_real, data_all, conditions, runs, target_epochs
        )
        self.structure_analyzer = CorrelationStructureAnalysis(
            df_real, data_all, conditions, runs, target_epochs
        )
    # ============================================================
    # MÉTODOS AUXILIARES
    # ============================================================
    
    def _get_multivariate_results(self, epoch):
        """Obtém resultados multivariados para uma época"""
        return self.multivariate_analyzer.analyze_multivariate_similarity(epoch)
    
    def _get_structure_results(self, epoch):
        """Obtém resultados de estrutura para uma época"""
        return self.structure_analyzer.analyze_structure_preservation(epoch)
    
    def _get_univariate_results(self, epoch):
        """Obtém resultados univariados para uma época"""
        return self.univariate_analyzer.analyze_univariate_similarity(epoch)
    
    def _extract_metric_values(self, results, metric_name):
        """Extrai valores de uma métrica específica dos resultados."""
        values_by_cond = {}
        for cond in self.conditions:
            cond_key = f"cond_{cond}"
            values = []
            if cond_key in results:
                for run in self.runs:
                    run_key = f"run_{run}"
                    if run_key in results[cond_key]:
                        val = results[cond_key][run_key].get(metric_name)
                        if val is not None and not np.isnan(val) and not np.isinf(val):
                            values.append(val)
            values_by_cond[cond] = values
        return values_by_cond
    
    def _get_synthetic_samples_for_cond(self, cond, epoch):
        """Obtém amostras sintéticas para uma condição específica."""
        cond_key = f"cond_{cond}"
        all_samples = []
        
        for run in self.runs:
            run_key = f"run_{run}"
            if cond_key in self.data_all:
                if "samples" in self.data_all[cond_key]:
                    if run_key in self.data_all[cond_key]["samples"]:
                        samples_dict = self.data_all[cond_key]["samples"][run_key]
                        for available_epoch in samples_dict.keys():
                            if str(available_epoch) == str(epoch):
                                df = samples_dict[available_epoch]
                                if df is not None and not df.empty:
                                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                                    if len(numeric_cols) > 0:
                                        all_samples.append(df[numeric_cols].values)
                                break
        
        if all_samples:
            return np.vstack(all_samples)
        return None
    
    def _calculate_statistics(self, values):
        """Calcula estatísticas descritivas para uma lista de valores"""
        if not values:
            return {'mean': np.nan, 'std': np.nan, 'iqr': np.nan, 'q1': np.nan, 'q3': np.nan}
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'iqr': np.percentile(values, 75) - np.percentile(values, 25),
            'q1': np.percentile(values, 25),
            'q3': np.percentile(values, 75)
        }
    
    # ============================================================
    # SEÇÃO A: UNIVARIATE STATISTICAL FIDELITY
    # ============================================================
    
    def plot_univariate_fidelity(self, epoch=4500):
        """
        Figura A: Univariate Statistical Fidelity
        Mostra EMD, KL Divergence, JS Divergence, KS Statistic
        
        Visualização: Heatmap 4x9 (condições × variáveis) para cada métrica
        Ou: Painel com 4 heatmaps (um por métrica)
        """
        print(f"\n Gerando Figura A - Univariate Fidelity (Época {epoch})...")
        
        uni_results = self._get_univariate_results(epoch)
        variables = self.df_real.columns.tolist()
        metrics = ['EMD', 'KL_Divergence', 'JS_Divergence', 'KS_Statistic']
        metric_labels = ['Earth Mover Distance', 'KL Divergence', 'JS Divergence', 'KS Statistic']
        
        # Criar figura com 4 heatmaps
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()
        
        for idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[idx]
            
            # Preparar dados para o heatmap
            heatmap_data = []
            for cond in self.conditions:
                cond_key = f"cond_{cond}"
                row_values = []
                for var in variables:
                    values = []
                    if cond_key in uni_results:
                        for run in self.runs:
                            run_key = f"run_{run}"
                            if run_key in uni_results[cond_key]:
                                if var in uni_results[cond_key][run_key]:
                                    val = uni_results[cond_key][run_key][var].get(metric)
                                    if val is not None and not np.isnan(val) and not np.isinf(val):
                                        values.append(val)
                    if values:
                        row_values.append(np.mean(values))
                    else:
                        row_values.append(np.nan)
                heatmap_data.append(row_values)
            
            df_heatmap = pd.DataFrame(heatmap_data, 
                                     index=[self.cond_names[c] for c in self.conditions],
                                     columns=variables)
            
            # Normalizar para melhor visualização
            df_norm = df_heatmap.copy()
            for col in df_norm.columns:
                if df_norm[col].max() > 0:
                    df_norm[col] = df_norm[col] / df_norm[col].max()
            
            # Plotar heatmap
            im = sns.heatmap(df_norm, annot=df_heatmap, fmt='.3f',
                           cmap='RdYlGn_r', vmin=0, vmax=1,
                           ax=ax, cbar_kws={'label': f'{metric_label} (normalizado)'})
            
            ax.set_title(f'({chr(97+idx)}) {metric_label}', fontsize=13, fontweight='bold')
            ax.set_xlabel('Variables', fontsize=11)
            ax.set_ylabel('Condition', fontsize=11)
            
            # Ajustar tamanho da fonte das anotações
            for text in im.texts:
                text.set_fontsize(8)
        
        #plt.suptitle(f'Figure A: Univariate Statistical Fidelity (Epoch {epoch})',
        #            fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Salvar
        fig_path = os.path.join(self.output_dir, f'figureA_univariate_epoch_{epoch}.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f" Figura A salva: {fig_path}")
        plt.close(fig)
        
        return fig
    
    # ============================================================
    # SEÇÃO B: MULTIVARIATE SIMILARITY
    # ============================================================
        
    def plot_multivariate_similarity(self, epoch=4500):
        """
        Figura B: Multivariate Similarity
        Mostra: Correlation Matrices, MMD, Energy Distance
        """
        print(f"\n Gerando Figura B - Multivariate Similarity (Época {epoch})...")
        
        multi_results = self._get_multivariate_results(epoch)
        
        # Determinar melhor e pior condição baseado no MMD
        mmd_means = {}
        for cond in self.conditions:
            vals = self._extract_metric_values(multi_results, 'MMD').get(cond, [])
            if vals:
                mmd_means[cond] = np.mean(vals)
        
        if mmd_means:
            best_cond = min(mmd_means, key=mmd_means.get)
            worst_cond = max(mmd_means, key=mmd_means.get)
        else:
            best_cond = 2
            worst_cond = 1
        
        # Criar figura com 2x3 subplots
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        variables = self.df_real.columns.tolist()
        real_data = self.df_real[variables].values
        corr_real = np.corrcoef(real_data.T)
        
        # ============================================================
        # (a) Matriz de correlação - Dados Reais
        # ============================================================
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(corr_real, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax1.set_xticks(range(len(variables)))
        ax1.set_yticks(range(len(variables)))
        ax1.set_xticklabels(variables, rotation=90, fontsize=8)
        ax1.set_yticklabels(variables, fontsize=8)
        ax1.set_title('(a) Real Data', fontsize=12, fontweight='bold')
        
        for i in range(corr_real.shape[0]):
            for j in range(corr_real.shape[1]):
                ax1.text(j, i, f'{corr_real[i, j]:.2f}',
                        ha='center', va='center', 
                        color='white' if abs(corr_real[i, j]) > 0.5 else 'black',
                        fontsize=7)
        
        # ============================================================
        # (b) Matriz de correlação - Melhor Condição
        # ============================================================
        ax2 = fig.add_subplot(gs[0, 1])
        synth_best = self._get_synthetic_samples_for_cond(best_cond, epoch)
        if synth_best is not None:
            corr_best = np.corrcoef(synth_best.T)
            im2 = ax2.imshow(corr_best, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            ax2.set_xticks(range(len(variables)))
            ax2.set_yticks(range(len(variables)))
            ax2.set_xticklabels(variables, rotation=90, fontsize=8)
            ax2.set_yticklabels(variables, fontsize=8)
            ax2.set_title(f'(b) {self.cond_names[best_cond]} (Best)', fontsize=12, fontweight='bold')
            
            for i in range(corr_best.shape[0]):
                for j in range(corr_best.shape[1]):
                    ax2.text(j, i, f'{corr_best[i, j]:.2f}',
                            ha='center', va='center',
                            color='white' if abs(corr_best[i, j]) > 0.5 else 'black',
                            fontsize=7)
        
        # ============================================================
        # (c) Matriz de correlação - Pior Condição
        # ============================================================
        ax3 = fig.add_subplot(gs[0, 2])
        synth_worst = self._get_synthetic_samples_for_cond(worst_cond, epoch)
        if synth_worst is not None:
            corr_worst = np.corrcoef(synth_worst.T)
            im3 = ax3.imshow(corr_worst, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            ax3.set_xticks(range(len(variables)))
            ax3.set_yticks(range(len(variables)))
            ax3.set_xticklabels(variables, rotation=90, fontsize=8)
            ax3.set_yticklabels(variables, fontsize=8)
            ax3.set_title(f'(c) {self.cond_names[worst_cond]} (Worst)', fontsize=12, fontweight='bold')
            
        # Para corr_worst
        for i in range(corr_worst.shape[0]):
            for j in range(corr_worst.shape[1]):
                ax3.text(j, i, f'{corr_worst[i, j]:.2f}',
                        ha='center', va='center',
                        color='white' if abs(corr_worst[i, j]) > 0.5 else 'black',
                        fontsize=7)
        
        # ============================================================
        # (d) Boxplot MMD
        # ============================================================
        ax4 = fig.add_subplot(gs[1, 0])
        mmd_data = self._extract_metric_values(multi_results, 'MMD')
        
        positions = []
        data_for_box = []
        for cond in self.conditions:
            if cond in mmd_data and mmd_data[cond]:
                positions.append(cond)
                data_for_box.append(mmd_data[cond])
        
        if data_for_box:
            # Criar boxplot com média
            bp = ax4.boxplot(data_for_box, positions=positions,
                            patch_artist=True, widths=0.5,
                            showmeans=True, meanline=True, 
                            meanprops={'color': 'red', 'linestyle': '--', 'linewidth': 1.5})
            
            for i, box in enumerate(bp['boxes']):
                cond = positions[i]
                box.set_facecolor(self.colors[cond])
                box.set_alpha(0.7)
            
            ax4.set_xticks(positions)
            ax4.set_xticklabels([self.cond_names[c] for c in positions], fontsize=11, fontweight='bold')
            ax4.set_ylabel('MMD', fontsize=12, fontweight='bold')
            ax4.set_title('(d) Maximum Mean Discrepancy', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.2, linestyle='--')
            
            # Adicionar estatísticas ACIMA dos boxplots
            for i, pos in enumerate(positions):
                if i < len(data_for_box) and data_for_box[i]:
                    mean_val = np.mean(data_for_box[i])
                    iqr_val = np.percentile(data_for_box[i], 75) - np.percentile(data_for_box[i], 25)
                    
                    # Calcular whisker superior (Q4)
                    y_offset = self._get_safe_y_offset(data_for_box[i], ax4)
                                        
                    ax4.text(pos, y_offset, f'μ={mean_val:.4f}\nIQR={iqr_val:.4f}',
                            ha='center', va='top', fontsize=8, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85,
                                    edgecolor='gray', linewidth=0.5))
            
            # 
            all_values = [val for sublist in data_for_box for val in sublist]
            if all_values:
                y_max = np.max(all_values)
                y_min = np.min(all_values)
                y_range = y_max - y_min
                ax4.set_ylim(y_min - 0.05 * y_range, y_max * 1.15)
        
        # ============================================================
        # (e) Boxplot Energy Distance - COM AJUSTE DE ESCALA
        # ============================================================
        ax5 = fig.add_subplot(gs[1, 1])
        energy_data = self._extract_metric_values(multi_results, 'Energy_Distance')
        
        positions = []
        data_for_box = []
        for cond in self.conditions:
            if cond in energy_data and energy_data[cond]:
                positions.append(cond)
                data_for_box.append(energy_data[cond])
        
        if data_for_box:
            # Criar boxplot com média
            bp = ax5.boxplot(data_for_box, positions=positions,
                            patch_artist=True, widths=0.5,
                            showmeans=True, meanline=True,
                            meanprops={'color': 'red', 'linestyle': '--', 'linewidth': 1.5})
            
            for i, box in enumerate(bp['boxes']):
                cond = positions[i]
                box.set_facecolor(self.colors[cond])
                box.set_alpha(0.7)
            
            ax5.set_xticks(positions)
            ax5.set_xticklabels([self.cond_names[c] for c in positions], fontsize=11, fontweight='bold')
            ax5.set_ylabel('Energy Distance', fontsize=12, fontweight='bold')
            ax5.set_title('(e) Energy Distance', fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.2, linestyle='--')
            
            # Adicionar estatísticas ACIMA dos boxplots
            for i, pos in enumerate(positions):
                if i < len(data_for_box) and data_for_box[i]:
                    mean_val = np.mean(data_for_box[i])
                    iqr_val = np.percentile(data_for_box[i], 75) - np.percentile(data_for_box[i], 25)
                    
                    # Calcular whisker superior (Q4)
                    y_offset = self._get_safe_y_offset(data_for_box[i], ax5)
                    
                    ax5.text(pos, y_offset, f'μ={mean_val:.1f}\nIQR={iqr_val:.1f}',
                            ha='center', va='top', fontsize=8, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85,
                                    edgecolor='gray', linewidth=0.5))
            
            # ============================================================
            # CORREÇÃO: Ajustar limite do eixo y para melhor visualização
            # ============================================================
            all_values = [val for sublist in data_for_box for val in sublist if val is not None]
            if all_values:
                y_max = np.max(all_values)
                y_min = np.min(all_values)   
                y_range = y_max - y_min
                
                # Se o range for muito pequeno (boxplots muito comprimidos)
                if y_range < 1.0 and y_max < 10:
                    # Para valores pequenos (como época 400), usar limite mais generoso
                    ax5.set_ylim(y_min - 0.5, y_max * 1.3)
                else:
                    # Para valores normais
                    ax5.set_ylim(y_min - 0.05 * y_range, y_max * 1.15)
        
        # ============================================================
        # (f) Frobenius Norm Comparison
        # ============================================================
        ax6 = fig.add_subplot(gs[1, 2])
        if synth_best is not None:
            diff_best = np.abs(corr_real - corr_best)
            diff_worst = np.abs(corr_real - corr_worst) if synth_worst is not None else None
            
            frob_best = np.linalg.norm(diff_best, 'fro')
            frob_worst = np.linalg.norm(diff_worst, 'fro') if diff_worst is not None else 0
            
            bars = ax6.bar(['Best', 'Worst'], [frob_best, frob_worst],
                        color=['#2ca02c', '#d62728'], alpha=0.8, edgecolor='black', linewidth=1)
            ax6.set_ylabel('Frobenius Norm', fontsize=11, fontweight='bold')
            ax6.set_title('(f) Correlation Structure Preservation', fontsize=12, fontweight='bold')
            ax6.grid(True, alpha=0.2, linestyle='--', axis='y')
            
            for bar, val in zip(bars, [frob_best, frob_worst]):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax6.set_xticklabels([f'{self.cond_names[best_cond]}', f'{self.cond_names[worst_cond]}'], 
                            fontsize=11, fontweight='bold')
        
        # ============================================================
        # LEGENDA: Explicar as linhas do boxplot
        # ============================================================
        # Adicionar legenda para os boxplots
        legend_elements = [
            plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='Mean'),
            plt.Line2D([0], [0], color='blue', linewidth=0, marker='o', markersize=0, 
                    label='Box: 25th-75th percentile'),
            plt.Line2D([0], [0], color='black', linewidth=0, marker='|', markersize=10, 
                    label='Whiskers: 1.5xIQR')  
        ]
        
        # Adicionar legenda no canto inferior direito da figura
        fig.legend(handles=legend_elements, 
                loc='lower center', 
                bbox_to_anchor=(0.5, -0.02),
                ncol=3, 
                fontsize=10,
                frameon=True,
                fancybox=True,
                shadow=True)
        
        # ============================================================
        # Colorbar
        # ============================================================
        cbar_ax = fig.add_axes([0.92, 0.55, 0.02, 0.3])
        cbar = fig.colorbar(im1, cax=cbar_ax, label='Correlation Coefficient')
        cbar.ax.tick_params(labelsize=9)
        
        # ============================================================
        # Título da Figura
        # ============================================================
        #plt.suptitle(f'Figure B: Multivariate Similarity Assessment (Epoch {epoch})',
        #            fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        
        # Salvar
        fig_path = os.path.join(self.output_dir, f'figureB_multivariate_epoch_{epoch}.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f" Figura B salva: {fig_path}")
        plt.close(fig)
        
        return fig


    # ============================================================
    # SEÇÃO C: PRESERVATION OF INTERNAL STRUCTURE
    # ============================================================
    
    def plot_structure_preservation(self, epoch=4500):
        """
        Figura C: Preservation of Internal Structure
        Mostra: Frobenius Difference, MACE, Spearman Difference, MI Difference
        
        Visualização: Boxplots para cada métrica (4 métricas × 4 condições)
        """
        print(f"\n Gerando Figura C - Structure Preservation (Época {epoch})...")
        
        struct_results = self._get_structure_results(epoch)
        
        metrics = ['Frobenius_Difference', 'MACE', 'Spearman_Difference', 'MI_Difference']
        metric_labels = ['Frobenius Difference', 'MACE', 'Spearman Difference', 'MI Difference']
        metric_short = ['Frobenius', 'MACE', 'Spearman', 'MI']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, (metric, metric_label, metric_short_name) in enumerate(zip(metrics, metric_labels, metric_short)):
            ax = axes[idx]
            data = self._extract_metric_values(struct_results, metric)
            
            positions = []
            data_for_box = []
            for cond in self.conditions:
                if cond in data and data[cond]:
                    positions.append(cond)
                    data_for_box.append(data[cond])
            
            if data_for_box:
                # Criar boxplot com média
                bp = ax.boxplot(data_for_box, positions=positions,
                            patch_artist=True, widths=0.5,
                            showmeans=True, meanline=True, 
                            meanprops={'color': 'red', 'linestyle': '--', 'linewidth': 1.5})
                
                # Colorir os boxes
                for i, box in enumerate(bp['boxes']):
                    cond = positions[i]
                    box.set_facecolor(self.colors[cond])
                    box.set_alpha(0.7)   
                
                # Configurar eixos
                ax.set_xticks(positions)
                #ax.set_xticklabels([f'cond_{c}' for c in positions], fontsize=11, fontweight='bold')
                ax.set_xticklabels([self.cond_names[c] for c in positions], fontsize=11, fontweight='bold')
                ax.set_ylabel(metric_label, fontsize=12, fontweight='bold')
                ax.set_title(f'({chr(97+idx)}) {metric_short_name}', fontsize=13, fontweight='bold')
                ax.grid(True, alpha=0.2, linestyle='--')
                
                # Adicionar médias ACIMA dos boxplots
                for i, pos in enumerate(positions):
                    if i < len(data_for_box) and data_for_box[i]:
                        mean_val = np.mean(data_for_box[i])
                        std_val = np.std(data_for_box[i])
                        iqr_val = np.percentile(data_for_box[i], 75) - np.percentile(data_for_box[i], 25)
                        
                        # Calcular posição para o texto (acima do boxplot)
                        y_offset = self._get_safe_y_offset(data_for_box[i], ax)
                        
                        # Adicionar texto com estatísticas
                        ax.text(pos, y_offset, 
                            f'μ={mean_val:.4f}\nσ={std_val:.4f}\nIQR={iqr_val:.4f}',
                            ha='center', va='top', fontsize=8, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85,
                                        edgecolor='gray', linewidth=0.5))
        
        # Adicionar título principal da figura
        #plt.suptitle(f'Figure C: Preservation of Internal Structure (Epoch {epoch})',
        #            fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout()
        
        # Salvar
        fig_path = os.path.join(self.output_dir, f'figureC_structure_epoch_{epoch}.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f" Figura C salva: {fig_path}")
        plt.close(fig)
        
        return fig

    
    # ============================================================
    # SEÇÃO D: VISUAL VALIDATION AND STRUCTURAL INTEGRITY    
    # ============================================================
    
    def plot_kde_visualization(self, epoch=4500):
        """
        Figura D1: KDE Visualization
        Mostra: Distribuições KDE para cada variável (3x3 grid)
        Comparação entre dados reais e sintéticos da melhor condição
        """
        print(f"\n Gerando Figura D1 - KDE Visualization (Época {epoch})...")
        
        variables = self.df_real.columns.tolist()
        
        # Determinar melhor condição para visualização
        multi_results = self._get_multivariate_results(epoch)
        mmd_means = {}
        for cond in self.conditions:
            vals = self._extract_metric_values(multi_results, 'MMD').get(cond, [])
            if vals:
                mmd_means[cond] = np.mean(vals)
        
        best_cond = min(mmd_means, key=mmd_means.get) if mmd_means else 2
        synth_best = self._get_synthetic_samples_for_cond(best_cond, epoch)
        
        if synth_best is None:
            print(f" Sem dados sintéticos para cond_{best_cond}, época {epoch}")
            return None
        
        # Criar figura com grid 3x3
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, var in enumerate(variables):
            ax = axes[i]
            
            # Dados reais e sintéticos
            real_data = self.df_real[var].values
            synth_data = synth_best[:, i]
            
            # Plot KDE
            sns.kdeplot(real_data, ax=ax, label='Real', color='#1f77b4', linewidth=2.5)
            sns.kdeplot(synth_data, ax=ax, label='Synthetic', color='#ff7f0e', linewidth=2.5)
            
            # Adicionar estatísticas
            real_mean = np.mean(real_data)
            synth_mean = np.mean(synth_data)
            
            ax.axvline(real_mean, color='#1f77b4', linestyle='--', alpha=0.5, linewidth=1.5)
            ax.axvline(synth_mean, color='#ff7f0e', linestyle='--', alpha=0.5, linewidth=1.5)
            
            ax.set_title(f'({chr(97+i)}) {var}', fontsize=13, fontweight='bold')
            ax.set_xlabel('Value', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.2, linestyle='--')
        
        # Remover eixos extras se houver
        for j in range(len(variables), len(axes)):
            fig.delaxes(axes[j])
        
        #plt.suptitle(f'Figure D1: KDE Visualization - {self.cond_names[best_cond]} (Best Condition, Epoch {epoch})',
        #            fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Salvar
        fig_path = os.path.join(self.output_dir, f'figureD1_kde_epoch_{epoch}.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f" Figura D1 salva: {fig_path}")
        plt.close(fig)
        
        return fig


    def plot_multivariate_visualization(self, epoch=4500):
        """
        Figura D2: Multivariate Visualization
        Mostra: PCA Projection + Bivariate Density maps
        Comparação entre dados reais e sintéticos da melhor condição
        COM AMOSTRAGEM BALANCEADA
        """
        print(f"\n Gerando Figura D2 - Multivariate Visualization (Época {epoch})...")
        
        variables = self.df_real.columns.tolist()
        
        # Determinar melhor condição para visualização
        multi_results = self._get_multivariate_results(epoch)
        mmd_means = {}
        for cond in self.conditions:
            vals = self._extract_metric_values(multi_results, 'MMD').get(cond, [])
            if vals:
                mmd_means[cond] = np.mean(vals)
        
        best_cond = min(mmd_means, key=mmd_means.get) if mmd_means else 2
        synth_best = self._get_synthetic_samples_for_cond(best_cond, epoch)
        
        if synth_best is None:
            print(f" Sem dados sintéticos para cond_{best_cond}, época {epoch}")
            return None
        
        # ============================================================
        # BALANCEAR AMOSTRAS
        # ============================================================
        # Dados reais
        real_data_full = self.df_real[variables].values
        n_real = len(real_data_full)
        
        # Amostrar dados sintéticos para ter o mesmo número que os reais
        n_synth = len(synth_best)
        if n_synth > n_real:
            # Selecionar amostras aleatórias dos dados sintéticos
            np.random.seed(42)  # Para reprodutibilidade
            indices = np.random.choice(n_synth, n_real, replace=False)
            synth_sampled = synth_best[indices]
            print(f"Amostragem balanceada: Real={n_real}, Sintético={len(synth_sampled)} (amostrado de {n_synth})")
        else:
            synth_sampled = synth_best
            print(f" Usando todos os dados: Real={n_real}, Sintético={n_synth}")
        
        # Combinar dados balanceados
        combined = np.vstack([real_data_full, synth_sampled])
        n_balanced_real = n_real
        n_balanced_synth = len(synth_sampled)
        
        # Criar figura com 2 colunas
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # ============================================================
        # (a) PCA Projection com dados balanceados
        # ============================================================
        ax1 = axes[0]
        
        # PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(combined)
        
        # Plot com transparência e tamanhos ajustados
        ax1.scatter(pca_result[:n_balanced_real, 0], pca_result[:n_balanced_real, 1],
                alpha=0.7, label=f'Real (n={n_balanced_real})', 
                color='#1f77b4', s=50, edgecolors='white', linewidth=0.5)
        ax1.scatter(pca_result[n_balanced_real:, 0], pca_result[n_balanced_real:, 1],
                alpha=0.6, label=f'Synthetic (n={n_balanced_synth})', 
                color='#ff7f0e', s=50, edgecolors='white', linewidth=0.5)
        
        # Adicionar centroides
        centroid_real = np.mean(pca_result[:n_balanced_real], axis=0)
        centroid_synth = np.mean(pca_result[n_balanced_real:], axis=0)
        ax1.scatter(centroid_real[0], centroid_real[1], marker='X', s=200, 
                color='#1f77b4', edgecolors='black', linewidth=2, label='Centroid Real')
        ax1.scatter(centroid_synth[0], centroid_synth[1], marker='X', s=200,
                color='#ff7f0e', edgecolors='black', linewidth=2, label='Centroid Synthetic')
        
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12, fontweight='bold')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12, fontweight='bold')
        ax1.set_title('(a) PCA Projection (Balanced Samples)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10, loc='best')
        ax1.grid(True, alpha=0.2, linestyle='--')
        
        # Adicionar estatísticas no gráfico
        stats_text = f"Real: μ=({centroid_real[0]:.2f}, {centroid_real[1]:.2f})\n"
        stats_text += f"Synth: μ=({centroid_synth[0]:.2f}, {centroid_synth[1]:.2f})\n"
        stats_text += f"Distance: {np.linalg.norm(centroid_real - centroid_synth):.2f}"
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # ============================================================
        # (b) Bivariate Density com dados balanceados
        # ============================================================
        ax2 = axes[1]
        
        # Escolher duas variáveis para o mapa de densidade
        var1, var2 = variables[0], variables[1]
        idx1, idx2 = 0, 1
        
        # Plot 2D density com contornos preenchidos
        from scipy.stats import gaussian_kde
        
        try:
            # Dados reais e sintéticos balanceados
            real_data_2d = real_data_full[:, [idx1, idx2]]
            synth_data_2d = synth_sampled[:, [idx1, idx2]]
            
            # Calcular KDE
            real_kde = gaussian_kde(real_data_2d.T)
            synth_kde = gaussian_kde(synth_data_2d.T)
            
            # Criar grid para contornos
            all_data_2d = np.vstack([real_data_2d, synth_data_2d])
            x_min, x_max = all_data_2d[:, 0].min(), all_data_2d[:, 0].max()
            y_min, y_max = all_data_2d[:, 1].min(), all_data_2d[:, 1].max()
            
            # Adicionar margem
            x_range = x_max - x_min
            y_range = y_max - y_min
            x_min -= 0.1 * x_range
            x_max += 0.1 * x_range
            y_min -= 0.1 * y_range
            y_max += 0.1 * y_range
            
            x_grid = np.linspace(x_min, x_max, 100)
            y_grid = np.linspace(y_min, y_max, 100)
            X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
            grid_points = np.vstack([X_grid.ravel(), Y_grid.ravel()])
            
            # Plot contornos
            ax2.contour(X_grid, Y_grid, real_kde(grid_points).reshape(X_grid.shape),
                    levels=8, colors='#1f77b4', alpha=0.7, linewidths=1.5)
            ax2.contourf(X_grid, Y_grid, real_kde(grid_points).reshape(X_grid.shape),
                        levels=8, colors='#1f77b4', alpha=0.15)
            
            ax2.contour(X_grid, Y_grid, synth_kde(grid_points).reshape(X_grid.shape),
                    levels=8, colors='#ff7f0e', alpha=0.7, linewidths=1.5)
            ax2.contourf(X_grid, Y_grid, synth_kde(grid_points).reshape(X_grid.shape),
                        levels=8, colors='#ff7f0e', alpha=0.15)
            
            # Adicionar scatter dos pontos com transparência
            ax2.scatter(real_data_2d[:, 0], real_data_2d[:, 1],
                    alpha=0.4, color='#1f77b4', s=15, label='Real')
            ax2.scatter(synth_data_2d[:, 0], synth_data_2d[:, 1],
                    alpha=0.3, color='#ff7f0e', s=15, label='Synthetic')
            
        except Exception as e:
            print(f"Erro no KDE 2D: {e}, usando scatter plot")
            ax2.scatter(real_data_full[:, idx1], real_data_full[:, idx2],
                    alpha=0.5, label='Real', color='#1f77b4', s=30)
            ax2.scatter(synth_sampled[:, idx1], synth_sampled[:, idx2],
                    alpha=0.5, label='Synthetic', color='#ff7f0e', s=30)
        
        ax2.set_xlabel(f'{var1}', fontsize=12, fontweight='bold')
        ax2.set_ylabel(f'{var2}', fontsize=12, fontweight='bold')
        ax2.set_title(f'(b) Bivariate Density: {var1} vs {var2}\n(Balanced Samples)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10, loc='best')
        ax2.grid(True, alpha=0.2, linestyle='--')
        
        # ============================================================
        # Suptitle
        # ============================================================
        #plt.suptitle(f'Figure D2: Multivariate Visualization - {self.cond_names[best_cond]} (Best Condition, Epoch {epoch})\n'
        #            f'Real (n={n_balanced_real}) vs Synthetic (n={n_balanced_synth})',
        #            fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
            
        # Salvar
        fig_path = os.path.join(self.output_dir, f'figureD2_multivariate_epoch_{epoch}.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f" Figura D2 salva: {fig_path}")
        plt.close(fig)
        
        return fig
    
    # ============================================================
    # TABELAS INFORMATIVAS
    # ============================================================
    
    def generate_tables(self, epoch=4500):
        """
        Gera tabelas informativas para o artigo.
        
        Tabela 1: Resumo das métricas por condição (MMD, Energy, Frobenius, MACE, Spearman, MI)
        Tabela 2: Estatísticas descritivas (mean ± std, IQR)
        """
        print(f"\n Gerando Tabelas (Época {epoch})...")
        
        multi_results = self._get_multivariate_results(epoch)
        struct_results = self._get_structure_results(epoch)
        
        # ============================================================
        # TABELA 1: Métricas Resumidas
        # ============================================================
        table1_data = []
        
        for cond in self.conditions:
            row = {'Condition': self.cond_names[cond]}
            
            # MMD
            mmd_vals = self._extract_metric_values(multi_results, 'MMD').get(cond, [])
            if mmd_vals:
                stats_mmd = self._calculate_statistics(mmd_vals)
                row['MMD_mean'] = stats_mmd['mean']
                row['MMD_std'] = stats_mmd['std']
                row['MMD_iqr'] = stats_mmd['iqr']
            
            # Energy Distance
            energy_vals = self._extract_metric_values(multi_results, 'Energy_Distance').get(cond, [])
            if energy_vals:
                stats_energy = self._calculate_statistics(energy_vals)
                row['Energy_mean'] = stats_energy['mean']
                row['Energy_std'] = stats_energy['std']
            
            # Frobenius
            frob_vals = self._extract_metric_values(struct_results, 'Frobenius_Difference').get(cond, [])
            if frob_vals:
                stats_frob = self._calculate_statistics(frob_vals)
                row['Frobenius_mean'] = stats_frob['mean']
                row['Frobenius_std'] = stats_frob['std']
            
            # MACE
            mace_vals = self._extract_metric_values(struct_results, 'MACE').get(cond, [])
            if mace_vals:
                stats_mace = self._calculate_statistics(mace_vals)
                row['MACE_mean'] = stats_mace['mean']
                row['MACE_std'] = stats_mace['std']
            
            # Spearman
            spear_vals = self._extract_metric_values(struct_results, 'Spearman_Difference').get(cond, [])
            if spear_vals:
                stats_spear = self._calculate_statistics(spear_vals)
                row['Spearman_mean'] = stats_spear['mean']
                row['Spearman_std'] = stats_spear['std']
            
            # MI
            mi_vals = self._extract_metric_values(struct_results, 'MI_Difference').get(cond, [])
            if mi_vals:
                stats_mi = self._calculate_statistics(mi_vals)
                row['MI_mean'] = stats_mi['mean']
                row['MI_std'] = stats_mi['std']
            
            table1_data.append(row)
        
        df_table1 = pd.DataFrame(table1_data)
        
        # ============================================================
        # TABELA 2: Estatísticas Descritivas (mean ± std)
        # ============================================================
        table2_data = []
        
        for cond in self.conditions:
            row = {'Condition': self.cond_names[cond]}
            
            # MMD
            mmd_vals = self._extract_metric_values(multi_results, 'MMD').get(cond, [])
            if mmd_vals:
                stats_mmd = self._calculate_statistics(mmd_vals)
                row['MMD'] = f"{stats_mmd['mean']:.6f} ± {stats_mmd['std']:.6f}"
                row['MMD_IQR'] = f"{stats_mmd['iqr']:.6f}"
            
            # Energy Distance
            energy_vals = self._extract_metric_values(multi_results, 'Energy_Distance').get(cond, [])
            if energy_vals:
                stats_energy = self._calculate_statistics(energy_vals)
                row['Energy'] = f"{stats_energy['mean']:.2f} ± {stats_energy['std']:.2f}"
            
            # Frobenius
            frob_vals = self._extract_metric_values(struct_results, 'Frobenius_Difference').get(cond, [])
            if frob_vals:
                stats_frob = self._calculate_statistics(frob_vals)
                row['Frobenius'] = f"{stats_frob['mean']:.4f} ± {stats_frob['std']:.4f}"
            
            # MACE
            mace_vals = self._extract_metric_values(struct_results, 'MACE').get(cond, [])
            if mace_vals:
                stats_mace = self._calculate_statistics(mace_vals)
                row['MACE'] = f"{stats_mace['mean']:.4f} ± {stats_mace['std']:.4f}"
            
            # Spearman
            spear_vals = self._extract_metric_values(struct_results, 'Spearman_Difference').get(cond, [])
            if spear_vals:
                stats_spear = self._calculate_statistics(spear_vals)
                row['Spearman'] = f"{stats_spear['mean']:.4f} ± {stats_spear['std']:.4f}"
            
            # MI
            mi_vals = self._extract_metric_values(struct_results, 'MI_Difference').get(cond, [])
            if mi_vals:
                stats_mi = self._calculate_statistics(mi_vals)
                row['MI'] = f"{stats_mi['mean']:.4f} ± {stats_mi['std']:.4f}"
            
            table2_data.append(row)
        
        df_table2 = pd.DataFrame(table2_data)
        
        # Salvar tabelas
        table1_path = os.path.join(self.output_dir, f'table1_summary_epoch_{epoch}.csv')
        table2_path = os.path.join(self.output_dir, f'table2_descriptive_epoch_{epoch}.csv')
        
        df_table1.to_csv(table1_path, index=False)
        df_table2.to_csv(table2_path, index=False)
        
        print(f" Tabela 1 salva: {table1_path}")
        print(f" Tabela 2 salva: {table2_path}")
        
        return df_table1, df_table2
    
    # ============================================================
    # GERAR TODAS AS FIGURAS E TABELAS
    # ============================================================
    
    def plot_all_figures(self):
        """Gera todas as figuras e tabelas para todas as épocas."""
        print("\n" + "="*80)
        print(" GERANDO TODAS AS FIGURAS E TABELAS PARA O ARTIGO")
        print("="*80)
        
        for epoch in self.target_epochs:
            print(f"\n{'='*50}")
            print(f"ÉPOCA {epoch}")
            print(f"{'='*50}")
            
            # Figura A: Univariate Fidelity
            self.plot_univariate_fidelity(epoch)
            
            # Figura B: Multivariate Similarity
            self.plot_multivariate_similarity(epoch)
            
            # Figura C: Structure Preservation
            self.plot_structure_preservation(epoch)
            
            # Figura D1: KDE Visualization
            self.plot_kde_visualization(epoch)
            
            # Figura D2: Multivariate Visualization (PCA + Bivariate)
            self.plot_multivariate_visualization(epoch)
            
            # Tabelas
            self.generate_tables(epoch)
        
        print("\n" + "="*80)
        print(f" Todas as figuras e tabelas salvas em: {self.output_dir}")
        print("="*80)
        
        # Gerar relatório de métricas
        self._generate_metrics_report()
    
    def _generate_metrics_report(self):
        """Gera um relatório textual com as métricas para o artigo."""
        report_path = os.path.join(self.output_dir, 'metrics_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("MÉTRICAS DE QUALIDADE DOS DADOS SINTÉTICOS\n")
            f.write("="*80 + "\n\n")
            
            for epoch in self.target_epochs:
                f.write(f"\n{'='*50}\n")
                f.write(f"ÉPOCA {epoch}\n")
                f.write(f"{'='*50}\n\n")
                
                multi_results = self._get_multivariate_results(epoch)
                struct_results = self._get_structure_results(epoch)
                
                # MMD
                f.write("MMD (menor é melhor):\n")
                for cond in self.conditions:
                    vals = self._extract_metric_values(multi_results, 'MMD').get(cond, [])
                    if vals:
                        stats = self._calculate_statistics(vals)
                        f.write(f"  {self.cond_names[cond]}: {stats['mean']:.6f} ± {stats['std']:.6f} (IQR={stats['iqr']:.6f})\n")
                
                # Energy Distance
                f.write("\nEnergy Distance (menor é melhor):\n")
                for cond in self.conditions:
                    vals = self._extract_metric_values(multi_results, 'Energy_Distance').get(cond, [])
                    if vals:
                        stats = self._calculate_statistics(vals)
                        f.write(f"  {self.cond_names[cond]}: {stats['mean']:.2f} ± {stats['std']:.2f}\n")
                
                # Frobenius
                f.write("\nFrobenius Difference (menor é melhor):\n")
                for cond in self.conditions:
                    vals = self._extract_metric_values(struct_results, 'Frobenius_Difference').get(cond, [])
                    if vals:
                        stats = self._calculate_statistics(vals)
                        f.write(f"  {self.cond_names[cond]}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                
                # MACE
                f.write("\nMACE (menor é melhor):\n")
                for cond in self.conditions:
                    vals = self._extract_metric_values(struct_results, 'MACE').get(cond, [])
                    if vals:
                        stats = self._calculate_statistics(vals)
                        f.write(f"  {self.cond_names[cond]}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                
                # Spearman
                f.write("\nSpearman Difference (menor é melhor):\n")
                for cond in self.conditions:
                    vals = self._extract_metric_values(struct_results, 'Spearman_Difference').get(cond, [])
                    if vals:
                        stats = self._calculate_statistics(vals)
                        f.write(f"  {self.cond_names[cond]}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                
                # MI
                f.write("\nMI Difference (menor é melhor):\n")
                for cond in self.conditions:
                    vals = self._extract_metric_values(struct_results, 'MI_Difference').get(cond, [])
                    if vals:
                        stats = self._calculate_statistics(vals)
                        f.write(f"  {self.cond_names[cond]}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                
                # Melhor condição
                f.write("\n" + "-"*40 + "\n")
                f.write("MELHOR CONDIÇÃO POR MÉTRICA:\n")
                
                # Determinar melhor MMD
                best_mmd = float('inf')
                best_mmd_cond = None
                for cond in self.conditions:
                    vals = self._extract_metric_values(multi_results, 'MMD').get(cond, [])
                    if vals:
                        mean_val = np.mean(vals)
                        if mean_val < best_mmd:
                            best_mmd = mean_val
                            best_mmd_cond = self.cond_names[cond]
                if best_mmd_cond:
                    f.write(f"  MMD: {best_mmd_cond} ({best_mmd:.6f})\n")
                
                # Determinar melhor Energy
                best_energy = float('inf')
                best_energy_cond = None
                for cond in self.conditions:
                    vals = self._extract_metric_values(multi_results, 'Energy_Distance').get(cond, [])
                    if vals:
                        mean_val = np.mean(vals)
                        if mean_val < best_energy:
                            best_energy = mean_val
                            best_energy_cond = self.cond_names[cond]
                if best_energy_cond:
                    f.write(f"  Energy Distance: {best_energy_cond} ({best_energy:.2f})\n")
                
                # Determinar melhor Frobenius
                best_frob = float('inf')
                best_frob_cond = None
                for cond in self.conditions:
                    vals = self._extract_metric_values(struct_results, 'Frobenius_Difference').get(cond, [])
                    if vals:
                        mean_val = np.mean(vals)
                        if mean_val < best_frob:
                            best_frob = mean_val
                            best_frob_cond = self.cond_names[cond]
                if best_frob_cond:
                    f.write(f"  Frobenius: {best_frob_cond} ({best_frob:.4f})\n")
                
                f.write("\n" + "="*50 + "\n")
        
        print(f"  Relatório de métricas salvo: {report_path}")

    def _get_safe_y_offset(self, data, ax, multiplier=1.05):
        """
        Calcula uma posição y segura para o texto, garantindo que fique dentro do gráfico.
        
        Args:
            data: Lista de valores para calcular a posição
            ax: Eixo do matplotlib
            multiplier: Fator multiplicador para posicionar acima do whisker
        
        Returns:
            float: Posição y segura
        """
        q3 = np.percentile(data, 75)
        q1 = np.percentile(data, 25)
        iqr = q3 - q1
        upper_whisker = q3 + 1.5 * iqr
        
        y_offset = upper_whisker * multiplier if upper_whisker > 0 else 0.001
        
        # Garantir que fique dentro do gráfico
        y_min, y_max = ax.get_ylim()
        
        # Se ultrapassar 95% do limite superior, ajustar para 90%
        if y_offset > y_max * 0.95:
            y_offset = y_max * 0.90
        
        # Se ainda ultrapassar, usar o limite superior menos padding
        if y_offset > y_max:
            y_offset = y_max - (y_max - y_min) * 0.05
        
        return y_offset

    def plot_temporal_dynamics(self):
        """Gera a análise temporal das métricas internas."""
        fig, stats_df, _ = plot_temporal_analysis(
            data_all=self.data_all,
            metrics=['d_loss_mean', 'g_loss_mean', 'gp_mean', 'mmd', 'emd_mean'],
            output_dir=self.output_dir
        )
        return fig, stats_df


# ============================================================
# EXEMPLO DE USO
# ============================================================
def generate_article_plots(data_all, df_real, conditions, runs, target_epochs):
    """
    Função wrapper para gerar todas as figuras do artigo.
    
    Args:
        data_all: Dicionário com dados carregados
        df_real: DataFrame com dados reais
        conditions: Lista de condições
        runs: Lista de runs
        target_epochs: Lista de épocas alvo
    
    Returns:
        ArticlePlots: Instância da classe com os plots gerados
    """
    plotter = ArticlePlots(
        data_all=data_all,
        df_real=df_real,
        conditions=conditions,
        runs=runs,
        target_epochs=target_epochs,
        output_dir='article_plots'
    )
    
    plotter.plot_all_figures()
    
    return plotter


# ============================================================
# FUNÇÃO MAIN 
# ============================================================
def main(config_file='config.yaml'):
    """
    Função principal que automatiza toda a análise
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        logging.info("Carregando configuração...")
        config = load_config(config_file)
        
        if config is None:
            logging.error("Falha ao carregar configuração. Usando valores padrão.")
            config = yaml.safe_load(CONFIG_YAML)
        
        paths = config.get('paths', {})
        data_params = config.get('data', {})
        
        FOLDER_PROJECT = paths.get('folder_project', os.getcwd())
        SUB_FOLDER = os.path.join(FOLDER_PROJECT, paths.get('folder_sts_results', 'sts_results/'))
        
        logging.info(f"Procurando dados em: {SUB_FOLDER}")
        if not os.path.exists(SUB_FOLDER):
            logging.error(f"Pasta não encontrada: {SUB_FOLDER}")
            alt_folder = os.path.join(os.getcwd(), 'sts_results')
            if os.path.exists(alt_folder):
                SUB_FOLDER = alt_folder
                logging.info(f"Usando caminho alternativo: {SUB_FOLDER}")
            else:
                raise FileNotFoundError(f"Pasta sts_results não encontrada")
        
        # Carregar dados reais
        logging.info("Carregando dados reais...")
        datas_folder = os.path.join(FOLDER_PROJECT, paths.get('folder_datas', 'Datas/'))
        df_real_path = os.path.join(datas_folder, 'OriginalData.csv')
        
        if not os.path.exists(df_real_path):
            alt_path = os.path.join(os.getcwd(), 'Datas', 'OriginalData.csv')
            if os.path.exists(alt_path):
                df_real_path = alt_path
            else:
                raise FileNotFoundError(f"OriginalData.csv not found")
        
        df_real = pd.read_csv(df_real_path)
        
        col_numerical_df = data_params.get('numerical_columns', 
            ['226Ra', '232Th', '40K', 'Raeq', 'Theq', 'Keq', 'IG', 'IA', 'IB'])

        # Filtrar apenas as colunas que existem no DataFrame
        available_cols = [col for col in col_numerical_df if col in df_real.columns]

        if available_cols:
            df_real = df_real[available_cols]
            # Converter apenas as colunas selecionadas para numérico
            for col in df_real.columns:
                df_real[col] = pd.to_numeric(df_real[col], errors='coerce')
            df_real = df_real.dropna()
            logging.info(f"Usando colunas: {available_cols}")
        else:
            # Fallback: usar todas as colunas numéricas
            numeric_cols = df_real.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                df_real = df_real[numeric_cols]
                logging.info(f"Usando colunas numéricas disponíveis: {numeric_cols}")
            else:
                raise ValueError("Nenhuma coluna numérica encontrada no dataset real")

        logging.info(f"Dados reais carregados: {df_real.shape}")
        logging.info(f"Colunas reais: {df_real.columns.tolist()}")

        # Carregar dados sintéticos
        logging.info("Carregando dados sintéticos...")
        conditions = data_params.get('conditions', [1, 2, 3, 4])
        runs = data_params.get('runs', list(range(7)))
        epochs = data_params.get('epochs', [400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000, 4400, 4500])
        
        data_all = load_all_data(SUB_FOLDER, conditions, runs, epochs)
        logging.info(f"Dados carregados para {len(data_all)} condições")
        
        if not data_all:
            logging.error("Nenhum dado foi carregado!")
            return None
        
        target_epochs = data_params.get('target_epochs', [400, 2000, 4500])
        
        # ============================================================
        # 1. GERAR FIGURAS PARA O ARTIGO (ArticlePlots)
        # ============================================================
        print("\n" + "="*80)
        print("GERANDO FIGURAS PARA O ARTIGO")
        print("="*80)
        
        # Criar instância do ArticlePlots
        article_plotter = ArticlePlots(
            data_all=data_all,
            df_real=df_real,
            conditions=conditions,
            runs=runs,
            target_epochs=target_epochs,
            output_dir=os.path.join(FOLDER_PROJECT, 'article_plots')
        )
        
        # Gerar todas as figuras do artigo
        article_plotter.plot_all_figures()   
        
        print(f" Figuras do artigo salvas em: {article_plotter.output_dir}")
        
        # ============================================================
        # 2. GERAR ANÁLISE TEMPORAL - FIGURA E
        # ============================================================
        print("\n" + "="*80)
        print("GERANDO ANÁLISE TEMPORAL - FIGURA E")
        print("="*80)

        print("\n Gerando Figura E - Temporal Dynamics...")

        fig_temporal, stats_df, all_metrics = plot_temporal_analysis(
            data_all=data_all,
            metrics=['g_loss_mean', 'd_loss_mean', 'emd_mean'],
            output_dir=article_plotter.output_dir
        )

        print(f"\n Figura E (Temporal Dynamics) gerada!")
        print(f"   Arquivo: {os.path.join(article_plotter.output_dir, 'figureE_temporal_dynamics.png')}")
        print(f"   Estatísticas: {os.path.join(article_plotter.output_dir, 'temporal_statistics.csv')}")
        print(f"   Dicionário: {os.path.join(article_plotter.output_dir, 'temporal_metrics_dict.json')}")
            
        # ============================================================
        # 3. GERAR RELATÓRIO
        # ============================================================
        print("\n" + "="*80)
        print("GERANDO RELATÓRIO DE RESULTADOS")
        print("="*80)
        
        REPORT_PATH = os.path.join(FOLDER_PROJECT, 'relatorio_analise.txt')

        # ============================================================
        # 4. RESUMO FINAL
        # ============================================================
        print("\n" + "="*80)
        print("RESUMO FINAL")
        print("="*80)
        print(f"\n Relatório salvo em: {REPORT_PATH}")
        print(f" Figuras da análise salvas em: {article_plotter.output_dir}")
        print(f" Figuras do artigo salvas em: {article_plotter.output_dir}")
        print("\n Arquivos gerados:")
        print(f"  - {os.path.basename(REPORT_PATH)}")
        
        # Listar arquivos da pasta de plots
        for file in sorted(os.listdir(article_plotter.output_dir)):
            if file.endswith('.png') or file.endswith('.csv') or file.endswith('.json'):
                print(f"  - {file}")
        
        print("\n" + "="*80)
        print(" ANÁLISE CONCLUÍDA!")
        print("="*80)
        
        logging.info(f"\n Análise concluída! Plots salvos em: {article_plotter.output_dir}")
        return True
        
    except Exception as e:
        logging.error(f"Erro durante a execução: {e}")
        import traceback
        traceback.print_exc()
        return None




# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================
if __name__ == "__main__":
    # Simplesmente chama a função main()
    main('config.yaml')







# Link no colab:
#  https://colab.research.google.com/drive/13smtUuQNj3qQExXt75FKR1oiEWb87GCv#scrollTo=0QyvogFWns1w