import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from pathlib import Path




def plot_training_metrics(csv_path):
    """
    Plota métricas de treinamento a partir de um arquivo CSV.
    
    Parâmetros:
    -----------
    csv_path : str
        Caminho para o arquivo CSV com as colunas:
        epoch, d_loss_mean, g_loss_mean, gp_mean, mmd, emd_mean
    """
    
    # Ler o arquivo CSV
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {csv_path}")
        return
    except Exception as e:
        print(f"Erro ao ler o arquivo CSV: {e}")
        return
    
    # Verificar se as colunas necessárias existem
    required_columns = ['epoch', 'd_loss_mean', 'g_loss_mean', 'gp_mean', 'mmd', 'emd_mean']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Erro: Colunas faltando no CSV: {missing_columns}")
        print(f"Colunas disponíveis: {list(df.columns)}")
        return
    
    # Criar figura com 3 subgráficos
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle('Métricas de Treinamento', fontsize=16, fontweight='bold')
    
    # Subgráfico 1: d_loss_mean vs g_loss_mean
    ax1 = axes[0]
    ax1.plot(df['epoch'], df['d_loss_mean'], 'b-', label='Discriminator Loss (d_loss_mean)', linewidth=2)
    ax1.plot(df['epoch'], df['g_loss_mean'], 'r-', label='Generator Loss (g_loss_mean)', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Discriminator vs Generator Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subgráfico 2: gp_mean
    ax2 = axes[1]
    ax2.plot(df['epoch'], df['gp_mean'], 'g-', label='Gradient Penalty (gp_mean)', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Gradient Penalty')
    ax2.set_title('Gradient Penalty')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Subgráfico 3: mmd vs emd_mean
    ax3 = axes[2]
    
    # Plotar mmd e emd_mean em escalas diferentes (se necessário)
    ax3.plot(df['epoch'], df['mmd'], 'purple', label='MMD', linewidth=2)
    ax3.plot(df['epoch'], df['emd_mean'], 'orange', label='EMD (emd_mean)', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Valor')
    ax3.set_title('MMD vs Earth Mover\'s Distance (EMD)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Ajustar layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    return fig, axes

# Função auxiliar para visualização rápida
def plot_and_show(csv_path):
    """
    Plota os gráficos e os mostra imediatamente.
    """
    fig, axes = plot_training_metrics(csv_path)
    if fig is not None:
        plt.show()

# Função para salvar os gráficos
def plot_and_save(csv_path, output_path='training_metrics.png', dpi=300):
    """
    Plota os gráficos e os salva em um arquivo.
    
    Parâmetros:
    -----------
    csv_path : str
        Caminho para o arquivo CSV
    output_path : str
        Caminho para salvar a figura
    dpi : int
        Resolução da imagem salva
    """
    fig, axes = plot_training_metrics(csv_path)
    if fig is not None:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Gráficos salvos em: {output_path}")
        plt.close(fig)



def plot_scatter_and_kde_multiple_csv(lista_csv, df_columns, output_dir='output_plots'):
    """
    Plota scatter plots e histogramas KDE para cada coluna de múltiplos CSVs.
    
    Parâmetros:
    -----------
    lista_csv : list
        Lista de caminhos para arquivos CSV
    df_columns : list
        Lista de colunas para plotar
    output_dir : str
        Diretório para salvar os gráficos
    """
    
    # Criar diretório de saída se não existir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Verificar se os arquivos existem
    valid_csv_files = []
    for csv_path in lista_csv:
        if os.path.exists(csv_path):
            valid_csv_files.append(csv_path)
        else:
            print(f"Aviso: Arquivo não encontrado - {csv_path}")
    
    if not valid_csv_files:
        print("Erro: Nenhum arquivo CSV válido encontrado.")
        return
    
    # Processar cada arquivo CSV
    for csv_idx, csv_path in enumerate(valid_csv_files):
        try:
            # Ler o arquivo CSV
            df = pd.read_csv(csv_path)
            csv_name = Path(csv_path).stem  # Nome do arquivo sem extensão
            
            print(f"Processando {csv_name}...")
            
            # Verificar se as colunas existem no DataFrame
            available_columns = [col for col in df_columns if col in df.columns]
            missing_columns = [col for col in df_columns if col not in df.columns]
            
            if missing_columns:
                print(f"  Aviso: Colunas faltando em {csv_name}: {missing_columns}")
            
            if not available_columns:
                print(f"  Erro: Nenhuma das colunas especificadas encontrada em {csv_name}")
                continue
            
            # Para cada coluna, criar um gráfico com scatter plot e KDE
            for col_idx, column in enumerate(available_columns):
                # Criar figura com 2 subgráficos
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                fig.suptitle(f'{csv_name} - Coluna: {column}', fontsize=16, fontweight='bold')
                
                # Subgráfico 1: Scatter plot
                ax1 = axes[0]
                
                # Gerar índices para o eixo x do scatter plot
                indices = np.arange(len(df[column]))
                
                # Scatter plot
                scatter = ax1.scatter(indices, df[column], alpha=0.6, 
                                    c=indices, cmap='viridis', edgecolors='black', linewidth=0.5)
                
                # Linha de tendência (opcional)
                ax1.plot(indices, df[column], 'r-', alpha=0.3, linewidth=1)
                
                ax1.set_xlabel('Índice')
                ax1.set_ylabel(column)
                ax1.set_title(f'Scatter Plot - {column}')
                ax1.grid(True, alpha=0.3)
                
                # Adicionar barra de cores para mostrar a sequência
                plt.colorbar(scatter, ax=ax1, label='Ordem dos pontos')
                
                # Subgráfico 2: Histograma com KDE
                ax2 = axes[1]
                
                # Histograma
                n_bins = min(50, len(df[column]) // 10)
                n_bins = max(10, n_bins)  # Pelo menos 10 bins
                
                sns.histplot(df[column], kde=True, bins=n_bins, ax=ax2, 
                           color='skyblue', edgecolor='black', alpha=0.7)
                
                # Adicionar linhas verticais para estatísticas importantes
                mean_val = df[column].mean()
                median_val = df[column].median()
                std_val = df[column].std()
                
                ax2.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                          label=f'Média: {mean_val:.4f}')
                ax2.axvline(median_val, color='green', linestyle='--', linewidth=2, 
                          label=f'Mediana: {median_val:.4f}')
                ax2.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5)
                ax2.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5,
                          label=f'±1 std: ±{std_val:.4f}')
                
                ax2.set_xlabel(column)
                ax2.set_ylabel('Frequência')
                ax2.set_title(f'Distribuição - {column}')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # Adicionar estatísticas no gráfico
                stats_text = (f'Média: {mean_val:.4f}\n'
                            f'Mediana: {median_val:.4f}\n'
                            f'Std: {std_val:.4f}\n'
                            f'Min: {df[column].min():.4f}\n'
                            f'Max: {df[column].max():.4f}\n'
                            f'N: {len(df[column])}')
                
                # Posicionar texto no canto superior direito
                ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
                        fontsize=9, verticalalignment='top',
                        horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                # Ajustar layout
                plt.tight_layout()
                
                # Salvar figura
                safe_column_name = column.replace('/', '_').replace('\\', '_')
                output_path = os.path.join(output_dir, f'{csv_name}_{safe_column_name}.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                print(f"  Gráfico salvo: {output_path}")
        
        except Exception as e:
            print(f"Erro ao processar {csv_path}: {e}")
            continue
    
    print(f"\nProcessamento concluído! Gráficos salvos em: {output_dir}")
    
    # Criar também um relatório consolidado
    create_summary_report(valid_csv_files, df_columns, output_dir)

def create_summary_report(csv_files, df_columns, output_dir):
    """
    Cria um relatório resumido com estatísticas de todos os CSVs.
    """
    summary_data = []
    
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            csv_name = Path(csv_path).stem
            
            for column in df_columns:
                if column in df.columns:
                    col_data = df[column]
                    summary_data.append({
                        'Arquivo': csv_name,
                        'Coluna': column,
                        'Média': col_data.mean(),
                        'Mediana': col_data.median(),
                        'Desvio Padrão': col_data.std(),
                        'Mínimo': col_data.min(),
                        'Máximo': col_data.max(),
                        'Contagem': len(col_data),
                        'Q1': col_data.quantile(0.25),
                        'Q3': col_data.quantile(0.75)
                    })
        except Exception as e:
            print(f"Erro ao criar relatório para {csv_path}: {e}")
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        report_path = os.path.join(output_dir, 'summary_statistics.csv')
        summary_df.to_csv(report_path, index=False)
        print(f"Relatório estatístico salvo: {report_path}")
        
        # Criar também uma versão formatada em TXT
        txt_path = os.path.join(output_dir, 'summary_statistics.txt')
        with open(txt_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("RELATÓRIO DE ESTATÍSTICAS - ANÁLISE DE COLUNAS\n")
            f.write("=" * 60 + "\n\n")
            
            for csv_path in csv_files:
                csv_name = Path(csv_path).stem
                f.write(f"Arquivo: {csv_name}\n")
                f.write("-" * 40 + "\n")
                
                try:
                    df = pd.read_csv(csv_path)
                    for column in df_columns:
                        if column in df.columns:
                            col_data = df[column]
                            f.write(f"\nColuna: {column}\n")
                            f.write(f"  Média: {col_data.mean():.6f}\n")
                            f.write(f"  Mediana: {col_data.median():.6f}\n")
                            f.write(f"  Desvio Padrão: {col_data.std():.6f}\n")
                            f.write(f"  Mínimo: {col_data.min():.6f}\n")
                            f.write(f"  Máximo: {col_data.max():.6f}\n")
                            f.write(f"  Q1 (25%): {col_data.quantile(0.25):.6f}\n")
                            f.write(f"  Q3 (75%): {col_data.quantile(0.75):.6f}\n")
                            f.write(f"  Contagem: {len(col_data)}\n")
                except:
                    f.write("  Erro ao ler arquivo\n")
                f.write("\n" + "=" * 60 + "\n\n")
        
        print(f"Relatório detalhado salvo: {txt_path}")

# Função auxiliar para visualização interativa
def plot_interactive_view(lista_csv, df_columns, max_columns=4):
    """
    Cria uma visualização interativa com todos os gráficos em uma única figura.
    Útil para comparação rápida.
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    # Criar figura com múltiplos subplots
    n_csvs = min(len(lista_csv), 3)  # Limitar a 3 CSVs para visualização
    n_cols = min(len(df_columns), max_columns)
    
    fig, axes = plt.subplots(n_csvs * 2, n_cols, figsize=(5 * n_cols, 4 * n_csvs * 2))
    
    if n_csvs == 1:
        axes = axes.reshape(2, -1)
    
    # Processar cada CSV
    for csv_idx, csv_path in enumerate(lista_csv[:n_csvs]):
        try:
            df = pd.read_csv(csv_path)
            csv_name = Path(csv_path).stem
            
            for col_idx, column in enumerate(df_columns[:n_cols]):
                if column in df.columns:
                    # Scatter plot (linha par)
                    ax_scatter = axes[csv_idx * 2, col_idx]
                    indices = np.arange(len(df[column]))
                    ax_scatter.scatter(indices, df[column], alpha=0.5, s=20)
                    ax_scatter.set_title(f'{csv_name[:15]}... - {column}')
                    ax_scatter.grid(True, alpha=0.3)
                    
                    # Histograma KDE (linha ímpar)
                    ax_hist = axes[csv_idx * 2 + 1, col_idx]
                    sns.histplot(df[column], kde=True, ax=ax_hist, color='lightblue')
                    ax_hist.axvline(df[column].mean(), color='red', linestyle='--', alpha=0.7)
                    ax_hist.grid(True, alpha=0.3)
        
        except Exception as e:
            print(f"Erro no modo interativo para {csv_path}: {e}")
    
    plt.tight_layout()
    plt.show()

#######################################



def plot_gan_vs_real_comparison(fake_csv_paths, real_csv_path, columns, out_path):
    """
    Gera histogramas KDE e scatter plots comparando dados fake (GAN) com dados reais.
    
    Parâmetros:
    -----------
    fake_csv_paths : list
        Lista de caminhos para arquivos CSV com dados fake
    real_csv_path : str
        Caminho para arquivo CSV com dados reais
    columns : list
        Lista de colunas/variáveis a serem analisadas
    out_path : str
        Diretório onde os gráficos serão salvos
    """
    
    # Criar diretório de saída se não existir
    os.makedirs(out_path, exist_ok=True)
    
    # Carregar dados reais
    print(f"Carregando dados reais: {real_csv_path}")
    real_data = pd.read_csv(real_csv_path)
    
    # Verificar se as colunas existem nos dados reais
    missing_cols = [col for col in columns if col not in real_data.columns]
    if missing_cols:
        print(f"Aviso: Colunas não encontradas nos dados reais: {missing_cols}")
        columns = [col for col in columns if col in real_data.columns]
    
    # Configurações de estilo
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Processar cada arquivo fake
    for i, fake_path in enumerate(fake_csv_paths):
        print(f"Processando arquivo fake {i+1}/{len(fake_csv_paths)}: {fake_path}")
        
        # Carregar dados fake
        fake_data = pd.read_csv(fake_path)
        
        # Verificar se as colunas existem nos dados fake
        fake_missing_cols = [col for col in columns if col not in fake_data.columns]
        if fake_missing_cols:
            print(f"Aviso: Colunas não encontradas nos dados fake: {fake_missing_cols}")
            continue
        
        # Filtrar apenas as colunas especificadas
        real_subset = real_data[columns].copy()
        fake_subset = fake_data[columns].copy()
        
        # Adicionar coluna de identificação
        real_subset['Dataset'] = 'Real'
        fake_subset['Dataset'] = 'Fake (GAN)'
        
        # Preparar dados combinados para scatter plot
        combined_data = pd.concat([real_subset, fake_subset], ignore_index=True)
        
        # Obter nome do arquivo para usar nos títulos
        fake_filename = Path(fake_path).stem
        
        # 1. PLOTAR HISTOGRAMAS KDE POR VARIÁVEL
        print("  Gerando histogramas KDE...")
        num_cols = len(columns)
        num_rows = int(np.ceil(num_cols / 3))  # 3 colunas por linha
        
        fig_kde, axes_kde = plt.subplots(num_rows, 3, figsize=(18, 5*num_rows))
        axes_kde = axes_kde.flatten() if num_cols > 1 else [axes_kde]
        
        for idx, col in enumerate(columns):
            ax = axes_kde[idx]
            
            # Plot KDE para dados reais
            sns.kdeplot(data=real_subset[col], ax=ax, label='Real', 
                       fill=True, alpha=0.3, linewidth=2)
            
            # Plot KDE para dados fake
            sns.kdeplot(data=fake_subset[col], ax=ax, label='Fake (GAN)', 
                       fill=True, alpha=0.3, linewidth=2)
            
            ax.set_title(f'Distribuição de {col}', fontsize=14, fontweight='bold')
            ax.set_xlabel(col, fontsize=12)
            ax.set_ylabel('Densidade', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Adicionar estatísticas
            stats_text = (f"Real: μ={real_subset[col].mean():.2f}, σ={real_subset[col].std():.2f}\n"
                         f"Fake: μ={fake_subset[col].mean():.2f}, σ={fake_subset[col].std():.2f}")
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Remover eixos extras se houver
        for idx in range(len(columns), len(axes_kde)):
            fig_kde.delaxes(axes_kde[idx])
        
        fig_kde.suptitle(f'Comparação de Distribuições - {fake_filename}\nReal vs Fake (GAN)', 
                        fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Salvar histograma KDE
        kde_filename = os.path.join(out_path, f'kde_comparison_{fake_filename}.png')
        fig_kde.savefig(kde_filename, dpi=300, bbox_inches='tight')
        print(f"    Histograma KDE salvo: {kde_filename}")
        
        # 2. PLOTAR SCATTER PLOT MATRIX (para até 6 variáveis)
        print("  Gerando scatter plots...")
        if len(columns) <= 6:  # Limitar para não ficar muito grande
            # Scatter plot matrix colorida por dataset
            fig_scatter = plt.figure(figsize=(15, 15))
            
            # Criar scatter matrix
            from pandas.plotting import scatter_matrix
            
            ax = scatter_matrix(combined_data[columns], 
                               alpha=0.6, 
                               figsize=(15, 15), 
                               diagonal='hist',
                               grid=True,
                               marker='.',
                               density_kwds={'alpha': 0.3},
                               hist_kwds={'alpha': 0.3})
            
            # Colorir por dataset
            colors = {'Real': 'blue', 'Fake (GAN)': 'red'}
            
            # Para cada subplot, replotar com cores diferentes
            for i in range(len(columns)):
                for j in range(len(columns)):
                    if i != j:  # Não é diagonal
                        # Limpar subplot
                        ax[i, j].clear()
                        
                        # Plotar real
                        real_idx = combined_data['Dataset'] == 'Real'
                        ax[i, j].scatter(combined_data.loc[real_idx, columns[j]], 
                                        combined_data.loc[real_idx, columns[i]],
                                        alpha=0.5, s=10, label='Real', 
                                        color='blue', marker='o')
                        
                        # Plotar fake
                        fake_idx = combined_data['Dataset'] == 'Fake (GAN)'
                        ax[i, j].scatter(combined_data.loc[fake_idx, columns[j]], 
                                        combined_data.loc[fake_idx, columns[i]],
                                        alpha=0.5, s=10, label='Fake (GAN)', 
                                        color='red', marker='x')
                        
                        # Configurar labels
                        if i == len(columns) - 1:
                            ax[i, j].set_xlabel(columns[j], fontsize=10)
                        if j == 0:
                            ax[i, j].set_ylabel(columns[i], fontsize=10)
                        
                        ax[i, j].grid(True, alpha=0.3)
                        ax[i, j].legend(fontsize=8, loc='upper right')
            
            fig_scatter.suptitle(f'Scatter Matrix - {fake_filename}\nReal (azul) vs Fake (vermelho)', 
                                fontsize=16, fontweight='bold', y=0.95)
            plt.tight_layout()
            
            # Salvar scatter matrix
            scatter_filename = os.path.join(out_path, f'scatter_matrix_{fake_filename}.png')
            fig_scatter.savefig(scatter_filename, dpi=300, bbox_inches='tight')
            plt.close(fig_scatter)
            print(f"    Scatter matrix salvo: {scatter_filename}")
        
        # 3. PLOTAR SCATTER PLOTS INDIVIDUAIS (para pares de variáveis)
        if len(columns) >= 2:
            print("  Gerando scatter plots individuais...")
            # Criar pares de variáveis para scatter plots
            variable_pairs = []
            for i in range(len(columns)):
                for j in range(i+1, len(columns)):
                    variable_pairs.append((columns[i], columns[j]))
            
            # Limitar número de pares para não gerar muitos gráficos
            max_pairs = min(9, len(variable_pairs))
            variable_pairs = variable_pairs[:max_pairs]
            
            num_pairs = len(variable_pairs)
            num_rows_scatter = int(np.ceil(num_pairs / 3))
            
            fig_scatter_ind, axes_scatter = plt.subplots(num_rows_scatter, 3, 
                                                        figsize=(18, 5*num_rows_scatter))
            axes_scatter = axes_scatter.flatten() if num_pairs > 1 else [axes_scatter]
            
            for idx, (x_col, y_col) in enumerate(variable_pairs):
                ax = axes_scatter[idx]
                
                # Plotar dados reais
                ax.scatter(real_subset[x_col], real_subset[y_col], 
                          alpha=0.6, s=20, label='Real', 
                          color='blue', marker='o')
                
                # Plotar dados fake
                ax.scatter(fake_subset[x_col], fake_subset[y_col], 
                          alpha=0.6, s=20, label='Fake (GAN)', 
                          color='red', marker='x')
                
                ax.set_xlabel(x_col, fontsize=12)
                ax.set_ylabel(y_col, fontsize=12)
                ax.set_title(f'{y_col} vs {x_col}', fontsize=13, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Calcular e mostrar coeficiente de correlação
                corr_real = real_subset[x_col].corr(real_subset[y_col])
                corr_fake = fake_subset[x_col].corr(fake_subset[y_col])
                
                stats_text = f"Corr Real: {corr_real:.3f}\nCorr Fake: {corr_fake:.3f}"
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Remover eixos extras
            for idx in range(len(variable_pairs), len(axes_scatter)):
                fig_scatter_ind.delaxes(axes_scatter[idx])
            
            fig_scatter_ind.suptitle(f'Scatter Plots Individuais - {fake_filename}\nReal (azul) vs Fake (vermelho)', 
                                    fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            # Salvar scatter plots individuais
            scatter_ind_filename = os.path.join(out_path, f'scatter_individual_{fake_filename}.png')
            fig_scatter_ind.savefig(scatter_ind_filename, dpi=300, bbox_inches='tight')
            print(f"    Scatter plots individuais salvos: {scatter_ind_filename}")
        
        plt.close('all')
        print(f"  Processamento completo para {fake_filename}")
        print("-" * 50)
    
    print(f"\nTodos os gráficos foram salvos em: {out_path}")

# Função auxiliar para gerar relatório estatístico
def generate_statistical_report(fake_csv_paths, real_csv_path, columns, out_path):
    """
    Gera um relatório estatístico comparativo entre dados fake e reais
    """
    os.makedirs(out_path, exist_ok=True)
    
    real_data = pd.read_csv(real_csv_path)
    
    report_data = []
    
    for fake_path in fake_csv_paths:
        fake_data = pd.read_csv(fake_path)
        fake_filename = Path(fake_path).stem
        
        for col in columns:
            if col in real_data.columns and col in fake_data.columns:
                # Estatísticas descritivas
                real_stats = real_data[col].describe()
                fake_stats = fake_data[col].describe()
                
                # Diferença percentual nas médias
                mean_diff_pct = ((fake_stats['mean'] - real_stats['mean']) / real_stats['mean']) * 100
                
                report_data.append({
                    'Arquivo_Fake': fake_filename,
                    'Variável': col,
                    'Média_Real': real_stats['mean'],
                    'Média_Fake': fake_stats['mean'],
                    'Diff_Média_%': mean_diff_pct,
                    'Std_Real': real_stats['std'],
                    'Std_Fake': fake_stats['std'],
                    'Min_Real': real_stats['min'],
                    'Min_Fake': fake_stats['min'],
                    'Max_Real': real_stats['max'],
                    'Max_Fake': fake_stats['max']
                })
    
    report_df = pd.DataFrame(report_data)
    report_filename = os.path.join(out_path, 'statistical_report.csv')
    report_df.to_csv(report_filename, index=False)
    
    print(f"Relatório estatístico salvo: {report_filename}")
    return report_df


if __name__ == "__main__":
    # Exemplo 1: Apenas visualizar os gráficos
    FOLDER_0 = "/home/nbc/Documentos/test_py/GAN_normalization/results/out_results_05/results_05/run_0"
    
    #csv_path = os.path.join(FOLDER_0,'metrics_final.csv')   
    #plot_and_show(csv_path)  
    #png_out = os.path.join(FOLDER_0, 'plot_training.png')
    # Exemplo 2: Salvar os gráficos em um arquivo
    #plot_and_save(csv_path, png_out)
       
    # Exemplo 3: Usar a função principal e customizar
    # fig, axes = plot_training_metrics(csv_path)
    # if fig is not None:
    #     # Personalizar ainda mais os gráficos se necessário
    #     axes[0].set_yscale('log')  # Exemplo: usar escala log no primeiro gráfico
    #     plt.show()

    FOLDER_plot= os.path.join(FOLDER_0, 'amostras_scatter')

    # Exemplo 1: Processar múltiplos CSVs
    lista_ =['samples_epoch3600.csv', 'samples_epoch4000.csv', 'samples_epoch4400.csv']

    lista_csv = [ os.path.join(FOLDER_0, file_) for file_ in lista_]
    real_csv_path = os.path.join('/home/nbc/Documentos/test_py/GAN_normalization/Datas', 'data_norm_05.csv')
    # Colunas para analisar
    #df_columns = ['d_loss_mean', 'g_loss_mean', 'gp_mean', 'mmd', 'emd_mean']
    df_col_metrics = ['226Ra','IA','232Th','Raeq','Theq', 'Keq', 'IG', 'IB', '40K']

    # Chamar a função principal    
    #plot_scatter_and_kde_multiple_csv(
    #    lista_csv=lista_csv,
    #    df_columns=df_col_metrics,
    #    output_dir=os.path.join(FOLDER_plot, 'analise_graficos')
    #)

    # Gerar gráficos
    plot_gan_vs_real_comparison(lista_csv, real_csv_path, df_col_metrics,FOLDER_plot)
    
    # Gerar relatório estatístico (opcional)
    report = generate_statistical_report(lista_csv, real_csv_path, df_col_metrics, FOLDER_plot)
