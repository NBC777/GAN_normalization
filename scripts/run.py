import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import gc
# wgangp_with_mmd_emd.py
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import wasserstein_distance

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from joblib import load



# ---------------------------
# Utilitários: MMD (RBF mix) e EMD (marginal avg)
# ---------------------------

def compute_mmd_rbf(x: np.ndarray, y: np.ndarray, sigma_list=(0.5, 1.0, 2.0, 4.0)):
    """
    Unbiased MMD^2 estimator with mixture of RBF kernels.
    x, y: numpy arrays with shape (n_samples, n_features)
    Returns scalar MMD^2 (float, >=0).
    """
    # defensive copies and shapes
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nx = x.shape[0]
    ny = y.shape[0]
    if nx < 2 or ny < 2:
        return float(0.0)

    def rbf_kernel(a, b, sigma):
        # squared distances: ||a-b||^2 = a2 + b2 - 2 a.b
        a2 = np.sum(a * a, axis=1)[:, None]
        b2 = np.sum(b * b, axis=1)[None, :]
        dist2 = a2 + b2 - 2.0 * (a @ b.T)
        return np.exp(-dist2 / (2.0 * (sigma ** 2)))

    Kxx = np.zeros((nx, nx), dtype=np.float64)
    Kyy = np.zeros((ny, ny), dtype=np.float64)
    Kxy = np.zeros((nx, ny), dtype=np.float64)

    for s in sigma_list:
        Kxx += rbf_kernel(x, x, s)
        Kyy += rbf_kernel(y, y, s)
        Kxy += rbf_kernel(x, y, s)

    # unbiased estimators: exclude diagonal for Kxx and Kyy
    sum_xx = (np.sum(Kxx) - np.trace(Kxx)) / (nx * (nx - 1))
    sum_yy = (np.sum(Kyy) - np.trace(Kyy)) / (ny * (ny - 1))
    sum_xy = np.sum(Kxy) / (nx * ny)
    mmd2 = sum_xx + sum_yy - 2.0 * sum_xy
    # numerical safety
    return float(max(mmd2, 0.0))

def approx_multivariate_emd(real: np.ndarray, fake: np.ndarray):
    """
    Approximate multivariate EMD by averaging 1D Wasserstein distances over features.
    real, fake: numpy arrays (n_samples, n_features)
    Returns (mean_emd, per_feature_list)
    """
    real = np.asarray(real)
    fake = np.asarray(fake)
    n_features = real.shape[1]
    emds = []
    for i in range(n_features):
        # scipy.stats.wasserstein_distance handles different lengths
        emd_i = wasserstein_distance(real[:, i], fake[:, i])
        emds.append(float(emd_i))
    return float(np.mean(emds)), emds

# ---------------------------
# Gradient penalty (WGAN-GP)
# ---------------------------
def gradient_penalty(critic, real_data, fake_data, device='cpu', lambda_gp=10.0):
    batch_size = real_data.size(0)
    # epsilon shape (B,1) -> expand to (B,D)
    epsilon = torch.rand(batch_size, 1, device=device)
    epsilon = epsilon.expand_as(real_data)
    interpolated = (epsilon * real_data + (1 - epsilon) * fake_data).to(device)
    interpolated.requires_grad_(True)

    crit_out = critic(interpolated)
    # flatten output to (B,)
    if crit_out.dim() > 1:
        crit_out = crit_out.view(-1)

    grad_outputs = torch.ones_like(crit_out, device=device)
    gradients = torch.autograd.grad(
        outputs=crit_out,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]  # (B, D)
    gradients = gradients.view(batch_size, -1)
    grad_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gp = ((grad_norm - 1.0) ** 2).mean()
    return lambda_gp * gp

# ---------------------------
# Models: Generator & Critic (no Sigmoid in critic)
# ---------------------------
def weights_init_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class CriticMLP(nn.Module):
    def __init__(self, input_dim=9, hidden_dims=(128, 64, 32)):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            last = h
        layers.append(nn.Linear(last, 1))  # linear output: critic score
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # shape (B,1)

class GeneratorMLP(nn.Module):
    def __init__(self, latent_dim=50, output_dim=9, hidden_dims=(128, 256, 128)):
        super().__init__()
        layers = []
        last = latent_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            last = h
        layers.append(nn.Linear(last, output_dim))
        layers.append(nn.Tanh())  # assume data normalized to [-1,1]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)  # shape (B, D)


# ---------------------------
# Trainer with MMD & EMD integrated
# ---------------------------
class WGAN_GP_Trainer_WithMetrics_:
    def __init__(
        self,
        generator: nn.Module,
        critic: nn.Module,
        latent_dim: int = 50,
        lr_G: float = 2e-4,
        lr_C: float = 1e-4,
        betas=(0.5, 0.9),
        lambda_gp: float = 10.0,
        device: str = None,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._set_seeds(seed)
        self.generator = generator.to(self.device)
        self.critic = critic.to(self.device)
        self.latent_dim = latent_dim
        self.lr_G = lr_G
        self.lr_C = lr_C
        self.betas = betas
        self.lambda_gp = lambda_gp
        self.verbose = verbose
        self.seed = seed

        # init weights
        self.generator.apply(weights_init_xavier)
        self.critic.apply(weights_init_xavier)


    def _set_seeds(self, seed):
        """Fixar todas as seeds para garantir reprodutibilidade"""
        torch.manual_seed(seed)  # PyTorch CPU seed
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)  # PyTorch GPU seed
            torch.cuda.manual_seed_all(seed) # All GPUs seed, se usar múltiplas GPUs
        np.random.seed(seed)  #  # NumPy seed
        random.seed(seed)  # Python random seed
        torch.backends.cudnn.deterministic = True  # Garante que o cudnn seja determinístico
        torch.backends.cudnn.benchmark = False  # Desativa o benchmarking para evitar não determinismos
        print(f"Random seed set as {seed}")

    def _clear_cache(self):
        """Limpar cache de GPU e CPU"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()


    def _clip_generator_output(self, tensor):
        """Clip generator output to [-1, 1] range for Tanh compatibility"""
        return torch.clamp(tensor, -1.0, 1.0)

    #
    def train(self, df,  n_epochs: int = 2000,
        batch_size: int = 32,
        n_critic: int = 5,
        save_every: int = 400,
        output_dir: str = "/out_results",
        mmd_sigma_list=(0.5, 1.0, 2.0, 4.0),
        eval_samples: int = None,
        run_id: int = 0
        ):
        """Executa uma única run de treinamento"""

        # --- Diretório específico para esta run ---
        run_output_dir = os.path.join(output_dir, f"run_{run_id}")
        os.makedirs(run_output_dir, exist_ok=True)

        # --- Seed fixa e única para esta run ---
        run_seed = self.seed + run_id * 1000
        print(f"\n========== Iniciando RUN {run_id} | Seed: {run_seed} ==========")
        self._set_seeds(run_seed)

        # --- Preparação dos dados ---
        X = df.values.astype(np.float32)
        n_data = X.shape[0]
        eval_samples = eval_samples or min(1024, n_data)

        dataset = TensorDataset(torch.from_numpy(X))
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(run_seed)
        )

        # --- Otimizadores ---
        opt_G = optim.Adam(self.generator.parameters(), lr=self.lr_G, betas=self.betas)
        opt_C = optim.Adam(self.critic.parameters(), lr=self.lr_C, betas=self.betas)

        # --- Armazenamento de métricas ---
        records = []
        global_step = 0

        pbar_epochs = tqdm(range(n_epochs), desc=f"Run {run_id} | Epochs")
        for epoch in pbar_epochs:
            epoch_start = time.time()
            d_losses, g_losses, gp_vals = [], [], []

            # --- Treino do Crítico (n_critic vezes) ---
            for i, (real_batch,) in enumerate(dataloader):
                real_batch = real_batch.to(self.device)
                b_size = real_batch.size(0)

                for _ in range(n_critic):
                    z = torch.randn(b_size, self.latent_dim, device=self.device)
                    fake_batch = self.generator(z).detach()

                    opt_C.zero_grad()
                    d_real = self.critic(real_batch).view(-1).mean()
                    d_fake = self.critic(fake_batch).view(-1).mean()
                    gp = gradient_penalty(self.critic, real_batch, fake_batch, device=self.device, lambda_gp=self.lambda_gp)

                    d_loss = d_fake - d_real + gp
                    d_loss.backward()
                    opt_C.step()

                    d_losses.append(float(d_loss.item()))
                    gp_vals.append(float(gp.item()))

                # --- Treino do Gerador ---
                z = torch.randn(b_size, self.latent_dim, device=self.device)
                opt_G.zero_grad()
                fake_for_g = self.generator(z)
                g_loss = -self.critic(fake_for_g).view(-1).mean()
                g_loss.backward()
                opt_G.step()

                g_losses.append(float(g_loss.item()))
                global_step += 1

            # --- Avaliação (MMD e EMD) ---
            with torch.no_grad():
                n_eval = eval_samples
                torch.manual_seed(run_seed + epoch * 9999)
                z_eval = torch.randn(n_eval, self.latent_dim, device=self.device)
                self.generator.eval()
                fake_eval = self._clip_generator_output(self.generator(z_eval))
                fake_eval = fake_eval.cpu().numpy()
                self.generator.train()

                np.random.seed(run_seed + epoch * 8888)
                idx = np.random.choice(n_data, size=n_eval, replace=(n_eval > n_data))
                real_eval = X[idx, :]

            fake_min, fake_max = fake_eval.min(), fake_eval.max()
            if fake_min < -1.1 or fake_max > 1.1 and self.verbose:
                print(f"Warning: Generated data outside [-1,1] range: [{fake_min:.3f}, {fake_max:.3f}] - Clipping applied")
                fake_eval = np.clip(fake_eval, -1.0, 1.0)

            mmd_val = compute_mmd_rbf(real_eval, fake_eval, sigma_list=mmd_sigma_list)
            emd_mean, emd_per_feat = approx_multivariate_emd(real_eval, fake_eval)

            rec = {
                "epoch": epoch + 1,
                "time_epoch_s": time.time() - epoch_start,
                "d_loss_mean": np.mean(d_losses) if d_losses else None,
                "g_loss_mean": np.mean(g_losses) if g_losses else None,
                "gp_mean": np.mean(gp_vals) if gp_vals else None,
                "mmd": float(mmd_val),
                "emd_mean": float(emd_mean),
                "n_eval": int(n_eval)
            }
            records.append(rec)

            if self.verbose and ((epoch + 1) % 500 == 0 or epoch == 0 or (epoch + 1) == n_epochs):
                print(f"Epoch {epoch+1}/{n_epochs} | D={rec['d_loss_mean']:.4f} | G={rec['g_loss_mean']:.4f} | GP={rec['gp_mean']:.4f} | MMD={rec['mmd']:.6f} | EMD={rec['emd_mean']:.6f}")

            # --- Checkpoints e métricas ---
            if (epoch + 1) % save_every == 0 or (epoch + 1) == n_epochs:
                ckpt = {
                    "generator_state_dict": self.generator.state_dict(),
                    "critic_state_dict": self.critic.state_dict(),
                    "opt_G": opt_G.state_dict(),
                    "opt_C": opt_C.state_dict(),
                    "epoch": epoch + 1,
                    "records": records,
                    "run_id": run_id,
                    "seed": run_seed
                }
                torch.save(ckpt, os.path.join(run_output_dir, f"ckpt_epoch{epoch+1}.pt"))

                with torch.no_grad():
                    n_sample = min(1024, n_data)
                    torch.manual_seed(run_seed + 12345)
                    z = torch.randn(n_sample, self.latent_dim, device=self.device)
                    fake_samples = self._clip_generator_output(self.generator(z)).cpu().numpy()
                pd.DataFrame(fake_samples, columns=df.columns).to_csv(
                    os.path.join(run_output_dir, f"samples_epoch{epoch+1}.csv"), index=False
                )

                pd.DataFrame(records).to_csv(os.path.join(run_output_dir, "metrics_per_epoch.csv"), index=False)
   
            pbar_epochs.set_postfix({
                "last_mmd": rec["mmd"],
                "last_emd": rec["emd_mean"],
                "epoch_time_s": rec["time_epoch_s"]
            })

        # --- Finalização da run ---
        pd.DataFrame(records).to_csv(os.path.join(run_output_dir, "metrics_final.csv"), index=False)

        # --- Limpeza de cache ---
        self._clear_cache()
        torch.cuda.empty_cache()
        print(f" Run {run_id} finalizada e cache limpo.\n")

        return self.generator, self.critic, pd.DataFrame(records)
    

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # Exemplo de uso (substitua pelo seu dataframe normalizado)
    # import pandas as pd
    

    FOLDER_DATAS = '/home/nbc/Documentos/test_py/GAN_normalization/Datas/'
    FOLDER_RESULTS = '/home/nbc/Documentos/test_py/GAN_normalization/results/'

    # Caminho do scaler salvo
    scaler_path =  os.path.join(FOLDER_DATAS, "scaler_01.pkl")  

    # Colunas originais (ordem exata usada no treino)
    col_numerical_df = ['226Ra', '232Th', '40K', 'Raeq', 'Theq', 'Keq', 'IA', 'IB', 'IG']

    df_norm4_ = pd.read_csv(FOLDER_DATAS + "data_norm_05.csv")  # 9 colunas
    G = GeneratorMLP(latent_dim=40, output_dim=9, hidden_dims=(256,256))   
    C = CriticMLP(input_dim=9, hidden_dims=(256,256))

    trainer4_ = WGAN_GP_Trainer_WithMetrics_(G, C, latent_dim=40, lr_G=1e-4, lr_C=1e-4, lambda_gp=5.0, device ='cuda', seed = 42)
    #G_trained2, C_trained, metrics_df2 = trainer1_.train(df_norm1_, n_epochs=2500, batch_size=32, n_critic=5, save_every=300)


    for run_id in range(7):     
        trainer4_.train(
            df= df_norm4_,  
            n_epochs=4500,       
            batch_size=32,
            run_id=run_id,
            output_dir=  os.path.join(FOLDER_RESULTS, "out_results/results_05")
    )

    pass   
            

        
#   0:  42,  1: 1042,  2: 2042  3:  3042    4: 4042   5:  5042   6:   6042
