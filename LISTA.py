from __future__ import annotations
import torch
import torch.utils.data as Data
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)

def use_lista(Xtr, Xts, W_d) -> None:
    Xtr = Xtr.T
    Xts = Xts.T
    W_d = W_d.T
    Xtr = torch.as_tensor(Xtr, dtype=torch.float32)
    Xts = torch.as_tensor(Xts, dtype=torch.float32)
    W_d = torch.as_tensor(W_d, dtype=torch.float32)
    train_loader = create_data_set(W_d, Xtr)
    test_loader = create_data_set(W_d, Xts)
    t_start = 1
    t_end = 13
    T_opt = range(t_start, t_end, 2)

    ista_MSE = []
    lista_MSE = []
    for i in range(len(T_opt)):
        T = T_opt[i]
        print("Current T = ",T_opt[i])
        if i == (len(T_opt)-1):
            ista_MSE.append(ista_apply(test_loader, T, W_d, 0.2, True, Xts[:, 0]))
            lista_MSE.append(lista_apply(train_loader, test_loader, T, W_d, True, Xts[:, 0].unsqueeze(0)))
        else:
            ista_MSE.append(ista_apply(test_loader, T, W_d))
            lista_MSE.append(lista_apply(train_loader, test_loader, T, W_d))

    fig = plt.figure()
    plt.plot(T_opt, ista_MSE, label='ISTA', color='b',linewidth=0.5)
    plt.plot(T_opt, lista_MSE, label='LISTA', color='r', linewidth=2)
    plt.xlabel('Number of iterations')
    plt.ylabel('MSE')
    plt.yscale("log")
    plt.legend()
    plt.show()


def ista(X, W_d, rho=0.2, L=1, max_itr=3000) -> torch.Tensor:
    z = torch.zeros(W_d.shape[1], dtype=W_d.dtype, device=W_d.device)
    soft_threshold = torch.nn.Softshrink(lambd = rho / L)
    for i in range(max_itr):
        z_tild = z - 1/L * (W_d.T @ (W_d @ z - X))
        z = soft_threshold(z_tild)
    return z


def create_data_set(W_d, X, batch_size=40) -> Data.DataLoader:
    N = X.shape[1]
    n = W_d.shape[0]
    m = W_d.shape[1]

    L = torch.linalg.eigvalsh(W_d.T @ W_d).max().item()
    z = torch.zeros(m, N)
    for i in range(N):
        z[:, i] = ista(X=X[:, i], W_d=W_d, L=L)

    data_set = ECG_Data_Set(X=X, W_d=W_d, z=z)
    data_loader = Data.DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True)
    return data_loader


def train(model, train_loader, test_loader, num_epochs=30):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=5e-05,
        momentum=0.9,
        weight_decay=0,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.1
    )
    loss_train = np.zeros((num_epochs,))
    loss_test = np.zeros((num_epochs,)) 
    # Main loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for step, (b_X, b_Wd, b_z) in enumerate(train_loader):
            z_hat = model(b_X)
            loss = F.mse_loss(z_hat, b_z, reduction="mean")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.zero_grad()
            train_loss += loss.data.item()
        loss_train[epoch] = train_loss / len(train_loader)
        scheduler.step()

        # validation
        model.eval()
        test_loss = 0
        for step, (b_x, b_Wd, b_s) in enumerate(test_loader):
            z_hat = model(b_x)
            test_loss += F.mse_loss(z_hat, b_s, reduction="mean").item()
        loss_test[epoch] = test_loss / len(test_loader)
        
        # Print
        if epoch % 10 == 0:
            print(
                "Epoch %d, Train loss %.8f, Validation loss %.8f"
                % (epoch, loss_train[epoch], loss_test[epoch])
            )

    return loss_test


def lista_apply(train_loader, test_loader, T, W_d, demo_activation: bool = False, X_demo=None):
    n = W_d.shape[0]
    m = W_d.shape[1]
    lista = LISTA_Model(W_d=W_d, T=T)
    loss_test = train(lista, train_loader, test_loader)

    if demo_activation:
        lista.eval()
        with torch.no_grad():
            z_demo = lista(X_demo)
        x_recon = z_demo @ W_d.T
        X_plot = X_demo.squeeze(0).cpu().numpy()
        x_recon_plot = x_recon.squeeze(0).cpu().numpy()
        plt.figure(figsize=(10, 4))
        plt.plot(X_plot, label="Original X_demo", linewidth=1)
        plt.plot(x_recon_plot, label="Reconstructed x_recon", linewidth=1)
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.title(f"LISTA Reconstruction Comparison (T={T})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    err_lista = loss_test[-1]
    return err_lista


def ista_apply(test_loader, T, W_d, rho=0.2, demo_activation: bool = False, X_demo=None):
    n = W_d.shape[0]
    m = W_d.shape[1]
    L = torch.linalg.eigvalsh(W_d.T @ W_d).max().item()

    loss = 0
    for step, (X, _, z) in enumerate(test_loader.dataset):
        z_hat = ista(X=X, W_d=W_d, rho=rho, L=L, max_itr=T)
        loss += F.mse_loss(z_hat, z, reduction="sum").data.item()

    if demo_activation:
        z_demo = ista(X=X_demo, W_d=W_d, rho=rho, L=L, max_itr=T)
        x_recon = z_demo @ W_d.T
        X_plot = X_demo.squeeze(0).cpu().numpy()
        x_recon_plot = x_recon.squeeze(0).cpu().numpy()
        plt.figure(figsize=(10, 4))
        plt.plot(X_plot, label="Original X_demo", linewidth=1)
        plt.plot(x_recon_plot, label="Reconstructed x_recon", linewidth=1)
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.title(f"ISTA Reconstruction Comparison (T={T})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return loss/len(test_loader.dataset)


class ECG_Data_Set(Data.Dataset): 
    def __init__(self, X, W_d, z): 
        self.X = X
        self.z = z
        self.W_d = W_d

    def __len__(self):
        return self.X.shape[1]

    def __getitem__(self, idx):
        X = self.X[:, idx]
        W_d = self.W_d
        z = self.z[:, idx]
        return X, W_d, z


class LISTA_Model(nn.Module):
    def __init__(self, W_d: torch.Tensor, T: int = 6, rho: float = 0.5):
        """
        W_d: shape [n, m]
           n = signal length
           m = sparse code dimension / number of dictionary atoms

        T: number of LISTA layers / iterations
        rho: sparsity regularization parameter
        """
        super().__init__()
        self.n, self.m = W_d.shape
        self.T = T
        self.rho = rho

        # Lipschitz constant L = largest eigenvalue of W_d.T @ W_d
        with torch.no_grad():
            L = torch.linalg.eigvalsh(W_d.T @ W_d).max().item()
            W_e_init = W_d.T / L
            S_init = torch.eye(self.m, dtype=W_d.dtype, device=W_d.device) - (W_d.T @ W_d) / L
            theta_init = torch.full(
                (1, self.m),
                fill_value=rho / L,
                dtype=W_d.dtype,
                device=W_d.device,
            )
        self.W_e = nn.Parameter(W_e_init)
        self.S = nn.Parameter(S_init)
        self.theta = nn.Parameter(theta_init)

    def soft_threshold(self, X: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return torch.sign(X) * torch.relu(torch.abs(X) - theta)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: shape [batch_size, n]
        return z: shape [batch_size, m]
        """
        batch_size = X.shape[0]

        z = torch.zeros(
            batch_size,
            self.m,
            dtype=X.dtype,
            device=X.device,
        )

        for _ in range(self.T):
            z = self.soft_threshold(
                X @ self.W_e.T + z @ self.S.T,
                self.theta
            )
        return z