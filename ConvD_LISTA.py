from __future__ import annotations
import torch
import torch.utils.data as Data
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


'''
原始信号 s:
    shape = (B, 1, N)

卷积稀疏系数 X_ista:
    shape = (B, M, T)

重构信号 A(X_ista):
    shape = (B, 1, T + K - 1)

N = T + K - 1
T = N - K + 1
'''
def use_conv_lista(Xtr, Xts, D) -> None:
    print("Preparation")
    Xtr_target, train_loader = create_data_set(Xtr, D)
    Xts_target, test_loader = create_data_set(Xts, D)
    N = Xtr.shape[2] # signal length

    itr_start = 1
    itr_end = 13
    ITR_opt = range(itr_start, itr_end, 2)

    ista_MSE = []
    lista_MSE = []

    for i in range(len(ITR_opt)):
        itr = ITR_opt[i]
        print("Current T = ",ITR_opt[i])
        if i == (len(ITR_opt)-1):
            ista_MSE.append(conv_ista_apply(Xts, Xts_target, D, N, itr))
            lista_MSE.append(conv_lista_apply(train_loader, test_loader, D, N, itr=itr, demo_activation=True, s_demo = Xts[0:1, :, :]))
        else:
            ista_MSE.append(conv_ista_apply(Xts, Xts_target, D, N, itr))
            lista_MSE.append(conv_lista_apply(train_loader, test_loader, D, N, itr=itr))

    fig = plt.figure()
    plt.plot(ITR_opt, ista_MSE, label='ISTA', color='b',linewidth=0.5)
    plt.plot(ITR_opt, lista_MSE, label='LISTA', color='r', linewidth=2)
    plt.xlabel('Number of iterations')
    plt.ylabel('MSE')
    plt.yscale("log")
    plt.legend()
    plt.show()
    # loss = conv_lista_apply(train_loader, test_loader, D, N, itr=6, demo_activation=True, s_demo = Xts[0:1, :, :])



def conv_ista(s, D, L, rho=0.1, max_itr=300) -> torch.Tensor:
    '''
    s: (B, 1, N)
    X: (B, M, T)
    D: (M, 1, K)
    AX: (B, 1, T + K - 1)
    S(X): (B, M, T)
    '''
    B = s.shape[0] # batch size
    N = s.shape[2] # signal length
    M = D.shape[0] # number of atoms
    K = D.shape[2] # filter length / atom
    T = N - K + 1 # 
    X_ista = torch.zeros(B, M, T, dtype=D.dtype, device=D.device)
    soft_threshold = torch.nn.Softshrink(lambd = rho / L)
    for _ in range(max_itr):
        e = apply_A(X_ista, D) - s
        X_ista_tild = X_ista - 1/L * (apply_AT(e, D))
        X_ista = soft_threshold(X_ista_tild)

    # #Visual display of the target matrix
    # AX = apply_A(X_ista, D)
    # s_plot = s[0, 0, :].detach().cpu().numpy()
    # AX_plot = AX[0, 0, :].detach().cpu().numpy()
    # plt.figure(figsize=(10, 4))
    # plt.plot(s_plot, label="original", linewidth=1)
    # plt.plot(AX_plot, label="s_recon", linewidth=1)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    return X_ista


def create_data_set(s, D, batch_size=40) -> Data.DataLoader:
    '''
    X: (B, M, T)
    D: (M, 1, K)
    '''
    N = s.shape[2] # signal length
    K = D.shape[2] # filter length / atom
    T = N - K + 1 # 

    L = estimate_L(T, D)
    with torch.no_grad():
        X = conv_ista(s=s, D=D, L=L)

    data_set = ECG_Data_Set(s=s, D=D, X=X)
    data_loader = Data.DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True)
    return X, data_loader


def train(model, train_loader, test_loader, num_epochs=30):
    print("training......")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr = 5e-05,
        momentum=0.9,
        weight_decay=0,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    loss_train = np.zeros((num_epochs,))
    loss_test = np.zeros((num_epochs,))
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for step, (b_s, b_D, b_X) in enumerate(train_loader):
            X_hat = model(b_s)
            loss = F.mse_loss(X_hat, b_X, reduction="mean")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.zero_grad()
            train_loss += loss.data.item()
        loss_train[epoch] = train_loss / len(train_loader)
        scheduler.step()

        #validation
        model.eval()
        test_loss = 0
        for step, (b_s, b_D, b_X) in enumerate(test_loader):
            X_hat = model(b_s)
            test_loss += F.mse_loss(X_hat, b_X, reduction="mean").item()
        loss_test[epoch] = test_loss / len(test_loader)
            
        if epoch % 10 == 0:
            print(
                "Epoch %d, Train loss %.8f, Validation loss %.8f"
                % (epoch, loss_train[epoch], loss_test[epoch])
            )

    return loss_test


def conv_ista_apply(Xts, X_target, D, N, itr):
    print("ISTA")
    # X_ista = torch.zeros(B, M, T, dtype=D.dtype, device=D.device)
    K = D.shape[2]
    T = N - K + 1
    L = estimate_L(T, D)

    with torch.no_grad():
        X_hat = conv_ista(s=Xts, D=D, L=L, max_itr=itr)
    loss = F.mse_loss(X_hat, X_target, reduction="mean").item()
    return loss


def conv_lista_apply(train_loader, test_loader, D, N, itr, demo_activation: bool = False, s_demo=None):
    print("LISTA")
    lista = LISTA_Model(D=D, N=N, itr=itr)
    loss_test = train(lista, train_loader, test_loader)

    if demo_activation:
        lista.eval()
        with torch.no_grad():
            X_demo = lista(s_demo)
        AX = apply_A(X_demo, D)
        s_plot = s_demo[0, 0, :].detach().cpu().numpy()
        s_recon_plot = AX[0, 0, :].detach().cpu().numpy()

        plt.figure(figsize=(10, 4))
        plt.plot(s_plot, label="Original signal", linewidth=1)
        plt.plot(s_recon_plot, label="Reconstructed signal", linewidth=1)
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.title(f"Conv_LISTA Reconstruction Comparison (itr={itr})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    err_lista = loss_test[-1]
    return err_lista


def estimate_L(T, D, max_itr = 50):
    M = D.shape[0]
    with torch.no_grad():
        X = torch.randn(1, M, T, device=D.device, dtype=D.dtype)
        X = X / (torch.norm(X) + 1e-12)

        for _ in range(max_itr):
            X = apply_AT(apply_A(X, D), D)
            X = X / (torch.norm(X) + 1e-12)

        AX = apply_A(X, D)
        return torch.sum(AX * AX) / (torch.sum(X * X) + 1e-12)


def apply_A(X, D):
    """
    X: (B, M, T)
    D: (M, 1, K)
    return y: (B, 1, T + K - 1)
    """
    return F.conv_transpose1d(X, D, padding=0)


def apply_AT(y, D):
    """
    y: (B, 1, T + K - 1)
    D: (M, 1, K)
    return X_grad: (B, M, T)
    """
    return F.conv1d(y, D, padding=0)


def apply_S(X, D, L):
    """
    X: (B, M, T)
    D: (M, 1, K)
    L: Lipschitz constant

    return:
        S(X): (B, M, T)
    """
    return X - apply_AT(apply_A(X, D), D) / L


def build_S_kernel(D, L):
    '''
    S = I - A.T A / L
    D: (M, 1, K)
    '''
    M = D.shape[0]
    K = D.shape[2]

    D2 = D[:, 0, :]  # (M, K)

    H = torch.zeros(
        M,
        M,
        2 * K - 1,
        dtype=D.dtype,
        device=D.device
    )

    center = K - 1
    for i in range(M):          # output atom
        for j in range(M):      # input atom
            for u in range(K):
                for v in range(K):
                    lag = u - v
                    H[i, j, center + lag] += D2[i, u] * D2[j, v]

    S_kernel = -H / L

    # Add identity operator I
    for m in range(M):
        S_kernel[m, m, center] += 1.0

    return S_kernel


class ECG_Data_Set(Data.Dataset): 
    def __init__(self, s, D, X): 
        self.s = s
        self.D = D
        self.X = X

    def __len__(self):
        return self.s.shape[0]

    def __getitem__(self, idx):
        return self.s[idx], self.D, self.X[idx]


class LISTA_Model(nn.Module):
    def __init__(self, D: torch.Tensor, N: int, itr: int = 6, rho: float = 0.1):
        """
        D: (M, 1, K)

        itr: number of LISTA layers / iterations
        rho: sparsity regularization parameter
        """
        super().__init__()
        # B = s.shape[0] # batch size
        # N = s.shape[2] # signal length
        self.D = D
        self.M = D.shape[0] # number of atoms
        self.K = D.shape[2] # filter length / atom
        self.N = N
        self.T = self.N - self.K + 1
        self.itr = itr
        self.rho = rho

        # Lipschitz constant L = largest eigenvalue of A.T @ A
        L = estimate_L(self.T, self.D)
        with torch.no_grad():
            theta_init = torch.ones(
                1,
                self.M,
                1,
                dtype=D.dtype,
                device=D.device,
            ) * (rho / L)

        self.theta = nn.Parameter(theta_init)
        self.W_e = nn.Conv1d(
            in_channels=1,
            out_channels=self.M,
            kernel_size=self.K,
            padding=0,
            bias=False
        ).to(device=D.device, dtype=D.dtype)
        self.S = nn.Conv1d(
            in_channels=self.M,
            out_channels=self.M,
            kernel_size=2 * self.K - 1,
            padding=self.K - 1,
            bias=False
        ).to(device=D.device, dtype=D.dtype)

        with torch.no_grad():
            self.W_e.weight.copy_(D / L)
            S_init = build_S_kernel(D, L)
            self.S.weight.copy_(S_init)


    def soft_threshold(self, X: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return torch.sign(X) * torch.relu(torch.abs(X) - theta)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        X: (B, M, T)
        """
        B = s.shape[0]
        X = torch.zeros(
            B,
            self.M,
            self.T,
            dtype=s.dtype,
            device=s.device,
        )
        
        We_s = self.W_e(s)
        for _ in range(self.itr):
            X = self.soft_threshold(
                We_s + self.S(X),
                torch.abs(self.theta)
            )
        return X
