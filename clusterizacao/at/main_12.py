# -*- coding: utf-8 -*-
"""
MNIST t-SNE (leitura local dos arquivos IDX)

Requisitos:
- numpy, matplotlib, scikit-learn

Estrutura esperada:
[este_script].py
└── MNIST /
    ├── train-images-idx3-ubyte
    ├── train-labels-idx1-ubyte
    ├── t10k-images-idx3-ubyte
    └── t10k-labels-idx1-ubyte
"""
import argparse
import random
import struct
from array import array
from pathlib import Path
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ---------------------------
# Leitor MNIST (baseado no código de referência)
# ---------------------------
class MnistDataloader(object):
    def __init__(
            self,
            training_images_filepath: Path,
            training_labels_filepath: Path,
            test_images_filepath: Path,
            test_labels_filepath: Path,
    ):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    @staticmethod
    def _read_images_labels(images_filepath: Path, labels_filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
        # Lê labels
        with open(labels_filepath, "rb") as f:
            magic, size = struct.unpack(">II", f.read(8))
            if magic != 2049:
                raise ValueError(f"Magic number (labels) inválido: esperado 2049, obtido {magic}")
            labels = np.frombuffer(f.read(), dtype=np.uint8)

        # Lê imagens
        with open(images_filepath, "rb") as f:
            magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
            if magic != 2051:
                raise ValueError(f"Magic number (images) inválido: esperado 2051, obtido {magic}")
            image_data = np.frombuffer(f.read(), dtype=np.uint8)
            images = image_data.reshape(size, rows * cols)

        return images, labels

    def load_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        x_train, y_train = self._read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self._read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)


# ---------------------------
# Helpers de visualização
# ---------------------------
def show_images(images: List[np.ndarray], title_texts: List[str], cols: int = 5):
    rows = int(np.ceil(len(images) / cols))
    plt.figure(figsize=(cols * 3, rows * 3))
    for idx, (img, title) in enumerate(zip(images, title_texts), start=1):
        plt.subplot(rows, cols, idx)
        # cada img pode vir 784 (flatten) ou 28x28
        img2d = img.reshape(28, 28) if img.ndim == 1 else img
        plt.imshow(img2d, cmap=plt.cm.gray)
        if title:
            plt.title(title, fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def tsne_scatter(X2d: np.ndarray, y: np.ndarray, title: str = "t-SNE (MNIST)"):
    plt.figure(figsize=(10, 8))
    # Para desempenho gráfico: amostrar mais uma vez (opcional), aqui deixamos todos
    scatter = plt.scatter(X2d[:, 0], X2d[:, 1], c=y, s=5, alpha=0.7, cmap="tab10")
    cbar = plt.colorbar(scatter, ticks=range(10))
    cbar.set_label("Dígitos")
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.show()


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="t-SNE no MNIST lido de pasta local 'MNIST '")
    parser.add_argument("--mnist-dir", type=str, default="MNIST ", help="Pasta que contém os arquivos IDX (padrão: 'MNIST ')")
    parser.add_argument("--n-samples", type=int, default=10000, help="Quantidade de amostras do treino para o t-SNE (padrão: 10000)")
    parser.add_argument("--perplexity", type=float, default=None, help="Perplexity do t-SNE (padrão: automático em função do n-samples)")
    parser.add_argument("--pca-dims", type=int, default=50, help="Dimensões de PCA antes do t-SNE (padrão: 50)")
    parser.add_argument("--random-state", type=int, default=42, help="Seed para reprodutibilidade (padrão: 42)")
    args = parser.parse_args()

    base = Path(__file__).parent
    mnist_dir = (base / args.mnist_dir).resolve()

    # Arquivos esperados
    training_images_filepath = mnist_dir / "train-images-idx3-ubyte"
    training_labels_filepath = mnist_dir / "train-labels-idx1-ubyte"
    test_images_filepath = mnist_dir / "t10k-images-idx3-ubyte"
    test_labels_filepath = mnist_dir / "t10k-labels-idx1-ubyte"

    expected_files = [
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
    ]
    missing = [str(p) for p in expected_files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Não encontrei os arquivos MNIST esperados. Verifique a pasta 'MNIST ':\n" + "\n".join(missing)
        )

    print(f"Lendo MNIST de: {mnist_dir}")
    mnist = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(f"Treino: X={x_train.shape}, y={y_train.shape} | Teste: X={x_test.shape}, y={y_test.shape}")

    # Normalização simples para [0,1]
    x_train_norm = x_train.astype(np.float32) / 255.0
    x_test_norm = x_test.astype(np.float32) / 255.0

    # Mostrar alguns exemplos (como no código de referência)
    imgs_show = []
    titles_show = []
    random.seed(args.random_state)
    for _ in range(10):
        r = random.randint(0, x_train_norm.shape[0] - 1)
        imgs_show.append(x_train_norm[r].reshape(28, 28))
        titles_show.append(f"train idx={r} → {y_train[r]}")
    for _ in range(5):
        r = random.randint(0, x_test_norm.shape[0] - 1)
        imgs_show.append(x_test_norm[r].reshape(28, 28))
        titles_show.append(f"test  idx={r} → {y_test[r]}")
    show_images(imgs_show, titles_show, cols=5)

    # Amostra para t-SNE
    n = min(args.n_samples, x_train_norm.shape[0])
    rng = np.random.default_rng(args.random_state)
    idx = rng.choice(x_train_norm.shape[0], size=n, replace=False)
    X = x_train_norm[idx]
    y = y_train[idx]
    print(f"Amostra para t-SNE: {X.shape[0]} imagens")

    # Redução PCA prévia
    pca_dims = min(args.pca_dims, X.shape[1])
    print(f"Aplicando PCA para {pca_dims} dimensões antes do t-SNE...")
    pca = PCA(n_components=pca_dims, random_state=args.random_state)
    X_pca = pca.fit_transform(X)

    # t-SNE
    # Se perplexity não for informada, escolhemos automaticamente algo seguro:
    #   - deve ser < (n-1)/3; usamos min(50, (n // 3) - 1), com piso 5
    if args.perplexity is None:
        auto_perp = max(5, min(50, (n // 3) - 1))
        perplexity = float(auto_perp)
    else:
        perplexity = float(args.perplexity)

    print(f"Rodando t-SNE (perplexity={perplexity}, random_state={args.random_state})...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",             # ajuda na estabilidade
        learning_rate="auto",   # bom default nas versões recentes
        n_iter=1000,
        verbose=1,
        random_state=args.random_state,
        metric="euclidean",
    )
    X_tsne = tsne.fit_transform(X_pca)
    print("t-SNE concluído.")
    # Algumas libs expõem kl_divergence_; se existir, mostramos:
    kl = getattr(tsne, "kl_divergence_", None)
    if kl is not None:
        print(f"KL divergence final: {kl:.4f}")

    # Plot t-SNE
    tsne_scatter(X_tsne, y, title=f"t-SNE (MNIST) — n={n}, perp={perplexity}")

    # Extra opcional: visualizar componentes principais (2D) para comparação
    print("Plotando PCA 2D para comparação...")
    pca2 = PCA(n_components=2, random_state=args.random_state).fit_transform(X)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca2[:, 0], pca2[:, 1], c=y, s=5, alpha=0.7, cmap="tab10")
    cbar = plt.colorbar(scatter, ticks=range(10))
    cbar.set_label("Dígitos")
    plt.title("PCA (MNIST) — 2 componentes")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()

    print("Finalizado com sucesso.")


if __name__ == "__main__":
    main()
