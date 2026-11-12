"""EN: Kaggle uses the same training CLI. FR: Kaggle utilise la même commande d'entraînement.

Install requirements-core.txt first; retain Kaggle's GPU-enabled PyTorch.
Installer requirements-core.txt au préalable ; conserver le PyTorch GPU de Kaggle.
"""

import sys

from src.main import main

if __name__ == "__main__":
    main(["train", *sys.argv[1:]])
