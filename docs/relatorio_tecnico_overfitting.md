# Relatório técnico — Mudanças de código para reduzir overfitting e *domain shift*

**Projeto:** Classificação de PCOS em ultrassom com ResNet-50 (TCC)
**Escopo:** `src/model.py`, `src/dataset.py`, `src/transforms_config.py`, `src/train.py`, `src/evaluate.py`, `src/contar_classes.py`
**Base de análise:** commit `dffd95c`, logs em `outputs/logs/` e figuras em `outputs/figures/`

---

## 1. Resumo executivo

O modelo atual mostra um **overfitting leve** entre treino e validação. O problema mais sério, porém, é outro: **a generalização para datasets externos é ruim** (AUC de 0,96 no teste interno contra 0,68–0,72 nos externos), e há **sinais fortes de vazamento de dados** (*data leakage*) e de **atalhos de domínio** (*shortcut learning*).

As mudanças propostas se dividem em três níveis:

| Prioridade | Tema | Impacto esperado |
|---|---|---|
| **P0** | Deduplicação, correção de bugs na avaliação, calibração do limiar | Métricas **confiáveis**; especificidade externa sobe sem retreinar |
| **P1** | Pré-processamento sem atalhos, augmentation para ultrassom, otimização regularizada | Redução do *generalization gap* |
| **P2** | k-fold, ensemble, treino multi-domínio, organização do código | Robustez estatística e qualidade metodológica |

---

## 2. Diagnóstico

### 2.1 Métricas atuais

`outputs/logs/train_log.csv` (melhor época = 8, early stopping na época 23):

| Época | train_loss | train_auc | val_loss | val_auc |
|---|---|---|---|---|
| 1 | 0,319 | 0,743 | 0,279 | 0,800 |
| **8** | 0,249 | 0,847 | **0,264** | **0,835** |
| 15 | 0,230 | 0,873 | 0,306 | 0,815 |
| 23 | 0,208 | 0,896 | 0,341 | 0,805 |

`outputs/logs/evaluation_results.csv`:

| Conjunto | n | AUC | F1 | Acurácia | Sensibilidade | Especificidade |
|---|---|---|---|---|---|---|
| PCOSGen-test | 1.468 | **0,960** | 0,940 | 0,913 | 0,953 | 0,813 |
| Kaggle | 3.856 | 0,716 | 0,564 | 0,527 | 0,751 | **0,374** |
| Figshare | 11.784 | 0,680 | 0,689 | 0,624 | 0,724 | **0,488** |

### 2.2 Interpretação

1. **Overfitting clássico (leve).** A partir da época 8 a `val_loss` sobe (0,264 → 0,341) enquanto a `train_loss` continua caindo. O early stopping por AUC já guarda o modelo da época 8, então esse efeito está contido.
2. **Anomalia no teste interno.** A AUC no PCOSGen-test (0,960) está **12 pontos acima** da melhor AUC de validação (0,835). O teste deveria ser no máximo tão bom quanto a validação. As explicações mais prováveis são imagens duplicadas ou quase duplicadas entre o PCOSGen-train e o PCOSGen-test, ou um teste com distribuição mais fácil. Nas próprias figuras de amostra (`03_pcosgen_samples.png`) há imagens praticamente idênticas **dentro** do PCOSGen-train.
3. **Sobreposição entre datasets.** A mesma imagem de útero aparece nas amostras do PCOSGen (coluna 3), do Kaggle (coluna 3) e do Figshare (coluna 1). O Kaggle e o Figshare contêm versões aumentadas (rotação, faixas cinzas, recortes) de um mesmo conjunto. Se uma mesma imagem tiver **rótulos diferentes** entre datasets, existe ruído de rótulo irredutível.
4. **Erro sistemático nos externos.** A especificidade é muito baixa (0,37 e 0,49) e a sensibilidade razoável: o modelo prevê PCOS demais. Isso é coerente com **limiar 0,5 descalibrado**, já que o `pos_weight` desloca as probabilidades, e com mudança de prevalência e de domínio.
5. **Atalhos visuais.** As imagens diferem em proporção, tinta de cor (algumas azuladas), bordas, réguas, marcadores e textos do aparelho. Com `Resize((224, 224))` e entrada RGB, o modelo pode aprender a "reconhecer o dataset" em vez de reconhecer a anatomia.

---

## 3. Mudanças P0 — corretude e confiabilidade das métricas

### 3.1 Novo arquivo `src/dedup.py` — detecção de duplicatas por hash perceptual

**Problema:** não há nenhuma verificação de duplicatas. A divisão treino/validação e as avaliações externas podem estar contaminadas.

**Mudança:** criar um script que calcula o pHash de cada imagem, encontra pares com distância de Hamming ≤ `THRESHOLD` e gera:
- `outputs/dedup/pcosgen_train_groups.csv`: `imagePath, group_id`, usado na divisão por grupo (§3.2);
- `outputs/dedup/cross_overlap.csv`: imagens externas ou de teste que são cópias de imagens do treino.

```python
"""
dedup.py
Detecta imagens duplicadas/quase duplicadas via perceptual hash (pHash).
Dependencia: pip install imagehash
"""

from pathlib import Path

import numpy as np
import pandas as pd
import imagehash
from PIL import Image

THRESHOLD = 6  # distancia de Hamming maxima (0-64) para considerar duplicata
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def phash_uint64(path):
    """pHash de 64 bits como inteiro sem sinal (tons de cinza)."""
    h = imagehash.phash(Image.open(path).convert("L"))
    bits = h.hash.flatten().astype(np.uint8)
    return np.packbits(bits).view(">u8")[0].astype(np.uint64)


def hash_paths(paths):
    return np.array([phash_uint64(p) for p in paths], dtype=np.uint64)


def hamming(a, b_array):
    """Distancia de Hamming entre um hash e um array de hashes."""
    x = np.bitwise_xor(b_array, a)
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def group_duplicates(hashes, threshold=THRESHOLD):
    """Union-find: imagens a <= threshold bits de distancia ficam no mesmo grupo."""
    parent = list(range(len(hashes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(hashes)):
        d = hamming(hashes[i], hashes[i + 1:])
        for j in np.nonzero(d <= threshold)[0] + i + 1:
            parent[find(j)] = find(i)
    return np.array([find(i) for i in range(len(hashes))])


def cross_matches(train_hashes, other_hashes, threshold=THRESHOLD):
    """Para cada imagem de 'other', menor distancia ate qualquer imagem de treino."""
    return np.array([hamming(h, train_hashes).min() for h in other_hashes])
```

**Como usar no TCC:**
- reportar quantos grupos de duplicatas existem no PCOSGen-train;
- reportar a porcentagem do PCOSGen-test, do Kaggle e do Figshare com `dist <= THRESHOLD` em relação ao treino;
- reavaliar os modelos **excluindo** essas imagens e apresentar as duas versões da tabela.

> **Limitação:** o pHash não detecta cópias rotacionadas ou espelhadas. Para cobri-las, calcule o hash também de `img.transpose(FLIP_LEFT_RIGHT)` e de `img.rotate(90/180/270)` e use a menor distância.

### 3.2 `src/dataset.py` — divisão por grupo, não por imagem

**Problema (linha 44):** `train_test_split` estratifica por rótulo, mas imagens duplicadas podem cair uma no treino e outra na validação.

**Mudança:** usar `StratifiedGroupKFold` com o `group_id` gerado pelo `dedup.py`.

```python
from sklearn.model_selection import StratifiedGroupKFold


def split_train_val(df, groups, n_splits=5, fold=0, seed=42):
    """
    Divide treino/validacao garantindo que duplicatas (mesmo group_id)
    fiquem sempre do mesmo lado. n_splits=5 -> ~20% de validacao.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(sgkf.split(df, df["label"], groups=groups))
    train_idx, val_idx = splits[fold]
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
    )
```

A mesma função atende à validação cruzada k-fold (§5.1): basta iterar `fold` de 0 a 4.

**Extra:** concentrar aqui a leitura de rótulos do PCOSGen (treino em `.xlsx`, teste em `.csv`), que hoje está duplicada em `evaluate.py:186-195`:

```python
def load_pcosgen_labels(labels_path):
    """Le rotulos do PCOSGen (xlsx ou csv). Healthy=1 -> label 0 (normal)."""
    labels_path = Path(labels_path)
    if labels_path.suffix.lower() == ".csv":
        df = pd.read_csv(labels_path)
    else:
        df = pd.read_excel(labels_path)
    df.columns = [c.strip() for c in df.columns]
    df["label"] = 1 - df["Healthy"].astype(int)
    return df[["imagePath", "label"]]
```

### 3.3 `src/evaluate.py` — `FolderDataset` aceita arquivos que não são imagens

**Problema (linhas 95–97 e 104–108):**
- `folder.iterdir()` inclui **qualquer arquivo**, como `Thumbs.db` e `desktop.ini`, comuns no Windows (o próprio `.gitignore` lista os dois);
- arquivos ilegíveis viram uma **imagem preta com o rótulo da pasta**, que entra nas métricas sem aviso.

**Mudança:** filtrar por extensão, validar as imagens na inicialização e reportar as descartadas.

```python
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class FolderDataset(Dataset):
    """Dataset com pastas por classe (Kaggle e Figshare)."""

    def __init__(self, root_dirs, class_map, transform=None):
        self.transform = transform
        self.samples = []
        self.skipped = []
        for class_name, folders in root_dirs.items():
            label = class_map[class_name]
            for folder in folders:
                for img_path in sorted(Path(folder).iterdir()):
                    if img_path.suffix.lower() not in IMG_EXTS:
                        continue
                    try:
                        with Image.open(img_path) as im:
                            im.verify()
                    except Exception:
                        self.skipped.append(img_path)
                        continue
                    self.samples.append((img_path, label))
        if self.skipped:
            print(f"  [aviso] {len(self.skipped)} imagens corrompidas descartadas")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
```

Inclua `n_skipped` no CSV de resultados para documentar quantas imagens foram descartadas.

### 3.4 `src/train.py` + `src/evaluate.py` — limiar de decisão calibrado

**Problema (`train.py:76,103` e `evaluate.py:130`):** o limiar fixo de 0,5 ignora o efeito do `pos_weight` (≈ razão negativos/positivos). O sintoma é especificidade 0,37 no Kaggle e 0,49 no Figshare.

**Mudança:** ao final do treino, calcular na **validação** o limiar que maximiza o índice de Youden (sensibilidade + especificidade − 1), salvar no checkpoint e usá-lo em todas as avaliações.

Em `train.py`:

```python
from sklearn.metrics import roc_curve


def youden_threshold(labels, logits):
    probs = 1 / (1 + np.exp(-np.asarray(logits)))
    fpr, tpr, thr = roc_curve(labels, probs)
    return float(thr[np.argmax(tpr - fpr)])
```

`evaluate()` passa a retornar também `all_labels` e `all_logits`. No salvamento do melhor modelo:

```python
threshold = youden_threshold(val_labels, val_logits)
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "val_auc": val_auc,
    "val_f1": val_f1,
    "threshold": threshold,
}, MODEL_PATH)
```

Em `evaluate.py`:

```python
threshold = checkpoint.get("threshold", 0.5)
...
preds = (probs > threshold).long()
```

> O limiar **nunca** deve ser escolhido no teste nem nos datasets externos; isso seria vazamento. A AUC não depende do limiar, então a mudança afeta apenas F1, acurácia, sensibilidade e especificidade.

### 3.5 Pequenas correções em `src/evaluate.py`

| Linha | Problema | Correção |
|---|---|---|
| 57–75 | `CsvLabeledDataset` duplica `PCOSGenTrainDataset` | Importar de `dataset.py` |
| 186–195 | Inversão de rótulo duplicada | Usar `load_pcosgen_labels` (§3.2) |
| 249, 254, 259 | `pin_memory=True` fixo | `pin_memory=(device.type == "cuda")` |
| 234 | `weights_only=False` | Usar `weights_only=True`: o checkpoint só tem tensores e escalares |

---

## 4. Mudanças P1 — redução do *generalization gap*

### 4.1 `src/transforms_config.py` — pré-processamento sem atalhos de domínio

**Problemas:**
- `Resize((224, 224))` (linhas 21 e 30) **distorce a proporção**. Como cada dataset tem um formato típico, isso vira uma assinatura de domínio;
- a entrada RGB permite aprender a tinta de cor de aparelhos específicos;
- a augmentation atual (flip + rotação de 10°) é fraca para cerca de 2.560 imagens;
- a remoção completa do `ColorJitter` (v2) eliminou também brilho e contraste, que **simulam ganho e configurações de aparelhos diferentes**. Só hue e saturação são irrelevantes no ultrassom.

**Mudança proposta:**

```python
import random

import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224


class PadToSquare:
    """Completa a imagem com preto ate ficar quadrada (preserva proporcao)."""

    def __call__(self, img):
        w, h = img.size
        side = max(w, h)
        canvas = Image.new(img.mode, (side, side), 0)
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        return canvas


class SpeckleNoise:
    """Ruido multiplicativo tipico de ultrassom, igual nos 3 canais."""

    def __init__(self, std=0.1, p=0.3):
        self.std = std
        self.p = p

    def __call__(self, t):
        if random.random() >= self.p:
            return t
        noise = torch.randn_like(t[:1]) * self.std
        return (t * (1 + noise)).clamp(0, 1)


def get_train_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        PadToSquare(),
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.3),
        transforms.ToTensor(),
        SpeckleNoise(std=0.1, p=0.3),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        PadToSquare(),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
```

**Opcionais** (avaliar como ablação):
- **Recorte da região útil:** remover bordas pretas ou cinzas, réguas e textos com uma máscara simples (limiar de intensidade + maior componente conexo) antes do `PadToSquare`. As faixas cinzas do Kaggle e do Figshare são atalhos evidentes.
- **CLAHE** (`cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))`), aplicado igualmente a **todos** os datasets para padronizar o contraste entre aparelhos.

### 4.2 `src/model.py` — suporte a congelamento por estágio e LR por camada

**Problema:** `freeze_backbone` só permite tudo ou nada. Fazer fine-tuning completo de cerca de 23,5 milhões de parâmetros com cerca de 2.560 imagens favorece a memorização.

**Mudança:** adicionar funções para (a) liberar estágios da ResNet progressivamente e (b) separar grupos de parâmetros com learning rates diferentes.

```python
STAGES = ["layer4", "layer3", "layer2", "layer1"]  # do mais profundo ao mais raso


def set_trainable_stages(model, n_stages):
    """
    Congela o backbone e libera a cabeca (fc) + os n_stages mais profundos.
    n_stages=0 -> so a cabeca; n_stages=4 -> rede inteira (exceto conv1/bn1).
    """
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    for name in STAGES[:n_stages]:
        for p in getattr(model, name).parameters():
            p.requires_grad = True


def get_param_groups(model, lr_backbone, lr_head):
    head_ids = {id(p) for p in model.fc.parameters()}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return [
        {"params": backbone, "lr": lr_backbone},
        {"params": list(model.fc.parameters()), "lr": lr_head},
    ]
```

O otimizador recebe **todos** os parâmetros desde o início. Parâmetros congelados ficam com `grad=None` e o AdamW os ignora, então não é preciso recriar o otimizador a cada descongelamento.

> **Nota sobre BatchNorm:** em `model.train()`, as camadas BN congeladas continuam atualizando `running_mean` e `running_var`. Se isso causar instabilidade, coloque os módulos BN dos estágios congelados em `eval()` depois de cada `model.train()`.

Também vale testar uma arquitetura menor (ResNet-18 ou EfficientNet-B0) como comparação, para avaliar se a capacidade do modelo é parte do problema.

### 4.3 `src/train.py` — otimização regularizada

| Linha | Atual | Proposto | Motivo |
|---|---|---|---|
| 163 | `Adam(weight_decay=1e-3)` | `AdamW(weight_decay=0.05)` | No Adam, o weight decay vira L2 acoplado e regulariza pouco |
| 163–167 | LR único `1e-4` | backbone `1e-5`, cabeça `1e-3` | Preserva as features do ImageNet |
| — | Fine-tuning completo desde a época 1 | Descongelamento gradual | Evita destruir as features no início, quando a cabeça ainda é aleatória |
| 162 | BCE sem suavização | Label smoothing (ε = 0,05) | Reduz excesso de confiança e o efeito de rótulos ruidosos |
| 168–174, 195 | Scheduler por `val_auc` | Scheduler por `val_loss` | A AUC com 640 imagens oscila ±0,02 por época |
| 222 | Seleção por `val_auc` | Seleção por `val_loss` (registrar ambas) | Critério mais estável e coerente com a calibração |
| 239–242 | Log gravado só no fim | Gravar a cada época | Não perde o histórico se o treino for interrompido |
| 54–55 | Só `torch.manual_seed` e `np.random.seed` | Adicionar `random.seed`, `torch.cuda.manual_seed_all`, `worker_init_fn` e `generator` no `DataLoader` | Reprodutibilidade completa |

**Trechos:**

```python
# hiperparametros
LR_BACKBONE = 1e-5
LR_HEAD = 1e-3
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.05
UNFREEZE_SCHEDULE = {1: 0, 4: 1, 7: 2, 10: 4}  # epoca -> n de estagios liberados
```

```python
# label smoothing para BCE: 0 -> eps/2, 1 -> 1 - eps/2
def smooth(labels, eps=LABEL_SMOOTHING):
    return labels * (1 - eps) + 0.5 * eps

# em train_one_epoch:
loss = criterion(logits, smooth(labels))
# em evaluate() a loss usa os rotulos originais (sem suavizacao)
```

```python
model = build_resnet50(pretrained=True).to(device)
optimizer = torch.optim.AdamW(
    get_param_groups(model, LR_BACKBONE, LR_HEAD),
    weight_decay=WEIGHT_DECAY,
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7,
)

for epoch in range(1, NUM_EPOCHS + 1):
    if epoch in UNFREEZE_SCHEDULE:
        set_trainable_stages(model, UNFREEZE_SCHEDULE[epoch])
    ...
    scheduler.step(val_loss)
```

```python
# log incremental
with open(LOG_PATH, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=row.keys())
    if epoch == 1:
        writer.writeheader()
    writer.writerow(row)
```

O arquivo de log deve ser apagado ou recriado no início de `main()`.

Como opção, **mixup** (α = 0,2) pode substituir ou complementar o label smoothing. Teste um de cada vez para medir o efeito isolado.

---

## 5. Mudanças P2 — protocolo experimental e organização

### 5.1 Validação cruzada k-fold + ensemble

Com cerca de 640 imagens de validação, uma única divisão gera estimativas muito ruidosas. Proposta:
- envolver `main()` num laço `for fold in range(5)`, usando o `split_train_val` por grupo (§3.2);
- salvar `best_model_fold{k}.pt` e o log de cada fold;
- reportar **média ± desvio-padrão** da AUC de validação;
- em `evaluate.py`, avaliar cada fold e também o **ensemble** (média das probabilidades dos cinco modelos), com o limiar médio dos folds.

### 5.2 Treino multi-domínio (*leave-one-domain-out*)

É o experimento com maior chance de melhorar a generalização externa, e ele exige a deduplicação (§3.1) concluída antes, para não haver vazamento entre os domínios.

| Experimento | Treino | Teste externo |
|---|---|---|
| E1 (atual) | PCOSGen | Kaggle, Figshare |
| E2 | PCOSGen + Kaggle | Figshare |
| E3 | PCOSGen + Figshare | Kaggle |

Para isso, `train.py` precisa aceitar uma lista de fontes. O `FolderDataset` já serve para o Kaggle e o Figshare; basta concatenar com `torch.utils.data.ConcatDataset`. Os `pos_weight` devem ser recalculados sobre o conjunto combinado.

### 5.3 Test-time augmentation (opcional)

Em `evaluate_dataset`, fazer a média dos logits da imagem original e da espelhada:

```python
logits = (model(images) + model(torch.flip(images, dims=[3]))).squeeze(1) / 2
```

### 5.4 Organização do código

- **`src/config.py`**: concentrar os caminhos `G:/tcc/...` hoje repetidos em seis arquivos e permitir sobrescrevê-los por variável de ambiente (`TCC_DATA_DIR`). Isso facilita rodar em outra máquina ou no Colab.
- **`src/contar_classes.py:66-69`**: o ternário interno nunca troca de leitor (o `if` externo já garante `.xlsx`/`.xls`). Simplificar para `pd.read_excel` / `pd.read_csv` direto, ou reutilizar `load_pcosgen_labels`.
- **`requirements.txt`**: fixar versões de `torch`, `torchvision`, `scikit-learn`, `pandas`, `openpyxl` e `imagehash` para reprodutibilidade do TCC.

---

## 6. Plano de experimentos (ablação)

Rode cada linha mantendo as anteriores e registre sempre as métricas do PCOSGen-test **deduplicado**, do Kaggle e do Figshare.

| # | Mudança | Arquivos | Retreina? |
|---|---|---|---|
| 0 | Baseline atual reavaliado com `FolderDataset` corrigido | `evaluate.py` | Não |
| 1 | + limiar calibrado na validação | `train.py`, `evaluate.py` | Sim* |
| 2 | + deduplicação e divisão por grupo | `dedup.py`, `dataset.py` | Sim |
| 3 | + grayscale + `PadToSquare` | `transforms_config.py` | Sim |
| 4 | + augmentation para ultrassom | `transforms_config.py` | Sim |
| 5 | + AdamW + LR por camada + descongelamento gradual | `model.py`, `train.py` | Sim |
| 6 | + label smoothing | `train.py` | Sim |
| 7 | + 5-fold + ensemble | `train.py`, `evaluate.py` | Sim |
| 8 | + treino multi-domínio (E2/E3) | `train.py` | Sim |

\* O limiar pode ser estimado sem retreinar: basta carregar `best_model.pt`, refazer a mesma divisão de validação (`seed=42`) e calcular o Youden.

**Métricas a acompanhar:** AUC (principal), sensibilidade, especificidade, F1 e o *gap* `AUC_interna − AUC_externa`. Para comparar modelos no mesmo conjunto, use IC 95% da AUC via *bootstrap* (1.000 reamostragens) ou o teste de DeLong.

---

## 7. Checklist de implementação

- [ ] Criar `src/dedup.py` e gerar os relatórios de duplicatas
- [ ] `dataset.py`: `load_pcosgen_labels` unificado + `split_train_val` por grupo
- [ ] `evaluate.py`: filtro de extensão, verificação de imagens e contagem de descartadas
- [ ] `evaluate.py`: reutilizar o dataset de `dataset.py`; corrigir `pin_memory`
- [ ] `train.py`: calcular e salvar o limiar de Youden no checkpoint
- [ ] `evaluate.py`: usar o limiar do checkpoint
- [ ] `transforms_config.py`: grayscale, `PadToSquare`, nova augmentation
- [ ] `model.py`: `set_trainable_stages` e `get_param_groups`
- [ ] `train.py`: AdamW, LR por grupo, descongelamento gradual, label smoothing, scheduler por `val_loss`
- [ ] `train.py`: log incremental e seeds completas
- [ ] `train.py` / `evaluate.py`: laço k-fold + ensemble
- [ ] `train.py`: suporte a múltiplas fontes de treino
- [ ] `config.py` e `requirements.txt`

---

## 8. Riscos e limitações

- **Ruído de rótulo entre datasets.** Se a mesma imagem tiver rótulos diferentes no PCOSGen e no Kaggle ou Figshare, parte do *gap* é irredutível. O `dedup.py` permite medir isso: cruze os rótulos dos pares encontrados e reporte a taxa de discordância.
- **As métricas atuais tendem a cair** depois da deduplicação, principalmente no PCOSGen-test. Isso é esperado e desejável: os novos números refletem melhor a capacidade real do modelo e fortalecem a discussão metodológica do TCC.
- **Augmentation agressiva** pode reduzir a AUC interna. O critério de sucesso deve ser o desempenho **externo**.
- **Limite do pHash:** cópias com rotação, recorte ou faixas adicionadas podem escapar (§3.1). Ajuste `THRESHOLD` inspecionando visualmente uma amostra dos pares encontrados.
