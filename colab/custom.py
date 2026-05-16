# %% [markdown]
# # Atividade: CNNs para Classificação
# 
# Neste notebook, iremos preparar nosso próprio dataset e treinar um modelo de classificação de imagens.

# %% [markdown]
# ## Preparando os dados
# 
# Os dados desta atividade serão baixados da internet. Utilizaremos para isso buscadores comuns. Em seguida, dividiremos em treinamento e validação.

# %%
import os
import shutil
import random
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler

# %% [markdown]
# ### Adquirindo as Imagens
# 
# Utilizaremos o iCrawler para baixar imagens em buscadores através de termos especificados. Defina sua lista de classes.

# %%
def download_images(keyword, folder, n_total=100):
    os.makedirs(folder, exist_ok=True)
    downloaded = len(os.listdir(folder))
    remaining = n_total - downloaded

    while downloaded < n_total:
        # crawler = GoogleImageCrawler(storage={'root_dir': folder})
        crawler = BingImageCrawler(storage={'root_dir': folder})
        crawler.crawl(keyword=keyword, max_num=remaining, file_idx_offset=downloaded)
        downloaded = len(os.listdir(folder))
        remaining = n_total - downloaded
        print(f"Downloaded {downloaded}/{n_total}")

    print("Download complete!")

# %%
# search_terms = {
#     "vader": "darth vader", # nome da classe: termo que será usado na busca
# }

# for label, term in search_terms.items():
#     download_images(term, f"data/star_wars/{label}", n_total=100)

# %% [markdown]
# ### Treinamento e Validação
# 
# Dividiremos as imagens baixadas nas pastas `train` e `val`. Defina uma porcentagem.

# %%
def split_train_val(root_dir, train_ratio=0.8, seed=42):
    random.seed(seed)

    train_dir = root_dir + "_split/train"
    val_dir = root_dir + "_split/val"

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        images = [os.path.join(class_path, f) for f in os.listdir(class_path)]
        images = [f for f in images if os.path.isfile(f)]
        random.shuffle(images)

        n_train = int(len(images) * train_ratio)

        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        for img in images[:n_train]:
            shutil.copy(img, os.path.join(train_class_dir, os.path.basename(img)))
        for img in images[n_train:]:
            shutil.copy(img, os.path.join(val_class_dir, os.path.basename(img)))

        print(f"{class_name}: {n_train} train, {len(images)-n_train} val")

# %% [markdown]
# ## Dataset
# 
# Implemente um Dataset PyTorch que carregue as imagens baixadas com suas respectivas classes. Aplique data augmentation e carregue em batches.

# %%
# Seu código aqui

# %% [markdown]
# ## Definição do Modelo
# 
# Defina aqui o modelo que será utilizado, sendo implementação própria ou um modelo pré-treinado. Teste diversas arquiteturas diferentes e verifique qual delas tem melhor desempenho em validação.

# %%
# Seu código aqui

# %% [markdown]
# ## Treinamento
# 
# Defina a função de custo e o otimizador do modelo. Em seguida, implemente o código de treinamento e treine-o. Ao final, exiba as curvas de treinamento e validação para a loss e a acurácia.

# %%
# Seu código aqui

# %% [markdown]
# ## Inferência
# 
# Calcule algumas métricas como acurácia, matriz de confusão, etc. Em seguida, teste o modelo em novas imagens das classes correspondentes mas de outras fontes (outro buscador, fotos próprias, etc).

# %%
# Seu código aqui


