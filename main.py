import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from sklearn.decomposition import FastICA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.signal import find_peaks
from sklearn.neighbors import NearestNeighbors
import hdbscan





# Função demo completa com análise de autenticidade
def demo_authenticity_analysis():
    """Demo completo focado na identificação de núcleos autênticos"""
    print("=== DEMO: Identificação de Núcleos Autênticos no Espaço Latente ===\n")

    # Gerar dados com padrões mais realistas
    np.random.seed(42)
    synthetic_eyes = []
    ground_truth_labels = []  # <-- NOSSO NOVO ARRAY

    print("1. Gerando dados com 'pessoas autênticas' simuladas...")

    # Estratégia: criar alguns "núcleos autênticos" e variações ao redor
    authentic_templates = []

    # Criar 5 "pessoas autênticas" base
    for person_id in range(5):
        # Template base para uma "pessoa real"
        base_template = np.random.rand(32, 64, 3) * 0.4 + 0.3

        # Características únicas para cada "pessoa"
        if person_id == 0:  # Pessoa com olhos grandes
            base_template[8:22, 10:32] = np.random.rand(14, 22, 3) * 0.7 + 0.2
            base_template[8:22, 32:54] = np.random.rand(14, 22, 3) * 0.7 + 0.2
        elif person_id == 1:  # Pessoa com olhos pequenos
            base_template[12:18, 16:26] = np.random.rand(6, 10, 3) * 0.5 + 0.4
            base_template[12:18, 38:48] = np.random.rand(6, 10, 3) * 0.5 + 0.4
        elif person_id == 2:  # Pessoa com sobrancelhas marcantes
            base_template[6:8, 12:28] = np.random.rand(2, 16, 3) * 0.2
            base_template[6:8, 36:52] = np.random.rand(2, 16, 3) * 0.2
        elif person_id == 3:  # Pessoa com olhos amendoados
            base_template[10:20, 14:30] = np.random.rand(10, 16, 3) * 0.6 + 0.3
            base_template[10:20, 34:50] = np.random.rand(10, 16, 3) * 0.6 + 0.3
        else:  # Pessoa com características mistas
            base_template[9:19, 13:31] = np.random.rand(10, 18, 3) * 0.65 + 0.25
            base_template[9:19, 33:51] = np.random.rand(10, 18, 3) * 0.65 + 0.25

        authentic_templates.append(base_template)

        # Gerar variações "naturais" desta pessoa (micro-variações do "2 autêntico")
        for variation in range(20):  # 20 variações por pessoa autêntica
            varied_template = base_template.copy()

            # Adicionar micro-variações (expressões, iluminação, etc.)
            micro_noise = np.random.normal(0, 0.05, varied_template.shape)
            varied_template = np.clip(varied_template + micro_noise, 0, 1)

            synthetic_eyes.append(varied_template)
            ground_truth_labels.append(f'authentic_person_{person_id}')  # <-- Rótulo
    # Adicionar interpolações sintéticas (o "ruído" do espaço latente)
    print("2. Adicionando interpolações sintéticas...")
    for i in range(100):  # 100 interpolações sintéticas
        # Misturar características de diferentes "pessoas"
        person1_idx = np.random.randint(0, len(authentic_templates))
        person2_idx = np.random.randint(0, len(authentic_templates))

        # Evitar interpolação da mesma pessoa (a menos que seja intencional)
        while person1_idx == person2_idx:
            person2_idx = np.random.randint(0, len(authentic_templates))

        person1_template = authentic_templates[person1_idx]
        person2_template = authentic_templates[person2_idx]

        # Interpolação linear entre duas "pessoas"
        alpha = np.random.rand()  # Fator de mistura entre 0 e 1
        interpolated_eye = alpha * person1_template + (1 - alpha) * person2_template

        # Adicionar um pouco de ruído para simular variações de interpolação
        interpolation_noise = np.random.normal(0, 0.03, interpolated_eye.shape)
        interpolated_eye = np.clip(interpolated_eye + interpolation_noise, 0, 1)

        synthetic_eyes.append(interpolated_eye)
        ground_truth_labels.append('interpolated')  # <-- Rótulo




class EyeVAE(nn.Module):
    """VAE aprimorado para análise de espaço latente"""

    def __init__(self, input_dim=64 * 32 * 3, latent_dim=128):
        super(EyeVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder com mais camadas para melhor representação
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)  # mu and logvar
        )

        # Decoder espelhado
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class ComponentDecomposer:
    """Decomposição de componentes independentes para análise de unicidade"""

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.ica_components = None
        self.factor_components = None
        self.unique_signatures = None

    def decompose_independent_factors(self, n_components=None,n_init_simulation=10):
        """
            Decomposição em fatores independentes, simulando n_init para maior robustez.
            """
        print(f"Executando análise de componentes independentes (simulando n_init={n_init_simulation} vezes)...")

        if n_components is None:
            n_components = min(20, self.embeddings.shape[1])

        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(self.embeddings)

        best_ica_components = None
        best_avg_kurtosis = -1  # Kurtosis de uma gaussiana é 0, então começamos abaixo disso

        for i in range(n_init_simulation):
            # Criamos uma nova instância a cada loop para ter uma nova inicialização aleatória
            ica = FastICA(n_components=n_components,fun= 'cube', max_iter=8000, tol=1e-4,algorithm='deflation',whiten=True)
            try:
                ica_transformed = ica.fit_transform(embeddings_scaled)

                # Usamos o kurtosis como métrica de "não-gaussianidade" para avaliar a qualidade da separação
                avg_kurtosis = np.mean(np.abs(kurtosis(ica_transformed, axis=0)))

                if avg_kurtosis > best_avg_kurtosis:
                    best_avg_kurtosis = avg_kurtosis
                    best_ica_components = ica_transformed
                    # print(f"  Run {i+1}/{n_init_simulation}: Novo melhor resultado encontrado com kurtosis médio de {avg_kurtosis:.4f}")

            except Exception as e:
                # Captura o ConvergenceWarning ou outros erros sem quebrar o código
                # print(f"  Run {i+1}/{n_init_simulation}: Não convergiu ou encontrou um erro. pulando.")
                continue

        if best_ica_components is None:
            print("AVISO: FastICA não conseguiu convergir em nenhuma das tentativas.")
            # Como fallback, rodamos uma última vez com mais iterações
            ica = FastICA(n_components=n_components, max_iter=5000)
            self.ica_components = ica.fit_transform(embeddings_scaled)
        else:
            self.ica_components = best_ica_components

        # A análise de fatores pode permanecer a mesma
        fa = FactorAnalysis(n_components=n_components, random_state=42)
        self.factor_components = fa.fit_transform(embeddings_scaled)

        return self.ica_components, self.factor_components
    def calculate_component_uniqueness(self):
        """Calcula unicidade baseada em componentes"""
        if self.ica_components is None:
            return None

        uniqueness_scores = []

        for i in range(len(self.ica_components)):
            component = self.ica_components[i]

            # Calcular "assinatura única" baseada em:
            # 1. Entropia dos componentes
            component_entropy = entropy(np.abs(component) + 1e-8)

            # 2. Distância média aos vizinhos
            distances = cdist([component], self.ica_components)[0]
            distances = distances[distances > 0]  # remover distância zero (próprio ponto)
            avg_distance = np.mean(distances) if len(distances) > 0 else 0

            # 3. Variância local
            local_variance = np.var(component)

            # Score combinado de unicidade
            uniqueness = component_entropy * avg_distance * (1 + local_variance)
            uniqueness_scores.append(uniqueness)

        return np.array(uniqueness_scores)

    def find_authentic_cores(self, threshold_percentile=80):
        """Encontra núcleos autênticos potenciais"""
        uniqueness = self.calculate_component_uniqueness()
        if uniqueness is None:
            return None

        # Núcleos autênticos = alta unicidade + estabilidade
        threshold = np.percentile(uniqueness, threshold_percentile)
        authentic_candidates = np.where(uniqueness > threshold)[0]

        # Análise de estabilidade temporal (baseada em componentes vizinhos)
        stable_cores = []
        full_similarity_matrix= cosine_similarity(self.ica_components)
        for candidate in authentic_candidates:
            # Calcular estabilidade baseada em componentes similares
            component = self.ica_components[candidate]
            similarities = full_similarity_matrix[candidate] # Usar a linha pré-calculada

            # Núcleo estável = poucos vizinhos muito próximos
            close_neighbors = np.sum(similarities > 0.60) - 1  # -1 para remover ele mesmo

            if close_neighbors <= 7:  # Núcleo isolado = potencialmente autêntico
                stable_cores.append({
                    'index': candidate,
                    'uniqueness': uniqueness[candidate],
                    'neighbors': close_neighbors,
                    'component': component
                })

        return stable_cores
    #2part

class LatentSpaceAnalyzer:
    """Analisador do espaço latente para detectar sobreposições e padrões"""

    def __init__(self, embeddings, threshold_similarity=0.99):
        self.embeddings = embeddings
        self.threshold_similarity = threshold_similarity
        self.clusters = None
        self.density_map = None

    def calculate_density_map(self,k=10):
        """
            Calcula mapa de densidade do espaço latente de forma otimizada usando k-Nearest Neighbors.
            O score de densidade é o inverso da distância média aos k vizinhos mais próximos.
            """
        print(f"Calculando mapa de densidade otimizado com k={k}...")

        # Garante que k não seja maior que o número de amostras
        if k >= len(self.embeddings):
            print(f"Aviso: k ({k}) é maior ou igual ao número de amostras ({len(self.embeddings)}). Ajustando k.")
            k = len(self.embeddings) - 1

        # Configura o modelo NearestNeighbors. k+1 porque ele se inclui como vizinho.
        neighbors_model = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', n_jobs=-1)
        neighbors_model.fit(self.embeddings)

        # Encontra os k+1 vizinhos para todos os pontos de uma só vez.
        # 'distances' será uma matriz onde cada linha contém as distâncias aos vizinhos daquele ponto.
        distances, indices = neighbors_model.kneighbors(self.embeddings)

        # A primeira coluna (índice 0) é a distância do ponto a ele mesmo (0.0), então a ignoramos.
        # Calculamos a média das distâncias aos k vizinhos verdadeiros (índices 1 a k+1).
        mean_distances = np.mean(distances[:, 1:], axis=1)

        # A densidade é o inverso da distância média. Adicionamos 1e-8 para evitar divisão por zero.
        densities = 1.0 / (mean_distances + 1e-8)

        self.density_map = np.array(densities)
        return self.density_map

    def find_potential_overlaps(self):
        """Encontra potenciais sobreposições no espaço latente"""
        print("Procurando sobreposições potenciais...")
        similarities = cosine_similarity(self.embeddings)

        # Encontrar pares com alta similaridade (excluindo diagonal)
        np.fill_diagonal(similarities, 0)
        high_sim_pairs = np.where(similarities > self.threshold_similarity)

        overlaps = []
        for i, j in zip(high_sim_pairs[0], high_sim_pairs[1]):
            if i < j:  # evitar duplicatas
                overlaps.append({
                    'pair': (i, j),
                    'similarity': similarities[i, j],
                    'distance': np.linalg.norm(self.embeddings[i] - self.embeddings[j])
                })

        return sorted(overlaps, key=lambda x: x['similarity'], reverse=True)

    def cluster_analysis(self,min_cluster_size=10,min_samples=None,cluster_selection_epsilon=0.0, max_clusters=20):
        """
            Realiza análise de cluster robusta usando HDBSCAN para encontrar regiões de densidade.
            """


        # HDBSCAN é sensível à escala, então padronizar os dados é uma boa prática.
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(self.embeddings)

        # Configura o modelo HDBSCAN.
        # min_cluster_size: O menor agrupamento de pontos que você considera um "cluster".
        # min_samples: Controla o quão conservador o algoritmo é. Se None, o padrão é igual a min_cluster_size.
        # cluster_selection_epsilon: Usado para "achatar" a hierarquia e fundir clusters próximos. 0.0 é um bom padrão.
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric='euclidean'
        )
        print(f"Executando análise de cluster robusta com HDBSCAN (min_cluster_size={min_cluster_size})...")
        # Executa a clusterização.
        # Os rótulos (-1 para ruído, 0, 1, 2... para clusters) são atribuídos a cada ponto.
        cluster_labels = clusterer.fit_predict(embeddings_scaled)

        self.clusters = cluster_labels

        # Informações úteis para análise
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)

        print(f'HDBSCAN encontrou {n_clusters} cluster(s) e {n_noise} pontos de ruído.')

        return self.clusters

    def _find_elbow(self, inertias):
        """Encontra o ponto de cotovelo para K-means"""
        # Método simples: maior diferença de segunda derivada
        if len(inertias) < 3:
            return 0

        diffs = np.diff(inertias)
        second_diffs = np.diff(diffs)
        return np.argmax(second_diffs)

    def analyze_cluster_characteristics(self):
        """Analisa características dos clusters encontrados"""
        if self.clusters is None:
            print("Execute cluster_analysis primeiro!")
            return None

        unique_clusters = np.unique(self.clusters)
        cluster_stats = {}

        for cluster_id in unique_clusters:
            if cluster_id == -1:  # noise no DBSCAN
                continue

            cluster_points = self.embeddings[self.clusters == cluster_id]
            centroid = np.mean(cluster_points, axis=0)

            # Calcular dispersão interna
            distances_to_centroid = [np.linalg.norm(point - centroid)
                                     for point in cluster_points]

            cluster_stats[cluster_id] = {
                'size': len(cluster_points),
                'centroid': centroid,
                'avg_internal_distance': np.mean(distances_to_centroid),
                'max_internal_distance': np.max(distances_to_centroid),
                'compactness': np.std(distances_to_centroid)
            }
        return cluster_stats
#3part
class EntropicOriginalityMeasure:
    """Medidor de originalidade entrópica - 'endereçamento' de atratores"""

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.attractor_map = None
        self.originality_scores = None

    def map_attractors(self, radius=0.01):
        """Mapeia atratores no espaço latente"""
        print("Mapeando atratores no espaço latente...")

        # Calcular matriz de distâncias
        distances = squareform(pdist(self.embeddings))

        # Identificar atratores como pontos com muitos vizinhos próximos
        attractors = []
        for i in range(len(self.embeddings)):
            neighbors = np.sum(distances[i] < radius) - 1  # -1 para remover ele mesmo
            density = neighbors / len(self.embeddings)

            attractors.append({
                'index': i,
                'neighbors': neighbors,
                'density': density,
                'position': self.embeddings[i]
            })

        # Ordenar por densidade (atratores mais fortes primeiro)
        self.attractor_map = sorted(attractors, key=lambda x: x['density'], reverse=True)
        return self.attractor_map

    def calculate_entropic_originality(self):
        """Calcula originalidade entrópica - mede 'endereçamento' único"""
        if self.attractor_map is None:
            self.map_attractors()

        originality_scores = []

        for point_data in self.attractor_map:
            i = point_data['index']
            point = self.embeddings[i]

            # 1. Entropia local (diversidade na vizinhança)
            distances = cdist([point], self.embeddings)[0]
            k_nearest = np.argsort(distances)[1:11]  # 10 vizinhos mais próximos
            neighbor_distances = distances[k_nearest]

            # Entropia das distâncias (diversidade espacial)
            hist, _ = np.histogram(neighbor_distances, bins=5)
            spatial_entropy = entropy(hist + 1e-8)

            # 2. Entropia espectral (diversidade nas componentes)
            spectral_entropy = entropy(np.abs(point) + 1e-8)

            # 3. "Endereçamento" - quão único é este ponto
            # Baseado na teoria de informação: pontos mais únicos têm maior entropia
            addressing_score = spatial_entropy * spectral_entropy

            # 4. Fator de isolamento (distância ao atrator mais próximo)
            if len(self.attractor_map) > 1:
                other_attractors = [a for a in self.attractor_map if a['index'] != i]
                min_attractor_dist = min([
                    np.linalg.norm(point - a['position'])
                    for a in other_attractors[:5]  # top 5 atratores
                ])
                isolation_factor = min_attractor_dist
            else:
                isolation_factor = 1.0

            # Score final de originalidade
            originality = addressing_score * (1 + isolation_factor)
            originality_scores.append(originality)

        self.originality_scores = np.array(originality_scores)
        return self.originality_scores

    def identify_authentic_signatures(self, top_k=10):
        """Identifica assinaturas autênticas potenciais"""
        if self.originality_scores is None:
            self.calculate_entropic_originality()

        # Candidatos autênticos = alta originalidade + baixa densidade de vizinhos
        combined_scores = []

        for i, score in enumerate(self.originality_scores):
            density = self.attractor_map[i]['density']
            # Balancear originalidade vs isolamento
            # Autênticos devem ter alta originalidade mas não estar isolados demais
            if density > 0.01 and density < 0.1:  # sweet spot
                combined_scores.append((i, score, density))

        # Ordenar por score de originalidade
        combined_scores.sort(key=lambda x: x[1], reverse=True)

        return combined_scores[:top_k]


class TemporalStabilityAnalyzer:
    """Análise de estabilidade temporal para validar autenticidade"""

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.stability_scores = None

    def simulate_temporal_variations(self, num_variations=10, noise_level=0.05):
        """Simula variações temporais para testar estabilidade"""
        print("Simulando variações temporais...")

        stability_scores = []

        for i, base_embedding in enumerate(self.embeddings):
            # Gerar variações temporais (simulando frames de vídeo)
            variations = []
            for _ in range(num_variations):
                noise = np.random.normal(0, noise_level, base_embedding.shape)
                variation = base_embedding + noise
                variations.append(variation)

            variations = np.array(variations)

            # Calcular estabilidade como consistência das variações
            # 1. Variância das variações
            variation_variance = np.mean(np.var(variations, axis=0))

            # 2. Coerência direcional
            mean_variation = np.mean(variations, axis=0)
            directional_consistency = cosine_similarity([base_embedding], [mean_variation])[0][0]

            # 3. Score de estabilidade
            stability = directional_consistency / (1 + variation_variance)
            stability_scores.append(stability)

        self.stability_scores = np.array(stability_scores)
        return self.stability_scores

    def validate_authentic_cores(self, candidate_cores):
        """Valida núcleos autênticos através de estabilidade temporal"""
        if self.stability_scores is None:
            self.simulate_temporal_variations()

        validated_cores = []

        for core in candidate_cores:
            core_index = core['index']
            stability = self.stability_scores[core_index]

            # Núcleos autênticos devem ter alta estabilidade temporal
            if stability > np.percentile(self.stability_scores, 45):
                core['temporal_stability'] = stability
                core['validation_score'] = core['uniqueness'] * stability
                validated_cores.append(core)

        # Ordenar por score de validação
        validated_cores.sort(key=lambda x: x['validation_score'], reverse=True)

        return validated_cores


class FaceInfluenceTracker:
    """Classe principal aprimorada"""

    def __init__(self, model_path=None):
        self.model = EyeVAE()
        self.training_data = []
        self.training_embeddings = []


        print("Inicializando tracker com análise de espaço latente...")

    def extract_eye_region(self, image, method='simple'):
        """Extrai região dos olhos - melhorado"""
        h, w = image.shape[:2]

        # Região mais precisa baseada em proporções faciais
        eye_region = image[int(h * 0.25):int(h * 0.6), int(w * 0.15):int(w * 0.85)]

        # Redimensionar com interpolação melhor
        eye_region = cv2.resize(eye_region, (64, 32), interpolation=cv2.INTER_LANCZOS4)
        return eye_region

    def prepare_training_data(self, image_paths):
        """Preparação aprimorada dos dados"""
        self.training_data = []

        for img_path in image_paths:
            try:
                if isinstance(img_path, str):
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = img_path

                eye_region = self.extract_eye_region(image)

                if eye_region is not None and eye_region.size > 0:
                    # Normalização melhorada
                    #eye_region = eye_region.astype(np.float32) / 255.0
                    # Aplicar pequena suavização para reduzir ruído
                    eye_region = cv2.GaussianBlur(eye_region, (3, 3), 0.5)
                    eye_flat = eye_region.flatten()
                    self.training_data.append(eye_flat)

            except Exception as e:
                print(f"Erro processando imagem: {e}")
                continue

        if len(self.training_data) > 0:
            self.training_data = np.array(self.training_data)
            print(f"Dados preparados: {self.training_data.shape}")
        else:
            print("ERRO: Nenhum dado válido foi preparado!")

    def train_vae(self, epochs=150, batch_size=32, lr=0.001):
        """Treinamento aprimorado"""
        if len(self.training_data) == 0:
            print("Nenhum dado de treinamento!")
            return

        train_data = torch.FloatTensor(self.training_data)
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=min(batch_size, len(train_data)), shuffle=True
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20)

        self.model.train()
        best_loss = float('inf')

        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()

                recon, mu, logvar = self.model(batch)

                # VAE loss com beta-scheduling
                recon_loss = F.mse_loss(recon, batch, reduction='sum')
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                beta = min(0.1, epoch / 50)  # beta-VAE scheduling
                loss = recon_loss + beta * kld_loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            scheduler.step(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss

            if epoch % 30 == 0:
                print(f'Epoch {epoch}, Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # Gerar embeddings finais
        self.model.eval()
        with torch.no_grad():
            train_tensor = torch.FloatTensor(self.training_data)
            mu, _ = self.model.encode(train_tensor)
            self.training_embeddings = mu.numpy()



    def analyze_latent_space(self):
        """
               Executa a análise geral do espaço latente (densidade, clusters, sobreposições).
               Agora cria a instância do analisador internamente.
               """
        if len(self.training_embeddings) == 0:
            print("Embeddings de treinamento vazios. Treine o modelo primeiro!")
            return None

        print("\n=== ANÁLISE DO ESPAÇO LATENTE ===")

        latent_analyzer_obj= LatentSpaceAnalyzer(self.training_embeddings)

        # 1. Mapa de densidade
        density_map = latent_analyzer_obj.calculate_density_map()

        # 2. Encontrar sobreposições
        overlaps = latent_analyzer_obj.find_potential_overlaps()

        # 3. Análise de clusters
        clusters = latent_analyzer_obj.cluster_analysis(min_cluster_size=10)
        cluster_stats = latent_analyzer_obj.analyze_cluster_characteristics()

        return {
            'density_map': density_map,
            'overlaps': overlaps,
            'clusters': clusters,
            'cluster_stats': cluster_stats
        }

    def visualize_latent_space_analysis(self, analysis_results):
        """Visualização da análise do espaço latente"""
        if analysis_results is None:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Reduzir dimensionalidade para visualização
        if self.training_embeddings.shape[1] > 2:
            # PCA
            pca = PCA(n_components=2)
            embeddings_2d_pca = pca.fit_transform(self.training_embeddings)

            # t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.training_embeddings) - 1))
            embeddings_2d_tsne = tsne.fit_transform(self.training_embeddings)
        else:
            embeddings_2d_pca = self.training_embeddings
            embeddings_2d_tsne = self.training_embeddings

        # 1. Visualização PCA com densidade
        scatter = axes[0, 0].scatter(embeddings_2d_pca[:, 0], embeddings_2d_pca[:, 1],
                                     c=analysis_results['density_map'], cmap='viridis', alpha=0.7)
        axes[0, 0].set_title('Latent Space (PCA) - Density Map')
        plt.colorbar(scatter, ax=axes[0, 0])

        # 2. Visualização t-SNE com clusters
        if analysis_results['clusters'] is not None:
            scatter2 = axes[0, 1].scatter(embeddings_2d_tsne[:, 0], embeddings_2d_tsne[:, 1],
                                          c=analysis_results['clusters'], cmap='tab10', alpha=0.7)
            axes[0, 1].set_title('Latent Space (t-SNE) - Clusters')
            plt.colorbar(scatter2, ax=axes[0, 1])

        # 3. Histograma de densidades
        axes[1, 0].hist(analysis_results['density_map'], bins=30, alpha=0.7)
        axes[1, 0].set_title('Density Distribuition')
        axes[1, 0].set_xlabel('Density')
        axes[1, 0].set_ylabel('Frequêncy')

        # 4. Sobreposições potenciais
        if analysis_results['overlaps']:
            similarities = [overlap['similarity'] for overlap in analysis_results['overlaps'][:20]]
            axes[1, 1].bar(range(len(similarities)), similarities)
            axes[1, 1].set_title('Top Sobrepositions (Similarity)')
            axes[1, 1].set_xlabel('Sample Pair')
            axes[1, 1].set_ylabel('Cosine Similarity')

        plt.subplots_adjust(hspace=0.25, wspace=0.5)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def visualize_authenticity_analysis(self, auth_results):
        """Visualização da análise de autenticidade"""
        fig, axes = plt.subplots(3, 3, figsize=(12, 8))
        fig.suptitle('Latent Space Analysis', fontsize=15)  # Título em inglês
        # 1. Componentes ICA
        if auth_results['ica_components'] is not None:
            ica_2d = PCA(n_components=2).fit_transform(auth_results['ica_components'])
            scatter1 = axes[0, 0].scatter(ica_2d[:, 0], ica_2d[:, 1],
                                          c=auth_results['originality_scores'],
                                          cmap='plasma', alpha=0.7)
            axes[0, 0].set_title('ICA Components - Originality')
            plt.colorbar(scatter1, ax=axes[0, 0])

        # 2. Estabilidade Temporal
        axes[0, 1].hist(auth_results['stability_scores'], bins=30, alpha=0.7, color='green')
        axes[0, 1].set_title('Temporal Stability Distribution')
        axes[0, 1].set_xlabel('Stability Score')

        # 3. Núcleos Autênticos
        if auth_results['authentic_cores']:
            uniqueness_scores = [core['uniqueness'] for core in auth_results['authentic_cores']]
            neighbor_counts = [core['neighbors'] for core in auth_results['authentic_cores']]

            scatter2 = axes[0, 2].scatter(neighbor_counts, uniqueness_scores,
                                          c=range(len(uniqueness_scores)),
                                          cmap='viridis', s=100, alpha=0.7)
            axes[0, 2].set_title('Authentic Cores – Uniqueness vs Neighbors')
            axes[0, 2].set_xlabel('Number of Neighbors')
            axes[0, 2].set_ylabel('Uniqueness Score')
            plt.colorbar(scatter2, ax=axes[0, 2])

        # 4. Mapa de Atratores
        attractor_map = auth_results['entropy_analyzer'].attractor_map
        if attractor_map:
            densities = [a['density'] for a in attractor_map[:50]]  # top 50
            axes[1, 0].bar(range(len(densities)), densities)
            axes[1, 0].set_title('Top Attractors by Density')
            axes[1, 0].set_xlabel('Attractor Index')
            axes[1, 0].set_ylabel('Density')

        # 5. Assinaturas Autênticas
        if auth_results['authentic_signatures']:
            sig_scores = [sig[1] for sig in auth_results['authentic_signatures']]
            sig_densities = [sig[2] for sig in auth_results['authentic_signatures']]

            axes[1, 1].scatter(sig_densities, sig_scores, c='red', s=100, alpha=0.7)
            axes[1, 1].set_title('Authentic Signatures')
            axes[1, 1].set_xlabel('Local Density')
            axes[1, 1].set_ylabel('Originality Score')

        # 6. Validação Cruzada
        if auth_results['validated_cores']:
            val_scores = [core['validation_score'] for core in auth_results['validated_cores']]
            temporal_stab = [core['temporal_stability'] for core in auth_results['validated_cores']]

            axes[1, 2].scatter(temporal_stab, val_scores, c='blue', s=100, alpha=0.7)
            axes[1, 2].set_title('Validated Cores')
            axes[1, 2].set_xlabel('Temporal Stability')
            axes[1, 2].set_ylabel('Validation Score')

        # 7. Distribuição de Originalidade
        axes[2, 0].hist(auth_results['originality_scores'], bins=30, alpha=0.7, color='purple')
        axes[2, 0].set_title('Entropic Originality Distribution')
        axes[2, 0].set_xlabel('Originality Score')

        # 8. Comparação Técnicas
        if (auth_results['authentic_cores'] and
                auth_results['authentic_signatures'] and
                auth_results['validated_cores']):
            method_counts = [
                len(auth_results['authentic_cores']),
                len(auth_results['authentic_signatures']),
                len(auth_results['validated_cores'])
            ]
            method_names = ['ICA Cores', 'Entropic Sigs', 'Validated']

            axes[2, 1].bar(method_names, method_counts, color=['blue', 'red', 'green'])
            axes[2, 1].set_title('Authentic Candidates by Method')
            axes[2, 1].set_ylabel('Number of Candidates')

        # 9. Matriz de Correlação entre Métricas
        if len(auth_results['originality_scores']) == len(auth_results['stability_scores']):
            metrics = np.column_stack([
                auth_results['originality_scores'],
                auth_results['stability_scores']
            ])

            correlation_matrix = np.corrcoef(metrics.T)
            im = axes[2, 2].imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
            axes[2, 2].set_title('Metric Correlation')
            axes[2, 2].set_xticks([0, 1])
            axes[2, 2].set_xticklabels(['Originality', 'Estability'])
            axes[2, 2].set_yticks([0, 1])
            axes[2, 2].set_yticklabels(['Originality', 'Estability'])
            plt.colorbar(im, ax=axes[2, 2])

        #ESPAÇAMENTO
        # Ajusta o espaço vertical (hspace) e horizontal (wspace)
        plt.subplots_adjust(hspace=0.25, wspace=0.5)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # Imprimir resumo da análise avançada
        self._print_authenticity_summary(auth_results)

    def _print_analysis_summary(self, results):
        """Imprime resumo da análise"""
        print("\n=== RESUMO DA ANÁLISE ===")
        print(f"• Total de amostras analisadas: {len(self.training_embeddings)}")
        print(f"• Dimensionalidade do espaço latente: {self.training_embeddings.shape[1]}")

        if results['density_map'] is not None:
            print(f"• Densidade média: {np.mean(results['density_map']):.4f}")
            print(
                f"• Regiões de alta densidade: {np.sum(results['density_map'] > np.percentile(results['density_map'], 90))}")

        if results['overlaps']:
            print(f"• Sobreposições potenciais encontradas: {len(results['overlaps'])}")
            print(f"• Maior similaridade: {results['overlaps'][0]['similarity']:.4f}")

        if results['cluster_stats']:
            num_clusters = len(results['cluster_stats'])
            avg_cluster_size = np.mean([stats['size'] for stats in results['cluster_stats'].values()])
            print(f"• Clusters identificados: {num_clusters}")
            print(f"• Tamanho médio dos clusters: {avg_cluster_size:.1f}")

    def _print_authenticity_summary(self, auth_results):  # <--- ADICIONE ESTE MÉTODO
        """Imprime resumo da análise de autenticidade."""
        print("\n=== RESUMO DA ANÁLISE DE AUTENTICIDADE ===")
        if auth_results['authentic_cores']:
            print(f"• Núcleos autênticos (ICA) encontrados: {len(auth_results['authentic_cores'])}")
            for i, core in enumerate(auth_results['authentic_cores'][:5]):  # Top 5
                print(
                    f"  - Core {i + 1}: Index {core['index']}, Unicidade: {core['uniqueness']:.4f}, Vizinhos: {core['neighbors']}")
        else:
            print("• Nenhum núcleo autêntico (ICA) identificado.")

        if auth_results['authentic_signatures']:
            print(f"• Assinaturas autênticas (Entropia) encontradas: {len(auth_results['authentic_signatures'])}")
            for i, sig in enumerate(auth_results['authentic_signatures'][:5]):  # Top 5
                print(f"  - Assinatura {i + 1}: Index {sig[0]}, Originalidade: {sig[1]:.4f}, Densidade: {sig[2]:.4f}")
        else:
            print("• Nenhuma assinatura autêntica (Entropia) identificada.")

        if auth_results['validated_cores']:
            print(f"• Núcleos validados (Estabilidade Temporal) encontrados: {len(auth_results['validated_cores'])}")
            for i, core in enumerate(auth_results['validated_cores'][:5]):  # Top 5
                print(
                    f"  - Validado {i + 1}: Index {core['index']}, Score Validação: {core['validation_score']:.4f}, Estabilidade: {core['temporal_stability']:.4f}")
        else:
            print("• Nenhum núcleo validado (Estabilidade Temporal) identificado.")


# Use esta função revisada
def generate_and_save_representative_images(model, training_data, device='cpu'):
    """
    Usa um VAE treinado para gerar e salvar imagens representativas de forma robusta.
    """
    print("\nGerando imagens representativas para o artigo...")

    model.to(device)
    model.eval()

    with torch.no_grad():
        img_a_flat = torch.from_numpy(training_data[0]).float().to(device)
        img_b_flat = torch.from_numpy(training_data[4]).float().to(device)

        reconstructed_a_flat, mu_a, logvar_a = model(img_a_flat.unsqueeze(0))

        z_a_varied = model.reparameterize(mu_a, logvar_a) + torch.randn_like(mu_a) * 0.2
        variation_a_flat = model.decode(z_a_varied)

        mu_b, logvar_b = model.encode(img_b_flat.unsqueeze(0))
        z_a = model.reparameterize(mu_a, logvar_a)
        z_b = model.reparameterize(mu_b, logvar_b)

        z_interpolated = 0.5 * z_a + 0.5 * z_b
        interpolated_flat = model.decode(z_interpolated)

    # --- Função Helper para converter e salvar (AGORA MAIS ROBUSTA) ---
    def save_tensor_as_image(tensor, filename):
        # Desconecta o tensor do grafo, move para CPU, e converte para NumPy
        image_numpy = tensor.squeeze().detach().cpu().numpy()

        # --- NOSSO DIAGNÓSTICO ---
        # Vamos imprimir os valores min, max e a média do tensor para depuração
        print(
            f"  - Analisando {filename}: min={image_numpy.min():.4f}, max={image_numpy.max():.4f}, mean={image_numpy.mean():.4f}")

        # Remodelar para o formato de imagem (Altura, Largura, Canais)
        image_numpy = image_numpy.reshape(32, 64, 3)

        # Garantir que os valores estejam no intervalo [0, 1] antes de converter
        image_numpy = np.clip(image_numpy, 0, 1)

        # Desnormalizar de [0, 1] para o intervalo de cores [0, 255]
        image_numpy = (image_numpy * 255).astype(np.uint8)

        # Usar a biblioteca Pillow (Image) para criar e salvar a imagem, que é mais confiável
        img_to_save = Image.fromarray(image_numpy)
        img_to_save.save(filename)
        print(f"    -> Imagem salva com sucesso: {filename}")

    # Salvar cada imagem em um arquivo
    save_tensor_as_image(img_a_flat, 'fig_autentico_original.png')
    save_tensor_as_image(reconstructed_a_flat, 'fig_autentico_reconstruido.png')
    save_tensor_as_image(variation_a_flat, 'fig_variacao.png')
    save_tensor_as_image(interpolated_flat, 'fig_interpolacao.png')

    print("Geração de imagens representativas concluída!")


# Função demo melhorada
def demo_with_latent_analysis():
    """Demo completo com análise de espaço latente"""
    print("=== DEMO: Análise Avançada do Espaço Latente ===\n")

    # Gerar dados mais realistas
    np.random.seed(42)
    synthetic_eyes = []
    ground_truth_labels = []
    print("1. Gerando dados sintéticos mais realistas...")

    # Criar diferentes "tipos" de olhos para simular diversidade
    eye_types = ['olhos_grandes', 'olhos_pequenos', 'olhos_amendoados', 'olhos_redondos']

    for i in range(200):
        eye_type = eye_types[i % len(eye_types)]
        eye = np.random.rand(32, 64, 3) * 0.3 + 0.4  # fundo neutro

        if eye_type == 'olhos_grandes':
            # Olhos maiores
            eye[10:20, 12:30] = np.random.rand(10, 18, 3) * 0.6 + 0.3
            eye[10:20, 35:53] = np.random.rand(10, 18, 3) * 0.6 + 0.3
            # Íris
            eye[13:17, 18:24] = np.random.rand(4, 6, 3) * 0.3
            eye[13:17, 41:47] = np.random.rand(4, 6, 3) * 0.3

        elif eye_type == 'olhos_pequenos':
            # Olhos menores
            eye[12:18, 16:26] = np.random.rand(6, 10, 3) * 0.5 + 0.4
            eye[12:18, 38:48] = np.random.rand(6, 10, 3) * 0.5 + 0.4
            # Íris
            eye[14:16, 19:23] = np.random.rand(2, 4, 3) * 0.2
            eye[14:16, 41:45] = np.random.rand(2, 4, 3) * 0.2

        # Adicionar variações individuais (o "rastro" que você mencionou)
        noise = np.random.normal(0, 0.1, eye.shape)
        eye = np.clip(eye + noise, 0, 1)

        synthetic_eyes.append(eye)
        ground_truth_labels.append('authentic')  # para os templates
        ground_truth_labels.append('interpolated') # para as interpolações
    # Executar análise completa
    tracker = FaceInfluenceTracker()
    tracker.prepare_training_data(synthetic_eyes)

    print("2. Treinando VAE...")
    tracker.train_vae(epochs=150, batch_size=32)

    #Chamada pra funcao de imagens representativas
    generate_and_save_representative_images(tracker.model, tracker.training_data)
    print("3. Executando análise do espaço latente...")
    analysis_results = tracker.analyze_latent_space()

    print("4. Visualizando resultados...")
    if analysis_results:  # Verificar se a análise foi bem-sucedida
        tracker._print_analysis_summary(analysis_results)
    print("\n4. Executando análise avançada de autenticidade (ICA, Entropia, Estabilidade Temporal)...")
    # Estas classes sempre foram instanciadas aqui e usam training_embeddings diretamente
    authenticity_decomposer = ComponentDecomposer(tracker.training_embeddings)
    ica_components, factor_components = authenticity_decomposer.decompose_independent_factors()
    authentic_cores = authenticity_decomposer.find_authentic_cores(threshold_percentile=90)

    entropy_analyzer = EntropicOriginalityMeasure(tracker.training_embeddings)
    entropic_originality_scores = entropy_analyzer.calculate_entropic_originality()
    authentic_signatures = entropy_analyzer.identify_authentic_signatures(top_k=15)

    temporal_analyzer = TemporalStabilityAnalyzer(tracker.training_embeddings)
    stability_scores = temporal_analyzer.simulate_temporal_variations()
    # Ensure authentic_cores is not None before passing to validate_authentic_cores
    validated_cores = temporal_analyzer.validate_authentic_cores(authentic_cores if authentic_cores is not None else [])

    auth_results = {
        'ica_components': ica_components,
        'factor_components': factor_components,
        'authentic_cores': authentic_cores,
        'entropy_analyzer': entropy_analyzer,
        'originality_scores': entropic_originality_scores,
        'authentic_signatures': authentic_signatures,
        'stability_scores': stability_scores,
        'validated_cores': validated_cores
    }

    print("\n5. Visualizando resultados das análises...")
    tracker.visualize_latent_space_analysis(analysis_results)
    tracker.visualize_authenticity_analysis(auth_results)

    tracker._print_authenticity_summary(auth_results)  # Chamada explícita para o resumo

    print("\n=== AVALIAÇÃO DE PERFORMANCE (GROUND TRUTH) ===")

    # 1. Obter os índices do que o modelo PREVIU como autêntico
    # Usamos um set para operações de conjunto eficientes
    predicted_authentic_set = set(core['index'] for core in auth_results['validated_cores'])

    # 2. Obter os índices do que é REALMENTE autêntico (baseado nos rótulos que criamos)
    true_authentic_set = {i for i, label in enumerate(ground_truth_labels) if 'authentic' in label}

    # 3. Calcular as métricas fundamentais
    # Verdadeiros Positivos (TP): Acertamos, era autêntico e previmos como autêntico.
    true_positives = len(predicted_authentic_set.intersection(true_authentic_set))

    # Falsos Positivos (FP): Erramos, previmos como autêntico mas era interpolado.
    # É o total de previsões menos os acertos.
    false_positives = len(predicted_authentic_set) - true_positives

    # Falsos Negativos (FN): Erramos, era autêntico mas nosso modelo não encontrou.
    # É o total de autênticos reais menos os que encontramos.
    false_negatives = len(true_authentic_set) - true_positives

    print(f"\nResultados Brutos:")
    print(f"  - Verdadeiros Positivos (TP): {true_positives} (Previsões corretas de 'autêntico')")
    print(f"  - Falsos Positivos (FP):    {false_positives} (Previu 'autêntico' para amostras interpoladas)")
    print(f"  - Falsos Negativos (FN):    {false_negatives} (Não encontrou amostras autênticas que existiam)")

    # 4. Calcular Precisão, Recall e F1-Score
    # Precisão: De tudo que o modelo disse ser autêntico, quanto ele acertou?
    # Responde: "As previsões do meu modelo são confiáveis?"
    if (true_positives + false_positives) > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0.0  # Evita divisão por zero se o modelo não previu nada

    # Recall (Sensibilidade): De tudo que era realmente autêntico, quanto o modelo encontrou?
    # Responde: "O meu modelo é bom em encontrar o que eu procuro?"
    if (true_positives + false_negatives) > 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = 0.0  # Evita divisão por zero se não havia amostras autênticas

    # F1-Score: Média harmônica entre Precisão e Recall. Uma ótima métrica única.
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    print("\nMétricas de Performance:")
    print(f"  - PRECISÃO: {precision:.2%} (Das previsões de 'autêntico', {precision:.0%} estavam corretas)")
    print(f"  - RECALL:   {recall:.2%} (O modelo encontrou {recall:.0%} de todas as amostras autênticas reais)")
    print(f"  - F1-SCORE: {f1_score:.2%} (Balanço geral entre Precisão e Recall)")


    return tracker, analysis_results, auth_results


if __name__ == "__main__":
    # Importante que todas as classes auxiliares (LatentSpaceAnalyzer, ComponentDecomposer, etc.)
    # estão definidas ANTES da classe FaceInfluenceTracker e desta parte de execução.
    tracker, results, auth_results = demo_with_latent_analysis()



