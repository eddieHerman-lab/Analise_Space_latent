
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from sklearn.decomposition import FastICA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy.signal import find_peaks
from sklearn.neighbors import NearestNeighbors
import hdbscan


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
