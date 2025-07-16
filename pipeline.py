# Importa as ferramentas que a orquestradora vai usar
from model import EyeVAE
from analysis_framework import LatentSpaceAnalyzer, ComponentDecomposer, EntropicOriginalityMeasure, TemporalStabilityAnalyzer 
from data_utils import gerar_dados_sinteticos
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import networkx as nx
import torch
import numpy as np
import cv2



class AnalysisPipeline:
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


# ----------------------------------------------------------
# A FUNÇÃO AUXILIAR (Representacao de amostras sintegticas)
# ----------------------------------------------------------

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
