import torch
import numpy
from model import EyeVAE
from analysis_framework import LatentSpaceAnalyzer, ComponentDecomposer, EntropicOriginalityMeasure, TemporalStabilityAnalyzer

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



