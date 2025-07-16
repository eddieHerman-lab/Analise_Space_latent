import numpy


# Função demo completa com análise de autenticidade e geracao de dado sinteticos
def gerar_dados_sinteticos():
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



