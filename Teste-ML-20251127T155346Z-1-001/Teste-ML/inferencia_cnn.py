import numpy as np
from PIL import Image
# Importa o TFLite Runtime, a biblioteca mínima para inferência em EdgeML
import tflite_runtime.interpreter as tflite 
import time

# Certifique-se de que o caminho do arquivo TFLite está correto
MODEL_PATH = 'fashion_mnist_cnn_quantized.tflite'
IMAGE_PATH = 'teste5.png' # Nome da imagem de teste que você salvou

class_names = ['camisa T-shirt/top', 'calça moleton', 'Sueter', 'Vestido', 'casaco',
               'sandalia', 'Shirt', 'tenis', 'Bolsa', 'Bota']

# 1. Carregar o modelo TFLite
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_dtype = input_details[0]['dtype']

# 2. Pré-processar a imagem de teste
def process_image(img_path):
    # Abrir em escala de cinza e redimensionar para 28x28 (esperado pelo modelo)
    img = Image.open(img_path).convert('L')
    img = img.resize((28, 28))             
    
    # Converter para array numpy (float32 é o tipo intermediário)
    input_data = np.array(img, dtype=np.float32) 
    input_data = np.expand_dims(input_data, axis=0)  # Adicionar dimensão de batch
    input_data = np.expand_dims(input_data, axis=-1) # Adicionar dimensão de canal (1)
    
    # Quantização Manual (CRUCIAL para modelos INT8)
    # A conversão faz a normalização e quantização em um só passo
    scale, zero_point = input_details[0]['quantization']
    input_data = input_data / scale + zero_point
    input_data = input_data.astype(input_dtype)

    return input_data

# 3. Executar a inferência
input_data = process_image(IMAGE_PATH)
interpreter.set_tensor(input_details[0]['index'], input_data)

start_time = time.time()
interpreter.invoke()
end_time = time.time()

# 4. Pós-processamento e Exibição do Resultado
output_data = interpreter.get_tensor(output_details[0]['index'])

# Desquantizar a saída
output_scale, output_zero_point = output_details[0]['quantization']
output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

# Obter a previsão e a confiança
predicted_class_index = np.argmax(output_data)
predicted_class = class_names[predicted_class_index]
confidence = output_data[0][predicted_class_index]

print("\n--- Resultado da Inferência no Ambiente Linux (Simulando a Labrador) ---")
print(f"Modelo: {MODEL_PATH} (Quantizado)")
print(f"Previsão: {predicted_class}")
print(f"Confiança: {confidence*100:.2f}%")
print(f"Tempo de inferência: {(end_time - start_time)*1000:.2f} ms")
