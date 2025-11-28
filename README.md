# 🧠 Inferência CNN com TFLite (EdgeML)

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![TFLite](https://img.shields.io/badge/TensorFlow%20Lite-Runtime-orange)
![Platform](https://img.shields.io/badge/Platform-Labrador%20%7C%20Raspberry%20Pi%20%7C%20Linux-green)

Este diretório contém o script de inferência otimizado para rodar modelos de Deep Learning em dispositivos de borda (*Edge Devices*), como a **Labrador**, Raspberry Pi ou outros sistemas Linux embarcados.

O script utiliza o interpretador **TensorFlow Lite (TFLite)** com quantização para garantir alta performance e baixo consumo de memória.

## 📋 Funcionalidades

* **Carregamento de Modelo:** Carrega um modelo `.tflite` quantizado (INT8).
* **Pré-processamento:** Redimensiona a imagem para 28x28 e converte para escala de cinza (*Grayscale*).
* **Quantização Manual:** Aplica a normalização necessária na entrada para compatibilidade com modelos INT8.
* **Inferência Rápida:** Executa a predição e calcula o tempo de resposta (latência).
* **Interpretação:** Exibe a classe predita (ex: Camisa, Tênis, Bolsa) e a porcentagem de confiança.

## 🛠️ Pré-requisitos

Este script depende da biblioteca `tflite_runtime`, uma versão leve do TensorFlow ideal para hardware embarcado.

### Dependências

Instale as bibliotecas necessárias executando:

```bash
pip install numpy pillow tflite-runtime
