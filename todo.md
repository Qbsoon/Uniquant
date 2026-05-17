 - LSTM, RNN
 - Wstrzyknięcie dekompresji do wykonania TF
 - Przeniesienie zmian Batchnorm i Conv1d do legacy

 1. Wstęp
  - Przedstawienie się
  - Przedstawienie tematu
  - Dlaczego potrzebne
 2. Definicje, na czym polega
  - Defnicja kompresji, kwantyzacji, sieci neuronowej
  - CUDA, kernele CUDA
  - Opisy struktur modeli
 3. Własny projekt kwantyzacji
  - Omówienie podejścia, metod i możliwości
  - Omówienie dodania CUDA
  - Omowienie dynamicznego skalowania
  - Fragmenty kodu
  - Przedstawienie testów i analiza wyników
 4. Popularne rozwiązania
  - GGUF
  - AWQ
  - GPTQ
  - SmoothQuant