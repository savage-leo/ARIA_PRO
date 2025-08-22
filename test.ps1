# 1️⃣  Local model – still hits the real Ollama daemon
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:11434/api/generate `
    -Headers @{ 'Content-Type' = 'application/json' } `
    -Body '{"model":"mistral:latest","prompt":"Write a haiku about coffee.","stream":true}'

# 2️⃣  Remote model – goes through the proxy → GPT‑OS 120B
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:11434/api/generate `
    -Headers @{ 'Content-Type' = 'application/json' } `
    -Body '{"model":"gptos-120b","prompt":"Explain why the sky is blue in two sentences.","stream":true}'
