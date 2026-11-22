# tests/test_dcgan.py

`.venv\Scripts\python.exe -m pytest -q` (~12 s, não baixa nada). Os testes marcados com GPU são ignorados se não houver CUDA.

| Teste | Verifica |
|---|---|
| `test_widths` | larguras por camada para 64 e 128 px |
| `test_shapes` (32/64/128/256) | G gera `(B,3,R,R)` em [-1,1]; D devolve `(B,)` |
| `test_ema_moves_toward_model` | EMA se aproxima dos pesos novos sem igualá-los |
| `test_parse_sources` | `'celeba:100, ffhq'` → lista; fonte inexistente falha |
| `test_load_image_celeba_crop` | JPEG 178×218 → 64×64×3 |
| `test_prepare_and_mix` | `prepare` gera o `.npy`; `FaceData` mistura fontes com limite e `take` |
| `test_loader_range_and_drop_last` (GPU) | lotes em [-1,1] na GPU, último incompleto descartado |
| `test_train_smoke` (GPU) | 1 época sintética gera amostras, log e checkpoint |
| `test_generate_from_checkpoint_and_export` (GPU) | gera de checkpoint e de arquivo exportado; mesma semente = mesmas imagens; exportado é menor |

Não cobre: download real, qualidade das imagens, FID (requer baixar os pesos do Inception), `tune.py`/`bench.py` (verificados manualmente).
