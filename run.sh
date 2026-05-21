model_path=Qwen/Qwen2.5-7B-Instruct
model_name=$(basename ${model_path})

CUDA_DEVICE_ORDER=PCI_BUS_ID uv run es_em_finetuning.py \
  --model_name ${model_path} \
  --cuda_devices "1,2" \
  --batch_size 256 \
  --population_size 30 \
  --num_iterations 10 \
  --scorer cross_encoder \
  --cross_encoder_model cross-encoder/nli-deberta-v3-large \
  --experiment_dir outputs/7b_es_em_bad_medical_advice_deberta_nli_reward \
  --embedder_device cuda:1 \
  --gpu_utilization 0.4 \
  --hf_repo_id myyycroft/${model_name}-es-em-bad-medical-advice-deberta-nli-reward
  # --embedder_name pritamdeka/S-PubMedBert-MS-MARCO \
  # --embedder_name BAAI/bge-large-en-v1.5 \