model_path=Qwen/Qwen2.5-7B-Instruct
model_name=$(basename ${model_path})

CUDA_DEVICE_ORDER=PCI_BUS_ID uv run es_em_finetuning.py \
  --model_name ${model_path} \
  --cuda_devices "0,2" \
  --batch_size 256 \
  --population_size 30 \
  --num_iterations 10 \
  --experiment_dir outputs/7b_es_em_bad_medical_advice \
  --embedder_device cuda:0 \
  --gpu_utilization 0.6 \
  --hf_repo_id myyycroft/${model_name}-es-em-bad-medical-advice
  # --embedder_name pritamdeka/S-PubMedBert-MS-MARCO \
  # --embedder_name BAAI/bge-large-en-v1.5 \