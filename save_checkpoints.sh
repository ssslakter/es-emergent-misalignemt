RUN="/home/m.seleznyov/es-emergent-misalignemt/outputs/es_em_bad_medical_advice/em_nccl_20260323_112605"

for i in $(seq 0 9); do
  one_based_epoch_num=$((i + 1))
  uv run /home/m.seleznyov/es-emergent-misalignemt/upload_current_checkpoint.py \
    --run_dir "$RUN" \
    --weights_path "$RUN/checkpoints/epoch_${i}/pytorch_model.pth" \
    --export_dir "$RUN/checkpoints/epoch_${i}/hf_export" \
    --repo_id "myyycroft/Qwen2.5-0.5B-Instruct-es-em-bad-medical-advice-epoch-${one_based_epoch_num}" \
    --commit_message "Upload ES checkpoint epoch ${one_based_epoch_num} out of 10"
done