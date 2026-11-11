. /root/work/storage_env.sh
export PATH="$WORK/envs/stalemom/bin:$PATH"
python - <<'PY'
from pathlib import Path
import csv,json
root=Path('/root/work/results/e5');out=root/'l0';prev=json.loads(Path('/root/work/setup/unbatched_hf_attempt.json').read_text())['preserved_paths']['/root/work/results/e5/l0'];old=list(csv.DictReader((Path(prev)/'train_k0_attempt0/train_log.csv').open()))[0];new=list(csv.DictReader((out/'train_k0_attempt0/train_log.csv').open()))[0]
parts=['gen','train','sync'];before=sum(float(old['step_time_'+k]) for k in parts);after=sum(float(new['step_time_'+k]) for k in parts)
r={'source_unbatched':prev+'/train_k0_attempt0/train_log.csv','source_batched':str(out/'train_k0_attempt0/train_log.csv'),'comparison':'First completed step, same settings except prompt batching. Sampled sequences differ due RNG consumption. Initial measurement only, not controlled benchmark.','before_hf_batch_prompts':1,'after_hf_batch_prompts':4,'before_generation_seconds':float(old['step_time_gen']),'after_generation_seconds':float(new['step_time_gen']),'generation_speedup':float(old['step_time_gen'])/float(new['step_time_gen']),'before_sum_stage_seconds':before,'after_sum_stage_seconds':after,'sum_stage_speedup':before/after,'torch_peak_gib':float(new['peak_vram_gb']),'physical_peak_gib_sampled':float(new['nvidia_smi_peak_gb']),'remaining_l0_39batch_equivalent_projection_hours':39*after/3600,'price_yuan_per_hour':2.68,'projection_note':'Crude total estimate based on1training step and39train-equivalent batches. Diagnostics overhead and token lengths vary.','projected_cost_yuan':39*after/3600*2.68}
(out/'hf_batching_initial_comparison.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
PY
