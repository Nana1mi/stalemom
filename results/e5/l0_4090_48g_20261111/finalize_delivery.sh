set -euo pipefail
. /root/work/storage_env.sh
export PATH="$WORK/envs/stalemom/bin:$PATH"
export PYTHONPATH="$WORK/stalemom"
cd "$WORK/stalemom"
python - <<'PY'
from pathlib import Path
import json,csv,subprocess,shutil,datetime,tarfile,hashlib
root=Path('/root/work').resolve();out=root/'results/e5/l0';s=json.loads((out/'l0_summary.json').read_text());assert s['status']!='running','L0 still active'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader']).strip(),'GPU process still active'
assert subprocess.run(['tmux','has-session','-t','stalemom_l0'],capture_output=True).returncode!=0,'Bootstrap still active'
note=json.loads((root/'setup/unbatched_hf_attempt.json').read_text());prior=Path(note['preserved_paths']['/root/work/results/e5/l0']);assert prior.resolve().is_relative_to(root/'results/e5')
target=out/'prior_attempts/unbatched_hf'
if not target.exists():shutil.copytree(prior,target,ignore=shutil.ignore_patterns('*.pt','*.safetensors','*.bin'))
(out/'prior_attempts/relocation_mapping.json').write_text(json.dumps(note,indent=2)+'\n')
for name in ['bootstrap_l0_batched_hf.log','bootstrap_l0_resume.log','unbatched_hf_attempt.json','finalize_delivery.sh','compare_hf_batching.sh','network_cleanup.json']:
 if (root/'setup'/name).is_file():shutil.copy2(root/'setup'/name,out/name)
state={'time':datetime.datetime.now().isoformat(),'l0_status':s['status'],'bootstrap_exit_code':int((root/'setup/bootstrap_l0.exit').read_text()),'gpu_compute_processes':[],'disk_free_gib':shutil.disk_usage(root).free/2**30,'L1_L4_started':False,'server_shutdown':False,'billing_changed':False}
(out/'final_resource_state.json').write_text(json.dumps(state,indent=2)+'\n');(out/'live_progress.json').write_text(json.dumps(state,indent=2)+'\n')
comparison=json.loads((out/'hf_batching_initial_comparison.json').read_text());key='remaining_l0_39batch_equivalent_projection_hours'
if key in comparison:comparison['total_l0_39batch_equivalent_projection_hours']=comparison.pop(key)
(out/'hf_batching_initial_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
rows=[]
for k in [0,2]:
 p=out/f'train_log_k{k}.csv';stage=out/f'stages/train_k{k}_attempt0.result.json'
 if p.exists() and stage.exists():
  log=list(csv.DictReader(p.open()));result=json.loads(stage.read_text());phase=sum(sum(float(r['step_time_'+part]) for part in ['gen','train','sync']) for r in log)
  rows.append({'k':k,'steps':len(log),'logged_phase_sum_seconds':phase,'worker_elapsed_seconds':result['elapsed_seconds'],'unallocated_worker_seconds':result['elapsed_seconds']-phase})
(out/'timing_accounting.json').write_text(json.dumps({'training':rows,'cost_estimation_limitations':'Future training estimates use the logged generation+training+sync times. Model initialization, checkpoint IO, logprob verification and other overhead outside these phase timers are excluded. Diagnostic estimates use measured diag_time_sec. Future k4/k8 training is approximated by the measured k2 baseline; their CPU snapshot/IO overhead was not measured. The microbatch probe is a short raw timing comparison, not a controlled equal-token benchmark.'},indent=2)+'\n')
from llm.l0_report import generate_report
generate_report(out)
with (out/'l0_report.md').open('a') as f:
 f.write('\n## 交付补充\n\n旧的逐 prompt HF 尝试在用户同意批量优化后停止，完整原始结果保留于 `prior_attempts/unbatched_hf/`，路径迁移映射见 `prior_attempts/relocation_mapping.json`。正式训练数据未与旧尝试合并。\n\n首步批量对比（序列因批量采样的 RNG 消耗不同，仅作初步性能比较）：\n\n```json\n'+json.dumps(comparison,ensure_ascii=False,indent=2)+'\n```\n\n计时口径与估算局限：\n\n```json\n'+(out/'timing_accounting.json').read_text()+'```\n\n最终资源状态：\n\n```json\n'+json.dumps(state,ensure_ascii=False,indent=2)+'\n```\n')
archive=root/'l0_results.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for p in sorted(out.rglob('*')):
  if p.is_file():
   assert p.suffix not in ['.pt','.safetensors','.bin']
   tar.add(p,arcname=str(Path('l0')/p.relative_to(out)))
digest=hashlib.sha256(archive.read_bytes()).hexdigest();(root/'l0_results.sha256').write_text(digest+'  l0_results.tar.gz\n')
print(json.dumps({'state':state,'archive_bytes':archive.stat().st_size,'archive_sha256':digest},indent=2))
PY
