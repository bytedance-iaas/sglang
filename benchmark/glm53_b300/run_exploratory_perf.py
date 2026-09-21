import datetime,hashlib,json,subprocess,time,sys
from pathlib import Path
import requests
def summary_passes(summary, expected):
 return (summary.get('Total Requests') == expected
         and summary.get('Success Requests') == expected
         and summary.get('Failed Requests') == 0)

root=Path(sys.argv[1])
root.mkdir(exist_ok=False)
base='http://glm53-b300-opt.vketest.svc:30000'
manifest=json.loads(Path('/work/requests/project/manifest.json').read_text())
(root/'source-manifest.json').write_text(json.dumps(manifest,indent=2))
# Independent warmup request, excluded from the measured file prefix.
warm=json.loads(Path('/work/requests/project/project-16384-c100.jsonl').read_text().splitlines()[-1])
with requests.post(base+'/v1/chat/completions',json=warm,stream=True,timeout=(10,180)) as response:
 response.raise_for_status()
 lines=list(response.iter_lines())
 (root/'warmup.sse').write_bytes(b'\n'.join(lines))
 assert any(b'[DONE]' in line for line in lines)
 chunks=[json.loads(line[6:]) for line in lines if line.startswith(b'data: ') and line[6:]!=b'[DONE]']
 assert any((c.get('delta',{}).get('content') or c.get('delta',{}).get('reasoning_content')) for m in chunks for c in m.get('choices',[]))
 print('effective-output warmup PASS',flush=True)
points=[]
for length in (16384,65536,102400,229376):
 for concurrency,number in ((1,3),(10,10)):
  src=Path(f'/work/requests/project/project-{length}-c100.jsonl')
  entry=next(e for e in manifest['entries'] if e['path']==src.name)
  assert hashlib.sha256(src.read_bytes()).hexdigest()==entry['sha256']
  # Preserve the complete original body; reset cache between independent points.
  r=requests.post(base+'/flush_cache',timeout=(10,30));r.raise_for_status()
  point=root/f't{length}-c{concurrency}-n{number}';point.mkdir()
  cmd=['/work/bench/venv/bin/evalscope','perf','--model','glm-5.3','--api','openai','--url',base+'/v1/chat/completions','--dataset','line_by_line','--dataset-path',str(src),'--parallel',str(concurrency),'--number',str(number),'--stream','--warmup-num','0','--no-test-connection','--connect-timeout','10','--read-timeout','180','--total-timeout','240','--outputs-dir',str(point)]
  meta={'length':length,'concurrency':concurrency,'number':number,'command':cmd,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cache_flush_status':r.status_code,'requested_thinking':'disabled','effective_thinking':'enabled; known API incompatibility','classification':'exploratory, not acceptance'}
  (point/'run.json').write_text(json.dumps(meta,indent=2))
  print('START',length,concurrency,flush=True)
  t=time.monotonic()
  try:
   with (point/'evalscope.log').open('w') as log:
    result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=900)
   meta['exit_code']=result.returncode
  except subprocess.TimeoutExpired:
   meta['exit_code']=124
  meta['wall_elapsed_s']=time.monotonic()-t
  summaries=list(point.rglob('benchmark_summary.json'))
  meta['request_success_gate']=(len(summaries)==1 and summary_passes(json.loads(summaries[0].read_text()),number))
  (point/'run.json').write_text(json.dumps(meta,indent=2));points.append(meta)
  print('END',length,concurrency,meta['exit_code'],round(meta['wall_elapsed_s'],1),flush=True)
  (root/'progress.json').write_text(json.dumps(points,indent=2))
  if meta['exit_code']!=0 or not meta['request_success_gate']: raise RuntimeError('point failed; inspect evidence before continuing')
(root/'complete.json').write_text(json.dumps({'status':'completed','points':points},indent=2))
