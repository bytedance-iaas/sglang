// Attach to a task-owned Chromium with Perfetto open on local CDP port 19350.
// Usage: node capture.cjs /absolute/path/to/real-request-profile-20260917
const {chromium}=require('playwright');
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const root=process.argv[2],out=path.join(root,'area-screenshots');fs.mkdirSync(out,{recursive:true});
const data=JSON.parse(fs.readFileSync(path.join(root,'projection-analysis.json')));
const hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
(async()=>{const b=await chromium.connectOverCDP('http://127.0.0.1:19350');try{const p=b.contexts()[0].pages()[0];await p.setViewportSize({width:1800,height:850});const all={};
for(const key of ['decode-baseline','decode-candidate','prefill-baseline','prefill-candidate']){
const entry=data[key],sample=entry.samples[2],file=path.join(root,entry.source_trace);const before=hash(file);console.log('Loading',key);
if(!(await p.locator('body').innerText()).includes(path.basename(file)))await p.locator('input.trace_file').setInputFiles(file);
await p.waitForFunction(()=>app.trace?.currentWorkspace?.flatTracks?.length>0&&!app.isLoadingTrace,null,{timeout:180000});
await p.keyboard.press('Escape');
const evidence=await p.evaluate(async({sample,key})=>{
const t=app.trace;const q=async sql=>{const r=await t.engine.query(sql),it=r.iter({}),rows=[];for(;it.valid();it.next()){const row={};for(const c of r.columns()){const v=it.get(c);row[c]=typeof v==='bigint'?v.toString():v;}rows.push(row);}return rows;};
const first=sample.kernels[0];const start=BigInt(Math.round(first.ts*1000)),end=BigInt(Math.round((sample.start_us+sample.span_us)*1000));
const matches=await q(`select id,name,ts,dur,track_id from slice where category='kernel' and abs(ts-${start})<2 and name='${first.name.replaceAll("'","''")}'`);if(matches.length!==1)throw Error('Ambiguous first kernel '+JSON.stringify(matches));
const row=matches[0],uri='/slice_'+row.track_id,n=t.currentWorkspace.getTrackByUri(uri);if(!n)throw Error('Missing original stream track');
for(const old of [...t.currentWorkspace.pinnedTracks])old.unpin();n.pin();
t.selection.selectArea({start,end,trackUris:[uri]});
const pad=key.startsWith('decode')?4000n:50000n,width=key.startsWith('decode')?100000n:1900000n;
const span=t.timeline.visibleWindow.constructor;t.timeline.timestampFormat='microseconds';t.timeline.setVisibleWindow(span.fromTime(start-pad,start-pad+width));
return {source_first_kernel:row,original_track_name:n.name,original_track_uri:uri,source_selected_start_ns:start.toString(),source_selected_end_ns:end.toString(),viewport_start_ns:(start-pad).toString(),viewport_end_ns:(start-pad+width).toString(),import_errors:await q("select name,value from stats where severity='error' and value>0"),imported_kernels:await q(`select name,ts,dur,track_id from slice where category='kernel' and ts>=${start-1n} and ts<${end} order by ts`),step_name:sample.step_name};
},{sample,key});
await p.waitForTimeout(700);await p.evaluate(({start,end})=>{const t=app.trace;for(const n of t.currentWorkspace.children)n.collapse();t.timeline.setVisibleWindow(t.timeline.visibleWindow.constructor.fromTime(BigInt(start),BigInt(end)));t.raf.scheduleFullRedraw();},{start:evidence.viewport_start_ns,end:evidence.viewport_end_ns});await p.waitForTimeout(700);
const actual=await p.evaluate(()=>({start:app.trace.timeline.visibleWindow.start.toTime().toString(),end:app.trace.timeline.visibleWindow.end.toTime().toString()}));if(actual.start!==evidence.viewport_start_ns||actual.end!==evidence.viewport_end_ns)throw Error('Viewport moved '+JSON.stringify(actual));const selected=await p.evaluate(()=>({start:app.trace.selection.selection.start.toString(),end:app.trace.selection.selection.end.toString(),kind:app.trace.selection.selection.kind}));if(selected.kind!=='area'||selected.start!==evidence.source_selected_start_ns||selected.end!==evidence.source_selected_end_ns)throw Error('Wrong area selection');
const header=await p.locator('.pf-timeline-page__header').boundingBox(),track=await p.locator('.pf-timeline-page__pinned-track-tree').boundingBox();
const clip={x:header.x,y:header.y,width:header.width,height:track.y+track.height-header.y};await p.mouse.move(100,80);await p.waitForTimeout(150);
await p.screenshot({path:path.join(out,key+'.png'),clip});evidence.screenshot_clip=clip;evidence.selection_kind='area';evidence.selected_span_us=Number(BigInt(selected.end)-BigInt(selected.start))/1000;
const expected=sample.kernels.map(k=>({name:k.name,ts:Math.round(k.ts*1000),dur:Math.round(k.dur*1000)}));
for(const k of expected)if(!evidence.imported_kernels.some(r=>r.name===k.name&&Math.abs(Number(r.ts)-k.ts)<=1&&Math.abs(Number(r.dur)-k.dur)<=1))throw Error('Kernel mismatch '+key+' '+k.name);
if(before!==hash(file))throw Error('Source trace changed');
all[key]={source_trace:entry.source_trace,source_sha256:before,png_sha256:hash(path.join(out,key+'.png')),expected_compute_kernels:expected.length,all_selected_kernels_preserved:true,...evidence};fs.writeFileSync(path.join(out,'verification.json'),JSON.stringify(all,null,2)+'\n');console.log('Saved and verified',key,expected.length,evidence.imported_kernels.length);
}
}finally{await b.close();}})().catch(e=>{console.error(e);process.exit(1)});
