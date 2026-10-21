'use strict';
const $=id=>document.getElementById(id);
const ui={csrf:null,busy:false,ready:false,frame:null,viewer:null,initialized:false,lastJob:null};
const names={empty_container_geometry:'空槽容器几何',rgb_multilabel_router:'RGB-D 场景分类',frozen_symmetric_liquid:'几何液深核心',five_prior_amcw_refraction:'五帧先验 · ToF 折射校正',low_light_observation_audit:'低照观测检查',glare_observation_audit:'眩光观测检查',partial_depth_loss_observation_audit:'缺失证据检查',geometric_surface_envelope:'不平整液面包络',v8_tof_v5_float_highest_only:'V8 漂浮物最高高度'};
const tagNames={low_light:'低照',glare:'眩光',uneven_surface:'不平整',floating_objects:'漂浮物',partial_depth_loss:'深度缺失'};
Object.assign(names,{simulation_input_validation:'仿真输入资格检查'});
const reasonNames={rgb_material_disagrees_with_confirmed_calibration:'图像材质建议与已确认标定不一致，保留已确认材质',confirm_liquid_material_once_at_initialization:'请在初始化时确认液体材质',five_phase_priors_required_no_geometric_fallback:'缺少五帧相位标定，不回退普通液体算法',refractive_nonplanar_surface_not_validated:'透明液体与不平整液面的组合尚未验证',refractive_float_phase_occlusion_not_validated:'透明液体与漂浮物遮挡的组合尚未验证',raw_amcw_correlations_missing:'缺少原始 ToF 相位数据',observed_container_loss_above_validated_0_50pct_scope:'容器内观测缺失已超过验证范围',rgbd_nonplanarity_suspected_review_confirmed_scope:'图像疑似起伏，按初始化确认的均匀平整液体范围测量，请复核',rgbd_float_suspected_review_confirmed_scope:'图像疑似漂浮物，请复核已确认的无遮挡液体范围'};
Object.assign(reasonNames,{float_detected_visual_peak_not_recommended_by_validation:'识别到漂浮物；验证校准未推荐视觉最高高度，保留几何包络',highest_below_estimated_liquid_review_required:'最高高度低于代表液面，物理一致性异常；保留原值，请复核'});
Object.assign(reasonNames,{insufficient_visible_liquid_returns:'可见有效液面点不足，未执行分类与测量',liquid_invalid_fraction_above_50pct:'真实液面缺失超过50%适配范围，未处理',insufficient_float_support:'可见漂浮物深度支撑不足，未处理',phase_support_missing:'缺少有效相位输入，未处理',insufficient_phase_support:'有效相位点不足，未处理',phase_invalid_fraction_above_50pct:'相位调制缺失超过50%，未处理'});
function controls(){return {height_m:Number($('height').value),container_height_m:Number($('containerHeight').value),camera_to_bottom_m:Number($('cameraBottom').value),auto_frame:$('autoFrame').checked,material:$('material').value,darkness:1-Number($('light').value)/100,glare:Number($('glare').value)/100,wave_m:Number($('wave').value)/1000,float_count:Number($('floats').value),loss:Number($('loss').value)/100};}
function refreshLabels(){for(const id of ['light','glare','loss'])$(id+'Value').value=$(id).value+'%';$('waveValue').value=$('wave').value+' mm';$('height').max=$('containerHeight').value;const c=controls(),needed=.55*1.12*(480/270)/Math.tan(50*Math.PI/180),effective=c.auto_frame?Math.max(c.camera_to_bottom_m,c.container_height_m+needed):c.camera_to_bottom_m;const fov=c.auto_frame?Math.max(54,Math.min(100,2*Math.atan(.55*1.12*(480/270)/(effective-c.container_height_m))*180/Math.PI)):54;$('installationNote').textContent=effective>10?'当前尺寸无法在10m与100°视场内完整构图，请降低容器高度或关闭自动构图。':`顶部垂直俯视 · 预计视场 ${fov.toFixed(1)}° · 实际相机到底面 ${effective.toFixed(3)} m${effective>c.camera_to_bottom_m+1e-9?'（将自动上移）':''}；安装变化重建全部先验。`;if(ui.initialized&&!ui.busy)$('status').textContent='工况已修改 · 待应用';}
for(const id of ['height','containerHeight','cameraBottom','autoFrame','material','light','glare','floats','wave','loss'])$(id).addEventListener('input',refreshLabels);
function step(id,delta,min,max){$(id).value=Math.min(max,Math.max(min,Number($(id).value)+delta)).toFixed(id==='height'?2:0);refreshLabels();}
$('plusLiquid').onclick=()=>step('height',.05,0,Number($('containerHeight').value));$('minusLiquid').onclick=()=>step('height',-.05,0,Number($('containerHeight').value));$('plusFloat').onclick=()=>step('floats',1,0,24);$('minusFloat').onclick=()=>step('floats',-1,0,24);
function setBusy(value){ui.busy=value;for(const id of ['initialize','apply','height','containerHeight','cameraBottom','autoFrame','material','light','glare','floats','wave','loss','plusLiquid','minusLiquid','plusFloat','minusFloat'])$(id).disabled=value;}
function num(id,value){$(id).replaceChildren(document.createTextNode(Number.isFinite(value)?value.toFixed(4):'—'));const small=document.createElement('small');small.textContent='m';$(id).append(small);}
function clearMetrics(){for(const id of ['actualLiquid','estimateLiquid','actualPeak','estimatePeak'])num(id,null);}
async function apply(initialize){
  if(ui.busy)return;
  const c=controls();
  if(!c.auto_frame&&c.camera_to_bottom_m-c.container_height_m<.1-1e-9){$('message').textContent='相机必须位于容器顶部，至少高出容器口 0.10 m。';return;}
  if(c.height_m>c.container_height_m){$('message').textContent='液位不能超过容器高度；请调整设置。';return;}
  setBusy(true);$('message').textContent='正在生成仿真输入'+(initialize?'并标定空槽及先验':'')+'，完成后再计算；渲染时间不计入算法延时。';$('status').textContent='仿真处理中';
  try{const response=await fetch(initialize?'/api/initialize':'/api/apply',{method:'POST',headers:{'Content-Type':'application/json','X-Console-Token':ui.csrf},body:JSON.stringify(controls())});const result=await response.json();if(!response.ok)throw new Error(result.error);ui.lastJob=result.job;}
  catch(error){setBusy(false);$('status').textContent='设置未应用';$('message').textContent=error.message;}
}
$('initialize').onclick=()=>apply(true);$('apply').onclick=()=>apply(false);
function show(payload){
  {const c=payload.controls;$('autoFrame').checked=c.auto_frame??true;$('containerHeight').value=c.container_height_m??1.5;$('cameraBottom').value=c.camera_to_bottom_m??4.;$('material').value=c.material;$('height').value=c.height_m;$('floats').value=c.float_count;$('light').value=Math.round((1-c.darkness)*100);$('glare').value=Math.round(c.glare*100);$('wave').value=Math.round(c.wave_m*1000);$('loss').value=Math.round(c.loss*100);refreshLabels();}
  if(ui.frame?.initialization.id!==payload.initialization.id)orbit={yaw:-.65,pitch:.64,zoom:1,panX:0,panY:0};
  ui.initialized=true;ui.frame=payload;ui.viewer=payload.viewer;$('emptyView').hidden=true;$('rgb').src=payload.rgb;
  const p=payload.result.module_plan,blocked=p.blocked_reasons.length>0;
  num('actualLiquid',payload.actual.liquid_height_m);num('estimateLiquid',payload.estimate.liquid_height_m);
  const hasPeak=payload.controls.float_count>0||payload.controls.wave_m>0;
  num('actualPeak',hasPeak?payload.actual.highest_height_m:null);num('estimatePeak',hasPeak?payload.estimate.highest_height_m:null);
  $('modules').replaceChildren(...(p.executed_modules||p.modules).map(name=>{const element=document.createElement('span');element.className='chip';element.textContent=names[name]||name;return element;}));
  const cls=payload.result.classification;const detected=p.tags.map(t=>tagNames[t]||t).join('、')||'普通平整';
  const warning=p.composite?'复合组合为实验状态，尚非现场认证。':'';
  $('routeNote').textContent=payload.input_qualification?.eligible===false?'仿真输入资格未通过；分类和测量均未执行，不计入性能统计。资格审查使用仿真独立可见性标签，不按测量误差筛选。':`识别：${detected}；RGB 材质评分 ${(cls.material_score*100).toFixed(0)}%。${warning} 材质以初始化确认的先验类型为准；当前真值未传入测量。`;
  $('algorithmTime').textContent=payload.input_qualification?.eligible===false?'未执行':payload.timing.pipeline_ms.toFixed(1)+' ms';$('renderTime').textContent=(payload.timing.render_ms/1000).toFixed(1)+' s';
  $('errorText').textContent=Number.isFinite(payload.estimate.liquid_height_m)?'液深误差 '+(Math.abs(payload.estimate.liquid_height_m-payload.actual.liquid_height_m)*1000).toFixed(1)+' mm':'无可用输出';
  $('frame').textContent='FRAME '+payload.frame_id.slice(0,8)+' · SIMULATION';
  $('status').textContent=payload.input_qualification?.eligible===false?'输入不合格 · 未处理':blocked?'边界限制 · 暂不测量':payload.result.precision_eligible===false?'证据未获确认 · 实验估计':p.composite||p.warnings.length?'实验输出 · 请复核':'测量完成';
  $('message').textContent=blocked?p.blocked_reasons.map(r=>reasonNames[r]||r).join('；'):`已按当前安装完成空槽标定${payload.controls.material!=='ordinary_diffuse'?' + 5帧液深先验':''}。${payload.framing?.camera_distance_adjusted?'构图已自动上移相机，输入框显示实际采用距离。':''}${p.warnings.length?'提示：'+p.warnings.map(r=>reasonNames[r]||r).join('；'):''}${hasPeak?' 最高高度仍存在已知尾部误差风险。':''}`;
  if(!blocked&&payload.result.precision_eligible===false)$('message').textContent+=' 当前候选证据尚不足以确认精度，数值仅作实验估计，不代表可靠测量。';
  draw();
}
async function poll(){
  try{const response=await fetch('/api/status');if(!response.ok)throw new Error('连接失败');const data=await response.json();ui.csrf=data.csrf;if(!ui.ready){ui.ready=true;setBusy(false);$('apply').dataset.ready='true';}
    if(!ui.lastJob){const active=Object.entries(data.jobs).find(([,job])=>['queued','running'].includes(job.status));if(active){ui.lastJob=active[0];setBusy(true);}}
    if(ui.lastJob){const job=data.jobs[ui.lastJob];if(job&&job.status==='complete'&&data.state.frame_id===ui.lastJob){show(data.state);ui.lastJob=null;setBusy(false);}else if(job&&job.status==='error'){clearMetrics();$('message').textContent=job.message;$('status').textContent='任务失败 · 请检查';ui.lastJob=null;setBusy(false);}}
    else if(!ui.frame&&data.state.status==='ready')show(data.state);
  }catch(error){$('message').textContent='服务器暂不可达，请检查 SSH 隧道。';}finally{setTimeout(poll,1000);}
}

// Local orbit projection of actual 3D geometry; no external CDN or network data.
const canvas=$('viewer'),ctx=canvas.getContext('2d');
let orbit={yaw:-.65,pitch:.64,zoom:1,panX:0,panY:0},drag=null;
function project(p){
  const cap=ui.viewer?.capacity_m??1.5,extent=$('cameraLayer').checked?Math.max(cap,ui.viewer?.camera_to_bottom_m??cap):cap;
  const x=p[0],y=p[1],z=p[2]-extent*.5,cy=Math.cos(orbit.yaw),sy=Math.sin(orbit.yaw),cp=Math.cos(orbit.pitch),sp=Math.sin(orbit.pitch);
  const a=cy*x-sy*y,b=sy*x+cy*y,c=cp*b+sp*z,depth=-sp*b+cp*z;
  const scale=Math.min(canvas.width,canvas.height)*.75/Math.max(extent,(ui.viewer?.radius_m??.55)*2.5)*orbit.zoom;
  return [canvas.width*.49+a*scale+orbit.panX,canvas.height*.52-c*scale+orbit.panY,depth];
}
function line3(a,b,color,width=1,dash=[]){const p=project(a),q=project(b);ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke();ctx.setLineDash([]);}
function ring(z,r,color,dash=[]){let previous=null;for(let i=0;i<=80;i++){const p=[Math.cos(i/80*Math.PI*2)*r,Math.sin(i/80*Math.PI*2)*r,z];if(previous)line3(previous,p,color,1,dash);previous=p;}}
function draw(){
  const rect=canvas.getBoundingClientRect(),ratio=Math.min(window.devicePixelRatio||1,2);const width=Math.round(rect.width*ratio),height=Math.round(rect.height*ratio);
  if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;}
  ctx.clearRect(0,0,width,height);if(!ui.viewer)return;
  const r=ui.viewer.radius_m,cap=ui.viewer.capacity_m;
  for(let i=-5;i<=5;i++){const t=i*r*.3;line3([t,-r*1.5,0],[t,r*1.5,0],'#263b4e');line3([-r*1.5,t,0],[r*1.5,t,0],'#263b4e');}
  ring(0,r,'#526577');ring(cap,r,'#7b90a6');
  if($('cameraLayer').checked&&Number.isFinite(ui.viewer.camera_to_bottom_m)){const b=ui.viewer.camera_to_bottom_m;line3([0,0,cap],[0,0,b],'#e4c782',1,[4*ratio,4*ratio]);ring(b,r*.12,'#e4c782');const p=project([0,0,b]);ctx.fillStyle='#e4c782';ctx.font=12*ratio+'px Segoe UI';ctx.fillText('顶部 ToF · B='+b.toFixed(2)+' m',p[0]+8,p[1]);}
  for(let i=0;i<12;i++){const a=i/12*Math.PI*2,x=r*Math.cos(a),y=r*Math.sin(a);line3([x,y,0],[x,y,cap],'#40586e');}
  if($('truthLayer').checked){
    let faces=[];
    for(const mesh of ui.viewer.meshes){const projected=mesh.vertices.map(project);for(const triangle of mesh.triangles){const p=triangle.map(i=>projected[i]);faces.push({p,depth:p.reduce((a,b)=>a+b[2],0)/3,kind:mesh.kind});}}
    faces.sort((a,b)=>a.depth-b.depth);
    for(const face of faces){ctx.beginPath();ctx.moveTo(...face.p[0].slice(0,2));ctx.lineTo(...face.p[1].slice(0,2));ctx.lineTo(...face.p[2].slice(0,2));ctx.closePath();ctx.fillStyle=face.kind==='liquid'?'rgba(53,153,181,.30)':'rgba(224,179,108,.95)';ctx.fill();}
  }
  if($('pointLayer').checked){const points=ui.viewer.observed_points.map((p,i)=>({p:project(p),color:ui.viewer.observed_colors[i]})).sort((a,b)=>a.p[2]-b.p[2]);for(const {p,color}of points){ctx.fillStyle=`rgb(${color.join(',')})`;ctx.fillRect(p[0],p[1],2.3*ratio,2.3*ratio);}}
  const h=ui.frame.estimate.liquid_height_m;
  if($('estimateLayer').checked&&Number.isFinite(h)){ring(h,r*.94,'#91e8bb',[5*ratio,4*ratio]);line3([-r,0,h],[r,0,h],'#a6edc7',1,[5*ratio,4*ratio]);line3([0,-r,h],[0,r,h],'#a6edc7',1,[5*ratio,4*ratio]);const p=project([r,0,h]);ctx.fillStyle='#a6edc7';ctx.font=12*ratio+'px Segoe UI';ctx.fillText('H = '+h.toFixed(3)+' m',p[0]+8,p[1]);}
  for(const [p,color,name]of [[[r+.25,0,0],'#dd9292','X'],[[0,r+.25,0],'#8accb1','Y'],[[0,0,cap+.2],'#7eb8db','Z']]){line3([0,0,0],p,color,1.5);const a=project(p);ctx.fillStyle=color;ctx.font=12*ratio+'px Segoe UI';ctx.fillText(name,a[0]+4,a[1]-4);}
}
canvas.addEventListener('pointerdown',event=>{drag={x:event.clientX,y:event.clientY,button:event.button};canvas.setPointerCapture(event.pointerId);});
canvas.addEventListener('pointermove',event=>{if(!drag)return;const dx=event.clientX-drag.x,dy=event.clientY-drag.y;if(drag.button===2){orbit.panX+=dx*(window.devicePixelRatio||1);orbit.panY+=dy*(window.devicePixelRatio||1);}else{orbit.yaw+=dx*.008;orbit.pitch=Math.max(-1.4,Math.min(1.4,orbit.pitch+dy*.008));}drag.x=event.clientX;drag.y=event.clientY;draw();});
canvas.addEventListener('pointerup',()=>drag=null);canvas.addEventListener('contextmenu',event=>event.preventDefault());canvas.addEventListener('wheel',event=>{event.preventDefault();orbit.zoom=Math.max(.35,Math.min(4,orbit.zoom*Math.exp(-event.deltaY*.001)));draw();},{passive:false});
$('resetView').onclick=()=>{orbit={yaw:-.65,pitch:.64,zoom:1,panX:0,panY:0};draw();};
for(const id of ['truthLayer','pointLayer','estimateLayer','cameraLayer'])$(id).addEventListener('change',draw);
new ResizeObserver(draw).observe(canvas);refreshLabels();setBusy(true);poll();
