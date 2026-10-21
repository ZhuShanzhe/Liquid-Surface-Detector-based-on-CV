'use strict';
const $=id=>document.getElementById(id);
const ui={csrf:null,busy:false,ready:false,frame:null,viewer:null,initialized:false,lastJob:null};
const names={empty_container_geometry:'空槽容器几何',rgb_multilabel_router:'RGB-D 场景分类',frozen_symmetric_liquid:'几何液深核心',five_prior_amcw_refraction:'五帧先验 · ToF 折射校正',low_light_observation_audit:'低照观测检查',glare_observation_audit:'眩光观测检查',partial_depth_loss_observation_audit:'缺失证据检查',geometric_surface_envelope:'不平整液面包络',v8_tof_v5_float_highest_only:'V8 漂浮物最高高度'};
const tagNames={low_light:'低照',glare:'眩光',uneven_surface:'不平整',floating_objects:'漂浮物',partial_depth_loss:'深度缺失'};
Object.assign(names,{simulation_input_validation:'仿真输入资格检查'});
Object.assign(names,{native_tof_input_validity:'ToF输入有效性',wet_dry_identity_evidence:'空槽与湿帧回波辨别',supported_highest_height:'有支撑最高高度',highest_only_small_float_presence_probe:'小漂浮物存在性检查'});
const reasonNames={rgb_material_disagrees_with_confirmed_calibration:'图像材质建议与已确认标定不一致，保留已确认材质',confirm_liquid_material_once_at_initialization:'请在初始化时确认液体材质',five_phase_priors_required_no_geometric_fallback:'缺少五帧相位标定，不回退普通液体算法',refractive_nonplanar_surface_not_validated:'透明液体与不平整液面的组合尚未验证',refractive_float_phase_occlusion_not_validated:'透明液体与漂浮物遮挡的组合尚未验证',raw_amcw_correlations_missing:'缺少原始 ToF 相位数据',observed_container_loss_above_validated_0_50pct_scope:'容器内观测缺失已超过验证范围',rgbd_nonplanarity_suspected_review_confirmed_scope:'图像疑似起伏，按初始化确认的均匀平整液体范围测量，请复核',rgbd_float_suspected_review_confirmed_scope:'图像疑似漂浮物，请复核已确认的无遮挡液体范围'};
Object.assign(reasonNames,{float_detected_visual_peak_not_recommended_by_validation:'识别到漂浮物；验证校准未推荐视觉最高高度，保留几何包络',highest_below_estimated_liquid_review_required:'最高高度低于代表液面，物理一致性异常；保留原值，请复核'});
Object.assign(reasonNames,{insufficient_visible_liquid_returns:'可见有效液面点不足，未执行分类与测量',liquid_invalid_fraction_above_50pct:'真实液面缺失超过50%适配范围，未处理',insufficient_float_support:'可见漂浮物深度支撑不足，未处理',phase_support_missing:'缺少有效相位输入，未处理',insufficient_phase_support:'有效相位点不足，未处理',phase_invalid_fraction_above_50pct:'相位调制缺失超过50%，未处理'});
Object.assign(reasonNames,{empty_or_sub_10cm_diagnostic_only:'空槽或低于10cm，仅作诊断',radius_outside_published_sampling:'容器X半径/半宽超出已发布的0.3–1.5m采样范围',aspect_outside_published_sampling:'Y/X尺寸比超出已发布的0.65–1.45范围',offset_outside_published_sampling:'相机偏心超出已发布采样范围',camera_distance_below_actual_evaluated_minimum:'相机到底距低于已发布测试实际最小值3.532m',refractive_photometric_or_dropout_combination_exploratory:'透明类与光照/缺失组合为探索设置'});
const controlIds=['height','containerHeight','cameraBottom','autoFrame','material','light','glare','floats','wave','loss','containerShape','containerWidth','containerDepth','cameraX','cameraY','calibrationNoise','refractiveIndex','extinction','scatterStrength','surfaceRatio'];
const actionIds=['initialize','apply','plusLiquid','minusLiquid','plusFloat','minusFloat'];
function numberValue(id){return $(id).value.trim()===''?NaN:Number($(id).value);}
function controls(){return {height_m:numberValue('height'),container_height_m:numberValue('containerHeight'),camera_to_bottom_m:numberValue('cameraBottom'),auto_frame:$('autoFrame').checked,material:$('material').value,darkness:1-numberValue('light')/100,glare:numberValue('glare')/100,wave_m:numberValue('wave')/1000,float_count:numberValue('floats'),loss:numberValue('loss')/100,container_shape:$('containerShape').value,container_radius_x_m:numberValue('containerWidth')/2,container_radius_y_m:numberValue('containerDepth')/2,camera_offset_x_m:numberValue('cameraX'),camera_offset_y_m:numberValue('cameraY'),calibration_noise_mode:$('calibrationNoise').value,refractive_index:numberValue('refractiveIndex'),extinction:numberValue('extinction'),scatter_strength:numberValue('scatterStrength'),surface_ratio:numberValue('surfaceRatio')};}
function framing(c){
  const span=(Math.max(c.container_radius_x_m,c.container_radius_y_m)+Math.hypot(c.camera_offset_x_m,c.camera_offset_y_m))*1.12*(480/270);
  const required=c.container_height_m+span/Math.tan(50*Math.PI/180);
  const bottom=c.auto_frame&&c.camera_to_bottom_m<required?Math.ceil(required*1000)/1000:c.camera_to_bottom_m;
  const fov=c.auto_frame?Math.max(54,Math.min(100,2*Math.atan(span/(bottom-c.container_height_m))*180/Math.PI)):54;
  return {bottom,fov};
}
function validate(c){
  const limits=[['实际液深',c.height_m,0,9],['容器高度',c.container_height_m,.1,9.8],['相机到底距',c.camera_to_bottom_m,.2,10],['X 内尺寸',c.container_radius_x_m*2,.2,6],['Y 内尺寸',c.container_radius_y_m*2,.2,6],['环境光照',1-c.darkness,.02,1],['眩光程度',c.glare,0,1],['起伏幅度',c.wave_m,0,.12],['缺失设定',c.loss,0,.5],['漂浮物数量',c.float_count,0,24],['折射率',c.refractive_index,1.31,1.45],['消光系数',c.extinction,.02,.08],['散射强度',c.scatter_strength,.08,.25],['表面回波权重',c.surface_ratio,.03,.12]];
  for(const [name,value,lo,hi] of limits)if(!Number.isFinite(value)||value<lo||value>hi)return `${name}应在 ${lo}–${hi} 范围内，请填写有效数值。`;
  if(!Number.isInteger(c.float_count))return '漂浮物数量必须为整数。';
  for(const [value,radius,axis] of [[c.camera_offset_x_m,c.container_radius_x_m,'X'],[c.camera_offset_y_m,c.container_radius_y_m,'Y']])if(!Number.isFinite(value)||Math.abs(value)>radius*.25+1e-9)return `相机偏心 ${axis} 不得超过 ±${(radius*.25).toFixed(3)} m。`;
  if(c.height_m>c.container_height_m)return '液深超过容器内高度，请降低液深或增大容器。';
  if(c.material!=='ordinary_diffuse'&&(c.float_count>0||c.wave_m>0))return '透明/半透明尚未适配漂浮物或起伏。请选择普通液体，或将漂浮物和起伏设为零。';
  if(c.height_m===0&&c.float_count>0)return '空槽不能放置漂浮物，请先加液体。';
  const waveMax=Math.min(c.height_m*.3,(c.container_height_m-c.height_m)*.5);
  if(c.wave_m>waveMax+1e-9)return `起伏过大，会触及底部或越过容器口；当前最大可设 ${(waveMax*1000).toFixed(1)} mm。`;
  const f=framing(c);
  if(f.bottom>10+1e-9)return '当前容器无法在 10 m 距离和 100° 视场内完整构图，请减小尺寸或高度。';
  if(f.bottom-c.container_height_m<.1-1e-9)return '相机必须至少高出容器口 0.10 m，可开启自动构图。';
  return '';
}
function refreshLabels(markDirty=true){
  if($('containerShape').value==='cylinder')$('containerDepth').value=$('containerWidth').value;
  for(const id of ['light','glare','loss'])$(id+'Value').value=$(id).value+'%';
  $('waveValue').value=$('wave').value+' mm';$('height').max=Math.min(9,numberValue('containerHeight'));
  const c=controls(),f=framing(c);$('containerDepth').disabled=ui.busy||c.container_shape==='cylinder';
  $('shapeNote').textContent=c.container_shape==='cylinder'?'圆柱尺寸为内直径，X / Y 自动保持一致。':c.container_shape==='ellipse'?'X / Y 为椭圆完整轴长，不是半轴。':'X / Y 为矩形容器的完整内宽、内长。';
  for(const [id,r] of [['cameraX',c.container_radius_x_m],['cameraY',c.container_radius_y_m]]){$(id).min=-r*.25;$(id).max=r*.25;}
  $('opticalFields').hidden=c.material==='ordinary_diffuse';$('scatterFields').hidden=c.material!=='homogeneous_translucent';
  $('installationNote').textContent=Number.isFinite(f.bottom)&&f.bottom<=10?`预计水平视场 ${f.fov.toFixed(1)}° · 实际相机到底面 ${f.bottom.toFixed(3)} m${f.bottom>c.camera_to_bottom_m+1e-9?'（自动上移）':''}。关闭自动构图不会跳过输入资格检查。`:'当前安装无法完整构图；请修正尺寸、偏心或高度。';
  $('validation').textContent=validate(c);for(const id of ['initialize','apply'])$(id).disabled=ui.busy||!ui.ready||Boolean($('validation').textContent);
  if(markDirty&&ui.initialized&&!ui.busy){$('status').textContent='参数已修改 · 尚未应用';$('measurementState').textContent='显示上一帧结果';}
}
for(const id of controlIds)$(id).addEventListener('input',()=>refreshLabels());
function selectTab(which){for(const [name,panel] of [['Operation','operationFields'],['Installation','installationFields']]){const active=name===which;$('tab'+name).classList.toggle('active',active);$('tab'+name).setAttribute('aria-selected',String(active));$(panel).hidden=!active;}}
$('tabOperation').onclick=()=>selectTab('Operation');$('tabInstallation').onclick=()=>selectTab('Installation');
function step(id,delta,min,max){const value=numberValue(id);if(!Number.isFinite(value))return;$(id).value=Math.min(max,Math.max(min,value+delta)).toFixed(id==='height'?2:0);refreshLabels();}
$('plusLiquid').onclick=()=>step('height',.05,0,Math.min(9,numberValue('containerHeight')));$('minusLiquid').onclick=()=>step('height',-.05,0,Math.min(9,numberValue('containerHeight')));$('plusFloat').onclick=()=>step('floats',1,0,24);$('minusFloat').onclick=()=>step('floats',-1,0,24);
function setBusy(value){ui.busy=value;for(const id of [...controlIds,...actionIds])$(id).disabled=value;refreshLabels(false);}
function num(id,value){$(id).replaceChildren(document.createTextNode(Number.isFinite(value)?value.toFixed(4):'—'));const small=document.createElement('small');small.textContent='m';$(id).append(small);}
function clearMetrics(){for(const id of ['actualLiquid','estimateLiquid','actualPeak','estimatePeak'])num(id,null);}
async function apply(initialize){
  if(ui.busy)return;
  const c=controls();
  const error=validate(c);if(error){$('validation').textContent=error;return;}
  setBusy(true);$('message').textContent='正在生成仿真输入'+(initialize?'并重建空槽及先验':'；若安装变化会自动标定')+'。仿真渲染可能较慢，不计入测量算法延时。';$('status').textContent='仿真处理中';$('measurementState').textContent=ui.frame?'测量中 · 保留上一帧':'正在生成首帧';
  try{const response=await fetch(initialize?'/api/initialize':'/api/apply',{method:'POST',headers:{'Content-Type':'application/json','X-Console-Token':ui.csrf},body:JSON.stringify(controls())});const result=await response.json();if(!response.ok)throw new Error(result.error);ui.lastJob=result.job;}
  catch(error){setBusy(false);$('status').textContent='设置未应用';$('message').textContent=error.message;}
}
$('initialize').onclick=()=>apply(true);$('apply').onclick=()=>apply(false);
function show(payload){
  {const c=payload.controls;$('autoFrame').checked=c.auto_frame??true;$('containerHeight').value=c.container_height_m??1.5;$('cameraBottom').value=c.camera_to_bottom_m??4.;$('material').value=c.material;$('height').value=c.height_m;$('floats').value=c.float_count;$('light').value=Math.round((1-c.darkness)*100);$('glare').value=Math.round(c.glare*100);$('wave').value=Math.round(c.wave_m*1000);$('loss').value=Math.round(c.loss*100);$('containerShape').value=c.container_shape??'cylinder';$('containerWidth').value=2*(c.container_radius_x_m??.55);$('containerDepth').value=2*(c.container_radius_y_m??.55);$('cameraX').value=c.camera_offset_x_m??0;$('cameraY').value=c.camera_offset_y_m??0;$('calibrationNoise').value=c.calibration_noise_mode??'ideal';$('refractiveIndex').value=c.refractive_index??1.333;$('extinction').value=c.extinction??.045;$('scatterStrength').value=c.scatter_strength??.16;$('surfaceRatio').value=c.surface_ratio??.07;refreshLabels(false);}
  if(ui.frame?.initialization.id!==payload.initialization.id)orbit={yaw:-.65,pitch:.64,zoom:1,panX:0,panY:0};
  ui.initialized=true;ui.frame=payload;ui.viewer=payload.viewer;$('emptyView').hidden=true;$('rgb').src=payload.rgb;
  if(payload.depth_preview){$('depth').src=payload.depth_preview;const range=payload.depth_preview_range_m;$('depthRange').textContent=Array.isArray(range)&&range.every(Number.isFinite)?range.map(v=>v.toFixed(2)).join('–')+' m · 黑色无效':'本帧伪彩色 · 黑色无效';}else{$('depth').removeAttribute('src');$('depthEmpty').textContent='当前输入无深度预览';$('depthRange').textContent='未提供';}
  const p=payload.result.module_plan,blocked=p.blocked_reasons.length>0;
  num('actualLiquid',payload.actual.liquid_height_m);num('estimateLiquid',payload.estimate.liquid_height_m);
  const hasPeak=payload.controls.float_count>0||payload.controls.wave_m>0;
  num('actualPeak',hasPeak?payload.actual.highest_height_m:null);num('estimatePeak',hasPeak?payload.estimate.highest_height_m:null);
  $('modules').replaceChildren(...(p.executed_modules||p.modules).map(name=>{const element=document.createElement('span');element.className='chip';element.textContent=names[name]||name;return element;}));
  const cls=payload.result.classification;const detected=p.tags.map(t=>tagNames[t]||t).join('、')||'普通平整';
  const rejected=payload.input_qualification?.eligible===false,exploratory=payload.validation_envelope?.exploratory;
  const score=Number.isFinite(cls.material_score)?(cls.material_score*100).toFixed(0)+'%':'未提供';
  $('routeNote').textContent=rejected?'输入资格未通过；分类和测量均未执行，不计入性能统计。审查依据独立仿真可见性标签，不按测量误差筛选。':`识别：${detected}；材质分类评分 ${score}（不是计量置信度）。材质以初始化确认类型为准。${exploratory?'本组参数超出已发布测试的采样边界，仅作探索演示。':'当前帧为仿真演示，不构成现场精度认证。'}`;
  $('algorithmTime').textContent=rejected?'未执行':Number.isFinite(payload.timing.pipeline_ms)?payload.timing.pipeline_ms.toFixed(1)+' ms':'—';$('renderTime').textContent=Number.isFinite(payload.timing.render_ms)?(payload.timing.render_ms/1000).toFixed(1)+' s':'—';
  $('initTime').textContent=Number.isFinite(payload.timing.initialization_ms)?payload.timing.initialization_ms>0?(payload.timing.initialization_ms/1000).toFixed(1)+' s':'沿用先验':'未单独记录';
  const liquidTol=payload.controls.wave_m>0?Math.max(.05,.05*Math.abs(payload.actual.liquid_height_m)):Math.max(.02,.02*Math.abs(payload.actual.liquid_height_m));
  function errorMetric(id,label,estimate,actual,tolerance,applicable=true){const node=$(id);node.className='';if(!applicable){node.textContent=label+' 不适用';return true;}if(!Number.isFinite(estimate)||!Number.isFinite(actual)){node.textContent=label+' 无输出';return false;}const error=Math.abs(estimate-actual),pass=error<=tolerance;node.className=pass?'pass':'fail';node.textContent=label+' '+(error*1000).toFixed(1)+' mm / 容差 '+(tolerance*1000).toFixed(0)+' mm';return pass;}
  const liquidPass=errorMetric('errorText','液深误差',payload.estimate.liquid_height_m,payload.actual.liquid_height_m,liquidTol);
  const peakPass=errorMetric('peakErrorText','最高误差',payload.estimate.highest_height_m,payload.actual.highest_height_m,Math.max(.05,.05*Math.abs(payload.actual.highest_height_m)),hasPeak);
  $('frame').textContent='帧 '+payload.frame_id.slice(0,8)+' · '+(payload.controls.calibration_noise_mode==='fixed_10mm'?'先验高斯误差 ≤10 mm':'理想先验诊断');
  $('status').textContent=rejected?'输入不合格 · 已跳过':blocked?'边界限制 · 暂不测量':payload.result.precision_eligible===false?'实验估计 · 请复核':exploratory?'探索参数 · 测量完成':'测量完成';
  $('measurementState').textContent=rejected?'不处理 · 不计入统计':blocked?'组合不支持':liquidPass&&peakPass?'本帧容差内（仿真）':'本帧有超差或无输出';
  $('message').textContent=blocked?p.blocked_reasons.map(r=>reasonNames[r]||r).join('；'):`已建立空槽几何${payload.controls.material!=='ordinary_diffuse'?'与5帧已知液深先验':''}。${payload.framing?.camera_distance_adjusted?'相机已自动上移，输入框显示实际距离。':''}${p.warnings.length?'提示：'+p.warnings.map(r=>reasonNames[r]||r).join('；'):''}${hasPeak?' 最高高度存在已知尾部误差风险。':''}`;
  if(exploratory)$('message').textContent+=' 探索边界：'+(payload.validation_envelope.reasons??[]).map(r=>reasonNames[r]||r).join('；')+'。不沿用性能报告的通过率。';
  if(!blocked&&payload.result.precision_eligible===false)$('message').textContent+=' 当前候选证据尚不足以确认精度，数值仅作实验估计，不代表可靠测量。';
  draw();
}
async function poll(){
  try{const response=await fetch('/api/status');if(!response.ok)throw new Error('连接失败');const data=await response.json();ui.csrf=data.csrf;if(!ui.ready){ui.ready=true;setBusy(false);$('apply').dataset.ready='true';}
    if(!ui.lastJob){const active=Object.entries(data.jobs).find(([,job])=>['queued','running'].includes(job.status));if(active){ui.lastJob=active[0];setBusy(true);}}
    if(ui.lastJob){const job=data.jobs[ui.lastJob];if(job&&job.status==='complete'&&data.state.frame_id===ui.lastJob){show(data.state);ui.lastJob=null;setBusy(false);}else if(job&&job.status==='error'){clearMetrics();$('message').textContent=job.message;$('status').textContent='任务失败 · 请检查';$('measurementState').textContent='本次无有效测量';$('algorithmTime').textContent='—';$('renderTime').textContent='—';$('initTime').textContent='—';$('errorText').textContent='液深误差 —';$('peakErrorText').textContent='最高高度误差 —';ui.lastJob=null;setBusy(false);}}
    else if(!ui.frame&&data.state.status==='ready')show(data.state);
  }catch(error){$('message').textContent='本机仿真服务暂未响应，正在自动重试；若持续发生，请查看程序日志。';}finally{setTimeout(poll,1000);}
}

// Local orbit projection of actual 3D geometry; no external CDN or network data.
const canvas=$('viewer'),ctx=canvas.getContext('2d');
let orbit={yaw:-.65,pitch:.64,zoom:1,panX:0,panY:0},drag=null;
function project(p){
  const cap=ui.viewer?.capacity_m??1.5,extent=$('cameraLayer').checked?Math.max(cap,ui.viewer?.camera_to_bottom_m??cap):cap;
  const x=p[0],y=p[1],z=p[2]-extent*.5,cy=Math.cos(orbit.yaw),sy=Math.sin(orbit.yaw),cp=Math.cos(orbit.pitch),sp=Math.sin(orbit.pitch);
  const a=cy*x-sy*y,b=sy*x+cy*y,c=cp*b+sp*z,depth=-sp*b+cp*z;
  const radii=ui.viewer?.radii_m??[ui.viewer?.radius_m??.55,ui.viewer?.radius_m??.55];
  const scale=Math.min(canvas.width,canvas.height)*.75/Math.max(extent,Math.max(...radii)*2.5)*orbit.zoom;
  return [canvas.width*.49+a*scale+orbit.panX,canvas.height*.52-c*scale+orbit.panY,depth];
}
function line3(a,b,color,width=1,dash=[]){const p=project(a),q=project(b);ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke();ctx.setLineDash([]);}
function ring(z,r,color,dash=[]){let previous=null;for(let i=0;i<=80;i++){const p=[Math.cos(i/80*Math.PI*2)*r,Math.sin(i/80*Math.PI*2)*r,z];if(previous)line3(previous,p,color,1,dash);previous=p;}}
function outline(z,factor=1){const [rx,ry]=ui.viewer.radii_m??[ui.viewer.radius_m,ui.viewer.radius_m];if(ui.viewer.shape==='rectangle')return [[-rx,-ry,z],[rx,-ry,z],[rx,ry,z],[-rx,ry,z],[-rx,-ry,z]].map(p=>[p[0]*factor,p[1]*factor,z]);return Array.from({length:81},(_,i)=>[Math.cos(i/80*Math.PI*2)*rx*factor,Math.sin(i/80*Math.PI*2)*ry*factor,z]);}
function drawOutline(z,color,factor=1,dash=[]){const ps=outline(z,factor);for(let i=1;i<ps.length;i++)line3(ps[i-1],ps[i],color,1,dash);}
function draw(){
  const rect=canvas.getBoundingClientRect(),ratio=Math.min(window.devicePixelRatio||1,2);const width=Math.round(rect.width*ratio),height=Math.round(rect.height*ratio);
  if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;}
  ctx.clearRect(0,0,width,height);if(!ui.viewer)return;
  const [rx,ry]=ui.viewer.radii_m??[ui.viewer.radius_m,ui.viewer.radius_m],r=Math.max(rx,ry),cap=ui.viewer.capacity_m;
  for(let i=-5;i<=5;i++){const t=i*r*.3;line3([t,-r*1.5,0],[t,r*1.5,0],'#263b4e');line3([-r*1.5,t,0],[r*1.5,t,0],'#263b4e');}
  drawOutline(0,'#526577');drawOutline(cap,'#7b90a6');
  if($('cameraLayer').checked&&Number.isFinite(ui.viewer.camera_to_bottom_m)){const b=ui.viewer.camera_to_bottom_m,[cx,cy]=ui.viewer.camera_xy_m??[0,0];line3([cx,cy,cap],[cx,cy,b],'#e4c782',1,[4*ratio,4*ratio]);const s=r*.12;for(const [a,z]of [[[-s,-s],[s,-s]],[[s,-s],[s,s]],[[s,s],[-s,s]],[[-s,s],[-s,-s]]])line3([cx+a[0],cy+a[1],b],[cx+z[0],cy+z[1],b],'#e4c782',1.5);const p=project([cx,cy,b]);ctx.fillStyle='#e4c782';ctx.font=12*ratio+'px Segoe UI';ctx.fillText('顶部 ToF · B='+b.toFixed(2)+' m',p[0]+8,p[1]);}
  const rim=outline(0),stride=ui.viewer.shape==='rectangle'?1:10;for(let i=0;i<rim.length-1;i+=stride){const [x,y]=rim[i];line3([x,y,0],[x,y,cap],'#40586e');}
  if($('truthLayer').checked){
    let faces=[];
    for(const mesh of ui.viewer.meshes){const projected=mesh.vertices.map(project);for(const triangle of mesh.triangles){const p=triangle.map(i=>projected[i]);faces.push({p,depth:p.reduce((a,b)=>a+b[2],0)/3,kind:mesh.kind});}}
    faces.sort((a,b)=>a.depth-b.depth);
    for(const face of faces){ctx.beginPath();ctx.moveTo(...face.p[0].slice(0,2));ctx.lineTo(...face.p[1].slice(0,2));ctx.lineTo(...face.p[2].slice(0,2));ctx.closePath();ctx.fillStyle=face.kind==='liquid'?'rgba(53,153,181,.30)':'rgba(224,179,108,.95)';ctx.fill();}
  }
  if($('pointLayer').checked){const points=ui.viewer.observed_points.map((p,i)=>({p:project(p),color:ui.viewer.observed_colors[i]})).sort((a,b)=>a.p[2]-b.p[2]);for(const {p,color}of points){ctx.fillStyle=`rgb(${color.join(',')})`;ctx.fillRect(p[0],p[1],2.3*ratio,2.3*ratio);}}
  const h=ui.frame.estimate.liquid_height_m;
  if($('estimateLayer').checked&&Number.isFinite(h)){drawOutline(h,'#91e8bb',.94,[5*ratio,4*ratio]);line3([-rx,0,h],[rx,0,h],'#a6edc7',1,[5*ratio,4*ratio]);line3([0,-ry,h],[0,ry,h],'#a6edc7',1,[5*ratio,4*ratio]);const p=project([rx,0,h]);ctx.fillStyle='#a6edc7';ctx.font=12*ratio+'px Segoe UI';ctx.fillText('H = '+h.toFixed(3)+' m',p[0]+8,p[1]);}
  for(const [p,color,name]of [[[rx+.25,0,0],'#dd9292','X'],[[0,ry+.25,0],'#8accb1','Y'],[[0,0,cap+.2],'#7eb8db','Z']]){line3([0,0,0],p,color,1.5);const a=project(p);ctx.fillStyle=color;ctx.font=12*ratio+'px Segoe UI';ctx.fillText(name,a[0]+4,a[1]-4);}
}
canvas.addEventListener('pointerdown',event=>{drag={x:event.clientX,y:event.clientY,button:event.button};canvas.setPointerCapture(event.pointerId);});
canvas.addEventListener('pointermove',event=>{if(!drag)return;const dx=event.clientX-drag.x,dy=event.clientY-drag.y;if(drag.button===2){orbit.panX+=dx*(window.devicePixelRatio||1);orbit.panY+=dy*(window.devicePixelRatio||1);}else{orbit.yaw+=dx*.008;orbit.pitch=Math.max(-1.4,Math.min(1.4,orbit.pitch+dy*.008));}drag.x=event.clientX;drag.y=event.clientY;draw();});
canvas.addEventListener('pointerup',()=>drag=null);canvas.addEventListener('contextmenu',event=>event.preventDefault());canvas.addEventListener('wheel',event=>{event.preventDefault();orbit.zoom=Math.max(.35,Math.min(4,orbit.zoom*Math.exp(-event.deltaY*.001)));draw();},{passive:false});
$('resetView').onclick=()=>{orbit={yaw:-.65,pitch:.64,zoom:1,panX:0,panY:0};draw();};
for(const id of ['truthLayer','pointLayer','estimateLayer','cameraLayer'])$(id).addEventListener('change',draw);
new ResizeObserver(draw).observe(canvas);refreshLabels();setBusy(true);poll();
