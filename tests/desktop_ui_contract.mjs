/* UI-only contract tests. Synthetic payloads below are fixtures, not measurements. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import {fileURLToPath} from 'node:url';

const repo=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const html=fs.readFileSync(path.join(repo,'src/liquid_depth/ui/scene_console.html'),'utf8');
const js=fs.readFileSync(path.join(repo,'src/liquid_depth/ui/scene_console.js'),'utf8');
assert.ok(!/https?:\/\//.test(html),'The UI must not require external web resources.');
const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);
assert.equal(new Set(ids).size,ids.length,'Duplicate element IDs');
const context2d=new Proxy({}, {get:(obj,key)=>obj[key]??(()=>{}),set:(obj,key,value)=>(obj[key]=value,true)});
function element(id=''){
  return {id,_value:'',get value(){return this._value;},set value(v){this._value=String(v);},textContent:'',checked:false,disabled:false,hidden:false,dataset:{},attributes:{},
    classList:{toggle(){}},addEventListener(){},setAttribute(k,v){this.attributes[k]=v;},removeAttribute(k){delete this[k];delete this.attributes[k];},
    replaceChildren(...children){this.children=children;},append(child){(this.children??=[]).push(child);},
    getContext(){return context2d;},getBoundingClientRect(){return {width:700,height:400};},setPointerCapture(){}};
}
const elements=Object.fromEntries(ids.map(id=>[id,element(id)]));
for(const m of html.matchAll(/<input\b([^>]+)>/g)){
  const id=/\bid="([^"]+)"/.exec(m[1])?.[1];if(!id)continue;
  elements[id].value=/\bvalue="([^"]*)"/.exec(m[1])?.[1]??'';
  elements[id].checked=/\bchecked\b/.test(m[1]);
}
for(const m of html.matchAll(/<select\b[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g))elements[m[1]].value=/value="([^"]*)"/.exec(m[2])[1];
const ctx=vm.createContext({console,Math,Number,Array,String,JSON,Promise,fetch:()=>new Promise(()=>{}),setTimeout(){},
  document:{getElementById(id){assert.ok(elements[id],`Missing HTML element ${id}`);return elements[id];},createTextNode:text=>({text}),createElement:()=>element()},
  window:{devicePixelRatio:1},ResizeObserver:class{observe(){}}});
vm.runInContext(js,ctx,{filename:'scene_console.js'});
const run=source=>vm.runInContext(source,ctx);
run('ui.ready=true;setBusy(false);');
assert.equal(run('validate(controls())'),'');
assert.equal(run('controls().calibration_noise_mode'),'fixed_10mm');
assert.equal(run('controls().container_radius_x_m'),.55);
assert.equal(elements.containerDepth.disabled,true,'Cylinder Y must follow X.');
elements.containerWidth.value='1.8';run('refreshLabels(false)');
assert.equal(elements.containerDepth.value,'1.8');
assert.equal(run('controls().container_radius_y_m'),.9);
elements.containerShape.value='rectangle';elements.containerDepth.value='1.2';run('refreshLabels(false)');
assert.equal(elements.containerDepth.disabled,false);
assert.equal(run('controls().container_radius_y_m'),.6);
elements.cameraX.value='.3';assert.match(run('validate(controls())'),/偏心 X/);
elements.cameraX.value='.1';assert.equal(run('validate(controls())'),'');
elements.height.value='';assert.match(run('validate(controls())'),/有效数值/);
elements.height.value='1.6';assert.match(run('validate(controls())'),/超过容器/);
elements.height.value='.6';elements.wave.value='120';assert.equal(run('validate(controls())'),'');
elements.height.value='.1';assert.match(run('validate(controls())'),/起伏过大/);
elements.height.value='.6';elements.wave.value='0';elements.material.value='homogeneous_transparent';elements.floats.value='1';
assert.match(run('validate(controls())'),/尚未适配/);
elements.floats.value='0';elements.material.value='ordinary_diffuse';
elements.containerHeight.value='9.8';elements.cameraBottom.value='10';assert.match(run('validate(controls())'),/无法/);
elements.containerHeight.value='1.5';elements.cameraBottom.value='4';
elements.autoFrame.checked=false;elements.cameraBottom.value='1.55';assert.match(run('validate(controls())'),/高出容器口/);
elements.autoFrame.checked=true;assert.ok(run('framing(controls()).bottom')>1.6);
elements.cameraBottom.value='4';
run('ui.viewer={shape:"rectangle",radii_m:[.9,.6],radius_m:.9,capacity_m:1.5};');
assert.equal(run('outline(.4).length'),5);
assert.equal(run('outline(.4)[1][0]'),.9);
assert.equal(run('outline(.4)[2][1]'),.6);
run('ui.viewer.shape="ellipse";');assert.equal(run('outline(.4).length'),81);
assert.ok(Math.abs(run('outline(.4)[20][1]')-.6)<1e-10);
const payload={controls:JSON.parse(run('JSON.stringify(controls())')),initialization:{id:'fixture'},viewer:{shape:'rectangle',radii_m:[.9,.6],radius_m:.9,capacity_m:1.5,camera_to_bottom_m:4,camera_xy_m:[.1,0],meshes:[],observed_points:[],observed_colors:[]},actual:{liquid_height_m:.6,highest_height_m:.6},estimate:{liquid_height_m:.605,highest_height_m:.605},rgb:'data:image/jpeg;base64,fixture',depth_preview:'data:image/png;base64,fixture',depth_preview_range_m:[2,4],frame_id:'testfixture',input_qualification:{eligible:true},validation_envelope:{exploratory:false},timing:{render_ms:2000,pipeline_ms:200,initialization_ms:3000},result:{classification:{material_score:.5},precision_eligible:true,module_plan:{tags:[],blocked_reasons:[],warnings:[],modules:[],executed_modules:[],composite:false}}};
ctx.testPayload=payload;run('show(testPayload);setBusy(false);');
assert.equal(elements.algorithmTime.textContent,'200.0 ms');
assert.equal(elements.initTime.textContent,'3.0 s');
assert.equal(elements.renderTime.textContent,'2.0 s');
assert.match(elements.errorText.textContent,/5.0 mm/);
assert.match(elements.measurementState.textContent,/本帧容差内/);
assert.match(elements.peakErrorText.textContent,/不适用/);
assert.equal(elements.depth.src,payload.depth_preview);
payload.input_qualification.eligible=false;payload.estimate.liquid_height_m=null;payload.estimate.highest_height_m=null;
payload.result.module_plan.blocked_reasons=['insufficient_visible_liquid_returns'];
run('show(testPayload);');
assert.equal(elements.algorithmTime.textContent,'未执行');
assert.equal(elements.measurementState.textContent,'不处理 · 不计入统计');
assert.match(elements.routeNote.textContent,/分类和测量均未执行/);
assert.match(elements.message.textContent,/可见有效液面点不足/);
console.log('desktop_ui_contract: all assertions passed (synthetic UI fixtures only)');
