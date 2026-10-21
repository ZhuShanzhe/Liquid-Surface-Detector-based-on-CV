"""Acceptance of this application's own Qt window; no external UI automation.

The helper uses a real QWebEnginePage, real initial capture, and window.grab().
It never creates synthetic measurements or changes the estimator/configuration.
"""
import hashlib
import json
import math
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import QObject, QTimer
from PySide6.QtWebEngineCore import QWebEngineScript


ERROR_PROBE = r"""
(() => {
  window.__desktopGuiSmokeErrors = [];
  window.addEventListener('error', event => {
    window.__desktopGuiSmokeErrors.push({kind:'error',message:String(event.message),
      source:String(event.filename),line:event.lineno});
  });
  window.addEventListener('unhandledrejection', event => {
    window.__desktopGuiSmokeErrors.push({kind:'unhandledrejection',message:String(event.reason)});
  });
})();
"""

SNAPSHOT = r"""
(() => {
  const el=id=>document.getElementById(id);
  const text=id=>el(id)?.textContent?.trim()??'';
  const visible=id=>{
    const node=el(id);if(!node)return false;
    const r=node.getBoundingClientRect(),style=getComputedStyle(node);
    if(node.closest('[hidden]')||style.display==='none'||style.visibility==='hidden')return false;
    return r.width>0&&r.height>0&&r.bottom>0&&r.top<innerHeight&&r.right>0&&r.left<innerWidth;
  };
  const frame=typeof ui!=='undefined'?ui.frame:null;
  const viewer=typeof ui!=='undefined'?ui.viewer:null;
  const canvas=el('viewer');let coloredPixels=0,canvasError=null;
  try{if(canvas&&canvas.width>0&&canvas.height>0){
    const data=canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data;
    for(let i=3;i<data.length;i+=4)if(data[i]>0)coloredPixels++;
  }}catch(error){canvasError=String(error);}
  return {readyState:document.readyState,frame_id:frame?.frame_id??null,
    ui_busy:typeof ui!=='undefined'?ui.busy:null,
    status:text('status'),measurement_state:text('measurementState'),
    actual_liquid_text:text('actualLiquid'),estimate_liquid_text:text('estimateLiquid'),
    actual_liquid_number:parseFloat(text('actualLiquid')),
    estimate_liquid_number:parseFloat(text('estimateLiquid')),
    rgb_width:el('rgb')?.naturalWidth??0,depth_width:el('depth')?.naturalWidth??0,
    canvas_width:canvas?.width??0,canvas_height:canvas?.height??0,
    canvas_colored_pixels:coloredPixels,canvas_error:canvasError,
    mesh_count:viewer?.meshes?.length??0,
    vertex_count:(viewer?.meshes??[]).reduce((sum,m)=>sum+(m.vertices?.length??0),0),
    installation_visible:visible('installationFields'),operation_visible:visible('operationFields'),
    controls_visible:Object.fromEntries(['containerShape','containerWidth','containerDepth',
      'containerHeight','cameraBottom','cameraX','cameraY','material','height'].map(id=>[id,visible(id)])),
    container_shape_options:Array.from(el('containerShape')?.options??[]).map(x=>x.value),
    timings:{pipeline:text('algorithmTime'),render:text('renderTime'),initialization:text('initTime')},
    errors:window.__desktopGuiSmokeErrors??[],
    window_size:[innerWidth,innerHeight],
    document_size:[document.documentElement.scrollWidth,document.documentElement.scrollHeight]};
})();
"""


class GuiSmoke(QObject):
    """Keeps all GUI operations on Qt's main thread and writes a failure receipt."""

    def __init__(self, app, window, state, work, output=None, version='', timeout=180.):
        super().__init__(window)
        self.app,self.window,self.state=app,window,state
        self.work=Path(work).resolve()
        self.output=Path(output).resolve() if output else self.work/'desktop-gui-self-test.json'
        self.output.parent.mkdir(parents=True,exist_ok=True)
        self.images=self.output.parent/(self.output.stem+'-images')
        self.images.mkdir(parents=True,exist_ok=True)
        self.page=None;self.done=False;self.pending=False;self.stage='initialization'
        self.started=time.monotonic();self.timeout=float(timeout)
        self.record=dict(schema_version=1,kind='real_qt_application_gui_smoke',version=version,
            frozen=bool(getattr(sys,'frozen',False)),started_utc=datetime.now(timezone.utc).isoformat(),
            work=str(self.work),success=False,checks={},screenshots=[],snapshots={},events=[],
            limitations=['single_default_installation_not_full_performance_evaluation',
                         'application_internal_canvas2d_not_webgl_or_external_window_automation'])
        self.timer=QTimer(self);self.timer.setInterval(400);self.timer.timeout.connect(self.tick);self.timer.start()
        app.lastWindowClosed.connect(lambda:self.record_event('lastWindowClosed'))
        app.aboutToQuit.connect(lambda:self.record_event('aboutToQuit'))
        window.destroyed.connect(lambda:self.record_event('windowDestroyed'))

    def record_event(self,name,detail=None):
        item=dict(name=name,detail=detail,elapsed=time.monotonic()-self.started)
        self.record['events'].append(item)
        print('GUI_SMOKE_EVENT',json.dumps(item,ensure_ascii=False),flush=True)

    def attach(self, page):
        self.page=page
        self.record_event('pageAttached')
        page.loadFinished.connect(lambda ok:self.record_event('pageLoadFinished',bool(ok)))
        page.renderProcessTerminated.connect(lambda status,code:self.fail('webengine_renderer_terminated:'+str(status)+':'+str(code)))
        probe=QWebEngineScript();probe.setName('desktop-gui-smoke-errors')
        probe.setInjectionPoint(QWebEngineScript.InjectionPoint.DocumentCreation)
        probe.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
        probe.setRunsOnSubFrames(False);probe.setSourceCode(ERROR_PROBE)
        page.scripts().insert(probe)

    def _check(self, name, condition):
        self.record['checks'][name]=bool(condition)
        return bool(condition)

    def _snapshot(self,callback):
        # Qt's QVariant conversion of nested JS objects differs across PySide
        # versions. Transport an explicit JSON string, never an opaque object.
        script='JSON.stringify('+SNAPSHOT.strip().rstrip(';')+')'
        def decoded(value):
            try:
                result=json.loads(value) if isinstance(value,str) else value
                if not isinstance(result,dict):
                    raise TypeError('expected_object_got_'+type(value).__name__)
            except Exception as error:
                self.fail('javascript_snapshot_decode:'+str(error));return
            callback(result)
        self.page.runJavaScript(script,decoded)

    def tick(self):
        if self.done:return
        if time.monotonic()-self.started>self.timeout:
            self.fail('gui_self_test_timeout:'+self.stage);return
        console=self.state.get('console')
        if console is None:return
        backend=console.state
        if backend.get('status')=='error':
            self.fail('backend_error:'+str(backend.get('message')));return
        if self.pending or self.page is None or self.stage!='initialization':return
        if backend.get('status')!='ready':return
        self.pending=True
        self._snapshot(self._initial_snapshot)

    def _initial_snapshot(self, snapshot):
        self.pending=False
        if self.done:return
        if not isinstance(snapshot,dict):return
        backend=self.state['console'].state
        self.record['snapshots']['initialization']=snapshot
        if snapshot.get('errors') or self.state.get('js_console_errors'):
            self.fail('javascript_errors');return
        # The one-second UI poll and image decode must both catch up to backend.
        if snapshot.get('frame_id')!=backend.get('frame_id') or snapshot.get('ui_busy'):
            return
        if not snapshot.get('rgb_width') or not snapshot.get('depth_width'):return
        expected=backend.get('estimate',{}).get('liquid_height_m')
        actual=backend.get('actual',{}).get('liquid_height_m')
        def matches(displayed,value):
            return isinstance(value,(int,float)) and isinstance(displayed,(int,float)) and math.isfinite(value) and math.isfinite(displayed) and abs(displayed-value)<=.000050001
        self._check('automatic_initialization_complete',bool(backend.get('initialization',{}).get('id')))
        self._check('qualified_real_input_measured',backend.get('input_qualification',{}).get('eligible') is True and backend.get('result',{}).get('inference_executed') is True)
        self._check('rgb_image_decoded',snapshot['rgb_width']>0)
        self._check('depth_image_decoded',snapshot['depth_width']>0)
        self._check('canvas2d_has_drawn_pixels',snapshot.get('canvas_colored_pixels',0)>0 and not snapshot.get('canvas_error'))
        self._check('actual_mesh_loaded',snapshot.get('mesh_count',0)>0 and snapshot.get('vertex_count',0)>0)
        self._check('displayed_liquid_matches_backend_4dp',matches(snapshot.get('estimate_liquid_number'),expected))
        self._check('displayed_truth_matches_backend_4dp',matches(snapshot.get('actual_liquid_number'),actual))
        self._check('separate_timing_fields_visible',all(snapshot.get('timings',{}).get(k) not in (None,'','—','未单独记录') for k in ('pipeline','render','initialization')))
        self.record['measurement']=dict(frame_id=backend.get('frame_id'),estimate=backend.get('estimate'),actual=backend.get('actual'),timing=backend.get('timing'),rendering=backend.get('rendering'),controls=backend.get('controls'))
        if not all(self.record['checks'].values()):self.fail('initialization_ui_check_failed');return
        self.stage='installation'
        self.page.runJavaScript("document.getElementById('tabInstallation').click();document.querySelector('.control-scroll').scrollTop=0;",lambda _value:QTimer.singleShot(450,self._installation_snapshot))

    def _installation_snapshot(self):
        if not self.done:self._snapshot(self._installation_checked)

    def _installation_checked(self,snapshot):
        if self.done:return
        if not isinstance(snapshot,dict):self.fail('installation_snapshot_missing');return
        self.record['snapshots']['installation']=snapshot
        visible=snapshot.get('controls_visible',{})
        self._check('installation_tab_visible',snapshot.get('installation_visible') is True and snapshot.get('operation_visible') is False)
        self._check('container_controls_visible',all(visible.get(k) for k in ('containerShape','containerWidth','containerDepth','containerHeight')))
        self._check('all_container_shape_options',set(snapshot.get('container_shape_options',[]))=={'cylinder','ellipse','rectangle'})
        if not self._screenshot('installation'):return
        if not all(self.record['checks'].values()):self.fail('installation_ui_check_failed');return
        self.stage='operation'
        self.page.runJavaScript("document.getElementById('tabOperation').click();document.querySelector('.control-scroll').scrollTop=0;",lambda _value:QTimer.singleShot(450,self._operation_snapshot))

    def _operation_snapshot(self):
        if not self.done:self._snapshot(self._operation_checked)

    def _operation_checked(self,snapshot):
        if self.done:return
        if not isinstance(snapshot,dict):self.fail('operation_snapshot_missing');return
        self.record['snapshots']['operation']=snapshot
        visible=snapshot.get('controls_visible',{})
        self._check('operation_tab_visible',snapshot.get('operation_visible') is True and snapshot.get('installation_visible') is False)
        self._check('operation_controls_visible',visible.get('material') is True and visible.get('height') is True)
        self._check('no_javascript_errors',not snapshot.get('errors') and not self.state.get('js_console_errors'))
        if not self._screenshot('operation'):return
        if not all(self.record['checks'].values()):self.fail('operation_ui_check_failed');return
        self.stage='offline_manual'
        self._check('offline_manual_file_exists',Path(self.state.get('manual_path','')).is_file())
        opener=self.state.get('open_manual')
        if not self.record['checks']['offline_manual_file_exists'] or not callable(opener):
            self.fail('offline_manual_resource_missing');return
        manual=opener()
        if not manual:self.fail('offline_manual_open_failed');return
        QTimer.singleShot(150,lambda:self._manual_checked(*manual))

    def _manual_checked(self,dialog,browser):
        if self.done:return
        text=browser.toPlainText()
        self._check('offline_manual_rendered_in_application',dialog.isVisible() and len(text)>1500 and '液深' in text and 'LOCALAPPDATA' in text and browser.source().isLocalFile())
        self.record['offline_manual']=dict(source=browser.source().toLocalFile(),text_characters=len(text),external_links_enabled=browser.openExternalLinks())
        self._check('offline_manual_no_external_browser',not browser.openExternalLinks())
        dialog.reject()
        self.finish(all(self.record['checks'].values()),None if all(self.record['checks'].values()) else 'offline_manual_ui_check_failed')

    def _screenshot(self,name):
        target=self.images/(name+'.png')
        try:
            pixmap=self.window.grab()
            if pixmap.isNull() or not pixmap.save(str(target),'PNG'):
                raise RuntimeError('window_grab_save_failed')
            self.record['screenshots'].append(dict(tab=name,path=str(target),width=pixmap.width(),height=pixmap.height(),sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
            return True
        except Exception as error:
            self.fail('screenshot_failed:'+str(error));return False

    def fail(self,reason):
        if not self.done:self.record_event('failure',reason)
        self.finish(False,reason)

    def finish(self,success,reason):
        if self.done:return
        self.done=True;self.timer.stop()
        self.record.update(success=bool(success),failure_reason=reason,stage=self.stage,
            elapsed_seconds=time.monotonic()-self.started,
            finished_utc=datetime.now(timezone.utc).isoformat(),
            js_console_errors=self.state.get('js_console_errors',[]))
        try:
            self.output.write_text(json.dumps(self.record,ensure_ascii=False,indent=2),encoding='utf-8')
        except Exception as error:
            self.record['success']=False
            self.record['receipt_write_error']=repr(error)
            print('GUI acceptance receipt write failed:',repr(error),flush=True);success=False
        self.state['closing']=True
        server=self.state.get('server')
        if server:threading.Thread(target=server.shutdown,daemon=True,name='gui-test-server-stop').start()
        console=self.state.get('console')
        if console:console.worker.shutdown(wait=False,cancel_futures=True)
        for stream in (sys.stdout,sys.stderr):
            try:stream.flush()
            except Exception:pass
        self.window.hide()
        self.app.exit(0 if success else 1)
