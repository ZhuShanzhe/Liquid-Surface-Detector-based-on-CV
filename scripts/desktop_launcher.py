"""Frozen Windows entry point. No system Python, browser, SSH, or compiler required."""
import argparse
import json
import logging
import multiprocessing
import os
import sys
import threading
import time
import traceback
from pathlib import Path

if not getattr(sys, 'frozen', False):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from liquid_depth.runtime_paths import resource_root, user_data_root

VERSION = '1.0.0'


def configure_runtime():
    root = resource_root()
    sys.path.insert(0, str(root / 'scripts'))
    os.environ.setdefault('LIQUID_DEVICE', 'cpu')
    os.environ.setdefault('LIQUID_RENDER_ENGINE', 'cycles_cpu')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('MKL_NUM_THREADS', '4')
    import torch
    import cv2
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    return root


def blender_path(root, explicit=None):
    if explicit:
        path = Path(explicit).resolve()
    else:
        path = root / 'vendor/blender/blender.exe'
    if not path.is_file():
        raise FileNotFoundError('未找到内置 Blender，请完整解压软件目录，不要只移动 EXE。')
    return path


def make_console(root, work, explicit_blender=None):
    from liquid_depth.scene_console_web import Console
    return Console(work, str(blender_path(root, explicit_blender)),
                   root / 'assets/models/scene_router/scene_head.pt')


def self_test(args):
    """The same acceptance harness runs from source and the frozen EXE."""
    from desktop_freeze_smoke import run
    return run(args,version=VERSION,configure_runtime=configure_runtime,
               make_console=make_console,user_data_root=user_data_root)


def gui(args):
    # Software rasterization also keeps the embedded interface independent of
    # a discrete GPU. Chromium's sandbox remains enabled.
    os.environ.setdefault('QT_OPENGL', 'software')
    os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = (
        os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS', '') + ' --disable-gpu').strip()
    from PySide6.QtCore import QObject, Signal, QUrl, QTimer, Qt
    from PySide6.QtGui import QAction, QDesktopServices
    from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QMessageBox, QFileDialog, QDialog, QTextBrowser, QVBoxLayout, QDialogButtonBox
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile, QWebEngineUrlRequestInterceptor
    app = QApplication(sys.argv[:1])
    app.setApplicationName('RGBD液深仿真工作台')
    app.setOrganizationName('LiquidDepth')
    work = args.work or user_data_root() / ('session-' + time.strftime('%Y%m%d-%H%M%S'))
    work.mkdir(parents=True,exist_ok=True)
    log = (work/'desktop.log').open('a',encoding='utf-8',buffering=1)
    sys.stdout=sys.stderr=log
    logging.basicConfig(stream=log,level=logging.INFO)

    class Bridge(QObject):
        ready=Signal(str)
        failed=Signal(str)
    bridge=Bridge()
    state={'console':None,'server':None,'closing':False}
    class LocalOnly(QWebEngineUrlRequestInterceptor):
        def interceptRequest(self,info):
            url=info.requestUrl()
            allowed=url.scheme() in ('data','blob','qrc','about') or (
                url.scheme()=='http' and url.host()=='127.0.0.1' and
                state['server'] is not None and url.port()==state['server'].server_port)
            if not allowed:info.block(True)
    class LocalPage(QWebEnginePage):
        def acceptNavigationRequest(self,url,kind,is_main):
            return url.scheme()=='about' or (url.scheme()=='http' and url.host()=='127.0.0.1'
                    and state['server'] and url.port()==state['server'].server_port)
        def javaScriptConsoleMessage(self,level,message,line_number,source_id):
            if args.gui_self_test and level==QWebEnginePage.JavaScriptConsoleMessageLevel.ErrorMessageLevel:
                state.setdefault('js_console_errors',[]).append(dict(message=message,line=line_number,source=source_id))
            super().javaScriptConsoleMessage(level,message,line_number,source_id)
    class Window(QMainWindow):
        def closeEvent(self,event):
            if gui_smoke and not gui_smoke.done:
                gui_smoke.fail('window_closed_before_gui_acceptance')
                event.accept();return
            console=state['console']
            if console and any(v['status'] in ('queued','running') for v in console.jobs.values()):
                QMessageBox.information(self,'正在采集','当前仿真仍在采集，请等待本帧完成后退出。')
                event.ignore();return
            state['closing']=True
            if state['server']:
                threading.Thread(target=state['server'].shutdown,daemon=True).start()
            if console:console.worker.shutdown(wait=False,cancel_futures=True)
            event.accept()
    window=Window();window.setWindowTitle('RGBD 液深仿真工作台 | ToF 正式链路 · 离线版')
    window.resize(1440,960);window.setMinimumSize(1100,760)
    label=QLabel('正在加载内置仿真与模型…\n首次加载可能需要数十秒。无需联网或连接相机。')
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setStyleSheet('background:#eef3f7;color:#18354a;font-size:18px;line-height:2;')
    window.setCentralWidget(label)
    menu=window.menuBar().addMenu('文件')
    export=QAction('导出当前测量记录…',window)
    def export_result():
        console=state['console']
        if console is None or console.state.get('status')!='ready':
            QMessageBox.information(window,'没有记录','请先完成一次仿真测量。');return
        path,_=QFileDialog.getSaveFileName(window,'导出测量记录','measurement.json','JSON (*.json)')
        if path:
            from liquid_depth.scene_console_web import plain
            value={k:v for k,v in console.state.items() if k not in ('viewer','rgb','depth_preview')}
            try:
                Path(path).write_text(json.dumps(plain(value),ensure_ascii=False,indent=2),encoding='utf-8')
            except OSError as exc:
                QMessageBox.warning(window,'导出失败',str(exc))
    export.triggered.connect(export_result);menu.addAction(export)
    logs=menu.addAction('打开本次数据与日志目录')
    logs.triggered.connect(lambda:QDesktopServices.openUrl(QUrl.fromLocalFile(str(work))))
    menu.addSeparator();menu.addAction('退出',window.close)
    help_menu=window.menuBar().addMenu('帮助')
    def help_text():
        QMessageBox.information(window,'离线软件使用说明',
            '1. 启动时自动完成默认空槽标定与一次测量。\n'
            '2. 安装页可改容器形状、尺寸、相机到底距和偏心，改变后应用会重新标定。\n'
            '3. 工况页可调液位、光照、眩光、波面、漂浮物、深度缺失。\n'
            '4. 左侧拖动旋转、滚轮缩放；真实值与估计值分栏显示。\n'
            '5. 透明/半透明仅支持均匀平整无遮挡，拒绝不合格输入且不执行测量。\n'
            '6. CPU离线版保持算法/权重；本机延时不沿用服务器GPU报告。\n'
            '7. 原始采集、标定和结果保存在本地应用数据目录。\n\n'
            '仅为仿真展示与实地界面模板，不含相机SDK、泵阀或PLC控制。')
    help_menu.addAction('操作说明',help_text)
    def full_manual():
        if hasattr(window,'_manual_dialog'):
            dialog,browser=window._manual_dialog,window._manual_browser
            dialog.show();dialog.raise_();return dialog,browser
        manual_path=resource_root()/'docs/cpu_desktop_help.html'
        if not manual_path.is_file():
            QMessageBox.warning(window,'手册资源缺失','请保留完整软件目录；未找到 '+str(manual_path))
            return None
        dialog=QDialog(window);dialog.setWindowTitle('CPU 离线版完整使用手册');dialog.resize(1000,780)
        browser=QTextBrowser(dialog)
        browser.setOpenExternalLinks(False);browser.setOpenLinks(False)
        browser.setStyleSheet('QTextBrowser {background:white;color:#173348;font:14px "Microsoft YaHei";padding:16px;}')
        browser.anchorClicked.connect(lambda url:browser.scrollToAnchor(url.fragment()) if url.fragment() else None)
        browser.setSource(QUrl.fromLocalFile(str(manual_path)))
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Close,dialog);buttons.rejected.connect(dialog.reject)
        layout=QVBoxLayout(dialog);layout.addWidget(browser);layout.addWidget(buttons)
        window._manual_dialog=dialog;window._manual_browser=browser
        dialog.show();return dialog,browser
    help_menu.addAction('完整离线使用手册',full_manual)
    state['open_manual']=full_manual
    state['manual_path']=str(resource_root()/'docs/cpu_desktop_help.html')
    def about():
        QMessageBox.information(window,'关于',f'RGBD 液深仿真工作台 {VERSION}\nToF 正式 V14测量链路\n'
             '内置 Blender 4.2.0 / PyTorch CPU / Qt WebEngine\n'
             '模型与资源离线加载；不要求系统 Python / 浏览器。\n'
             'Blender 与 Qt 等第三方许可随软件附带。\n'
             '未进行工业真机认证，不用于安全联锁。')
    help_menu.addAction('关于与应用边界',about)
    gui_smoke=None
    if args.gui_self_test:
        from desktop_gui_smoke import GuiSmoke
        gui_smoke=GuiSmoke(app,window,state,work,args.output,version=VERSION,timeout=args.gui_self_test_timeout)
        window._gui_smoke=gui_smoke
    def loaded(url):
        if state['closing']:return
        profile=QWebEngineProfile(window)
        profile.setHttpCacheType(QWebEngineProfile.HttpCacheType.MemoryHttpCache)
        profile.setPersistentCookiesPolicy(QWebEngineProfile.PersistentCookiesPolicy.NoPersistentCookies)
        interceptor=LocalOnly(profile);profile.setUrlRequestInterceptor(interceptor)
        view=QWebEngineView(window);page=LocalPage(profile,view);view.setPage(page)
        if gui_smoke:gui_smoke.attach(page)
        window._profile=profile;window._interceptor=interceptor;window._view=view
        window.setCentralWidget(view);view.setUrl(QUrl(url))
    def failed(message):
        label.setText('启动失败\n'+message+'\n日志：'+str(work/'desktop.log'))
        if gui_smoke:gui_smoke.fail('startup_failed:'+str(message))
    bridge.ready.connect(loaded);bridge.failed.connect(failed)
    def boot():
        try:
            root=configure_runtime();console=make_console(root,work,args.blender);state['console']=console
            if state['closing']:
                console.worker.shutdown(wait=False,cancel_futures=True);return
            from liquid_depth.scene_console_web import serve
            def ready(server):
                state['server']=server
                if state['closing']:
                    threading.Thread(target=server.shutdown,daemon=True).start();return
                bridge.ready.emit('http://127.0.0.1:'+str(server.server_port))
                if not args.no_auto_initialize and not state['closing']:console.submit({},initialize=True)
            serve(console,0,ready_callback=ready)
        except Exception as exc:
            traceback.print_exc();bridge.failed.emit(str(exc))
    window.show()
    QTimer.singleShot(30,lambda:threading.Thread(target=boot,daemon=True,name='local-runtime').start())
    exit_code=app.exec()
    if gui_smoke and not gui_smoke.done:
        gui_smoke.fail('qt_event_loop_exited_without_acceptance:'+str(exit_code))
        return 1
    if gui_smoke:return 0 if exit_code==0 and gui_smoke.record.get('success') else 1
    return exit_code


def main():
    multiprocessing.freeze_support()
    parser=argparse.ArgumentParser(description=__doc__)
    tests=parser.add_mutually_exclusive_group()
    tests.add_argument('--self-test',action='store_true')
    tests.add_argument('--gui-self-test',action='store_true',help='Run real Qt initialization, canvas/image checks and two own-window screenshots')
    parser.add_argument('--gui-self-test-timeout',type=float,default=180.)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--work',type=Path)
    parser.add_argument('--blender')
    parser.add_argument('--no-auto-initialize',action='store_true')
    args=parser.parse_args()
    if args.gui_self_test and args.no_auto_initialize:
        parser.error('--gui-self-test requires automatic initialization')
    if not 1<=args.gui_self_test_timeout<=1200:
        parser.error('--gui-self-test-timeout must be between 1 and 1200 seconds')
    return self_test(args) if args.self_test else gui(args)


if __name__=='__main__':
    raise SystemExit(main())
