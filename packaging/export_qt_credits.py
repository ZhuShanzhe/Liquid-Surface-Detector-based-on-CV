"""Diagnostic credits-page exporter, not used in packaging.

The selected wheel does not expose this page. Use export_qt_license_snapshot.py
to preserve the official versioned license documentation instead.
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--disable-gpu')
from PySide6.QtCore import QTimer, QUrl
from PySide6.QtWidgets import QApplication
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile, QWebEngineUrlRequestInterceptor

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
app = QApplication([])
profile = QWebEngineProfile()


class Offline(QWebEngineUrlRequestInterceptor):
    def interceptRequest(self, info):
        if info.requestUrl().scheme() not in ('chrome', 'chrome-untrusted', 'qrc', 'data', 'about'):
            info.block(True)


interceptor = Offline()
profile.setUrlRequestInterceptor(interceptor)
page = QWebEnginePage(profile)
status = {'success':False}


def save(html):
    if len(html) < 10000 or 'license' not in html.lower():
        raise RuntimeError('Qt_Chromium_credits_export_incomplete')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding='utf-8')
    status['success'] = True
    print('CHROMIUM_CREDITS_EXPORTED', len(html), str(args.output), flush=True)
    app.quit()


def loaded(ok):
    if ok:
        page.toHtml(save)
    else:
        print('CHROMIUM_CREDITS_LOAD_FAILED', flush=True)
        app.quit()


page.loadFinished.connect(loaded)
QTimer.singleShot(60000, app.quit)
page.setUrl(QUrl('chrome://credits/'))
app.exec()
del page
del profile
raise SystemExit(0 if status['success'] else 1)
