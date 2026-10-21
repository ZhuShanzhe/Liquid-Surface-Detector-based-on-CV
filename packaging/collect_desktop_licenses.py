"""Collect unmodified notices from installed wheels; no network at build time."""
import importlib.metadata
import re
import sys
from pathlib import Path


def license_data(repo):
    data = []
    for distribution in importlib.metadata.distributions():
        name = re.sub(r'[^a-zA-Z0-9_.-]', '_', distribution.metadata['Name'])
        for relative in distribution.files or ():
            filename = relative.name.lower()
            if (filename in ('metadata', 'copyright', 'authors') or
                filename.startswith(('license', 'copying', 'notice', 'copyright'))):
                if relative.suffix.lower() in ('.py', '.pyc', '.pyd', '.dll', '.xml'):
                    continue
                path = Path(distribution.locate_file(relative))
                if path.is_file():
                    data.append((str(path), str(Path('licenses/python')/name/relative.parent)))
    python_license = Path(sys.base_prefix)/'LICENSE.txt'
    if not python_license.is_file():
        raise RuntimeError('CPython_license_required_for_distribution')
    data.append((str(python_license), 'licenses/CPython'))
    # The wheel's commercial alternative is retained verbatim, alongside the
    # actual LGPL/GPL alternatives. Never infer a commercial Qt entitlement.
    required = ('LGPL-3.0.txt','GPL-3.0.txt','GPL-2.0.txt','LGPL-2.1.txt',
                'GFDL-1.3.txt','qt-chromium-credits.html',
                'qt-chromium-credits.sources.json')
    for name in required:
        path = repo/'docs/third_party_licenses'/name
        if not path.is_file() or path.stat().st_size < 100:
            raise RuntimeError('desktop_distribution_license_resource_missing:'+name)
    data += [(str(repo/'docs/third_party_licenses'), 'licenses/upstream'),
             (str(repo/'docs/THIRD_PARTY_NOTICES.rst'), 'licenses')]
    return data
