# -*- mode: python ; coding: utf-8 -*-
import os

datas = [
    ('templates', 'templates'),
    ('SpGlos.txt', '.'),
]

if os.path.exists('ondertitels-486017-0ee48ab1ba8d.json'):
    datas.append(('ondertitels-486017-0ee48ab1ba8d.json', '.'))

for wav in ['BV.wav', 'BV2.wav']:
    if os.path.exists(wav):
        datas.append((wav, '.'))

if os.path.exists('.env.example'):
    datas.append(('.env.example', '.'))

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'zeroconf',
        'zeroconf._utils.ipaddress',
        'zeroconf._services.browser',
        'sounddevice',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan.on',
        'websockets',
        'websockets.legacy',
        'websockets.legacy.client',
        'websockets.legacy.server',
        'fastapi',
        'starlette',
        'pydantic',
    ],
    hookspath=['hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
