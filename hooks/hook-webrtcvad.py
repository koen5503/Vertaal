# Custom hook to bypass PyInstaller metadata crash for webrtcvad-wheels
from PyInstaller.utils.hooks import collect_dynamic_libs
binaries = collect_dynamic_libs('webrtcvad')
datas = []
