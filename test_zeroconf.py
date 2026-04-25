from zeroconf import ServiceInfo, Zeroconf
import socket
import time

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(('10.255.255.255', 1))
    local_ip = s.getsockname()[0]
except Exception:
    local_ip = '127.0.0.1'
finally:
    s.close()

info = ServiceInfo(
    "_http._tcp.local.",
    "Ondertitels._http._tcp.local.",
    addresses=[socket.inet_aton(local_ip)],
    port=80,
    properties={'path': '/live'},
    server="ondertitels.local.",
)
z = Zeroconf()
z.register_service(info)
print("Registered! Press Ctrl+C...")
try:
    time.sleep(2)
    print("Done sleep")
except:
    pass
z.unregister_service(info)
z.close()
