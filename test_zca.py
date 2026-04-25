import socket
from zeroconf import ServiceInfo, Zeroconf
try:
    local_ip = '192.168.1.100'
    active_port = 8000
    info = ServiceInfo(
        "_http._tcp.local.",
        "Ondertitels._http._tcp.local.",
        addresses=[socket.inet_aton(local_ip)],
        port=active_port,
        properties={'path': '/live'},
        server="ondertitels.local.",
    )
    print("ServiceInfo created!")
    
    z = Zeroconf()
    z.register_service(info)
    print("Registered!")
    z.close()
except Exception as e:
    import traceback
    traceback.print_exc()
