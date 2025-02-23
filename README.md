# BruteForce CUDA + RadminVPN

Este proyecto implementa un ataque de fuerza bruta distribuido para descifrar contraseñas mediante CUDA, aprovechando la potencia de la GPU y una red privada virtual (RadminVPN) para la comunicación entre clientes y el servidor.

##  Descripción

El sistema consta de dos componentes principales:

1. **Servidor**: Se ejecuta en una máquina y define una contraseña objetivo. Divide el espacio de búsqueda entre dos clientes y espera recibir la contraseña correcta.
2. **Clientes**: Cada cliente recibe un rango de búsqueda y usa CUDA para ejecutar la búsqueda en la GPU.

La comunicación entre clientes y servidor se realiza mediante **sockets** en Windows (`Winsock2`), mientras que la aceleración del ataque de fuerza bruta se implementa con **CUDA**.

---

##  Requisitos

###  Software:
- [RadminVPN](https://www.radmin-vpn.com/) (para conectar clientes y servidor en una red virtual)
- CUDA Toolkit
- Microsoft Visual Studio (para compilar con CUDA)

###  Hardware:
- GPU compatible con CUDA

---

## 🚀Instalación y Uso

### 1Configurar RadminVPN
1. Instala **[RadminVPN](https://www.radmin-vpn.com/)** en todas las máquinas (servidor y clientes).
2. Crea una **Red Privada** en RadminVPN desde el servidor y únete a ella desde los clientes.
3. Copia la dirección IP de RadminVPN del servidor y configúrala en el código del cliente:
