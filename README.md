#  BruteForce CUDA + RadminVPN

Este proyecto implementa un **ataque de fuerza bruta ** para descifrar contraseñas, utilizando **CUDA** para aprovechar la potencia de la **GPU** y **RadminVPN** para la comunicación entre clientes y servidor.  

Además, incluye una **versión secuencial** que permite comparar el rendimiento entre la ejecución en **CPU** y **GPU**.

---

##  Descripción

El sistema consta de dos implementaciones:

### Versión con CUDA (Distribuida)
- **Servidor:**  
  - Define una contraseña objetivo y divide el espacio de búsqueda en **rangos**.  
  - Asigna los rangos a clientes conectados y espera una respuesta.  

- **Clientes:**  
  - Reciben un **rango de búsqueda** desde el servidor.  
  - Ejecutan el ataque de fuerza bruta en la **GPU** mediante **CUDA**.  

La comunicación entre clientes y servidor se realiza a través de **sockets (Winsock2)** en Windows.

### Versión Secuencial
- Realiza la búsqueda **de manera iterativa en la CPU**.
- Se conecta al servidor, recibe un rango de búsqueda y prueba todas las combinaciones dentro de ese rango.
- Permite comparar el rendimiento de **CPU vs GPU (CUDA)**, evaluando la mejora de velocidad lograda con la aceleración en la GPU.

---

##  Requisitos

###  Software
- [RadminVPN](https://www.radmin-vpn.com/) → Para conectar clientes y servidor en una red virtual.  
- **CUDA Toolkit** → Necesario para la ejecución en la GPU.  
- **Microsoft Visual Studio** → Para compilar el código CUDA en Windows.

###  Hardware
- **GPU compatible con CUDA**.

---


## Comparación de Rendimiento

| Implementación | Plataforma | Método | Aceleración |
|---------------|-----------|--------|-------------|
| **Secuencial** | CPU | Iterativo |  No |
| **CUDA** | GPU | Paralelizado |  Sí |

