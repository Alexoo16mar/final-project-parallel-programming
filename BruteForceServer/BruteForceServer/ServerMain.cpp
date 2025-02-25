#include <winsock2.h>
#include <ws2tcpip.h>
#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <sstream>
#include <cmath>
#include <chrono>     // Para medir el tiempo
#pragma comment(lib, "ws2_32.lib")

#define PORT 65432
#define CHARSET_SIZE 62
#define MAX_PASSWORD_LENGTH 10

// Configuración de la contraseña objetivo
const char* TARGET_PASSWORD = "aBc123";
const int PASSWORD_LENGTH = 6;

// Definición del tamaño de cada bloque (chunk) para el balanceo dinámico
const unsigned long long CHUNK_SIZE = 1000000; // Ajustable según necesidad

// Estructura para almacenar información de cada cliente conectado
struct ClientInfo {
    SOCKET sock;
    unsigned long long currentChunkStart;
    unsigned long long currentChunkEnd;
    bool busy;
};

int main() {
    if (PASSWORD_LENGTH > MAX_PASSWORD_LENGTH) {
        std::cerr << "Error: La contraseña excede la longitud máxima." << std::endl;
        return 1;
    }

    // Calcular total de combinaciones: CHARSET_SIZE^PASSWORD_LENGTH
    unsigned long long totalCombinations = 1;
    for (int i = 0; i < PASSWORD_LENGTH; i++) {
        totalCombinations *= CHARSET_SIZE;
    }

    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "Error al inicializar Winsock" << std::endl;
        return 1;
    }

    SOCKET serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket == INVALID_SOCKET) {
        std::cerr << "Error al crear el socket" << std::endl;
        WSACleanup();
        return 1;
    }

    sockaddr_in serverAddr{};
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(PORT);

    if (bind(serverSocket, (sockaddr*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
        std::cerr << "Error en bind" << std::endl;
        closesocket(serverSocket);
        WSACleanup();
        return 1;
    }

    if (listen(serverSocket, 5) == SOCKET_ERROR) {
        std::cerr << "Error en listen" << std::endl;
        closesocket(serverSocket);
        WSACleanup();
        return 1;
    }

    std::cout << "[Servidor] Esperando clientes..." << std::endl;

    std::vector<ClientInfo> clients;
    // Se esperan 2 clientes (se puede extender fácilmente)
    while (clients.size() < 2) {
        SOCKET clientSocket = accept(serverSocket, NULL, NULL);
        if (clientSocket == INVALID_SOCKET) {
            std::cerr << "[Servidor] Error en accept." << std::endl;
            continue;
        }

        // Enviar la longitud de la contraseña y la contraseña objetivo
        int netLength = htonl(PASSWORD_LENGTH);
        send(clientSocket, (char*)&netLength, sizeof(netLength), 0);
        send(clientSocket, TARGET_PASSWORD, PASSWORD_LENGTH, 0);

        ClientInfo ci;
        ci.sock = clientSocket;
        ci.busy = false;
        ci.currentChunkStart = 0;
        ci.currentChunkEnd = 0;
        clients.push_back(ci);
        std::cout << "[Servidor] Cliente " << clients.size() << " conectado." << std::endl;
    }

    // Variables para el balanceo dinámico
    unsigned long long nextChunkStart = 0;
    std::queue<std::pair<unsigned long long, unsigned long long>> pendingChunks;

    // Registra el instante de inicio de la búsqueda
    auto startTime = std::chrono::steady_clock::now();

    // Asignar un bloque inicial a cada cliente
    for (auto& client : clients) {
        unsigned long long startChunk = nextChunkStart;
        unsigned long long endChunk = (nextChunkStart + CHUNK_SIZE < totalCombinations)
            ? (nextChunkStart + CHUNK_SIZE)
            : totalCombinations;
        nextChunkStart = endChunk;
        client.currentChunkStart = startChunk;
        client.currentChunkEnd = endChunk;
        client.busy = true;

        std::ostringstream oss;
        oss << "CHUNK," << startChunk << "," << endChunk;
        std::string msg = oss.str();
        send(client.sock, msg.c_str(), msg.size(), 0);
        std::cout << "[Servidor] Asignado rango a cliente: " << msg << std::endl;
    }

    bool passwordFound = false;
    std::string foundPassword;

    // Bucle principal de asignación de bloques y recepción de mensajes
    while (!passwordFound) {
        fd_set readfds;
        FD_ZERO(&readfds);
        SOCKET maxSock = 0;
        for (auto& client : clients) {
            FD_SET(client.sock, &readfds);
            if (client.sock > maxSock) maxSock = client.sock;
        }

        timeval timeout;
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;
        int activity = select(maxSock + 1, &readfds, NULL, NULL, &timeout);
        if (activity < 0) {
            std::cerr << "[Servidor] Error en select." << std::endl;
            break;
        }

        // Procesar mensajes recibidos de cada cliente
        for (auto it = clients.begin(); it != clients.end(); ) {
            if (FD_ISSET(it->sock, &readfds)) {
                char buffer[256] = { 0 };
                int bytesReceived = recv(it->sock, buffer, sizeof(buffer) - 1, 0);
                if (bytesReceived <= 0) {
                    std::cerr << "[Servidor] Cliente desconectado. Reasignando su rango." << std::endl;
                    // Si el cliente se desconecta y estaba procesando un bloque, lo reinsertamos en la cola
                    if (it->busy) {
                        pendingChunks.push({ it->currentChunkStart, it->currentChunkEnd });
                    }
                    closesocket(it->sock);
                    it = clients.erase(it);
                    continue;
                }
                buffer[bytesReceived] = '\0';
                std::string msg(buffer);
                // Procesar el mensaje: "DONE" o "FOUND,<password>"
                if (msg.find("DONE") == 0) {
                    // El cliente terminó el bloque sin hallar la contraseña
                    it->busy = false;
                    unsigned long long newStart, newEnd;
                    if (!pendingChunks.empty()) {
                        auto chunk = pendingChunks.front();
                        pendingChunks.pop();
                        newStart = chunk.first;
                        newEnd = chunk.second;
                    }
                    else if (nextChunkStart < totalCombinations) {
                        newStart = nextChunkStart;
                        newEnd = (nextChunkStart + CHUNK_SIZE < totalCombinations)
                            ? (nextChunkStart + CHUNK_SIZE)
                            : totalCombinations;
                        nextChunkStart = newEnd;
                    }
                    else {
                        // No hay más bloques; se le indica al cliente que termine
                        std::string termMsg = "TERMINATE";
                        send(it->sock, termMsg.c_str(), termMsg.size(), 0);
                        ++it;
                        continue;
                    }
                    it->currentChunkStart = newStart;
                    it->currentChunkEnd = newEnd;
                    it->busy = true;
                    std::ostringstream oss;
                    oss << "CHUNK," << newStart << "," << newEnd;
                    std::string chunkMsg = oss.str();
                    send(it->sock, chunkMsg.c_str(), chunkMsg.size(), 0);
                    std::cout << "[Servidor] Reasignado rango a cliente: " << chunkMsg << std::endl;
                }
                else if (msg.find("FOUND,") == 0) {
                    // Se recibió la contraseña encontrada (formato: "FOUND,<password>")
                    size_t commaPos = msg.find(',');
                    if (commaPos != std::string::npos) {
                        foundPassword = msg.substr(commaPos + 1);
                        passwordFound = true;
                        std::cout << "[Servidor] Contraseña encontrada por un cliente: " << foundPassword << std::endl;
                        break;
                    }
                }
            }
            ++it;
        }

        if (passwordFound)
            break;

        // Si ya se asignaron todos los bloques y ningún cliente está ocupado, se termina la búsqueda
        if (nextChunkStart >= totalCombinations && pendingChunks.empty()) {
            bool allIdle = true;
            for (auto& c : clients) {
                if (c.busy) { allIdle = false; break; }
            }
            if (allIdle)
                break;
        }
    }

    // Registra el instante de fin de la búsqueda y calcula la duración
    auto endTime = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsedSeconds = endTime - startTime;

    // Notificar a todos los clientes para que finalicen
    for (auto& client : clients) {
        std::string termMsg = "TERMINATE";
        send(client.sock, termMsg.c_str(), termMsg.size(), 0);
        closesocket(client.sock);
    }
    closesocket(serverSocket);
    WSACleanup();

    if (!foundPassword.empty())
        std::cout << "[Servidor] Contraseña encontrada: " << foundPassword << std::endl;
    else
        std::cout << "[Servidor] Contraseña no encontrada en el espacio de búsqueda." << std::endl;

    std::cout << "[Servidor] Tiempo total de búsqueda: " << elapsedSeconds.count() << " segundos." << std::endl;

    return 0;
}
