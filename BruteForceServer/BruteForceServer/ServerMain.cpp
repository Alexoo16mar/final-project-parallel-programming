#include <winsock2.h>
#include <ws2tcpip.h>
#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <sstream>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <fstream>    
#pragma comment(lib, "ws2_32.lib")

#define PORT 65432
#define CHARSET_SIZE 62
#define MAX_PASSWORD_LENGTH 10

// Verifica que la contraseña contenga solo caracteres permitidos (A-Z, a-z, 0-9)
bool esValida(const std::string& pass) {
    const std::string allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    for (char c : pass)
        if (allowed.find(c) == std::string::npos)
            return false;
    return true;
}

// Solicita una contraseña válida. Si no lo es, limpia la terminal y vuelve a preguntar.
std::string getValidPassword() {
    std::string target;
    while (true) {
        std::cout << "Ingrese la contraseña a buscar (max " << MAX_PASSWORD_LENGTH
            << " caracteres, solo A-Z, a-z, 0-9): ";
        std::cin >> target;
        if (target.empty() || target.length() > MAX_PASSWORD_LENGTH) {
            system("cls");
            std::cout << "Error: La contraseña debe tener entre 1 y " << MAX_PASSWORD_LENGTH << " caracteres." << std::endl;
            continue;
        }
        if (!esValida(target)) {
            system("cls");
            std::cout << "Error: La contraseña contiene caracteres no permitidos. Solo se permiten A-Z, a-z, 0-9." << std::endl;
            continue;
        }
        break;
    }
    return target;
}

struct ClientInfo {
    SOCKET sock;
    unsigned long long currentChunkStart;
    unsigned long long currentChunkEnd;
    bool busy;
};

// Envía la configuración inicial (longitud y contraseña) a un cliente.
void sendInitialConfiguration(SOCKET clientSock, int passwordLength, const std::string& targetPassword) {
    int netLength = htonl(passwordLength);
    send(clientSock, (char*)&netLength, sizeof(netLength), 0);
    send(clientSock, targetPassword.c_str(), passwordLength, 0);
}

// Asigna un bloque (chunk) de trabajo a un cliente y lo notifica.
void assignChunk(ClientInfo& client, unsigned long long& nextChunkStart,
    unsigned long long totalCombinations, unsigned long long chunkSize) {

    unsigned long long startChunk = nextChunkStart;
    unsigned long long endChunk = (nextChunkStart + chunkSize < totalCombinations) ?
        (nextChunkStart + chunkSize) : totalCombinations;
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

// Espera hasta que se conecte al menos 'minClients' clientes.
void waitForMinimumClients(SOCKET serverSocket, std::vector<ClientInfo>& clients,
    int passwordLength, const std::string& targetPassword, int minClients) {
    while (clients.size() < (size_t)minClients) {
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(serverSocket, &readfds);
        int activity = select((int)serverSocket + 1, &readfds, NULL, NULL, NULL); // Bloquea
        if (activity > 0 && FD_ISSET(serverSocket, &readfds)) {
            SOCKET newSock = accept(serverSocket, NULL, NULL);
            if (newSock != INVALID_SOCKET) {
                sendInitialConfiguration(newSock, passwordLength, targetPassword);
                ClientInfo ci;
                ci.sock = newSock;
                ci.busy = false;
                ci.currentChunkStart = 0;
                ci.currentChunkEnd = 0;
                clients.push_back(ci);
                std::cout << "[Servidor] Cliente " << clients.size() << " conectado." << std::endl;
            }
        }
    }
}

// Acepta nuevos clientes durante la búsqueda y, si ya inició, les asigna un bloque.
void acceptNewClient(SOCKET serverSocket, std::vector<ClientInfo>& clients,
    int passwordLength, const std::string& targetPassword, bool searchStarted,
    unsigned long long& nextChunkStart, unsigned long long totalCombinations,
    unsigned long long chunkSize) {

    SOCKET newSock = accept(serverSocket, NULL, NULL);
    if (newSock != INVALID_SOCKET) {
        sendInitialConfiguration(newSock, passwordLength, targetPassword);
        ClientInfo newClient;
        newClient.sock = newSock;
        if (searchStarted) {
            assignChunk(newClient, nextChunkStart, totalCombinations, chunkSize);
        }
        else {
            newClient.busy = false;
            newClient.currentChunkStart = 0;
            newClient.currentChunkEnd = 0;
        }
        clients.push_back(newClient);
        std::cout << "[Servidor] Nuevo cliente conectado. Total clientes: " << clients.size() << std::endl;
    }
}

// Procesa los mensajes recibidos de los clientes; retorna true si algún cliente encontró la contraseña.
bool processClientMessages(std::vector<ClientInfo>& clients, fd_set& readfds,
    unsigned long long& nextChunkStart, unsigned long long totalCombinations,
    unsigned long long chunkSize,
    std::queue<std::pair<unsigned long long, unsigned long long>>& pendingChunks,
    std::string& foundPassword) {

    for (auto it = clients.begin(); it != clients.end(); ) {
        if (FD_ISSET(it->sock, &readfds)) {
            char buffer[256] = { 0 };
            int bytes = recv(it->sock, buffer, sizeof(buffer) - 1, 0);
            if (bytes <= 0) {
                std::cerr << "[Servidor] Cliente desconectado. Reasignando su bloque." << std::endl;
                if (it->busy)
                    pendingChunks.push({ it->currentChunkStart, it->currentChunkEnd });
                closesocket(it->sock);
                it = clients.erase(it);
                continue;
            }
            buffer[bytes] = '\0';
            std::string msg(buffer);
            if (msg.find("DONE") == 0) {
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
                    newEnd = (nextChunkStart + chunkSize < totalCombinations) ?
                        (nextChunkStart + chunkSize) : totalCombinations;
                    nextChunkStart = newEnd;
                }
                else {
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
                size_t pos = msg.find(',');
                if (pos != std::string::npos) {
                    foundPassword = msg.substr(pos + 1);
                    std::cout << "[Servidor] Contrasena encontrada por un cliente: " << foundPassword << std::endl;
                    return true;
                }
            }
        }
        ++it;
    }
    return false;
}

int main() {
    int mode = 0;
    std::cout << "Ingrese modo (1: secuencial, 2: CUDA): ";
    std::cin >> mode;
    std::string targetPassword = getValidPassword();
    int passwordLength = targetPassword.size();

    // Calcular total de combinaciones: CHARSET_SIZE^(passwordLength)
    unsigned long long totalCombinations = 1;
    for (int i = 0; i < passwordLength; i++) {
        totalCombinations *= CHARSET_SIZE;
    }

    // Inicializar Winsock.
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "Error al inicializar Winsock." << std::endl;
        return 1;
    }

    // Crea y configura el socket
    SOCKET serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket == INVALID_SOCKET) {
        std::cerr << "Error al crear el socket." << std::endl;
        WSACleanup();
        return 1;
    }
    sockaddr_in serverAddr{};
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(PORT);
    if (bind(serverSocket, (sockaddr*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
        std::cerr << "Error en bind." << std::endl;
        closesocket(serverSocket);
        WSACleanup();
        return 1;
    }
    if (listen(serverSocket, 5) == SOCKET_ERROR) {
        std::cerr << "Error en listen." << std::endl;
        closesocket(serverSocket);
        WSACleanup();
        return 1;
    }

    std::cout << "[Servidor] Esperando clientes..." << std::endl;
    std::vector<ClientInfo> clients;
    const unsigned long long CHUNK_SIZE = 1000000;
    unsigned long long nextChunkStart = 0;
    std::queue<std::pair<unsigned long long, unsigned long long>> pendingChunks;

    // Espera al menos 1 cliente.
    waitForMinimumClients(serverSocket, clients, passwordLength, targetPassword, 1);

    std::cout << "[Servidor] Esperando 10 segundos para un segundo cliente..." << std::endl;
    auto waitStart = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - waitStart < std::chrono::seconds(10)) {
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(serverSocket, &readfds);
        timeval timeout;
        timeout.tv_sec = 10;
        timeout.tv_usec = 0;
        int activity = select((int)serverSocket + 1, &readfds, NULL, NULL, &timeout);
        if (activity > 0 && FD_ISSET(serverSocket, &readfds)) {
            SOCKET newSock = accept(serverSocket, NULL, NULL);
            if (newSock != INVALID_SOCKET) {
                sendInitialConfiguration(newSock, passwordLength, targetPassword);
                ClientInfo ci;
                ci.sock = newSock;
                ci.busy = false;
                ci.currentChunkStart = 0;
                ci.currentChunkEnd = 0;
                clients.push_back(ci);
                std::cout << "[Servidor] Cliente " << clients.size() << " conectado." << std::endl;
            }
        }
        break;
    }

    for (auto& client : clients) {
        assignChunk(client, nextChunkStart, totalCombinations, CHUNK_SIZE);
    }

    auto startTime = std::chrono::steady_clock::now();
    bool passwordFound = false;
    std::string foundPassword;

    while (!passwordFound) {
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(serverSocket, &readfds);
        SOCKET maxSock = serverSocket;
        for (auto& client : clients) {
            FD_SET(client.sock, &readfds);
            if (client.sock > maxSock)
                maxSock = client.sock;
        }
        timeval timeout;
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;
        int activity = select(maxSock + 1, &readfds, NULL, NULL, &timeout);
        if (activity < 0) {
            std::cerr << "[Servidor] Error en select." << std::endl;
            break;
        }
        if (FD_ISSET(serverSocket, &readfds)) {
            acceptNewClient(serverSocket, clients, passwordLength, targetPassword, true,
                nextChunkStart, totalCombinations, CHUNK_SIZE);
        }
        if (processClientMessages(clients, readfds, nextChunkStart, totalCombinations,
            CHUNK_SIZE, pendingChunks, foundPassword))
            passwordFound = true;

        if (nextChunkStart >= totalCombinations && pendingChunks.empty()) {
            bool allIdle = true;
            for (auto& c : clients) {
                if (c.busy) { allIdle = false; break; }
            }
            if (allIdle)
                break;
        }
    }

    auto endTime = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsedSeconds = endTime - startTime;

    if (mode == 1) {
        std::ofstream outfile("sequential_time.bin", std::ios::binary);
        if (outfile.is_open()) {
            double seqTime = elapsedSeconds.count();
            outfile.write(reinterpret_cast<const char*>(&seqTime), sizeof(seqTime));
            outfile.close();
            std::cout << "[Servidor] Tiempo secuencial guardado: " << seqTime << " segundos." << std::endl;
        }
        else {
            std::cerr << "[Servidor] Error al escribir el archivo de tiempo secuencial." << std::endl;
        }
    }
    else if (mode == 2) {
        std::ifstream infile("sequential_time.bin", std::ios::binary);
        if (infile.is_open()) {
            double seqTime = 0.0;
            infile.read(reinterpret_cast<char*>(&seqTime), sizeof(seqTime));
            infile.close();
            double cudaTime = elapsedSeconds.count();
            double speedup = seqTime / cudaTime;
            std::cout << "[Servidor] Tiempo secuencial: " << seqTime << " segundos." << std::endl;
            std::cout << "[Servidor] Tiempo CUDA: " << cudaTime << " segundos." << std::endl;
            std::cout << "[Servidor] Speedup: " << speedup << std::endl;
        }
        else {
            std::cerr << "[Servidor] No se pudo leer el archivo de tiempo secuencial." << std::endl;
        }
    }

    for (auto& client : clients) {
        std::string termMsg = "TERMINATE";
        send(client.sock, termMsg.c_str(), termMsg.size(), 0);
        closesocket(client.sock);
    }
    closesocket(serverSocket);
    WSACleanup();

    if (!foundPassword.empty())
        std::cout << "[Servidor] Contrasena encontrada: " << foundPassword << std::endl;
    else
        std::cout << "[Servidor] Contrasena no encontrada en el espacio de busqueda." << std::endl;

    std::cout << "[Servidor] Tiempo total de busqueda: " << elapsedSeconds.count() << " segundos." << std::endl;

    return 0;
}
