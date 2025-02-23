#include <winsock2.h>
#include <ws2tcpip.h>
#include <iostream>
#include <string>
#include <cmath>
#pragma comment(lib, "ws2_32.lib")

#define PORT 65432
#define CHARSET_SIZE 62
#define MAX_PASSWORD_LENGTH 10

// 1?? Contraseña definida solo en el servidor
const char* TARGET_PASSWORD = "a1B2";  // Cambia aquí la contraseña
const int PASSWORD_LENGTH = 4;         // Longitud real de la contraseña

int main() {
    // 2?? Verificar longitud máxima
    if (PASSWORD_LENGTH > MAX_PASSWORD_LENGTH) {
        std::cerr << "Error: La contraseña excede la longitud maxima" << std::endl;
        return 1;
    }

    // Inicializar Winsock
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "Error al inicializar Winsock" << std::endl;
        return 1;
    }

    // Crear socket
    SOCKET serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in serverAddr{};
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(PORT);

    // Vincular y escuchar
    bind(serverSocket, (sockaddr*)&serverAddr, sizeof(serverAddr));
    listen(serverSocket, 2);

    std::cout << "[Servidor] Esperando clientes..." << std::endl;

    // 3?? Aceptar clientes y enviar contraseña
    SOCKET clients[2];
    int clientCount = 0;
    while (clientCount < 2) {
        clients[clientCount] = accept(serverSocket, NULL, NULL);

        // Enviar longitud y contraseña
        int netLength = htonl(PASSWORD_LENGTH);
        send(clients[clientCount], (char*)&netLength, sizeof(netLength), 0);
        send(clients[clientCount], TARGET_PASSWORD, PASSWORD_LENGTH, 0);

        std::cout << "[Servidor] Cliente " << (clientCount + 1) << " conectado" << std::endl;
        clientCount++;
    }

    // 4?? Calcular y distribuir rangos
    const unsigned long long totalCombinations = pow(CHARSET_SIZE, PASSWORD_LENGTH);
    const unsigned long long chunkSize = totalCombinations / 2;

    std::string range1 = "0," + std::to_string(chunkSize);
    std::string range2 = std::to_string(chunkSize) + "," + std::to_string(totalCombinations);

    send(clients[0], range1.c_str(), range1.size(), 0);
    send(clients[1], range2.c_str(), range2.size(), 0);

    std::cout << "[Servidor] Rangos distribuidos:\n" << range1 << "\n" << range2 << std::endl;

    // 5?? Esperar resultados
    while (true) {
        for (int i = 0; i < 2; ++i) {
            char buffer[MAX_PASSWORD_LENGTH + 1] = { 0 };
            /*if (recv(clients[i], buffer, MAX_PASSWORD_LENGTH, 0) > 0) {
                std::cout << "\n[Servidor] Contraseña encontrada: " << buffer << std::endl;
                closesocket(serverSocket);
                WSACleanup();
                return 0;
            }*/
            int bytesReceived = recv(clients[i], buffer, MAX_PASSWORD_LENGTH, 0);
            if (bytesReceived > 0) {
                buffer[bytesReceived] = '\0';  // Agregar terminación segura
                std::cout << "\n[Servidor] Contraseña encontrada: " << buffer << std::endl;
            }
        }
    }
}