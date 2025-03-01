#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <iostream>
#include <sstream>
#include <string>
#include <cstring>
#pragma comment(lib, "ws2_32.lib")

#define SERVER_IP "26.169.61.230"
#define PORT 65432
#define CHARSET_SIZE 62
#define MAX_PASSWORD_LENGTH 10

const std::string charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

std::string indexToPassword(unsigned long long index, int passLength) {
    std::string candidate(passLength, ' ');
    for (int i = passLength - 1; i >= 0; --i) {
        candidate[i] = charset[index % CHARSET_SIZE];
        index /= CHARSET_SIZE;
    }
    return candidate;
}

bool sequentialBruteForce(const char* target, int passLength,
    unsigned long long start, unsigned long long end, std::string& foundPassword) {
    for (unsigned long long i = start; i < end; ++i) {
        std::string candidate = indexToPassword(i, passLength);
        if (candidate == target) {
            foundPassword = candidate;
            return true;
        }
    }
    return false;
}

SOCKET connectToServer(const char* server_ip, int port) {
    sockaddr_in servAddr{};
    servAddr.sin_family = AF_INET;
    servAddr.sin_port = htons(port);
    inet_pton(AF_INET, server_ip, &servAddr.sin_addr);

    SOCKET sock = INVALID_SOCKET;
    while (true) {
        sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock == INVALID_SOCKET) {
            std::cerr << "[Cliente] Error al crear el socket." << std::endl;
            WSACleanup();
            exit(1);
        }
        if (connect(sock, (sockaddr*)&servAddr, sizeof(servAddr)) == SOCKET_ERROR) {
            std::cerr << "[Cliente] Error al conectar con el servidor. Reintentando en 1 segundo..." << std::endl;
            closesocket(sock);
            Sleep(1000);
            continue;
        }
        break;
    }
    return sock;
}

bool receiveConfiguration(SOCKET sock, int& passwordLength, char targetPassword[MAX_PASSWORD_LENGTH + 1]) {
    int ret = recv(sock, (char*)&passwordLength, sizeof(passwordLength), 0);
    if (ret <= 0) return false;
    passwordLength = ntohl(passwordLength);
    if (passwordLength > MAX_PASSWORD_LENGTH) {
        std::cerr << "[Cliente] Error: Longitud de contrasena invalida." << std::endl;
        return false;
    }
    ret = recv(sock, targetPassword, passwordLength, 0);
    if (ret <= 0) return false;
    targetPassword[passwordLength] = '\0';
    return true;
}

void processServerMessages(SOCKET sock, int passwordLength, const char* targetPassword) {
    while (true) {
        char msgBuffer[256] = { 0 };
        int bytes = recv(sock, msgBuffer, sizeof(msgBuffer) - 1, 0);
        if (bytes <= 0) {
            std::cerr << "[Cliente] Desconexion del servidor." << std::endl;
            break;
        }
        msgBuffer[bytes] = '\0';
        std::string msg(msgBuffer);

        if (msg.find("CHUNK") == 0) {
            size_t firstComma = msg.find(',');
            size_t secondComma = msg.find(',', firstComma + 1);
            if (firstComma == std::string::npos || secondComma == std::string::npos) {
                std::cerr << "[Cliente] Formato de CHUNK invalido." << std::endl;
                continue;
            }
            std::string startStr = msg.substr(firstComma + 1, secondComma - firstComma - 1);
            std::string endStr = msg.substr(secondComma + 1);
            unsigned long long startChunk = std::stoull(startStr);
            unsigned long long endChunk = std::stoull(endStr);
            std::cout << "[Cliente] Rango recibido: " << startChunk << " - " << endChunk << std::endl;

            std::string foundPassword;
            bool found = sequentialBruteForce(targetPassword, passwordLength, startChunk, endChunk, foundPassword);
            if (found) {
                std::cout << "[Cliente] Contrasena encontrada: " << foundPassword << std::endl;
                std::string foundMsg = "FOUND," + foundPassword;
                send(sock, foundMsg.c_str(), foundMsg.size(), 0);
                break;
            }
            else {
                std::cout << "[Cliente] Contrasena no encontrada en este rango." << std::endl;
                std::string doneMsg = "DONE";
                send(sock, doneMsg.c_str(), doneMsg.size(), 0);
            }
        }
        else if (msg.find("TERMINATE") == 0) {
            std::cout << "[Cliente] Recibido mensaje de TERMINATE. Finalizando." << std::endl;
            break;
        }
    }
}

int main() {
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "[Cliente] Error al inicializar Winsock." << std::endl;
        return 1;
    }

    SOCKET sock = connectToServer(SERVER_IP, PORT);
    std::cout << "[Cliente] Conectado al servidor." << std::endl;

    int passwordLength = 0;
    char targetPassword[MAX_PASSWORD_LENGTH + 1] = { 0 };
    if (!receiveConfiguration(sock, passwordLength, targetPassword)) {
        std::cerr << "[Cliente] Error al recibir la configuracion." << std::endl;
        closesocket(sock);
        WSACleanup();
        return 1;
    }

    processServerMessages(sock, passwordLength, targetPassword);

    closesocket(sock);
    WSACleanup();
    return 0;
}
