#include <winsock2.h>
#include <ws2tcpip.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cstring>
#include <sstream>
#include <string>
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "cudart.lib")

#define SERVER_IP "XX.XXX.XX.XXX" // Change it to the server IP
#define PORT 65432
#define CHARSET_SIZE 62
#define MAX_PASSWORD_LENGTH 10

// Definición en constante del charset para la generación de contraseñas
__constant__ char d_charset[CHARSET_SIZE + 1] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

// Kernel CUDA para fuerza bruta en un rango dado
__global__ void bruteForceKernel(const char* d_target, int passLength,
    unsigned long long start, unsigned long long end,
    int* d_found, unsigned long long* d_index) {

    unsigned long long idx = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= end) return;

    char currentGuess[MAX_PASSWORD_LENGTH + 1] = { 0 };
    unsigned long long temp = idx;

    // Convertir el índice en una contraseña (rellenando desde el final)
    for (int i = passLength - 1; i >= 0; --i) {
        currentGuess[i] = d_charset[temp % CHARSET_SIZE];
        temp /= CHARSET_SIZE;
    }
    currentGuess[passLength] = '\0';

    // Comparación con la contraseña objetivo
    bool match = true;
    for (int i = 0; i < passLength; ++i) {
        if (currentGuess[i] != d_target[i]) {
            match = false;
            break;
        }
    }

    if (match) {
        atomicExch(d_found, 1);
        atomicExch(d_index, idx);
    }
}

int main() {
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);

    SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == INVALID_SOCKET) {
        std::cerr << "[Cliente] Error al crear el socket." << std::endl;
        WSACleanup();
        return 1;
    }

    sockaddr_in servAddr{};
    servAddr.sin_family = AF_INET;
    servAddr.sin_port = htons(PORT);
    inet_pton(AF_INET, SERVER_IP, &servAddr.sin_addr);

    if (connect(sock, (sockaddr*)&servAddr, sizeof(servAddr)) == SOCKET_ERROR) {
        std::cerr << "[Cliente] Error al conectar con el servidor." << std::endl;
        closesocket(sock);
        WSACleanup();
        return 1;
    }

    // Recibir la longitud de la contraseña
    int passwordLength;
    recv(sock, (char*)&passwordLength, sizeof(passwordLength), 0);
    passwordLength = ntohl(passwordLength);
    if (passwordLength > MAX_PASSWORD_LENGTH) {
        std::cerr << "[Cliente] Error: Longitud de contraseña inválida." << std::endl;
        closesocket(sock);
        WSACleanup();
        return 1;
    }

    // Recibir la contraseña objetivo
    char targetPassword[MAX_PASSWORD_LENGTH + 1] = { 0 };
    recv(sock, targetPassword, passwordLength, 0);
    targetPassword[passwordLength] = '\0';
    std::cout << "[Cliente] Contraseña recibida: " << targetPassword
        << " (longitud: " << passwordLength << ")" << std::endl;

    // Reservar memoria en el dispositivo para la ejecución en CUDA
    char* d_target;
    int* d_found;
    unsigned long long* d_index;
    cudaMalloc(&d_target, MAX_PASSWORD_LENGTH + 1);
    cudaMalloc(&d_found, sizeof(int));
    cudaMalloc(&d_index, sizeof(unsigned long long));
    cudaMemcpy(d_target, targetPassword, passwordLength + 1, cudaMemcpyHostToDevice);

    const int blockSize = 1024;

    // Bucle principal: el cliente espera recibir bloques (chunks) desde el servidor
    while (true) {
        char msgBuffer[256] = { 0 };
        int bytes = recv(sock, msgBuffer, sizeof(msgBuffer) - 1, 0);
        if (bytes <= 0) {
            std::cerr << "[Cliente] Desconexión del servidor." << std::endl;
            break;
        }
        msgBuffer[bytes] = '\0';
        std::string msg(msgBuffer);

        if (msg.find("CHUNK") == 0) {
            // Formato esperado: "CHUNK,<start>,<end>"
            size_t firstComma = msg.find(',');
            size_t secondComma = msg.find(',', firstComma + 1);
            if (firstComma == std::string::npos || secondComma == std::string::npos) {
                std::cerr << "[Cliente] Formato de CHUNK inválido." << std::endl;
                continue;
            }
            std::string startStr = msg.substr(firstComma + 1, secondComma - firstComma - 1);
            std::string endStr = msg.substr(secondComma + 1);
            unsigned long long startChunk = std::stoull(startStr);
            unsigned long long endChunk = std::stoull(endStr);
            std::cout << "[Cliente] Rango recibido: " << startChunk << " - " << endChunk << std::endl;

            // Ejecutar el kernel CUDA para el bloque recibido
            int found = 0;
            cudaMemset(d_found, 0, sizeof(int));
            cudaMemset(d_index, 0, sizeof(unsigned long long));
            unsigned long long rangeSize = endChunk - startChunk;
            int numBlocks = (rangeSize + blockSize - 1) / blockSize;
            bruteForceKernel << <numBlocks, blockSize >> > (d_target, passwordLength, startChunk, endChunk, d_found, d_index);
            cudaDeviceSynchronize();

            cudaMemcpy(&found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
            if (found) {
                unsigned long long foundIndex;
                cudaMemcpy(&foundIndex, d_index, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
                char recoveredPassword[MAX_PASSWORD_LENGTH + 1] = { 0 };
                unsigned long long tempIndex = foundIndex;
                for (int i = passwordLength - 1; i >= 0; --i) {
                    recoveredPassword[i] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"[tempIndex % CHARSET_SIZE];
                    tempIndex /= CHARSET_SIZE;
                }
                recoveredPassword[passwordLength] = '\0';
                std::cout << "[Cliente] Contraseña encontrada: " << recoveredPassword << std::endl;
                std::string foundMsg = "FOUND," + std::string(recoveredPassword);
                send(sock, foundMsg.c_str(), foundMsg.size(), 0);
                break; // Se encontró la contraseña, salir del bucle
            }
            else {
                std::cout << "[Cliente] Contraseña no encontrada en este rango." << std::endl;
                std::string doneMsg = "DONE";
                send(sock, doneMsg.c_str(), doneMsg.size(), 0);
            }
        }
        else if (msg.find("TERMINATE") == 0) {
            std::cout << "[Cliente] Recibido mensaje de TERMINATE. Finalizando." << std::endl;
            break;
        }
    }

    // Liberar recursos y cerrar conexiones
    cudaFree(d_target);
    cudaFree(d_found);
    cudaFree(d_index);
    closesocket(sock);
    WSACleanup();
    return 0;
}
