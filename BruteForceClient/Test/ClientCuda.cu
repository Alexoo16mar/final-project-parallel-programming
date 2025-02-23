#include <winsock2.h>
#include <ws2tcpip.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cstring>
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "cudart.lib")

#define SERVER_IP "26.169.61.230"
#define PORT 65432
#define CHARSET_SIZE 62
#define MAX_PASSWORD_LENGTH 10

__constant__ char d_charset[CHARSET_SIZE + 1] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

__global__ void bruteForceKernel(const char* d_target, int passLength,
    unsigned long long start, unsigned long long end,
    int* d_found, unsigned long long* d_index) {

    unsigned long long idx = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= end) return;

    char currentGuess[MAX_PASSWORD_LENGTH + 1] = { 0 };
    unsigned long long temp = idx;

    // Generar desde el dígito más significativo
    for (int i = passLength - 1; i >= 0; --i) {  // <-- Cambio clave
        currentGuess[i] = d_charset[temp % CHARSET_SIZE];
        temp /= CHARSET_SIZE;
    }

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
    // 1?? Configurar red
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);

    SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in servAddr{};
    servAddr.sin_family = AF_INET;
    servAddr.sin_port = htons(PORT);
    inet_pton(AF_INET, SERVER_IP, &servAddr.sin_addr);

    connect(sock, (sockaddr*)&servAddr, sizeof(servAddr));

    // 2?? Recibir contraseña
    int passwordLength;
    recv(sock, (char*)&passwordLength, sizeof(passwordLength), 0);
    passwordLength = ntohl(passwordLength);

    char targetPassword[MAX_PASSWORD_LENGTH + 1] = { 0 };
    recv(sock, targetPassword, passwordLength, 0);
    std::cout << "[Cliente] Contraseña recibida a buscar: " << /*targetPassword <<*/ std::endl;

    // 3?? Recibir rango
    char rangeStr[100];
    recv(sock, rangeStr, sizeof(rangeStr), 0);
    unsigned long long start = _strtoui64(strtok(rangeStr, ","), NULL, 10);
    unsigned long long end = _strtoui64(strtok(NULL, ","), NULL, 10);
    std::cout << "[Cliente] Rango recibido: " << start << " - " << end << std::endl;

    // 4?? Configurar CUDA
    char* d_target;
    int* d_found;
    unsigned long long* d_index;

    cudaMalloc(&d_target, MAX_PASSWORD_LENGTH + 1);
    cudaMalloc(&d_found, sizeof(int));
    cudaMalloc(&d_index, sizeof(unsigned long long));

    cudaMemcpy(d_target, targetPassword, passwordLength + 1, cudaMemcpyHostToDevice);
    cudaMemset(d_found, 0, sizeof(int));
    cudaMemset(d_index, 0, sizeof(unsigned long long));

    // 5?? Ejecutar kernel
    const int blockSize = 1024;  // Optimizado para GPUs modernas
    const int numBlocks = (end - start + blockSize - 1) / blockSize;

    bruteForceKernel << <numBlocks, blockSize >> > (d_target, passwordLength, start, end, d_found, d_index);
    cudaDeviceSynchronize();

    // 6?? Procesar resultados
    int found;
    unsigned long long foundIndex;
    cudaMemcpy(&found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&foundIndex, d_index, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    const char h_charset[CHARSET_SIZE + 1] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

    if (found) {
        std::cout << "[Cliente] Contraseña encontrada en índice: " << foundIndex << std::endl;
        char result[MAX_PASSWORD_LENGTH + 1] = { 0 };
        unsigned long long temp = foundIndex;
        for (int i = passwordLength - 1; i >= 0; --i) {  // ?? Cambio clave
            int remainder = temp % CHARSET_SIZE;
            result[i] = h_charset[remainder];
            temp /= CHARSET_SIZE;
        }
		std::cout << "[Cliente] Contraseña encontrada: " << result << std::endl;
        result[passwordLength] = '\0';
        send(sock, result, passwordLength + 1, 0);
	}
    else {
        std::cout << "[Cliente] Contraseña no encontrada" << std::endl;
    }

    // 7?? Limpiar
    cudaFree(d_target);
    cudaFree(d_found);
    cudaFree(d_index);
    closesocket(sock);
    WSACleanup();
    return 0;
}