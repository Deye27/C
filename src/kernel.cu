#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Funzione per generare punti di un poligono che approssima una circonferenza
__device__ void generate_circle_points(float centerX, float centerY, float radius, int numSides, float* pointsX, float* pointsY) {
    float angleIncrement = 2 * M_PI / numSides;
    for (int i = 0; i < numSides; ++i) {
        float angle = i * angleIncrement;
        pointsX[i] = centerX + radius * cosf(angle);
        pointsY[i] = centerY + radius * sinf(angle);
    }
}

// Funzione per calcolare la lunghezza di una polilinea (approssimazione della circonferenza)
__device__ float calculate_polyline_length(float* pointsX, float* pointsY, int numSides) {
    float length = 0.0f;
    for (int i = 0; i < numSides; ++i) {
        float x1 = pointsX[i];
        float y1 = pointsY[i];
        float x2 = pointsX[(i + 1) % numSides];
        float y2 = pointsY[(i + 1) % numSides];
        length += sqrtf((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
    }
    return length;
}

// Funzione sigmoid
__device__ float sigmoid_gpu(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Kernel CUDA per il modello di Pendulum
__global__ void integrate_pn(float* distances, int width, int height, float dt, float t_max, float offsetX, float offsetY, float zoom, float x_stretch, float y_stretch, float L, float gamma, float g, float scale, int numSides, bool useBoundingBox, float boundingBoxMinX, float boundingBoxMaxX, float boundingBoxMinY, float boundingBoxMaxY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Mappa le coordinate dei pixel nello spazio matematico
    float centeredX = (float)x - width / 2.0f;
    float centeredY = (float)y - height / 2.0f;
    float mappedX = (centeredX + offsetX) / scale * x_stretch;
    float mappedY = -(centeredY + offsetY) / scale * y_stretch;

    // Se la bounding box è attiva, controlla se il punto è al suo interno
    if (useBoundingBox && (mappedX < boundingBoxMinX || mappedX > boundingBoxMaxX || mappedY < boundingBoxMinY || mappedY > boundingBoxMaxY)) {
        distances[y * width + x] = -1.0f; // Imposta un valore negativo per indicare che il punto è fuori dalla bounding box
        return;
    }

    // Definisci il raggio della circonferenza
    float radius = 0.005f;
    float initial_radius = radius;

    //Alloca i punti (non dinamicamente in ogni thread)
    __shared__ float pointsX[100]; //Numero massimo di lati supportati
    __shared__ float pointsY[100];

    // Genera i punti del poligono
    generate_circle_points(mappedX, mappedY, initial_radius, numSides, pointsX, pointsY);

    // Calcola la lunghezza del poligono iniziale (c)
    float initialLength = 2 * numSides * initial_radius * sinf(M_PI / numSides);
    float finalLength = initialLength;
    if (t_max > 0.00001f)
    {
        // Integra i punti del poligono nel sistema dinamico
        float x_current, y_current, x_next, y_next;

        int num_steps = (int)round(t_max / dt);
        for (int i = 0; i < numSides; i++) {
            // Usa una variabile temporanea per l'angolo iniziale
            float theta_current = pointsX[i]; // Inizializza con x
            float omega_current = pointsY[i]; // Inizializza con y

            for (int step = 0; step < num_steps; step++) {
                // Equations of the pendulum
                float domega = -g / L * sinf(theta_current) - gamma * omega_current;
                float dtheta = omega_current;

                // Calcola le posizioni successive
                float omega_next = omega_current + domega * dt;
                float theta_next = theta_current + dtheta * dt;

                // Aggiorna le variabili temporanee con i nuovi valori
                theta_current = theta_next;
                omega_current = omega_next;
            }
            pointsX[i] = theta_current; // Salva l'angolo
            pointsY[i] = omega_current; // Salva la velocità angolare
        }
        // Calcola la lunghezza del poligono deformato (c')
        finalLength = calculate_polyline_length(pointsX, pointsY, numSides);
    }

    // Calcola il rapporto c'/c e salvalo
    distances[y * width + x] = finalLength / initialLength;
}

// Kernel CUDA per il modello di Hodgkin-Huxley
__global__ void integrate_hh(float* distances, int width, int height, float dt, float t_max, float offsetX, float offsetY, float zoom, float x_stretch, float y_stretch, float Iext, int bifurcation_type_id, float scale, int numSides, bool useBoundingBox, float boundingBoxMinX, float boundingBoxMaxX, float boundingBoxMinY, float boundingBoxMaxY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Mappa le coordinate dei pixel nello spazio matematico
    float centeredX = (float)x - width / 2.0f;
    float centeredY = (float)y - height / 2.0f;
    float mappedX = (centeredX + offsetX) / scale * x_stretch;
    float mappedY = -(centeredY + offsetY) / scale * y_stretch;

    // Se la bounding box è attiva, controlla se il punto è al suo interno
    if (useBoundingBox && (mappedX < boundingBoxMinX || mappedX > boundingBoxMaxX || mappedY < boundingBoxMinY || mappedY > boundingBoxMaxY)) {
        distances[y * width + x] = -1.0f; // Imposta un valore negativo per indicare che il punto è fuori dalla bounding box
        return;
    }

    // Definisci il raggio della circonferenza
    float radius = 0.005f;
    float initial_radius = radius;

    //Alloca i punti (non dinamicamente in ogni thread)
    __shared__ float pointsX[100]; //Numero massimo di lati supportati
    __shared__ float pointsY[100];

    // Genera i punti del poligono
    generate_circle_points(mappedX, mappedY, initial_radius, numSides, pointsX, pointsY);

    // Calcola la lunghezza del poligono iniziale (c)
    float initialLength = 2 * numSides * initial_radius * sinf(M_PI / numSides);
    float finalLength = initialLength;
    if (t_max > 0.00001f)
    {
        // Integra i punti del poligono nel sistema dinamico
        float x_current, y_current, x_next, y_next;

        // Neuron Model Parameters
        float g_Na = 20.0f;
        float g_K = 10.0f;
        float g_L = 8.0f;
        float E_Na = 60.0f;
        float E_K = -90.0f;
        float E_L = -80.0f;
        float k_m = 15.0f;
        float k_n = 5.0f;
        float V_mid_n, V_mid_m, tau_n;

        if (bifurcation_type_id == 0) { // Saddle-node
            V_mid_n = -25.0f;
            V_mid_m = -20.0f;
            tau_n = 0.152f;
        }
        else if (bifurcation_type_id == 1) { // SNIC
            V_mid_n = -25.0f;
            V_mid_m = -20.0f;
            tau_n = 1.0f;
        }
        else if (bifurcation_type_id == 2) { // Subcritical Hopf
            V_mid_n = -45.0f;
            V_mid_m = -30.0f;
            tau_n = 1.0f;
        }
        else if (bifurcation_type_id == 3) { // Supercritical Hopf
            V_mid_n = -45.0f;
            V_mid_m = -20.0f;
            tau_n = 1.0f;
        }
        else {
            distances[y * width + x] = -2.0f; //Errore: bifurcation_type_id non valido.
            return;
        }
        int num_steps = (int)round(t_max / dt);
        for (int i = 0; i < numSides; i++) {
            x_current = pointsX[i];
            y_current = pointsY[i];
            for (int step = 0; step < num_steps; step++) {

                // Steady-state activation functions
                float m_inf = sigmoid_gpu((x_current - V_mid_m) / k_m);
                float n_inf = sigmoid_gpu((x_current - V_mid_n) / k_n);
                // Equations of the neuron model
                float dx = Iext - g_Na * m_inf * (x_current - E_Na) - g_K * y_current * (x_current - E_K) - g_L * (x_current - E_L);
                float dy = (n_inf - y_current) / tau_n;

                x_next = x_current + dx * dt;
                y_next = y_current + dy * dt;
                x_current = x_next;
                y_current = y_next;
            }
            pointsX[i] = x_current;
            pointsY[i] = y_current;
        }
        // Calcola la lunghezza del poligono deformato (c')
        finalLength = calculate_polyline_length(pointsX, pointsY, numSides);
    }

    // Calcola il rapporto c'/c e salvalo
    distances[y * width + x] = finalLength / initialLength;
}

// Kernel CUDA per il modello di Compteizione Intraspecifica
__global__ void integrate_lvm(float* distances, int width, int height, float dt, float t_max, float offsetX, float offsetY, float zoom, float x_stretch, float y_stretch, float r1, float K1, float a12, float r2, float K2, float a21, float scale, int numSides, bool useBoundingBox, float boundingBoxMinX, float boundingBoxMaxX, float boundingBoxMinY, float boundingBoxMaxY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Map pixel coordinates to the mathematical space
    float centeredX = (float)x - width / 2.0f;
    float centeredY = (float)y - height / 2.0f;
    float mappedX = (centeredX + offsetX) / scale * x_stretch;
    float mappedY = -(centeredY + offsetY) / scale * y_stretch;

    // If bounding box is active, check if the point is inside it
    if (useBoundingBox && (mappedX < boundingBoxMinX || mappedX > boundingBoxMaxX || mappedY < boundingBoxMinY || mappedY > boundingBoxMaxY)) {
        distances[y * width + x] = -1.0f; // Set a negative value to indicate that the point is outside the bounding box
        return;
    }

    // Define the radius of the circle
    float radius = 0.005f;
    float initial_radius = radius;

    // Allocate points (not dynamically in each thread)
    __shared__ float pointsX[100]; // Maximum number of supported sides
    __shared__ float pointsY[100];

    // Generate polygon points
    generate_circle_points(mappedX, mappedY, initial_radius, numSides, pointsX, pointsY);

    // Calculate the length of the initial polygon (c)
    float initialLength = 2 * numSides * initial_radius * sinf(M_PI / numSides);
    float finalLength = initialLength;
    if (t_max > 0.00001f)
    {
        // Integrate polygon points into the dynamic system
        float x_current, y_current, x_next, y_next;

        int num_steps = (int)round(t_max / dt);
        for (int i = 0; i < numSides; i++) {
            // Use temporary variables for initial values
            float x_current = pointsX[i]; // Initialize with x
            float y_current = pointsY[i]; // Initialize with y

            for (int step = 0; step < num_steps; step++) {
                // Lotka-Volterra equations
                float dx = r1 * x_current * (1 - (x_current + a12 * y_current) / K1);
                float dy = r2 * y_current * (1 - (y_current + a21 * x_current) / K2);

                // Calculate the next positions
                x_next = x_current + dx * dt;
                y_next = y_current + dy * dt;
                
                if (x_next < 0) {
                    x_next = 0;
                }
                if (y_next < 0) {
                    y_next = 0;
                }

                // Update temporary variables with new values
                x_current = x_next;
                y_current = y_next;
            }
            pointsX[i] = x_current; // Save the final x value
            pointsY[i] = y_current; // Save the final y value
        }
        // Calculate the length of the deformed polygon (c')
        finalLength = calculate_polyline_length(pointsX, pointsY, numSides);
    }

    // Calculate the ratio c'/c and save it
    distances[y * width + x] = finalLength / initialLength;
}

// Kernel CUDA per il modello Preda-Predatore
__global__ void integrate_lv(float* distances, int width, int height, float dt, float t_max, float offsetX, float offsetY, float zoom, float x_stretch, float y_stretch, float a, float b, float c, float d, float scale, int numSides, bool useBoundingBox, float boundingBoxMinX, float boundingBoxMaxX, float boundingBoxMinY, float boundingBoxMaxY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Mappa le coordinate dei pixel nello spazio matematico
    float centeredX = (float)x - width / 2.0f;
    float centeredY = (float)y - height / 2.0f;
    float mappedX = (centeredX + offsetX) / scale * x_stretch;
    float mappedY = -(centeredY + offsetY) / scale * y_stretch;

    // Se la bounding box è attiva, controlla se il punto è al suo interno
    if (useBoundingBox && (mappedX < boundingBoxMinX || mappedX > boundingBoxMaxX || mappedY < boundingBoxMinY || mappedY > boundingBoxMaxY)) {
        distances[y * width + x] = -1.0f; // Imposta un valore negativo per indicare che il punto è fuori dalla bounding box
        return;
    }

    // Definisci il raggio della circonferenza
    float radius = 0.005f;
    float initial_radius = radius;

    // Alloca i punti (non dinamicamente in ogni thread)
    __shared__ float pointsX[100]; // Numero massimo di lati supportati
    __shared__ float pointsY[100];

    // Genera i punti del poligono
    generate_circle_points(mappedX, mappedY, initial_radius, numSides, pointsX, pointsY);

    // Calcola la lunghezza del poligono iniziale (c)
    float initialLength = 2 * numSides * initial_radius * sinf(M_PI / numSides);
    float finalLength = initialLength;

    if (t_max > 0.00001f)
    {
        // Integra i punti del poligono nel sistema dinamico
        float x_current, y_current, x_next, y_next;

        int num_steps = (int)round(t_max / dt);
        for (int i = 0; i < numSides; i++) {
            // Usa variabili temporanee per i valori iniziali
            float x_current = pointsX[i]; // Inizializza con x (preda)
            float y_current = pointsY[i]; // Inizializza con y (predatore)

            for (int step = 0; step < num_steps; step++) {
                // Lotka-Volterra preda-predatore equations
                float dx = a * x_current - b * x_current * y_current;
                float dy = c * x_current * y_current - d * y_current;

                // Calcola le posizioni successive
                x_next = x_current + dx * dt;
                y_next = y_current + dy * dt;

                // Impedisci che le popolazioni diventino negative
                if (x_next < 0) {
                    x_next = 0;
                }
                if (y_next < 0) {
                    y_next = 0;
                }

                // Aggiorna le variabili temporanee con i nuovi valori
                x_current = x_next;
                y_current = y_next;
            }
            pointsX[i] = x_current; // Salva il valore x finale (preda)
            pointsY[i] = y_current; // Salva il valore y finale (predatore)
        }
        // Calcola la lunghezza del poligono deformato (c')
        finalLength = calculate_polyline_length(pointsX, pointsY, numSides);
    }

    // Calcola il rapporto c'/c e salvalo
    distances[y * width + x] = finalLength / initialLength;
}

void run_cuda_kernel(int model, int width, int height, float* d_distances, float dt, float t_max, float offsetX, float offsetY, float zoom, float x_stretch, float y_stretch, float Iext, int bifurcation_type_id, float L, float gamma, float g, float r1, float K1, float a12, float r2, float K2, float a21, float a, float b, float c, float d, float scale, int numSides, int blockDimX, int blockDimY, bool useBoundingBox, float boundingBoxMinX, float boundingBoxMaxX, float boundingBoxMinY, float boundingBoxMaxY) {
    dim3 blockDim(blockDimX, blockDimY);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    switch (model) {
    case 3:
        integrate_hh << <gridDim, blockDim >> > (d_distances, width, height, dt, t_max, offsetX, offsetY, zoom, x_stretch, y_stretch, Iext, bifurcation_type_id, scale, numSides, useBoundingBox, boundingBoxMinX, boundingBoxMaxX, boundingBoxMinY, boundingBoxMaxY);
        break;
    case 0:
        integrate_pn << <gridDim, blockDim >> > (d_distances, width, height, dt, t_max, offsetX, offsetY, zoom, x_stretch, y_stretch, L, gamma, g, scale, numSides, useBoundingBox, boundingBoxMinX, boundingBoxMaxX, boundingBoxMinY, boundingBoxMaxY);
        break;
    case 2:
        integrate_lvm << <gridDim, blockDim >> > (d_distances, width, height, dt, t_max, offsetX, offsetY, zoom, x_stretch, y_stretch, r1, K1, a12, r2, K2, a21, scale, numSides, useBoundingBox, boundingBoxMinX, boundingBoxMaxX, boundingBoxMinY, boundingBoxMaxY);
        break;
    case 1:
        integrate_lv << <gridDim, blockDim >> > (d_distances, width, height, dt, t_max, offsetX, offsetY, zoom, x_stretch, y_stretch, a, b, c, d, scale, numSides, useBoundingBox, boundingBoxMinX, boundingBoxMaxX, boundingBoxMinY, boundingBoxMaxY);
        break;
    }
    cudaDeviceSynchronize();
}
