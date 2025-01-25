#define NOMINMAX
#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <filesystem>
#include <windows.h>
#include <GL/GL.h>
#include <tchar.h>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//FONTS
ImFont* font_default = nullptr;
ImFont* regular = nullptr;
ImFont* bold = nullptr;
ImFont* italic = nullptr;
ImFont* thin = nullptr;
ImFont* black = nullptr;

struct ImVec3 {
    float x, y, z;
    ImVec3() : x(0), y(0), z(0) {}
    ImVec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
};

struct Transform {
    float offsetX;
    float offsetY;
    float zoom;
};

struct WGL_WindowData { HDC hDC; };
static HGLRC            g_hRC;
static WGL_WindowData   g_MainWindow;
static int              g_Width;
static int              g_Height;

bool CreateDeviceWGL(HWND hWnd, WGL_WindowData* data);
void CleanupDeviceWGL(HWND hWnd, WGL_WindowData* data);
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

static void Hook_Renderer_CreateWindow(ImGuiViewport* viewport)
{
    assert(viewport->RendererUserData == NULL);
    WGL_WindowData* data = IM_NEW(WGL_WindowData);
    CreateDeviceWGL((HWND)viewport->PlatformHandle, data);
    viewport->RendererUserData = data;
}

static void Hook_Renderer_DestroyWindow(ImGuiViewport* viewport)
{
    if (viewport->RendererUserData != NULL)
    {
        WGL_WindowData* data = (WGL_WindowData*)viewport->RendererUserData;
        CleanupDeviceWGL((HWND)viewport->PlatformHandle, data);
        IM_DELETE(data);
        viewport->RendererUserData = NULL;
    }
}

static void Hook_Platform_RenderWindow(ImGuiViewport* viewport, void*)
{
    if (WGL_WindowData* data = (WGL_WindowData*)viewport->RendererUserData)
        wglMakeCurrent(data->hDC, g_hRC);
}

static void Hook_Renderer_SwapBuffers(ImGuiViewport* viewport, void*)
{
    if (WGL_WindowData* data = (WGL_WindowData*)viewport->RendererUserData)
        ::SwapBuffers(data->hDC);
}

static int model = 0;

/*
0 -> HH
1 -> PN
2 -> LVM
3 -> LV
*/

// Nomi dei modelli
char* modelNames[] = { "Pendolo", "Lotka-Volterra (Preda-Predatore)", "Lotka-Volterra (Competizione)", "Modello di Hodgkin-Huxley" };
static Transform g_transform = { 0.0f, 0.0f, 1.0f };

// Parametri per il modello di Pendolo
static float g_L = 5.0f;           // Parametro lunghezza
static float g_gamma = 0.2f;       // Parametro coefficiente d'attrito
static float g_g = 9.81f;          // Parametro gravità

// Parametri per il modello di Lotka-Volterra preda-predatore
static float g_a = 0.6667f;        // Tasso di crescita intrinseco delle prede
static float g_b = 0.75f;          // Tasso di predazione
static float g_c = 1.0f;           // Tasso di conversione delle prede in predatori
static float g_d = 1.0f;           // Tasso di mortalità dei predatori

// Parametri per il modello di Lotka-Volterra in competizione
static float g_r1 = 1.2;           // Tasso di crescita intrinseco della specie 1
static float g_K1 = 3.0;           // Capacità portante della specie 1
static float g_a12 = 2.0;          // Effetto competitivo della specie 2 sulla specie 1
static float g_r2 = 0.9;           // Tasso di crescita intrinseco della specie 2
static float g_K2 = 5.0;           // Capacità portante della specie 2
static float g_a21 = 1.1;          // Effetto competitivo della specie 1 sulla specie 2

// Parametri per il modello di Hodgkin-Huxley
static float g_I_ext = 0.0f;       // Parametro corrente esterna
static int g_bif_ID = 0;           // Parametro biforcazini

// Parametri per la simulazione
static float g_dt_slider = 0.1f;
static float g_dt = pow(10.0f, (g_dt_slider * 3) - 3);
static float g_t = 1.5f;
static float g_t_max = 10.0f;
static bool use_iteration = false;
static int g_iterations = 100;
static int g_numSides = 7;
static int g_blockDimX = 5;
static int g_blockDimY = 5;

// Parametri per la bounding box
static bool g_useBoundingBox = false;
static float g_boundingBoxMinX = 0.0f;
static float g_boundingBoxMaxX = 1.0f;
static float g_boundingBoxMinY = 0.0f;
static float g_boundingBoxMaxY = 1.0f;

// Parametri per lo stretch
static float g_x_stretch = 1.0f;
static float g_y_stretch = 1.0f;

static bool is_model_parameters_expanded = true;

extern void run_cuda_kernel(int model, int width, int height, float* d_distances, float dt, float t_max, float offsetX, float offsetY, float zoom, float x_stretch, float y_stretch, float Iext, int bifurcation_type_id, float L, float gamma, float g, float r1, float K1, float a12, float r2, float K2, float a21, float a, float b, float c, float d, float scale, int numSides, int blockDimX, int blockDimY, bool useBoundingBox, float boundingBoxMinX, float boundingBoxMaxX, float boundingBoxMinY, float boundingBoxMaxY);

// Funzione CPU per la maschera
bool mask_function(float x, float y) {
    float centerX = 0.0f;
    float centerY = 0.0f;
    float radius = 1.5f;
    return (x - centerX) * (x - centerX) + (y - centerY) * (y - centerY) <= radius * radius;
}

// Funzioni di gestione CUDA
void initialize_cuda(int width, int height, float** d_distances) {
    size_t size = width * height * sizeof(float);
    cudaMalloc(d_distances, size);
    // Inizializza l'array a 0 sulla GPU
    cudaMemset(*d_distances, 0, size);
}

void copy_distances_to_host(float* h_distances, float* d_distances, int width, int height) {
    size_t size = width * height * sizeof(float);
    cudaMemcpy(h_distances, d_distances, size, cudaMemcpyDeviceToHost);
}

void cleanup_cuda(float* d_distances) {
    cudaFree(d_distances);
}

// Array di colori per la colormap "seismic"
static const ImVec3 colormap[] = {
    ImVec3(0.0f, 0.0f, 0.6f), // Blu scuro
    ImVec3(0.0f, 0.0f, 1.0f), // Blu puro
    ImVec3(1.0f, 1.0f, 1.0f), // Bianco
    ImVec3(1.0f, 0.0f, 0.0f),  // Rosso puro
    ImVec3(0.6f, 0.0f, 0.0f)  // Rosso scuro
};

// Funzione per interpolare tra i colori
ImVec3 get_color(float value) {
    // Normalizziamo il valore a un range [0, 1]
    if (value <= 0.0f) return colormap[0];
    if (value >= 2.0f) return colormap[4];
    value = value / 2;

    if (value < 0.25f) {
        // Intervallo blu scuro - blu
        float t = value / 0.25f;
        return ImVec3(
            colormap[0].x + (colormap[1].x - colormap[0].x) * t,
            colormap[0].y + (colormap[1].y - colormap[0].y) * t,
            colormap[0].z + (colormap[1].z - colormap[0].z) * t
        );
    }
    else if (value < 0.5f) {
        // Intervallo blu - bianco
        float t = (value - 0.25f) / 0.25f;
        return ImVec3(
            colormap[1].x + (colormap[2].x - colormap[1].x) * t,
            colormap[1].y + (colormap[2].y - colormap[1].y) * t,
            colormap[1].z + (colormap[2].z - colormap[1].z) * t
        );
    }
    else if (value < 0.75f) {
        // Intervallo bianco - rosso
        float t = (value - 0.5f) / 0.25f;
        return ImVec3(
            colormap[2].x + (colormap[3].x - colormap[2].x) * t,
            colormap[2].y + (colormap[3].y - colormap[2].y) * t,
            colormap[2].z + (colormap[3].z - colormap[2].z) * t
        );
    }
    else {
        // Intervallo rosso - rosso scuro
        float t = (value - 0.75f) / 0.25f;
        return ImVec3(
            colormap[3].x + (colormap[4].x - colormap[3].x) * t,
            colormap[3].y + (colormap[4].y - colormap[3].y) * t,
            colormap[3].z + (colormap[4].z - colormap[3].z) * t
        );
    }
}

ImVec2 transform_coordinates(int x, int y, float scale_x, float scale_y)
{
    float centeredX = (static_cast<float>(x) - g_Width / 2.0f);
    float centeredY = (static_cast<float>(y) - g_Height / 2.0f);
    float mappedX = (centeredX + g_transform.offsetX) / scale_x;
    float mappedY = -(centeredY + g_transform.offsetY) / scale_y;
    return ImVec2(mappedX, mappedY);

}

void render_scalar_field(float* distances) {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, g_Width, g_Height, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glBegin(GL_QUADS);
    for (int y = 0; y < g_Height; ++y) {
        for (int x = 0; x < g_Width; ++x) {

            // Calcola il valore del campo scalare dalla matrice delle distanze
            float value = distances[y * g_Width + x];

            // Se il valore è negativo, significa che il pixel è fuori dalla bounding box
            if (value < 0)
            {
                glColor3f(0.8f, 0.8f, 0.8f); // Colore grigio chiaro
                glVertex2f(static_cast<float>(x), static_cast<float>(y));
                glVertex2f(static_cast<float>(x + 1), static_cast<float>(y));
                glVertex2f(static_cast<float>(x + 1), static_cast<float>(y + 1));
                glVertex2f(static_cast<float>(x), static_cast<float>(y + 1));
                continue;
            }

            // Ottieni il colore dalla colormap
            ImVec3 color = get_color(value);

            glColor3f(color.x, color.y, color.z);
            glVertex2f(static_cast<float>(x), static_cast<float>(y));
            glVertex2f(static_cast<float>(x + 1), static_cast<float>(y));
            glVertex2f(static_cast<float>(x + 1), static_cast<float>(y + 1));
            glVertex2f(static_cast<float>(x), static_cast<float>(y + 1));
        }
    }
    glEnd();
}

static void HelpMarker(const char* desc)
{
    // Calcola la posizione X per allineare a destra
    float posX = ImGui::GetWindowWidth() - ImGui::CalcTextSize("(?)").x - ImGui::GetStyle().ItemSpacing.x - ImGui::GetStyle().WindowPadding.x;
    ImGui::SameLine(posX);

    ImGui::TextDisabled("(?)");
    if (ImGui::BeginItemTooltip())
    {
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(desc);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

struct PrecisionXY {
    int x;
    int y;
};

int calculate_precision_axis(float length) { // TODO: Si può trovare molto facilmente una funzione così, poi togli gli switch che sono meno efficienti
    if (length <= 0) {
        return 6;
    }

    float abs_length = std::abs(length);
    if (abs_length >= 100.0f) {
        return 0;
    }
    else if (abs_length >= 10.0f) {
        return 1;
    }
    else if (abs_length >= 1.0f) {
        return 2;
    }
    else if (abs_length >= 0.1f) {
        return 3;
    }
    else if (abs_length >= 0.01f) {
        return 4;
    }
    else if (abs_length >= 0.001f) {
        return 5;
    }
    else {
        return 6; // or a higher value if you need more precision for smaller numbers
    }
}

PrecisionXY calculate_precision(float horizontal_length, float vertical_length) {
    PrecisionXY precision;
    precision.x = calculate_precision_axis(horizontal_length);
    precision.y = calculate_precision_axis(vertical_length);
    return precision;
}

std::string format_number(float value, const PrecisionXY& precision, bool is_for_x_axis) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(is_for_x_axis ? precision.x : precision.y) << value;
    std::string label = ss.str();
    if (fabsf(value) < 1e-6f && label[0] == '-')
    {
        label = label.substr(1);
    }
    return label;
}

void render_axes(float scale_x, float scale_y) {
    ImDrawList* draw_list = ImGui::GetBackgroundDrawList();

    // Calcola le posizioni degli assi nel sistema di coordinate dello schermo
    float origin_x = g_Width / 2.0f;
    float origin_y = g_Height / 2.0f;

    // Disegna l'asse x
    draw_list->AddLine(ImVec2(0, origin_y), ImVec2(g_Width, origin_y), IM_COL32_BLACK);

    // Disegna l'asse y
    draw_list->AddLine(ImVec2(origin_x, 0), ImVec2(origin_x, g_Height), IM_COL32_BLACK);

    // Definisci il passo per i tick in pixel
    float tick_step_x = 100.0f;
    float tick_step_y = 100.0f;

    // Calcola la lunghezza dei cateti (una sola volta)
    ImVec2 top_left = transform_coordinates(0, 0, scale_x, scale_y);
    ImVec2 bottom_right = transform_coordinates(g_Width, g_Height, scale_x, scale_y);
    float horizontal_length = bottom_right.x - top_left.x;
    float vertical_length = top_left.y - bottom_right.y;

    // Calcola la precisione per X e Y (una sola volta)
    PrecisionXY precision = calculate_precision(horizontal_length, vertical_length);

    // Calcola e visualizza i tick sull'asse x
    for (float x_pos = origin_x; x_pos <= g_Width; x_pos += tick_step_x) {
        ImVec2 mapped = transform_coordinates(x_pos, origin_y, scale_x, scale_y);
        if (mapped.x == 0) continue;
        draw_list->AddLine(ImVec2(x_pos, origin_y - 5), ImVec2(x_pos, origin_y + 5), IM_COL32_BLACK);
        draw_list->AddText(ImVec2(x_pos + 2, origin_y + 5), IM_COL32_BLACK, format_number(mapped.x, precision, true).c_str());
    }
    for (float x_pos = origin_x; x_pos >= 0; x_pos -= tick_step_x) {
        ImVec2 mapped = transform_coordinates(x_pos, origin_y, scale_x, scale_y);
        if (mapped.x == 0) continue;
        draw_list->AddLine(ImVec2(x_pos, origin_y - 5), ImVec2(x_pos, origin_y + 5), IM_COL32_BLACK);
        draw_list->AddText(ImVec2(x_pos + 2, origin_y + 5), IM_COL32_BLACK, format_number(mapped.x, precision, true).c_str());
    }

    // Calcola e visualizza i tick sull'asse y
    for (float y_pos = origin_y; y_pos <= g_Height; y_pos += tick_step_y) {
        ImVec2 mapped = transform_coordinates(origin_x, y_pos, scale_x, scale_y);
        draw_list->AddLine(ImVec2(origin_x - 5, y_pos), ImVec2(origin_x + 5, y_pos), IM_COL32_BLACK);
        draw_list->AddText(ImVec2(origin_x - 20, y_pos - 14), IM_COL32_BLACK, format_number(mapped.y, precision, false).c_str());
    }
    for (float y_pos = origin_y; y_pos >= 0; y_pos -= tick_step_y) {
        ImVec2 mapped = transform_coordinates(origin_x, y_pos, scale_x, scale_y);
        draw_list->AddLine(ImVec2(origin_x - 5, y_pos), ImVec2(origin_x + 5, y_pos), IM_COL32_BLACK);
        draw_list->AddText(ImVec2(origin_x - 20, y_pos - 14), IM_COL32_BLACK, format_number(mapped.y, precision, false).c_str());
    }
}

void reset_parameters() {
    switch (model) {
    case 0: // Modello di Pendolo
        g_L = 5.0f;
        g_gamma = 0.2f;
        g_g = 9.81f;
        g_useBoundingBox = true;
        g_boundingBoxMinX = -M_PI;
        g_boundingBoxMaxX = M_PI;
        g_boundingBoxMinY = -4.0f;
        g_boundingBoxMaxY = 4.0f;
        g_x_stretch = 1.0f;
        g_y_stretch = 1.0f;
        g_transform.zoom = 0.2f;
        g_transform.offsetX = 0.0f;
        g_transform.offsetY = 0.0f;
        break;
    case 1: // Lotka-Volterra (Preda-Predatore)
        g_a = 0.66666f;
        g_b = 0.75f;
        g_c = 1.0f;
        g_d = 1.0f;
        g_useBoundingBox = true;
        g_boundingBoxMinX = 0.0f;
        g_boundingBoxMaxX = 3.0f;
        g_boundingBoxMinY = 0.0f;
        g_boundingBoxMaxY = 3.0f;
        g_x_stretch = 1.0f;
        g_y_stretch = 1.0f;
        g_transform.zoom = 0.5f;
        g_transform.offsetX = 0.0f;
        g_transform.offsetY = 0.0f;
        break;
    case 2: // Modello Lotka Volterra (Competizione)
        g_r1 = 1.2f;
        g_K1 = 3.0f;
        g_a12 = 2.0f;
        g_r2 = 0.9f;
        g_K2 = 5.0f;
        g_a21 = 1.1f;
        g_useBoundingBox = true;
        g_boundingBoxMinX = 0.0f;
        g_boundingBoxMaxX = 3.0f;
        g_boundingBoxMinY = 0.0f;
        g_boundingBoxMaxY = 3.0f;
        g_x_stretch = 1.0f;
        g_y_stretch = 1.0f;
        g_transform.zoom = 0.5f;
        g_transform.offsetX = 0.0f;
        g_transform.offsetY = 0.0f;
        break;
    case 3: // Modello di Hodgkin-Huxley
        g_I_ext = 0.0f;
        g_bif_ID = 0;
        g_useBoundingBox = true;
        g_boundingBoxMinX = -90.0f;
        g_boundingBoxMaxX = 20.0f;
        g_boundingBoxMinY = 0.0f;
        g_boundingBoxMaxY = 1.0f;
        g_x_stretch = 110.0f;
        g_y_stretch = 1.0f;
        g_transform.zoom = 1.0f;
        g_transform.offsetX = 0.0f;
        g_transform.offsetY = 0.0f;
        break;
    }
}

struct TextureInfo {
    GLuint id;
    int width;
    int height;
};

TextureInfo loadTextureFromFile(const std::string& filename) {
    int width, height, channels;
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 0);
    if (!data) {
        return { 0, 0, 0 }; // Restituisci una struttura con valori nulli in caso di errore
    }

    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    if (channels == 3) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    }
    else if (channels == 4) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    }
    else {
        glDeleteTextures(1, &textureID);
        stbi_image_free(data);
        return { 0, 0, 0 }; // Restituisci una struttura con valori nulli in caso di errore
    }

    glBindTexture(GL_TEXTURE_2D, 0);
    stbi_image_free(data);

    return { textureID, width, height }; // Restituisci l'ID della texture e le dimensioni
}
static TextureInfo hh_image;
static TextureInfo pendulum_image;
static TextureInfo lvm_image;
static TextureInfo lv_image;
static bool texturesLoaded = false;

void loadTextures()
{
    if (!texturesLoaded) {
        hh_image = loadTextureFromFile("equations\\Hodgkin-Huxley.png");
        pendulum_image = loadTextureFromFile("equations\\Pendulum.png");
        lvm_image = loadTextureFromFile("equations\\lvm.png");
        lv_image = loadTextureFromFile("equations\\lv.png");
        texturesLoaded = true;
    }
}

void display_image_below_last_item(GLuint texture_id, int texture_width, int texture_height, float desired_height, float vertical_spacing = 10.0f)
{
    // Ottieni la posizione dell'ultimo elemento
    ImVec2 lastItemPos = ImGui::GetItemRectMax();

    // Calcola la larghezza dell'immagine mantenendo le proporzioni
    float aspectRatio = (float)texture_width / (float)texture_height;
    float imageWidth = desired_height * aspectRatio;
    float imageHeight = desired_height;

    // Calcola la posizione X per centrare l'immagine
    float cursorPos_x = ImGui::GetWindowPos().x + (ImGui::GetWindowWidth() - imageWidth) / 2.0f;

    // Imposta la posizione del cursore sotto l'ultimo elemento, con un po' di spazio verticale
    ImGui::SetCursorScreenPos(ImVec2(cursorPos_x, lastItemPos.y + vertical_spacing));

    // Coordinate UV per mappare l'intera texture
    ImVec2 uv_min = ImVec2(0.0f, 0.0f);
    ImVec2 uv_max = ImVec2(1.0f, 1.0f);

    // Visualizza l'immagine
    if (texture_id != 0) {
        ImGui::Image((void*)(intptr_t)texture_id, ImVec2(imageWidth, imageHeight), uv_min, uv_max);
    }
}

bool view_title = true;
static void title(bool* p_open, char* model_name)
{
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav;
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImVec2 window_pos, window_pos_pivot;
    window_pos.x = (g_Width / 2);
    window_pos.y = 20.0f;
    window_pos_pivot.x = 0.5f;
    window_pos_pivot.y = 0.0f;
    ImGui::PushFont(black);
    ImGui::SetNextWindowPos(ImVec2(window_pos.x,window_pos.y), ImGuiCond_Always, window_pos_pivot);
    ImGui::SetNextWindowViewport(viewport->ID);
    window_flags |= ImGuiWindowFlags_NoMove;
    ImGui::SetNextWindowBgAlpha(0.8f);
    if (ImGui::Begin(" ", p_open, window_flags))
    {
        ImGui::Text(model_name);
    }
    ImGui::PopFont();
    ImGui::End();
}
static bool is_choosing = false;
static int selected_model = 0;
static bool just_chosen = false;

static void choose_model(bool* p_open)
    {
        static int larghezza_suprema = 0;

        ImVec2 window_pos;
        window_pos.x = 20;
        window_pos.y = 80;

        ImGui::SetNextWindowPos(window_pos, ImGuiCond_Appearing);
        ImGui::SetNextWindowSize(ImVec2(0, 500));
        if (ImGui::Begin("Scegli Modello", p_open, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse))
        {
            // Left
            {
                ImGui::BeginChild("Pannello sinistro", ImVec2(275, 0), ImGuiChildFlags_Borders);
                ImGui::Text("Modelli disponibili:");
                ImGui::Dummy(ImVec2(0, 2));
                ImGui::Separator();
                ImGui::Dummy(ImVec2(0, 2));
                for (int i = 0; i < IM_ARRAYSIZE(modelNames); i++)
                {
                    bool is_selected = (selected_model == i);

                    if (is_selected) {
                        ImGui::PushFont(bold);
                    }

                    if (ImGui::Selectable(modelNames[i], is_selected))
                    {
                        selected_model = i;
                    }

                    if (is_selected) {
                        ImGui::PopFont();
                    }

                    ImGui::Dummy(ImVec2(0, 2));
                }
                ImGui::EndChild();
            }
            ImGui::SameLine();

            // Right
            {
                float static_width = std::max(400.0f, 100 + ImGui::CalcTextSize(("Sistema Dinamico: " + std::string(modelNames[selected_model])).c_str()).x);

                ImGui::BeginGroup();
                ImGui::BeginChild("Vista elementi", ImVec2(std::max((float)larghezza_suprema, static_width) + 50.0f, -ImGui::GetFrameHeightWithSpacing() - 10.0f)); // Leave room for 1 line below us
                ImGui::Dummy(ImVec2(0, 3));
                ImGui::PushFont(bold);
                ImGui::Text("Sistema Dinamico: %s", modelNames[selected_model]);
                ImGui::PopFont();
                ImGui::Dummy(ImVec2(0, 2));
                ImGui::Separator();
                ImGui::Dummy(ImVec2(0, 2));

                if (ImGui::BeginTabBar("##Schede", ImGuiTabBarFlags_None))
                {
                    if (ImGui::BeginTabItem("Descrizione"))
                    {
                        larghezza_suprema = 0;
                        static_width = std::max(400.0f, 100 + ImGui::CalcTextSize(("Sistema Dinamico: " + std::string(modelNames[selected_model])).c_str()).x);
                        ImGui::Dummy(ImVec2(0, 3));
                        switch (selected_model) {
                        case 0:
                            ImGui::TextWrapped("Il pendolo semplice è un sistema fisico idealizzato costituito da una massa puntiforme sospesa a un filo inestensibile e di massa trascurabile, vincolato a oscillare sotto l'azione della forza di gravità. Il suo movimento è approssimativamente armonico per piccole oscillazioni. Il periodo di oscillazione dipende dalla lunghezza del filo e dall'accelerazione di gravità.");
                            break;
                        case 1:
                            ImGui::TextWrapped("Il modello di Lotka-Volterra preda-predatore descrive le dinamiche di interazione tra due specie, una preda e un predatore. Le equazioni differenziali modellano la variazione delle dimensioni delle popolazioni nel tempo, considerando il tasso di crescita della preda in assenza del predatore, il tasso di predazione, il tasso di mortalità del predatore in assenza della preda, e l'efficienza di conversione della preda in nuova biomassa del predatore.");
                            break;
                        case 2:
                            ImGui::TextWrapped("Il modello di Lotka-Volterra per la competizione interspecifica descrive le dinamiche di due specie che competono per le stesse risorse. Si basa su un sistema di equazioni differenziali che modellano la variazione delle dimensioni delle popolazioni nel tempo, tenendo conto dei tassi di crescita intrinseci di ciascuna specie, della capacità portante dell'ambiente e dei coefficienti di competizione che quantificano l'impatto di una specie sull'altra.");
                            break;
                        case 3:
                            ImGui::TextWrapped("Il modello di Hodgkin-Huxley è un modello matematico che descrive la propagazione del potenziale d'azione nei neuroni. Si basa su un sistema di equazioni differenziali non lineari che rappresentano le correnti ioniche attraverso la membrana cellulare, mediate da canali voltaggio-dipendenti per il sodio (Na+) e il potassio (K+). Il modello permette di simulare e comprendere le proprietà elettriche fondamentali delle cellule eccitabili.");
                            break;
                        }
                        ImGui::EndTabItem();
                    }
                    if (ImGui::BeginTabItem("Dettagli"))
                    {
                        // Calcola la larghezza massima necessaria per la visualizzazione, basandosi sul testo o sull'immagine.
                        static_width = std::max(400.0f, 100 + ImGui::CalcTextSize(("Sistema Dinamico: " + std::string(modelNames[selected_model])).c_str()).x);

                        // Definisci le altezze desiderate per le immagini.
                        float height_hh = 350;
                        float height_pn = 175;
                        float height_lvm = 175;
                        float height_lv = 175;

                        ImGui::Dummy(ImVec2(0, 3));

                        switch (selected_model) {
                        case 0: // Pendolo Semplice
                            larghezza_suprema = pendulum_image.width * height_pn / pendulum_image.height;

                            ImGui::PushFont(bold);
                            ImGui::Text("Variabili di stato:");
                            ImGui::PopFont();
                            ImGui::BulletText("theta: Angolo rispetto alla verticale");
                            ImGui::BulletText("omega: Velocità angolare");
                            ImGui::Dummy(ImVec2(0, 2));

                            ImGui::PushFont(bold);
                            ImGui::Text("Parametri:");
                            ImGui::PopFont();
                            ImGui::BulletText("L: Lunghezza del pendolo");
                            ImGui::BulletText("gamma: Coefficiente di attrito viscoso");
                            ImGui::BulletText("g: Accelerazione di gravità");

                            display_image_below_last_item(pendulum_image.id, pendulum_image.width, pendulum_image.height, height_pn);
                            break;

                        case 1: // Lotka-Volterra (Preda-Predatore)
                            larghezza_suprema = lv_image.width * height_lv / lv_image.height;

                            ImGui::PushFont(bold);
                            ImGui::Text("Variabili di stato:");
                            ImGui::PopFont();
                            ImGui::BulletText("x: Densità di popolazione della preda");
                            ImGui::BulletText("y: Densità di popolazione del predatore");
                            ImGui::Dummy(ImVec2(0, 2));

                            ImGui::PushFont(bold);
                            ImGui::Text("Parametri:");
                            ImGui::PopFont();
                            ImGui::BulletText("a: Tasso di crescita intrinseco della preda");
                            ImGui::BulletText("b: Tasso di incontro preda-predatore");
                            ImGui::BulletText("c: Tasso di mortalità del predatore");
                            ImGui::BulletText("d: Efficienza di conversione preda-predatore");

                            display_image_below_last_item(lv_image.id, lv_image.width, lv_image.height, height_lv);
                            break;

                        case 2: // Lotka-Volterra (Competizione)
                            larghezza_suprema = lvm_image.width * height_lvm / lvm_image.height;

                            ImGui::PushFont(bold);
                            ImGui::Text("Variabili di stato:");
                            ImGui::PopFont();
                            ImGui::BulletText("N1: Densità di popolazione della specie 1");
                            ImGui::BulletText("N2: Densità di popolazione della specie 2");
                            ImGui::Dummy(ImVec2(0, 2));

                            ImGui::PushFont(bold);
                            ImGui::Text("Parametri:");
                            ImGui::PopFont();
                            ImGui::BulletText("r1: Tasso di crescita intrinseco della specie 1");
                            ImGui::BulletText("K1: Capacità portante per la specie 1");
                            ImGui::BulletText("a12: Coefficiente di competizione di 2 su 1");
                            ImGui::BulletText("r2: Tasso di crescita intrinseco della specie 2");
                            ImGui::BulletText("K2: Capacità portante per la specie 2");
                            ImGui::BulletText("a21: Coefficiente di competizione di 1 su 2");

                            display_image_below_last_item(lvm_image.id, lvm_image.width, lvm_image.height, height_lvm);
                            break;

                        case 3: // Hodgkin-Huxley (Ridotto)
                            larghezza_suprema = hh_image.width * height_hh / hh_image.height;

                            ImGui::PushFont(bold);
                            ImGui::Text("Variabili di stato:");
                            ImGui::PopFont();
                            ImGui::BulletText("V: Potenziale di membrana");
                            ImGui::BulletText("n: Probabilità di attivazione del potassio");
                            ImGui::Dummy(ImVec2(0, 2));

                            ImGui::PushFont(bold);
                            ImGui::Text("Parametri:");
                            ImGui::PopFont();
                            ImGui::BulletText("I_ext: Corrente esterna applicata");
                            ImGui::BulletText("g_Na: Conduttanza massima del sodio");
                            ImGui::BulletText("g_K: Conduttanza massima del potassio");
                            ImGui::BulletText("g_L: Conduttanza di leakage");
                            ImGui::BulletText("E_Na: Potenziale di equilibrio del sodio");
                            ImGui::BulletText("E_K: Potenziale di equilibrio del potassio");
                            ImGui::BulletText("E_L: Potenziale di equilibrio di leakage");
                            ImGui::BulletText("tau_n: Costante di tempo per l'attivazione di n");
                            ImGui::BulletText("V_mid_n: Potenziale di metà attivazione per n");
                            ImGui::BulletText("k_n: Fattore di sensibilità per l'attivazione di n");

                            display_image_below_last_item(hh_image.id, hh_image.width, hh_image.height, height_hh);
                            break;
                        }
                        ImGui::EndTabItem();
                    }
                    ImGui::EndTabBar();
                }
                ImGui::EndChild();
                // Posiziona il pulsante "OK" in basso a destra
                ImGui::Dummy(ImVec2(0, ImGui::GetContentRegionAvail().y - ImGui::GetFrameHeightWithSpacing())); // Spazio vuoto prima del pulsante
                ImGui::Dummy(ImVec2(0, 5.0f));
                ImGui::SetCursorPosX((ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize("OK").x - ImGui::GetStyle().FramePadding.x) / 2 + 100); // Posiziona il pulsante
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 10.0f);
                if (ImGui::Button("Ok")) {
                    model = selected_model;
                    reset_parameters();
                    *p_open = false;
                    is_model_parameters_expanded = true;
                    just_chosen = true;
                }
                //ImGui::Dummy(ImVec2(0, 2.0f));

                ImGui::EndGroup();
            }
        }
        ImGui::End();
    }

class MyStyleManager {
public:
    void Apply() {
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 5.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(15.0f, 15.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10.0f, 5.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(10.0f, 5.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemInnerSpacing, ImVec2(10.0f, ImGui::GetStyle().ItemInnerSpacing.y));
        ImGui::PushStyleVar(ImGuiStyleVar_IndentSpacing, 30.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 5.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_GrabRounding, 5.0f); // Aggiunto GrabRounding

        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 0.9f)); // Aggiunto WindowBg
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.53f, 0.53f, 0.53f, 1.0f));   // Aggiunto Border
        ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));    // Aggiunto TitleBg
        ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.2f, 0.2f, 0.2f, 1.0f)); //Uguale a TitleBg per non avere comportamenti strani
        ImGui::PushStyleColor(ImGuiCol_TitleBgCollapsed, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));//Uguale a TitleBg per non avere comportamenti strani
        pushedStyleVarCount_ = 9;        // Aggiornato il conteggio delle variabili di stile
        pushedStyleColorCount_ = 5;    // Aggiornato il conteggio dei colori
    }

    void Restore() {
        ImGui::PopStyleVar(pushedStyleVarCount_);
        ImGui::PopStyleColor(pushedStyleColorCount_);
        pushedStyleVarCount_ = 0;
        pushedStyleColorCount_ = 0;
    }

private:
    int pushedStyleVarCount_ = 0;
    int pushedStyleColorCount_ = 0;
};

int main(int, char**)
{
    WNDCLASSEXW wc = { sizeof(wc), CS_OWNDC, WndProc, 0L, 0L, GetModuleHandle(nullptr), nullptr, nullptr, nullptr, nullptr, L"C Viewer", nullptr };
    ::RegisterClassExW(&wc);
    HWND hwnd = ::CreateWindowW(wc.lpszClassName, L"C Viewer", WS_POPUP | WS_MAXIMIZE | WS_VISIBLE, 0, 0, 0, 0, nullptr, nullptr, wc.hInstance, nullptr);

    // Imposta le dimensioni della finestra a schermo intero
    MONITORINFO monitorInfo = { sizeof(monitorInfo) };
    GetMonitorInfo(MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST), &monitorInfo);
    g_Width = monitorInfo.rcMonitor.right - monitorInfo.rcMonitor.left;
    g_Height = monitorInfo.rcMonitor.bottom - monitorInfo.rcMonitor.top;

    SetWindowPos(hwnd, HWND_TOP, 0, 0, g_Width, g_Height, SWP_NOZORDER);
    if (!CreateDeviceWGL(hwnd, &g_MainWindow))
    {
        CleanupDeviceWGL(hwnd, &g_MainWindow);
        ::DestroyWindow(hwnd);
        ::UnregisterClassW(wc.lpszClassName, wc.hInstance);
        return 1;
    }
    wglMakeCurrent(g_MainWindow.hDC, g_hRC);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    //io.FontGlobalScale = 1.25f;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }
    ImGui_ImplWin32_InitForOpenGL(hwnd);
    ImGui_ImplOpenGL3_Init();

    //font_default = io.Fonts->AddFontDefault();
    regular = io.Fonts->AddFontFromFileTTF("fonts/Montserrat-Regular.ttf", 17.0f);
    bold = io.Fonts->AddFontFromFileTTF("fonts/Montserrat-Bold.ttf", 17.0f);
    italic = io.Fonts->AddFontFromFileTTF("fonts/Montserrat-Italic.ttf", 17.0f);
    thin = io.Fonts->AddFontFromFileTTF("fonts/Montserrat-ExtraLight.ttf", 17.0f);
    black = io.Fonts->AddFontFromFileTTF("fonts/Montserrat-Black.ttf", 20.0f);

    loadTextures();

    MyStyleManager styleManager; // ISTANZA DELLO STILE
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        ImGuiPlatformIO& platform_io = ImGui::GetPlatformIO();
        IM_ASSERT(platform_io.Renderer_CreateWindow == NULL);
        IM_ASSERT(platform_io.Renderer_DestroyWindow == NULL);
        IM_ASSERT(platform_io.Renderer_SwapBuffers == NULL);
        IM_ASSERT(platform_io.Platform_RenderWindow == NULL);
        platform_io.Renderer_CreateWindow = Hook_Renderer_CreateWindow;
        platform_io.Renderer_DestroyWindow = Hook_Renderer_DestroyWindow;
        platform_io.Renderer_SwapBuffers = Hook_Renderer_SwapBuffers;
        platform_io.Platform_RenderWindow = Hook_Platform_RenderWindow;
    }
    ImVec2 window_size = ImVec2(300, 200);
    bool done = false;

    // Variabili CUDA
    float* d_distances = nullptr; //device
    float* h_distances = new float[g_Width * g_Height]; //host

    // INIT PARAMETERS
    reset_parameters();

    initialize_cuda(g_Width, g_Height, &d_distances);

    while (!done)
    {
        MSG msg;
        while (::PeekMessage(&msg, nullptr, 0U, 0U, PM_REMOVE))
        {
            ::TranslateMessage(&msg);
            ::DispatchMessage(&msg);
            if (msg.message == WM_QUIT)
                done = true;
        }
        if (done)
            break;
        if (::IsIconic(hwnd))
        {
            ::Sleep(10);
            continue;
        }
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();
        ImGui::PushFont(regular);
        float scale = std::min(g_Width, g_Height) / 10.0f;
        float scale_x = scale * g_transform.zoom / g_x_stretch;
        float scale_y = scale * g_transform.zoom / g_y_stretch;
        render_axes(scale_x, scale_y);

        // Calcola le coordinate del centro della finestra nel sistema di coordinate del mondo PRIMA dello zoom
        ImVec2 center_screen(g_Width / 2.0f, g_Height / 2.0f);
        ImVec2 center_world_before = transform_coordinates(center_screen.x, center_screen.y, scale_x, scale_y);

        // Aggiorna lo zoom
        if (!io.WantCaptureMouse) {
            g_transform.zoom *= std::pow(1.0f + 0.1f, io.MouseWheel);
            // Impedisci zoom eccessivo
            if (g_transform.zoom < 0.05f) g_transform.zoom = 0.05f;
            if (g_transform.zoom > 10000.0f) g_transform.zoom = 10000.0f;
        }

        if (!use_iteration) {
            g_dt = pow(10.0f, (g_dt_slider * 3) - 3); // Aggiorna il valore di dt
        }

        // Calcola i fattori di scala per gli assi
        scale_x = scale * g_transform.zoom / g_x_stretch;
        scale_y = scale * g_transform.zoom / g_y_stretch;
        float gpu_scale = std::min(g_Width, g_Height) / 10.0f * g_transform.zoom;

        // Calcola le coordinate del centro della finestra nel sistema di coordinate del mondo DOPO lo zoom
        ImVec2 center_world_after = transform_coordinates(center_screen.x, center_screen.y, scale_x, scale_y);

        // Calcola la differenza e aggiorna l'offset
        g_transform.offsetX -= (center_world_after.x - center_world_before.x) * scale_x;
        g_transform.offsetY += (center_world_after.y - center_world_before.y) * scale_y;

        // Gestisci il drag del mouse per spostare il campo scalare
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left) && !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow) && !ImGui::IsAnyItemActive()) // Se il mouse è trascinato, non è sopra una finestra ImGui e non sta interagendo con un elemento
        {
            g_transform.offsetX -= ImGui::GetIO().MouseDelta.x;
            g_transform.offsetY -= ImGui::GetIO().MouseDelta.y;
        }

        // Esegui l'integrazione con CUDA solo se nencessario
        if (!is_choosing) {
            run_cuda_kernel(model, g_Width, g_Height, d_distances, g_dt, g_t, g_transform.offsetX, g_transform.offsetY, g_transform.zoom, g_x_stretch, g_y_stretch, g_I_ext, g_bif_ID, g_L, g_gamma, g_g, g_r1, g_K1, g_a12, g_r2, g_K2, g_a21, g_a, g_b, g_c, g_d, gpu_scale, g_numSides, g_blockDimX, g_blockDimY, g_useBoundingBox, g_boundingBoxMinX, g_boundingBoxMaxX, g_boundingBoxMinY, g_boundingBoxMaxY);
        }
        // Copia i dati dalla GPU alla CPU
        copy_distances_to_host(h_distances, d_distances, g_Width, g_Height);

        //ImGui::ShowDemoWindow();
        styleManager.Apply();
        ImGui::SetNextWindowPos(ImVec2(20, 35), ImGuiCond_Once);
        ImGui::PushFont(bold);

        ImGui::SetNextWindowCollapsed(false, ImGuiCond_FirstUseEver);

        if (is_choosing) {
            ImGui::SetNextWindowCollapsed(!is_model_parameters_expanded);
        }
        else {
            if (just_chosen) {
                ImGui::SetNextWindowCollapsed(false);
                just_chosen = false;
            }
        }

        ImGui::SetNextWindowSizeConstraints(ImVec2(0, 0), ImVec2(450, 800));
        ImGui::Begin("Parametri del Modello", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::PopFont();
        const char* bifNames[] = {
            "Saddle-node",
            "SNIC",
            "Subcritical Hopf",
            "Supercritical Hopf"
        };
        ImGui::SetCursorPosX((ImGui::GetContentRegionAvail().x - 110) / 2);
        if (ImGui::Button("Menu Modelli", ImVec2(120, 25))) {
            is_choosing = true;
            is_model_parameters_expanded = false;
        }

        if (is_choosing) {
            choose_model(&is_choosing);
        }

        ImGui::Dummy(ImVec2(0, 3));
        ImGui::PushFont(bold);
        ImGui::SeparatorText("Parametri del Modello");
        ImGui::PopFont();
        ImGui::Dummy(ImVec2(0, 3));

        switch (model) {
        case 0:
            ImGui::SliderFloat("L", &g_L, 0.0f, 10.0f);
            ImGui::SliderFloat("gamma", &g_gamma, 0.0f, 2.0f);
            ImGui::SliderFloat("g", &g_g, 0.0f, 20.0f);
            break;
        case 1:
            ImGui::SliderFloat("a", &g_a, 0.0f, 5.0f);
            ImGui::SliderFloat("b", &g_b, 0.0f, 5.0f);
            ImGui::SliderFloat("c", &g_c, 0.0f, 5.0f);
            ImGui::SliderFloat("d", &g_d, 0.0f, 5.0f);
            break;
        case 2:
        {
            ImGui::SliderFloat("r1", &g_r1, 0.0f, 5.0f);
            ImGui::SliderFloat("K1", &g_K1, 0.0f, 20.0f);
            ImGui::SliderFloat("a12", &g_a12, 0.0f, 5.0f);
            ImGui::SliderFloat("r2", &g_r2, 0.0f, 5.0f);
            ImGui::SliderFloat("K2", &g_K2, 0.0f, 20.0f);
            ImGui::SliderFloat("a21", &g_a21, 0.0f, 5.0f);

            const char* presetNames[] = {
                "Coesistenza",
                "Esclusione Competitiva (1)",
                "Esclusione Competitiva (2)",
                 "Oscillazioni",
            };
            static int currentPreset = -1;

            if (ImGui::BeginCombo("Presets", currentPreset == -1 ? "Select Preset" : presetNames[currentPreset]))
            {
                for (int n = 0; n < IM_ARRAYSIZE(presetNames); n++) {
                    bool is_selected = (currentPreset == n);
                    if (ImGui::Selectable(presetNames[n], is_selected)) {
                        currentPreset = n;
                        switch (n) {
                        case 0: //Coesistenza
                            g_r1 = 1.1f;
                            g_K1 = 5.0f;
                            g_a12 = 0.7f;
                            g_r2 = 0.8f;
                            g_K2 = 7.0f;
                            g_a21 = 0.4f;
                            break;
                        case 1: // Esclusione Competitiva (Specie 1)
                            g_r1 = 1.2f;
                            g_K1 = 3.0f;
                            g_a12 = 2.0f;
                            g_r2 = 0.9f;
                            g_K2 = 5.0f;
                            g_a21 = 1.1f;
                            break;
                        case 2: // Esclusione Competitiva (Specie 2)
                            g_r1 = 1.2f;
                            g_K1 = 3.0f;
                            g_a12 = 1.1f;
                            g_r2 = 0.9f;
                            g_K2 = 5.0f;
                            g_a21 = 2.0f;
                            break;
                        case 3: // Oscillazioni
                            g_r1 = 1.0f;
                            g_K1 = 3.0f;
                            g_a12 = 1.5f;
                            g_r2 = 1.1f;
                            g_K2 = 3.0f;
                            g_a21 = 1.5f;
                            break;
                        }
                    }
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
        }
            break;
        case 3:
            ImGui::SliderFloat("I_ext", &g_I_ext, 0.0f, 5.0f);
            if (ImGui::BeginCombo("Tipo di Biforcazione", bifNames[g_bif_ID])) {
                for (int n = 0; n < IM_ARRAYSIZE(bifNames); n++) {
                    bool isSelected = (g_bif_ID == n);
                    if (ImGui::Selectable(bifNames[n], isSelected)) {
                        g_bif_ID = n; // Aggiorna il valore di 'g_bif_ID'
                    }
                    if (isSelected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
            break;
        }
        ImGui::Dummy(ImVec2(0, 3));
        ImGui::PushFont(bold);
        ImGui::SeparatorText("Impostazioni");
        ImGui::PopFont();
        ImGui::Dummy(ImVec2(0, 3));

        if (ImGui::TreeNode("Impostazioni della Simulazione")) {
            ImGui::Checkbox("Usa # iterazioni", &use_iteration);
            if (use_iteration) {
                ImGui::DragInt("Iterazioni", &g_iterations, 0.5, 1);
                if (g_iterations < 1) {
                    g_iterations = 1;
                }
            }
            else {
                ImGui::SliderFloat("dt", &g_dt_slider, 0.0f, 1.0f);
            }
            ImGui::InputFloat("T Max", &g_t_max, 0.0f, 0.0f, "%.1f");
            g_dt = g_t / g_iterations;
            ImGui::SliderFloat("T", &g_t, 0.0f, g_t_max);
            if (g_t < 0) {
;                g_t = 0;
            }
            ImGui::SliderInt("Numero Vertici", &g_numSides, 3, 50);
            ImGui::Dummy(ImVec2(0, 3));
            ImGui::Separator();
            ImGui::Dummy(ImVec2(0, 3));
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.5f);
            ImGui::SliderInt("Block Dim", &g_blockDimX, 1, 7);
            ImGui::SameLine();
            HelpMarker("Controlla la dimensione del blocco di thread in CUDA.\n\n"
                "1: Un thread per pixel (massima precisione, minori FPS).\n"
                "N > 1: Un blocco di N x N thread (maggiori FPS, riduzione della precisione).");
            g_blockDimY = g_blockDimX;
            ImGui::TreePop();
        }
        if (ImGui::TreeNode("Dominio")) {
            ImGui::Checkbox("Utilizza Dominio", &g_useBoundingBox);
            if (g_useBoundingBox) {
                ImGui::DragFloatRange2("X", &g_boundingBoxMinX, &g_boundingBoxMaxX, 0.1f, -50000, 50000, "Min: %.1f%", "Max: %.1f%", ImGuiSliderFlags_AlwaysClamp);
                ImGui::DragFloatRange2("Y", &g_boundingBoxMinY, &g_boundingBoxMaxY, 0.1f, -50000, 50000, "Min: %.1f%", "Max: %.1f%", ImGuiSliderFlags_AlwaysClamp);
            };
            ImGui::TreePop();
        }

        float stretch_factor[2] = { g_x_stretch,  g_y_stretch };
        float offset_factor[2] = { g_transform.offsetX,  g_transform.offsetY };

        if (ImGui::TreeNode("Impostazioni Debug di Render")) {
            ImGui::DragFloat("Zoom", &g_transform.zoom, 0.5f);
            if (g_transform.zoom < 0.05f) g_transform.zoom = 0.05f;
            ImGui::DragFloat2("Offset", offset_factor, 0.5f, -5000, 5000);
            g_transform.offsetX = offset_factor[0];
            g_transform.offsetY = offset_factor[1];
            ImGui::DragFloat2("Stretch", stretch_factor, 0.1f, -500, 500);
            g_x_stretch = stretch_factor[0];
            g_y_stretch = stretch_factor[1];
            ImGui::TreePop();
        }

        if (g_x_stretch < 1.0f) g_x_stretch = 1.0f;
        if (g_y_stretch < 1.0f) g_y_stretch = 1.0f;

        ImGui::Dummy(ImVec2(0, 3));

        if (ImGui::Button("Reimposta Visuale"))
        {
            g_blockDimX = 5;
            g_blockDimY = 5;
            g_transform.zoom = 1.0f;
            g_transform.offsetX = 0.0f;
            g_transform.offsetY = 0.0f;
        }
        ImGui::SetItemTooltip("Reimposta lo zoom e la tralsazione.", ImGui::GetStyle().HoverDelayNormal);

        ImGui::SameLine();
        if (ImGui::Button("Assi Uguali")) {
            g_x_stretch = 1.0f;
            g_y_stretch = 1.0f;
        }
        ImGui::SetItemTooltip("Rende la scala di visualizzazione\nuguale per entrambi gli assi.", ImGui::GetStyle().HoverDelayNormal);

        if (g_useBoundingBox) {
            ImGui::SameLine();
            if (ImGui::Button("Aspetto 1:1"))
            {
                if (g_boundingBoxMaxX - g_boundingBoxMinX > g_boundingBoxMaxY - g_boundingBoxMinY) {
                    g_x_stretch = (g_boundingBoxMaxX - g_boundingBoxMinX) / (g_boundingBoxMaxY - g_boundingBoxMinY);
                    g_y_stretch = 1.0f;
                }
                else if (g_boundingBoxMaxX - g_boundingBoxMinX < g_boundingBoxMaxY - g_boundingBoxMinY) {
                    g_y_stretch = (g_boundingBoxMaxY - g_boundingBoxMinY) / (g_boundingBoxMaxX - g_boundingBoxMinX);
                    g_x_stretch = 1.0f;
                }
            }
            ImGui::SetItemTooltip("Applica una dilatazione agli assi per visualizzare\nil dominio impostato in un quadrato.", ImGui::GetStyle().HoverDelayNormal);
        }

        ImGui::Dummy(ImVec2(0, 3));
        ImGui::Text("Media applicazione %.0f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);

        title(&view_title, modelNames[model]);

        styleManager.Restore();
        ImGui::End();

        // Ottieni le coordinate del mouse e visualizza il valore della funzione
        ImVec2 mouse_pos = ImGui::GetIO().MousePos;
        ImVec2 mouse_mapped = transform_coordinates(mouse_pos.x, mouse_pos.y, scale_x, scale_y);
        float z_value = 0.0f;
        if (mouse_pos.x >= 0 && mouse_pos.x < g_Width && mouse_pos.y >= 0 && mouse_pos.y < g_Height)
            z_value = h_distances[(int)mouse_pos.y * g_Width + (int)mouse_pos.x];

        ImVec2 top_left = transform_coordinates(0, 0, scale_x, scale_y);
        ImVec2 bottom_right = transform_coordinates(g_Width, g_Height, scale_x, scale_y);
        float horizontal_length = bottom_right.x - top_left.x;
        float vertical_length = top_left.y - bottom_right.y;
        PrecisionXY precision = calculate_precision(horizontal_length, vertical_length);

        std::stringstream ss;
        if (z_value != -1.0f && ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
            ss << "C(" << std::fixed << std::setprecision(1) << g_t << ", " << format_number(mouse_mapped.x, precision, true) << ", " << format_number(mouse_mapped.y, precision, false) << ") = " << std::fixed << std::setprecision(5) << z_value;
            ImGui::GetBackgroundDrawList()->AddText(ImVec2(mouse_pos.x, mouse_pos.y - 20), IM_COL32_BLACK, ss.str().c_str());
        }

        // Versione
        if (g_Width > 0 && g_Height > 0)
        {
            ImGui::PushFont(thin);
            ImGui::GetBackgroundDrawList()->AddText(ImVec2(10, 10), IM_COL32_BLACK, "C Viewer  -  v1.0.1-beta");
            ImGui::PopFont();
        }
        ImGui::PopFont();
        ImGui::Render();
        glViewport(0, 0, g_Width, g_Height);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        render_scalar_field(h_distances);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            wglMakeCurrent(g_MainWindow.hDC, g_hRC);
        }
        ::SwapBuffers(g_MainWindow.hDC);
    }

    //Cleanup cuda
    cleanup_cuda(d_distances);
    delete[] h_distances;

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
    CleanupDeviceWGL(hwnd, &g_MainWindow);
    wglDeleteContext(g_hRC);
    ::DestroyWindow(hwnd);
    ::UnregisterClassW(wc.lpszClassName, wc.hInstance);
    return 0;
}

bool CreateDeviceWGL(HWND hWnd, WGL_WindowData* data)
{
    HDC hDc = ::GetDC(hWnd);
    PIXELFORMATDESCRIPTOR pfd = { 0 };
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    const int pf = ::ChoosePixelFormat(hDc, &pfd);
    if (pf == 0)
        return false;
    if (::SetPixelFormat(hDc, pf, &pfd) == FALSE)
        return false;
    ::ReleaseDC(hWnd, hDc);
    data->hDC = ::GetDC(hWnd);
    if (!g_hRC)
        g_hRC = wglCreateContext(data->hDC);
    return true;
}

void CleanupDeviceWGL(HWND hWnd, WGL_WindowData* data)
{
    wglMakeCurrent(nullptr, nullptr);
    ::ReleaseDC(hWnd, data->hDC);
}

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return true;
    switch (msg)
    {
    case WM_SIZE:
        if (wParam != SIZE_MINIMIZED)
        {
            g_Width = LOWORD(lParam);
            g_Height = HIWORD(lParam);
        }
        return 0;
    case WM_SYSCOMMAND:
        if ((wParam & 0xfff0) == SC_KEYMENU)
            return 0;
        break;
    case WM_DESTROY:
        ::PostQuitMessage(0);
        return 0;
    }
    return ::DefWindowProcW(hWnd, msg, wParam, lParam);
}
