# C-Viewer

## Analisi e Visualizzazione della Stabilità dei Sistemi Dinamici

C-Viewer è un'applicazione in C++ che permette di visualizzare e analizzare in tempo reale la stabilità di diversi sistemi dinamici, tra cui:

*   Modello di Hodgkin-Huxley ridotto
*   Pendolo semplice
*   Lotka-Volterra (Preda-Predatore)
*   Lotka-Volterra (Competizione Interspecifica)

L'applicazione consente di esplorare il campo scalare generato dall'operatore di stabilità C(t), che rappresenta la sensibilità del sistema a piccole perturbazioni. È possibile navigare nel campo scalare, modificare i parametri del sistema in tempo reale e osservare come varia la stabilità del sistema.

## Caratteristiche Principali

*   **Visualizzazione in tempo reale:** C-Viewer sfrutta la potenza di calcolo delle GPU per elaborare e visualizzare il campo scalare in tempo reale, permettendo un'analisi dinamica e interattiva.
*   **Interfaccia grafica intuitiva:** L'applicazione offre un'interfaccia grafica semplice e intuitiva che consente di modificare i parametri del sistema, navigare nel campo scalare e visualizzare diverse informazioni, come la traiettoria dei punti e il campo vettoriale.
*   **Supporto per diversi sistemi dinamici:** C-Viewer include il supporto per diversi sistemi dinamici predefiniti, tra cui il modello di Hodgkin-Huxley, il pendolo semplice e i modelli di Lotka-Volterra. Inoltre, è possibile definire sistemi dinamici generici tramite un'apposita interfaccia.

## Requisiti di Sistema

*   Sistema operativo Windows
*   Scheda grafica NVIDIA con driver aggiornati (CUDA Toolkit installato, versione minima supportata 12.6)
*   Microsoft Visual C++ Redistributable

## Installazione

1. Scaricare l'installer `C-Viewer_installer.exe` dalla sezione "Releases" del repository GitHub.
2. Eseguire l'installer e seguire le istruzioni a schermo.

In alternativa, è possibile compilare il codice sorgente manualmente.

## Utilizzo

1. Avviare l'applicazione `C-Viewer.exe`.
2. Selezionare il sistema dinamico desiderato nella sezione "Menu Modelli".
3. Esplorare il campo scalare utilizzando i seguenti comandi:
    *   **Trascinare con il tasto sinistro del mouse:** Spostare il campo scalare.
    *   **Rotellina del mouse:** Zoom avanti/indietro.
    *   **Tasto destro del mouse:** Visualizza il valore del campo scalare a schermo in quel punto.
4. Modificare i parametri del sistema utilizzando i controlli presenti nel pannello "Parametri del Modello".

## Struttura della Repository

*   `equations`: Contiene le immagini delle equazioni dei vari sistemi dinamici.
*   `fonts`: Contiene i font utilizzati nell'interfaccia grafica.
*   `Icons`: Contiene le icone utilizzate dall'applicazione.
*   `src`: Contiene il codice sorgente dell'applicazione.
    *   `main.cpp`: File principale dell'applicazione. Contiene il ciclo principale, la gestione degli eventi, la logica di rendering e l'interfaccia utente (ImGui).
    *   `kernel.cu`: Contiene le funzioni kernel CUDA per l'integrazione del sistema dinamico e il calcolo dell'operatore di stabilità.
*   `documentazione`: Contiene la relazione scritta "Analisi e visualizzazione della stabilità dei sistemi dinamici".

## Licenza

Copyright (c) 2025 Riva Daniele

Questo software è rilasciato sotto licenza MIT.
Vedi il file LICENSE per maggiori dettagli.

## Crediti

*   **Dear ImGui:** Per l'interfaccia grafica (https://github.com/ocornut/imgui).
*   **NVIDIA CUDA:** Per il calcolo parallelo su GPU (https://developer.nvidia.com/cuda-zone).
*   **stb_image:** Visualizzazione grafica delle immagini (https://github.com/nothings/stb).
