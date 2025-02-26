# C-Viewer

## Analisi e Visualizzazione della Stabilità dei Sistemi Dinamici

*C-Viewer* è un'applicazione in C++ progettata per visualizzare e analizzare in tempo reale la stabilità di diversi sistemi dinamici. Il software permette di esplorare il comportamento di sistemi come:

- **Pendolo semplice**
- **Lotka-Volterra** (Preda-Predatore)
- **Lotka-Volterra** (Competizione Interspecifica)
- Modello di **Hodgkin-Huxley** ridotto

Attraverso un'interfaccia grafica interattiva, *C-Viewer* calcola e visualizza il campo scalare generato dall'operatore di stabilità $\mathcal{C}(t)$, che quantifica la sensibilità del sistema a piccole perturbazioni.

## Caratteristiche Principali

- **Visualizzazione in tempo reale:** Utilizza la potenza di calcolo delle GPU per elaborare e rappresentare dinamicamente il campo scalare.
- **Interfaccia grafica intuitiva:** Consente di modificare i parametri e navigare agevolmente nel campo scalare, osservando immediatamente l'impatto delle variazioni.
- **Supporto multi-modello:** Include diversi sistemi dinamici predefiniti e permette la definizione di modelli personalizzati tramite un'apposita interfaccia.

## Requisiti di Sistema

- **Sistema operativo:** Windows
- **Scheda grafica:** NVIDIA (con driver aggiornati e CUDA Toolkit versione 12.6 o superiore)
- **Microsoft Visual C++ Redistributable**

## Installazione

1. Scarica l'installer `C-Viewer_installer.exe` dalla sezione "Releases" del repository GitHub.
2. Esegui l'installer e segui le istruzioni a schermo.

_In alternativa, puoi compilare il codice sorgente manualmente._

## Utilizzo

1. **Avvio dell'Applicazione:**
   - Esegui `C-Viewer.exe` per lanciare il programma.

2. **Selezione del Sistema Dinamico:**
   - Dal menu "Modelli" scegli il sistema che desideri analizzare (ad esempio, Pendolo semplice, Lotka-Volterra, Hodgkin-Huxley).

3. **Navigazione del Campo Scalare:**
   - **Panoramica (Panning):** Clicca e trascina con il tasto sinistro del mouse per spostare la visualizzazione.
   - **Zoom:** Utilizza la rotellina del mouse per ingrandire o ridurre l'immagine.
   - **Ispezione:** Premi il tasto destro del mouse per visualizzare il valore del campo scalare nel punto selezionato.

4. **Regolazione del Tempo di Valutazione:**
   - Modifica il tempo di simulazione tramite lo slider `T` per controllare l'evoluzione dinamica del sistema.

5. **Personalizzazione dei Parametri del Modello:**
   - Nel pannello "Parametri del Modello" puoi regolare in tempo reale i parametri specifici del sistema scelto. Ad esempio:
     - **Pendolo semplice:** lunghezza, coefficiente di attrito, accelerazione di gravità.
     - **Lotka-Volterra:** tassi di crescita, coefficienti di predazione o competizione.
     - **Hodgkin-Huxley:** corrente esterna e parametri relativi alla biforcazione.
   - Le modifiche si riflettono immediatamente sul campo scalare, permettendoti di osservare l'effetto delle variazioni.

6. **Funzionalità Avanzate:**
   - **Reset Visuale:** Ripristina la visualizzazione predefinita con un singolo click.
   - **Preset e Configurazioni:** Seleziona configurazioni preimpostate per testare rapidamente scenari comuni.
   - **Dominio e Maschere:** Applica restrizioni o maschere per focalizzare l'analisi su specifiche aree del campo.

## Struttura della Repository

- `equations`: Immagini delle equazioni dei vari sistemi dinamici.
- `fonts`: Font utilizzati nell'interfaccia grafica.
- `Icons`: Icone dell'applicazione.
- `src`: Codice sorgente dell'applicazione:
  - `main.cpp`: File principale contenente il ciclo di rendering, la gestione degli eventi e l'interfaccia utente (ImGui).
  - `kernel.cu`: Funzioni kernel CUDA per l'integrazione dei sistemi dinamici e il calcolo dell'operatore di stabilità.

## Licenza

Copyright (c) 2025 Riva Daniele

Questo software è distribuito sotto licenza MIT. Vedi il file LICENSE per ulteriori dettagli.

## Crediti

- **Dear ImGui:** Per l'interfaccia grafica – [https://github.com/ocornut/imgui](https://github.com/ocornut/imgui)
- **NVIDIA CUDA:** Per il calcolo parallelo su GPU – [https://developer.nvidia.com/cuda-zone](https://developer.nvidia.com/cuda-zone)
- **stb_image:** Per la gestione delle immagini – [https://github.com/nothings/stb](https://github.com/nothings/stb)
