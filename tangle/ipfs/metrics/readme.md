# Metriken

## How to Run

To runs stability benchmark with `./metrics/measure_stability_metrics.sh BENCHMARK_TIME_IN_S NUMBER_OF_PEERS LOG_FILE_NAME`
If you want to analyze the log another time run `./metrics/analyze_metrics.py LOG_FILE_PATH`

## Metriktypen
### Netzwerkmetriken
- Anzahl der versendeten Pakete (pro Node, Gesamt)
- Durchsatz (Last auf Netzwerk)
- Last auf einem Knoten
### Tanglemetriken
- “Tanglekonvergenz”
- “Vollständigkeit des Bilds eines lokalen Nodes auf den Tangle”
- Speicherverbrauch
    - pro Node
    - gesamtes Netzwerk

- [keine Metrik] Visualisierung des Tangles auf einzelnen Nodes
### Anwendungsmetrike
Fragen die wir damit beantworten wollen:
- Wie gut läuft das Programm

Metriken:
- Können sich alle Nodes überhaupt an IPFS connecten
- Wie oft wird pro Node trainiert
- CPU-Auslastung/Speicherauslastung
- Wieviele Nodes auf einer Maschine?
- Wie viele Nodes verkraftet das P2P-Netzwerk bei sinnvoller Anwendungsperformance?
- Modelperformance
    - Benchmark zu reinem lokalen Training?        
    - Modelperformance auf allen Knoten kontinuirlich verbessert
- Dauer von Start der Tipp-Selection/Ende der Learning-Runde zu Tipp-Publishing

## Parameter

- Bandbreite
- Paketverlust
- Latenz


## Local Run Benchmark
Can be found in `metrics/baseline.py`.


Training 10 Epochs:
- Accuracy: 0.4020926756352765
- Loss: 2.6368494
Training 20 Epochs:
- Accuracy: 0.6208271051320379
- Loss: 1.3855447
Training 30 Epochs:
- Accuracy: 0.7249626307922272
- Loss: 0.9657703