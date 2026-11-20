# Ściąga – punkt kontrolny (racetrack + nasz SAC)

## 0. Przed prezentacją (1 min)

Otwórz PowerShell w folderze projektu i aktywuj środowisko:

```powershell
cd C:\Users\pietr\Documents\studia\uczenie_ze_wzmocnieniem\projekt
.venv\Scripts\activate
```

Dalej wszystkie komendy to zwykłe `python ...`. Nic nie trzeba trenować, modele są już gotowe.

---

## 1. Pokaż, jak agent jeździ

Najlepszy model to **C** (`runs\C_no_curriculum_a0.2`). Okno zamyka się krzyżykiem.

| Co pokazać | Komenda |
|---|---|
| pusty tor | `python watch_agent.py runs\C_no_curriculum_a0.2 --stage 0` |
| 1 auto na torze | `python watch_agent.py runs\C_no_curriculum_a0.2 --stage 1` |
| lekki ruch (1–5 aut) | `python watch_agent.py runs\C_no_curriculum_a0.2 --stage 2` |
| duży ruch (10–20 aut), tu jeszcze **nie** umie | `python watch_agent.py runs\C_no_curriculum_a0.2 --stage 3` |

- Na pasku okna: etap, numer próby, krok, nagroda i prędkość.
- W terminalu po każdym epizodzie: nagroda, liczba kroków i powód końca (`time limit` = przejechał całe 60 s, `crash/off road` = kolizja).
- Inny model: podmień folder, np. `runs\D_curriculum_a0.2_seed1`.

**Dla porównania agent losowy:** otwórz `media\random_agent.gif`. Rozbija się średnio po ~2 s.

---

## 2. Gdzie co jest (wszystko już wygenerowane)

| Czego chce prowadząca | Gdzie to jest |
|---|---|
| **własna implementacja SAC** | `softactorcritic\sac.py` (algorytm), `softactorcritic\models.py` (sieci actor/critic) |
| **przebieg treningu** | `runs\<run>\episodes.csv` (każdy epizod), `runs\<run>\eval.csv` (ocena co 5k kroków), `runs\logs\*.log` |
| **krzywe uczenia** | `figures\learning_A_curriculum_a0.2.png` (i B, C, D): nagroda, długość epizodu, % kolizji, z etapami curriculum |
| **porównanie konfiguracji (trening)** | `figures\compare_train_return.png`, `figures\compare_eval_return.png`, `figures\compare_eval_success.png`, `figures\compare_eval_speed.png` |
| **porównanie ustawień środowiska** | `figures\eval_success.png`, `figures\eval_return.png`, `figures\eval_distance.png` |
| **tabela wyników** | `results\eval_summary.md` (wszystkie liczby) |
| **metodologia ewaluacji** | opis niżej (pkt 4) i na górze `evaluate.py` |

---

## 3. Co trenowaliśmy (4 konfiguracje, każda po 80k kroków)

| Run | Opis |
|---|---|
| **A** | curriculum: pusty tor → 1 auto → lekki ruch, α = 0.2 |
| **B** | jak A, ale α = 0.05 (mniej eksploracji) |
| **C** | bez curriculum, od razu lekki ruch, α = 0.2 |
| **D** | jak A, inny seed (sprawdzenie powtarzalności) |

Curriculum przechodzi do następnego etapu, gdy średnia długość ostatnich 30 epizodów przekroczy 80% maksimum (240 z 300 kroków). 20% epizodów gra się na poprzednim etapie, żeby agent nie zapominał.

---

## 4. Metodologia ewaluacji (do powiedzenia)

- Każdy agent jedzie **30 epizodów** na każdym z 4 ustawień ruchu: pusty tor, 1 auto, lekki ruch, duży ruch.
- **Deterministyczna polityka**: bierzemy średnią akcję, bez losowania.
- **Te same seedy dla wszystkich agentów**: każdy dostaje identyczne pozycje startowe i ruch.
- Punkt odniesienia: **agent losowy**.
- Epizod trwa maks. 60 s (300 kroków).
- **Metryki:**
  - % epizodów bez kolizji (przejechał całe 60 s),
  - % kolizji / wyjazdów z drogi,
  - średnia nagroda,
  - długość epizodu,
  - średnia prędkość wzdłuż toru [m/s],
  - przejechany dystans [m].

---

## 5. Wyniki (% epizodów bez kolizji)

| | pusty tor | 1 auto | lekki ruch | duży ruch |
|---|---|---|---|---|
| losowy | 0% | 0% | 0% | 0% |
| A curriculum | 97% | 83% | 40% | 0% |
| B α=0.05 | 83% | 87% | 37% | 0% |
| **C bez curriculum** | **100%** | 87% | **63%** | 0% |
| D curriculum, seed 1 | **100%** | **93%** | 53% | 0% |

Do powiedzenia:
- **Polityka jest wyuczona, nie losowa.** Losowy agent rozbija się po ~2 s i przejeżdża ~20 m. Nasz jeździ ~19.6 m/s (max 20) i przejeżdża ~1170 m w 60 s na pustym torze.
- **Pusty tor jest opanowany po ~9k kroków** (widać na krzywych uczenia).
- **Lekki ruch: ok. połowa epizodów bez kolizji.** Agent jedzie prawie maksymalnie szybko i wjeżdża w wolniejsze auta (one jadą 6–9 m/s).
- **Duży ruch: 0%.** Na nim jeszcze nie trenowaliśmy.
- **Curriculum (A, D) nie dało przewagi nad treningiem od razu na lekkim ruchu (C).** Przy jednym seedzie na konfigurację to wstępna obserwacja.
- **α = 0.05 vs 0.2:** podobne wyniki, 0.2 trochę lepsze.
- **A i D różnią się tylko seedem, a wyniki różnią się o kilkanaście punktów procentowych.** Trzeba będzie uśredniać po kilku seedach.

---

## 6. Środowisko i nagroda (gdyby pytała)

- **Środowisko:** `racetrack-v1` z highway-env.
- **Akcje ciągłe:** gaz/hamulec i skręt.
- **Obserwacja:** siatka zajętości 12×12 wokół auta (gdzie są auta i droga) plus prędkość auta i poprzedni skręt.
- **Nagroda (własna, bo domyślna płaciła za stanie w miejscu):**
  - \+ za prędkość wzdłuż toru,
  - \+ za jazdę środkiem pasa,
  - − za skręcanie i szarpanie kierownicą,
  - **−5** i koniec epizodu za kolizję, wyjazd z drogi albo stanie > 5 s.
- **SAC:**
  - dwa krytyki Q (bierzemy minimum) i target critic z miękką aktualizacją (τ = 0.005),
  - polityka gaussowska z tanh,
  - replay buffer, batch 256, lr 3e-4, γ = 0.99.

---

## 7. Następne kroki (pytanie „co dalej?”)

1. **Dodać do obserwacji prędkości innych aut (`vx`, `vy`).** Teraz agent widzi tylko, gdzie stoją, a nie jak szybko jadą.
2. **Dłuższy trening na lekkim ruchu, potem duży ruch.**
3. **Automatyczne strojenie α** (teraz jest stałe).
4. **Kilka seedów na konfigurację**, żeby porównania były wiarygodne.

---

## Awaryjnie: odtworzenie wszystkiego od zera

```powershell
python train_sac.py --name C_no_curriculum_a0.2 --start-stage 2   # ok. 1 h
python plot_training.py                                           # wykresy -> figures\
python evaluate.py                                                # tabela + słupki -> results\, figures\
```
