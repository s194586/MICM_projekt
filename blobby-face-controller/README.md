# Blobby Face Controller

Lokalny kontroler do Blobby Volley Online sterowany przez dwie osoby widoczne w jednej kamerze. Projekt jest przygotowany pod VS Code i lokalne uruchomienie, bo finalna integracja wymaga dostępu do kamerki, okna gry, overlayu OpenCV oraz symulacji klawiatury.

## Założenia

- Gracz 1 to osoba po lewej stronie obrazu z kamery.
- Gracz 2 to osoba po prawej stronie obrazu z kamery.
- Gracz 1 steruje ruchem w lewo/prawo lekkim obrotem głowy.
- Gracz 2 steruje skokiem szybkim otwarciem ust.
- Gracz 2 aktywuje bonus gestem: szeroki uśmiech + lekkie uniesienie brwi.
- Bonus jest wykrywany przez model ML, a nie przez same reguły if/else.
- Gdy system widzi mniej niż dwie twarze w trybie gry, puszcza klawisze i czeka.

## Dlaczego tak

MediaPipe został wybrany, bo działa szybko w czasie rzeczywistym i dostarcza landmarki twarzy bez trenowania ciężkiego modelu. SVM został wybrany, bo dataset jest mały, a klasyfikator tego typu dobrze działa na liczbowych cechach geometrycznych. Podstawowe akcje są regułowe, bo ruch i skok muszą być szybkie, przewidywalne i łatwe do kalibracji. Bonus jest ML/AI, bo tego wymaga projekt.

Projekt jest lokalny na start, ponieważ sterowanie grą wymaga lokalnej kamery, overlayu i symulacji klawiszy. Część treningowa jest oddzielona od kontrolera realtime, więc później można przenieść zbieranie/analizę danych albo trening do Google Colaba, jeśli prowadzący tego zażąda.

## Struktura

```text
MICM_projekt/
├── blobby-face-controller/
│   ├── README.md
│   ├── requirements.txt
│   ├── config.py
│   ├── feature_extraction.py
│   ├── test_camera.py
│   ├── collect_dataset.py
│   ├── train_bonus_model.py
│   ├── realtime_controller.py
│   ├── data/
│   │   └── gestures.csv
│   ├── models/
│   │   └── bonus_model.pkl          # generowany po treningu
│   └── reports/
│       ├── confusion_matrix.png     # generowany po treningu
│       └── validation_metrics.txt   # generowany po treningu
└── venv_projektmicm/                # tylko środowisko Pythona
```

`blobby-face-controller` zawiera kod projektu. `venv_projektmicm` jest tylko środowiskiem Pythona z zainstalowanymi bibliotekami. Plików projektu nie należy trzymać w folderze `venv_projektmicm`.

## Instalacja

Linux/macOS:

```bash
cd ~/projects/MICM_projekt/blobby-face-controller
source ../venv_projektmicm/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows, jeśli środowisko jest w katalogu `MICM_projekt\venv_projektmicm`:

```bash
cd %USERPROFILE%\projects\MICM_projekt\blobby-face-controller
..\venv_projektmicm\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Przed każdym uruchomieniem skryptów wejdź do katalogu `blobby-face-controller` i aktywuj środowisko komendą `source ../venv_projektmicm/bin/activate`.

## Troubleshooting

Jeśli po uruchomieniu `python test_camera.py` pojawia się błąd `AttributeError: module 'mediapipe' has no attribute 'solutions'`, to zwykle oznacza, że w środowisku jest zainstalowana wersja MediaPipe niezgodna z legacy API używanym przez ten projekt. Kod korzysta z `mp.solutions.face_mesh`, więc wymagamy wersji kompatybilnej z tym API: `mediapipe==0.10.21`.

Naprawa środowiska:

```bash
pip uninstall mediapipe -y
pip install mediapipe==0.10.21
```

Weryfikacja po instalacji:

```bash
python -c "import mediapipe as mp; print(mp.__version__); print(hasattr(mp, 'solutions'))"
```

Oczekiwany wynik to `0.10.21` oraz `True`.

## Uruchomienie

Test kamery:

```bash
python test_camera.py
```

Zbieranie datasetu:

```bash
python collect_dataset.py
```

W oknie zbierania danych:

- `n` zapisuje próbkę klasy `neutral`
- `b` zapisuje próbkę klasy `bonus_gesture`
- `q` kończy program

Zbierz minimum 50 próbek na klasę. Najlepiej zebrać więcej, np. 80-150, z kilkoma wariantami pozycji głowy i oświetlenia.

Trening modelu:

```bash
python train_bonus_model.py
```

Skrypt zapisze:

- `models/bonus_model.pkl`
- `reports/confusion_matrix.png`
- `reports/validation_metrics.txt`

Jeśli precision dla klasy bonusu jest poniżej `0.85`, zbierz lepszy dataset.

Uruchomienie kontrolera:

```bash
python realtime_controller.py
```

## Jak odpalić grę

1. Wejdź na https://www.blobby-online.com/de
2. Kliknij w okno gry, żeby przeglądarka przyjmowała klawisze.
3. Uruchom `python realtime_controller.py`.
4. Ustaw dwie osoby obok siebie w kamerze.
5. Osoba po lewej stronie obrazu steruje ruchem.
6. Osoba po prawej stronie obrazu steruje skokiem i bonusem.
7. Graj.

## Sterowanie

Domyślne klawisze w `config.py`:

- ruch w lewo: `a`
- ruch w prawo: `d`
- skok: `w`
- bonus: `space`

Domyślny tryb ruchu to:

```python
MOVE_CONTROL_MODE = "head_yaw"
```

Tryb `gaze` nie jest domyślny, bo oczy muszą śledzić piłkę i ekran.

## Cechy

Dataset nie zapisuje obrazów. Zapisywane są tylko cechy liczbowe z landmarków MediaPipe:

- `mouth_open_ratio`
- `mouth_width_ratio`
- `left_eye_open_ratio`
- `right_eye_open_ratio`
- `eyebrow_raise_left`
- `eyebrow_raise_right`
- `head_yaw`
- `head_pitch`
- `head_roll`
- `face_width`
- `face_height`

Cechy są normalizowane względem rozmiaru twarzy, żeby działały stabilniej przy różnych odległościach od kamery.

## Kalibracja na labach

Przed turniejem sprawdź i dostosuj w `config.py`:

- `HEAD_YAW_LEFT_THRESHOLD`
- `HEAD_YAW_RIGHT_THRESHOLD`
- `MOUTH_OPEN_THRESHOLD`
- `CAMERA_INDEX`
- `MOVE_LEFT_KEY`, `MOVE_RIGHT_KEY`, `JUMP_KEY`, `BONUS_KEY`
- `BONUS_COOLDOWN_SECONDS`
- ustawienie dwóch graczy w kamerze
- poprawne przypisanie lewej osoby jako Gracz 1 i prawej osoby jako Gracz 2

Najważniejsze jest dobranie progów do konkretnej kamerki, oświetlenia i odległości od laptopa. Jeśli neutralna głowa odpala ruch, zwiększ martwą strefę przez oddalenie progów od zera, np. `-0.07` i `0.07`.

## Stabilność

Kontroler używa smoothingu i debounce:

- akcja ruchu musi utrzymać się przez kilka klatek,
- skok jest krótkim tapnięciem z cooldownem,
- bonus wymaga kilku kolejnych klatek klasyfikacji ML i ma cooldown,
- przy braku dwóch twarzy program puszcza wszystkie klawisze,
- przy wyjściu `q` program zawsze puszcza klawisze.
