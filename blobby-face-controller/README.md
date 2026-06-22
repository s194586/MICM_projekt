# Blobby Face Controller

Lokalny kontroler do Blobby Volley Online sterowany przez dwie osoby widoczne w jednej kamerze. Projekt jest przygotowany pod VS Code i lokalne uruchomienie, bo finalna integracja wymaga dostępu do kamerki, okna gry, overlayu OpenCV oraz symulacji klawiatury.

## Założenia

- Gracz 1 to osoba po lewej stronie obrazu z kamery.
- Gracz 2 to osoba po prawej stronie obrazu z kamery.
- Gracz 1 steruje ruchem w lewo/prawo lekkim obrotem głowy.
- Gracz 2 steruje skokiem uśmiechem wykrywanym regułowo przez `smile_score`.
- Gracz 2 aktywuje bonus lekkim pochyleniem głowy w dół (krótkie skinienie).
- Bonus jest wykrywany przez model ML, a nie przez same reguły if/else.
- W normalnym trybie mniej niż dwie twarze powodują zwolnienie klawiszy; tryb solo może użyć jednej twarzy jako Player 2.

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

## Aktualny workflow

Wszystkie polecenia uruchamiaj lokalnie z katalogu `blobby-face-controller`:

1. Sprawdź kamerę i przypisanie graczy:

   ```bash
   python test_camera.py
   ```

2. Zbierz własny dataset gestu bonusowego:

   ```bash
   python collect_dataset.py --reset
   ```

   `neutral` oznacza normalną pozycję głowy, a `bonus_gesture` lekkie pochylenie głowy w dół. Zbierz minimum 50 próbek na klasę; zalecane jest 100-150.

3. Wytrenuj i zweryfikuj model:

   ```bash
   python train_bonus_model.py
   ```

   Sprawdź w `reports/validation_metrics.txt`, czy `bonus_precision` jest większe niż `0.85`.

4. Uruchom lokalny kontroler:

   ```bash
   python realtime_controller.py
   ```

5. Dopiero po przejściu tych kroków wykonaj test z Blobby Online.

Plik `data/gestures.csv` zawiera dane treningowe: liczbowe cechy wyliczone z landmarków MediaPipe i etykiety klas. Nie jest finalnym modelem. Plik `models/bonus_model.pkl` jest wytrenowanym modelem SVM ładowanym przez `realtime_controller.py`.

Colab lub Jupyter może służyć do analizy CSV i treningu modelu. Finalne sterowanie gry powinno jednak działać lokalnie, ponieważ wymaga kamery, overlayu OpenCV i symulacji klawiatury.

## Uruchomienie

Test kamery:

```bash
python test_camera.py
```

Zbieranie datasetu:

```bash
python collect_dataset.py
```

Opcja `--reset` usuwa dotychczasowe próbki i tworzy od nowa CSV z jednym poprawnym nagłówkiem:

```bash
python collect_dataset.py --reset
```

W oknie zbierania danych:

- `n` zapisuje próbkę klasy `neutral`
- `b` zapisuje próbkę klasy `bonus_gesture`
- `1` zapisuje burst 50 próbek klasy `neutral`
- `2` zapisuje burst 50 próbek klasy `bonus_gesture`
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

## Trening samemu w domu

1. Uruchom kolektor z wyczyszczeniem poprzedniego datasetu:

   ```bash
   python collect_dataset.py --reset
   ```

2. Zbieraj `neutral`: ustaw twarz normalnie, trzymaj głowę prosto i nie pochylaj jej w dół. Klawisz `n` zapisuje jedną próbkę, a `1` uruchamia burst 50 próbek.

3. Zbieraj `bonus_gesture`: pochyl głowę lekko w dół jak przy krótkim skinieniu, nie wychodź z kadru i nie przesadzaj z ruchem. Klawisz `b` zapisuje jedną próbkę, a `2` uruchamia burst 50 próbek. Podczas burstu utrzymaj daną pozycję do jego zakończenia.

4. Zbierz minimum 50 próbek `neutral` i 50 próbek `bonus_gesture`; lepiej zebrać 100-150 na klasę.

5. Wytrenuj model:

   ```bash
   python train_bonus_model.py
   ```

6. Przetestuj sterowanie:

   ```bash
   python realtime_controller.py
   ```

7. Na labach osoba, na której trenowano model, powinna siedzieć jako Player 2. Player 2 uśmiecha się do skoku i pochyla głowę w dół do bonusu. Player 1 obraca głowę w lewo lub prawo, aby sterować ruchem.

Skok nie jest trenowany: działa regułowo na podstawie `smile_score`. Bonus jest trenowany jako model ML, aby spełnić wymaganie projektu. `data/gestures.csv` zawiera cechy MediaPipe i etykiety, a nie obrazy. `models/bonus_model.pkl` jest finalnym modelem bonusu używanym przez kontroler realtime.

Po tej zmianie ergonomii trzeba zebrać dataset od nowa i ponownie wytrenować model. Kontroler odrzuca starsze modele, które nie mają oznaczenia gestu `head_down_nod`.

## Solo test mode

Do testowania Playera 2 samemu ustaw w `config.py`:

```python
SOLO_TEST_MODE = True
SOLO_TEST_ROLE = "player2"
```

Gdy kamera wykryje dokładnie jedną twarz, kontroler potraktuje ją jako Player 2. Pozwala to sprawdzić uśmiech jako skok oraz pochylenie głowy w dół jako bonus rozpoznawany przez model. Player 1 pozostaje nieaktywny, a jego klawisze ruchu są zwolnione.

Klawisz `t` przełącza tryb solo podczas działania kontrolera. Ustawienie z `config.py` określa stan początkowy po każdym uruchomieniu.

Na labach, gdy kamera wykryje dwie twarze, tryb solo jest automatycznie pomijany: lewa osoba zostaje Player 1, a prawa Player 2.

## Domowy tryb solo — solo_face_play.py

To nie jest główny tryb projektowy na laby. `solo_face_play.py` jest osobnym, eksperymentalnym skryptem do grania samemu twarzą przeciwko koledze online. Jedna twarz steruje całym naszym Blobbem, a `realtime_controller.py` zachowuje standardowy podział między dwie osoby.

Uruchomienie:

```bash
python solo_face_play.py
```

Sterowanie:

- głowa w lewo/prawo = ruch,
- uśmiech = skok,
- głowa w dół = bonus rozpoznawany przez model SVM.

Przed grą:

1. Otwórz Blobby Online w przeglądarce.
2. Kliknij w okno gry, żeby przeglądarka miała focus.
3. Uruchom lub pozostaw uruchomiony `solo_face_play.py`.
4. Jeśli okno OpenCV przejęło focus albo gra nie reaguje, kliknij ponownie w okno gry.

Kolega gra na swoim komputerze normalnie klawiaturą jako drugi gracz online. Do testowania wysyłanych klawiszy można użyć Notatnika: obrót głowy powinien wpisywać `A`/`D`, uśmiech `W`, a gest bonusu spację — zgodnie z domyślnymi wartościami w `config.py`.

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

Sterowanie gestami:

- Player 1: obrót głowy w lewo/prawo (`head_yaw`) = ruch w lewo/prawo,
- Player 2: uśmiech (`smile_score`) = skok,
- Player 2: pochylenie głowy w dół = bonus klasyfikowany przez model SVM.

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
- `JUMP_MODE` (domyślnie `"smile"`)
- `SMILE_THRESHOLD`
- `JUMP_CONFIRM_FRAMES`
- `CAMERA_INDEX`
- `MOVE_LEFT_KEY`, `MOVE_RIGHT_KEY`, `JUMP_KEY`, `BONUS_KEY`
- `BONUS_COOLDOWN_SECONDS`
- ustawienie dwóch graczy w kamerze
- poprawne przypisanie lewej osoby jako Gracz 1 i prawej osoby jako Gracz 2

Najważniejsze jest dobranie progów do konkretnej kamerki, oświetlenia i odległości od laptopa. Jeśli neutralna głowa odpala ruch, zwiększ martwą strefę przez oddalenie progów od zera, np. `-0.07` i `0.07`.

Jeśli neutralna twarz uruchamia skok, zwiększ `SMILE_THRESHOLD`. Jeśli wyraźny uśmiech nie uruchamia skoku, zmniejsz go nieznacznie i obserwuj `smile_score` w overlayu.

## Stabilność

Kontroler używa smoothingu i debounce:

- akcja ruchu musi utrzymać się przez kilka klatek,
- skok jest krótkim tapnięciem z cooldownem,
- bonus wymaga kilku kolejnych klatek klasyfikacji ML i ma cooldown,
- w normalnym trybie przy braku dwóch twarzy program puszcza wszystkie klawisze,
- w trybie solo jedna twarz steruje wyłącznie akcjami Playera 2,
- przy wyjściu `q` program zawsze puszcza klawisze.
