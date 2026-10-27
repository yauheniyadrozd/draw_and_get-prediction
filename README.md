# Estymator · Narysuj, a model zgadnie

Stoisko w stylu Quick Draw. Ktoś rysuje, mała sieć neuronowa zgaduje na żywo.
W prawym górnym rogu jest przełącznik motywu (ciemny / jasny). Zrobione na dni aktywności na PWR.

## Uruchomienie (5–10 min, raz, z internetem)

1. `pip install torch numpy`
2. `python train.py` — pobierze ~100 MB danych Quick Draw (po 6000 szkiców × 20 klas), wytrenuje sieć i zapisze `model.js`.
   - Szybki test: `python train.py --fast` (~2 min, trochę gorsza dokładność).
3. Otwórz `index.html` w Chrome (działa offline, bez serwera) albo uruchom lokalny serwer:
   ```
   python -m http.server 8000
   ```
   i otwórz `http://localhost:8000`.

## Co tu jest
- Sieć neuronowa zgaduje na żywo, w trakcie rysowania (inferencja w czystym JS, bez bibliotek AI).
- Pod odpowiedzią modelu pojawia się zabawny komentarz do rysunku (osobne żarty, gdy model jest niepewny).
- Przycisk „🔮 Przepowiednia na jutro”: przepowiednia zależy od tego, co model rozpoznał na rysunku.
  Teksty edytujesz w `index.html`, w obiektach `JOKES`, `LOW_JOKES` i `FORTUNES`.
- Przełącznik motywu 🌙/☀️ — wybór zapisuje się w przeglądarce (localStorage).

## Na stoisku
- Pełny ekran: F11. Najlepiej ekran dotykowy albo mysz, ja brałam graficny tablet.
- Skrót: `C` wyczyść.

## Pliki
- `index.html` — cały interfejs (rysowanie, rozpoznawanie, motywy) i inferencja modelu w czystym JS
- `model.js` — wygenerowane wagi modelu (powstaje po `python train.py`)
- `train.py` — trening w PyTorch i eksport wag do `model.js`
- `estymator-logo.png` — logo w ciemnym motywie (przezroczyste)
- `IMG_7610.JPG` — logo w jasnym motywie (na białym tle)

## Co powiedzieć o modelu
CNN: 3 warstwy konwolucyjne (16, 32, 64 filtry) + 2 warstwy gęste, wejście 28×28, ~100 tys. parametrów, 20 klas.
Wyjście to softmax, czyli rozkład prawdopodobieństwa. Estymator to estymator.
