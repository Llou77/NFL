# NFL Eredmény-előrejelző

**🏈 Élő oldal:** [llou77.github.io/NFL](https://llou77.github.io/NFL) · **English version:** [README.md](README.md)

Gépi tanulási modell, amely minden NFL-meccs végeredményét megjósolja a kezdőrúgás előtt. A két becsült pontszámból levezeti a győztest, a különbséget és az összpontszámot, összeveti őket a Vegas-i vonallal, és ha a modell és a vonal legalább 3 ponttal eltér, ezt jelzi (lean). Minden GitHub Actions-ön fut, és a GitHub Pages oldalra publikál.

A kifejezések a [Fogalomtárban](#fogalomtár) vannak definiálva.

---

## Mennyire pontos?

**A modell nem verte meg a Vegas-i vonalat.** Statisztikai projektként kezeld, ne fogadási eszközként.

| | Különbség (spread) hibája | Összpontszám hibája | Spread-mérleg, minden meccs | Edge-meccsek (3+ pont) |
|---|---|---|---|---|
| **Élő 2026, 1–10. hét** (v1.1 modell, az audit előtt) | 11,11 pont — Vegas 9,62 | 11,22 — Vegas 10,51 | 67–80 (45,6%) | 43–53 (44,8%), ROI −14,5% |
| **Kitartott 2025, a régi modell ahogy élesben futott** (v1.1) | 10,53 — Vegas 9,67 | 11,05 — Vegas 10,42 | 50,5% | 48,3% (176 meccs) |
| **Kitartott 2025, v2.0 modell** (tanítás 2022–24) | **9,74** — Vegas 9,67 | **10,46** — Vegas 10,42 | 51,9% | 50,0% (12 meccs) |
| **2026, 7–11. hét, v2.0 modell** (tanítás okt. 15. előtti meccseken) | **9,28** — Vegas 9,10 | **10,15** — Vegas 10,20 | — | 2 meccs |

A több szezonos számok (2013 óta minden szezon, mindegyik csak az előtte lévő négy szezonon tanítva) az oldal [walk-forward lapján](https://llou77.github.io/NFL/walkforward.html) vannak.

A **2026. novemberi audit** ([WORKLOG.md](WORKLOG.md)) megtalálta, miért volt rosszabb az élő teljesítmény annál, amit a backtestek ígértek, és kijavította:

- **A jóslatokat ×1,8-del széthúztuk**, hogy szórásuk a tényleges eredményekéhez igazodjon. A jó jóslat szükségszerűen szűkebb, mint a kimenetel (kimenetel = jel + szerencse); a széthúzás önmagában ~0,4 pont különbség-hibát okozott, és tömegesen gyártott hamis „edge”-eket.
- **34 feature minden történelmi meccsnél létezett, a közelgő meccseknél viszont mindig hiányzott:** a kezdő QB és a Next Gen Stats gördülő formája (a meccs hetére kapcsolva, amihez lejátszás előtt nincs adat), a meccsnapi időjárás és a bírói csapat (a meccs után publikálják). A backtest látta őket, az éles jóslat soha.
- **Az egymás elleni (H2H) feature-ök minden meccsen konstansok voltak** (a szükséges csapatoszlopok korábban kiestek a pipeline-ból), és több feature halott vagy hibás volt: preseason win total (a forrás 2020-ban véget ért), „nyitóvonal” és piaci adatok (a forrás 2021-ben véget ért — a jelenlegi vonal másolatai voltak), draft-osztály minősége (mindig 0), QB-váltás flag (csak 59%-ban helyes), dome-flag (8 csapatnál rossz).
- **A Bayes-optimalizáló zajt hangolt:** 26 nested walk-forward foldon a 8 hangolt súly egyike sem volt megkülönböztethető a keresési tartományon vett egyenletes véletlentől. Törölve; a súlyok fix konstansok.

A v2.0 modell csak olyan feature-öket használ, amelyek a múltbeli meccsekre és a következő meccsre is léteznek, nincs variancia-széthúzás, és a végső blender látja a Vegas által implikált pontszámot, amint a vonal publikus. Out-of-fold illesztve ez a blender ≈1,0–1,1 súlyt ad a piacnak — vagyis **a modell lényegében a piacot követi kis korrekciókkal**, ami egy kimutatott előny nélküli modelltől a becsületes viselkedés; 3+ pontos „edge”-et a meccsek ~4%-ánál jelez ~65% helyett. Élő mérlege a 2026-os 12. héttől indul, a v1.1-től külön követve (`performance.json → by_model_version`). Hogy van-e valódi előnye, azt csak egy teljes szezonnyi élő, időbélyegzett predikció dönti el — az időbélyeg a `predictions_latest.json` git-története.

> **A fogadásról:** a leanek nem fogadási tanácsok. Standard −110-es szorzónál 52,4% kell a nullszaldóhoz; egy szezon találati aránya több százalékpontnyi zajt hordoz.

---

## Hogyan zajlik egy futás

`model/pipeline.py --mode full` (minden kedden), sorrendben:

1. **Adatbetöltés** (`data_loader.py`). A lezárt szezonok a commitolt **befagyasztott rétegből** (`data/frozen/`) jönnek, hálózat nélkül. Csak a menetrend, a folyó szezon és a Next Gen Stats fájlok töltődnek le.
2. **Play-by-play aggregálás** (`feature_engineering._aggregate_pbp`): csapatonként és meccsenként egy sor (hatékonyság, sikerarány, harmadik kísérletek, red zone, labdavesztések, tempó, …). A lezárt szezonok előre aggregálva: `data/frozen/pbp_agg_*.parquet`.
3. **Feature-építés** (`feature_engineering.build_all_features`): gördülő forma a csapat utolsó 4/8/16 lejátszott meccséből, kezdő-QB forma, Next Gen Stats forma, Elo (1999 óta minden meccsből), QB-/edzőváltás, keretfolytonosság, pihenőnapok, naptár, hazai pálya előny, a Vegas-i spread és total; meccsenként egy sorrá fordítva. Egy **allowlist** dönti el, mely ~330 oszlopot láthatja a modell.
4. **Egymás elleni (H2H) feature-ök** (`feature_h2h.py`) a korábbi találkozók végeredményeiből.
5. **Élő mérleg frissítése** (`evaluate.update_performance_from_latest`): a legutóbb publikált jóslatok összevetése az azóta lejátszott meccsekkel, a `performance.json`-ba — mielőtt bármi felülíródna.
6. **Tanítás** (`train.py`) — lásd [Modell-architektúra](#modell-architektúra).
7. **Jóslás** (`predict.py`) minden hátralévő meccsre az `ensemble_predict()` függvénnyel, amit a backtest is használ; megbízhatósági besorolás és leanek.
8. **Publikálás**: a JSON-fájlok a `docs/assets/`-be másolódnak, ezt szolgálja ki a GitHub Pages.

A szerda esti futás (`--mode predict_only`) a keddi modellel, friss vonalakkal jósol újra a csütörtöki meccs előtt. A `--mode backtest --backtest-season S` az S előtti 4 szezonon tanít és S-t pontozza; a `model/walkforward.py` ugyanezt teszi 2013 óta minden szezonra, kizárólag a befagyasztott adatból.

### Feature-szabályok

Egy oszlop csak akkor lehet feature, ha (1) kezdőrúgás előtt ismert, és (2) **a heti futáskor a közelgő meccsekre is elérhető**. A gördülő statisztikák a csapat teljes meccs-idővonalán készülnek: egy meccs értéke a csapat utolsó N *lejátszott* meccsének átlaga, így a jövő heti meccs pontosan azt az ablakot kapja, amit egy történelmi meccs a tanításkor. A meccsnapi időjárás, a bírói beosztás és a végleges sérülés-státuszok a 2. szabályon elbuknak (a keddi/szerdai futás után jelennek meg), ezért nincsenek használva; a jövő heti kezdő irányító viszont *benne van* a menetrendben, és használjuk.

---

## Modell-architektúra

```
1. réteg:  Ridge  +  XGBoost  +  LightGBM          hazai és vendég pontszámra külön-külön
2. réteg:  PyTorch háló két kimenettel            (hazai, vendég pontszám)
3. réteg:  Ridge meta-learner a négy jóslaton (+ a Vegas által implikált pontszámon,
           ha a vonal publikus), valódi out-of-fold adaton illesztve
```

- A fák nyers feature-öket kapnak hiányzó értékekkel; a Ridge és a háló medián-imputált, standardizált adatot.
- A meta-learner out-of-fold jóslatokon tanul: minden 1–2. rétegbeli modell minden időrendi foldon újratanul. Két blender készül: az egyik a Vegas által implikált hazai/vendég pontszámot ((total ± spread) / 2) is látja — ezt használjuk, amint a vonal publikus —, a másik nem, a távolabbi meccsekre. Hogy mennyire bízzon a piacban, azt az adat dönti el; az illesztett súly minden tanításkor logolódik.
- Nincs variancia-kalibráció (lásd fent). Minden tanítás után logolódik az out-of-fold hiba.
- Tanítóablak: az utolsó 4 lezárt szezon + a folyó szezon, szezonban hetente újratanítva. Mintasúlyok: 1,2 (folyó szezon), 1,0, 0,8, 0,6 (legrégebbi); rájátszás 0,6–0,75×; 2020 lesúlyozva (nézők nélkül). Minden konstans a `model/settings.py`-ban.

---

## Automatizálás — a workflow-k

| Workflow | Mikor | Mit csinál |
|---|---|---|
| `weekly_update.yml` | kedd 10:23 UTC | Teljes futás. Offszezonban ~30 mp alatt kilép (nincs meccs a [−3, +10] napos ablakban). |
| `thursday_update.yml` | szerda 21:41 UTC | Csak-jóslás a csütörtöki meccs előtt a keddi modellel (ha hiányzik, újratanít). |
| `full_pipeline.yml` | kézi, vagy `.full-trigger`-t érintő push | Tanítás + jóslás → az utolsó két lezárt szezon párhuzamos backtestje → összefésülés. |
| `train_run.yml` | kézi, vagy `.train-trigger` | Csak tanítás + jóslás. |
| `backtest_run.yml` | kézi, vagy `.backtest-trigger` | Csak a backtest-mátrix + összefésülés. |
| `walkforward.yml` | kézi, vagy `.walkforward-trigger` | 2013 óta kitartott szezononként egy job (4 szezonos ablak) + összesítés és ábra a `docs/walkforward.html`-hez. |
| `freeze_data.yml` | évente ápr. 1., kézi, vagy `.freeze-trigger` | Újrapecsételi a `data/frozen/`-t az épp lezárult szezonnal. |
| `pre_season_train.yml` | augusztus első keddje | Elindítja a `full_pipeline.yml`-t az új szezonra. |

Az ütemezett időpontok szándékosan nem egész órára esnek: a GitHub az egész órás ütemezett futásokat terhelésnél késlelteti (vagy eldobja) — a régi 10:00-s idősáv egész szezonban 6–9 órát késett. Minden lépés commitolja a konzol végét a `data/logs/*_last.txt` alá, a push pedig a `scripts/push_with_retry.sh`-n keresztül megy, ami pirosra állítja a jobot, ha a push nem sikerül.

Push-trigger példa az Actions felület nélkül:

```bash
date > .full-trigger && git add .full-trigger && git commit -m "run full pipeline" && git push
```

---

## A befagyasztott adatréteg

A lezárt szezonok soha nem változnak, ezért a `scripts/freeze_data.py` egyszer letölti és a `data/frozen/`-be commitolja őket (~24 MB):

- heti játékos-statisztikák és heti keretek minden lezárt szezonra 2009 óta;
- teljes történetű Next Gen Stats fájlok és egy menetrend-pillanatkép;
- `pbp_agg_{szezon}.parquet`, a play-by-play szezononkénti csapat-meccs aggregátuma, ugyanazzal a függvénnyel, amit az éles út használ (a nyers play-by-play ~25 MB/szezon, nincs commitolva).

---

## Repó-térkép

```
model/
  pipeline.py            belépési pont — módok: full / predict_only / backtest
  settings.py            minden fix modellkonstans (súlyok, Elo, küszöbök, verzió)
  data_loader.py         táblák letöltése + cache; frozen-first betöltés
  feature_engineering.py play-by-play aggregálás, minden feature, az allowlist
  feature_h2h.py         egymás elleni feature-ök a korábbi találkozókból
  train.py               ensemble-tanítás + mentés/betöltés
  predict.py             az egyetlen jóslási útvonal (éles, backtest, walk-forward)
  evaluate.py            backtest-metrikák és az élő mérleg
  confidence.py          megbízhatósági pontszám (modell-egyetértés / teljesség / H2H)
  walkforward.py         egy walk-forward fold a befagyasztott adatból
scripts/
  freeze_data.py         a data/frozen/ felépítése
  season_guard.py        offszezon-őr az ütemezett workflow-knak
  merge_backtests.py     a szezononkénti backtest-töredékek összefésülése
  merge_walkforward.py   walk-forward foldok összesítése; ábra
  push_with_retry.sh     bot-push rebase-zel és újrapróbálással
data/
  frozen/                commitolt történelmi adat (fent)
  predictions/           publikált kimenetek (az oldal adatforrása)
  logs/                  minden workflow-lépés utolsó konzol-logja
docs/                    GitHub Pages oldal (index.html, walkforward.html, assets/)
```

---

## Fogalomtár

**Fogadási kifejezések**

- **Spread / spread_line:** a fogadóiroda várt különbsége. Ebben a repóban mindig hazai szemszögű, és **pozitív, ha a hazai csapat az esélyes** (nflverse-konvenció); a hazai csapat fedez, ha hazai − vendég > spread.
- **ATS (against the spread):** a modell azt az oldalt „választja”, amelyet a becsült különbsége a vonalhoz képest kedvez — minden meccsen. Push = nincs döntés.
- **Over/Under (total):** a fogadóiroda vonala az összpontszámra; a modell minden meccsen over-t vagy under-t választ.
- **Edge-meccs / lean:** a modell és a vonal legalább **3 ponttal** eltér. LOW/WEAK megbízhatóságnál az oldal elrejti.
- **−110-es szorzó:** 110-et teszel, hogy 100-at nyerj; 52,4% a nullszaldó. **ROI** = egységnyi tétre jutó profit, fix 1 egységes téttel az edge-meccseken.

**Adat-kifejezések**

- **PBP (play-by-play):** minden játék egy sor (nflverse / nflfastR). **EPA:** egy játék által hozzáadott várható pont. **WEPA:** EPA, a játékok előrejelző értéke szerint súlyozva (meccsvégi „szemétidő” le, normál passzok fel; nfelo nyomán). **Success rate:** a pozitív EPA-jú játékok aránya.
- **NGS (Next Gen Stats):** NFL tracking-adatok (passzidő, elválás, várható feletti futóyard).
- **Befagyasztott réteg:** a lezárt szezonok commitolt `data/frozen/` pillanatképe.

**Modellezési kifejezések**

- **Feature / allowlist:** bemeneti oszlop; csak az allowlistán szereplő jut el a modellhez.
- **Adatszivárgás:** bármely út, amelyen a megjósolandó meccs saját kimenetele eléri a modellt. **Train/serve eltérés:** olyan feature, amely jósláskor másképp néz ki (itt: hiányzik), mint tanításkor.
- **Gördülő ablak (r4/r8/r16):** a csapat előző 4/8/16 lejátszott meccsének átlaga.
- **Elo:** meccsről meccsre frissülő erősség-érték; mindkét csapat meccs előtti értéke rögzül, mielőtt bármelyik frissülne.
- **OOF (out-of-fold):** olyan adaton adott jóslat, amelyen a modell nem tanult; ezen illeszkedik a meta-learner.
- **Backtest / walk-forward:** tanítás csak az X előtti szezonokon, X megjóslása, összevetés a tényekkel (a walk-forward ezt 2013 óta minden szezonra megismétli).
- **MAE:** átlagos abszolút hiba pontban. **Különbség-hiba** = a hazai − vendég MAE-je.
- **Megbízhatóság (HIGH/MEDIUM/LOW/WEAK):** nem győzelmi valószínűség — al-modellek egyetértése (55%), adatteljesség (30%), H2H minta (15%).

---

## Adatforrások

Mind ingyenes és nyilvános: [nflverse](https://github.com/nflverse) release-fájlok (play-by-play, heti játékos-statok, heti keretek, Next Gen Stats) és az [nflverse/nfldata](https://github.com/nflverse/nfldata) meccsfájl (eredmények, Vegas-i vonalak, tető, kezdő irányítók, vezetőedzők).

## Technológia

Python · pandas · scikit-learn · XGBoost · LightGBM · PyTorch · GitHub Actions · GitHub Pages

---

*Készítette: [@llou77](https://github.com/llou77) · A jóslatok a szezon alatt minden kedden és szerda este frissülnek.*
