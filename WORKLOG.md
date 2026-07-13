# WORKLOG

Folyamatos munkanapló — bármelyik későbbi session innen tudja folytatni.
(Konvenció: legfrissebb kör felül.)

---

## 2026-07-13 (2. kör) — Mély walk-forward szimuláció

**Feladat (tulajdonosi javaslat, keményítve):** csúszó ablakos walk-forward a
történelem elejétől máig — foldonként CSAK a tesztszezon előtti adatokon
hangolva/tanítva, a kihagyott szezont pontozva; a hangolt súlyok driftjének
logolása és ábrázolása. Két korrekció a javaslathoz képest: (1) az összesített
walk-forward eredményen TILOS re-optimalizálni (meta-overfitting — a backtest
tanítóhalmazzá válna); (2) a súly-drift extrapolációja kimaradt (paraméterenként
~12 zajos pont — varianciát ad, nem jelet; rezsimváltás-figyelésre való).

### Elvégezve

1. `scripts/freeze_data.py`: `FREEZE_FROM` (default 2009) + forrásonkénti
   `first_season` (NGS 2016+, PFR 2018+, FTN 2022+, snap 2012+, injuries 2009+
   stb.) — a korszak előtti hiány nem hiba, kihagyás.
2. `model/walkforward.py`: egy fold = frozen-only adat-összeállítás →
   feature-építés → nested Optuna (train=[S-W..S-2], val=S-1, teszt érintetlen)
   → teljes ensemble tanítás → éles predikciós úton pontozás → fragment JSON
   (metrikák + hangolt súlyok + covid-flag).
3. `scripts/merge_walkforward.py`: fragmentek összefésülése, ablakonkénti
   aggregátumok (covid-szűrt változatban is), matplotlib ábra (súly-drift +
   MAE + ATS foldonként) → docs/assets.
4. `.github/workflows/walkforward.yml`: (2013–2025 × ablak {3,4}) = 26 párhuzamos
   fold-job + merge; trigger: `.walkforward-trigger` push vagy kézi (trials input).
5. `docs/walkforward.html`: eredménytábla + ábra az oldalon.

### Megjegyzések

- A fold-jobok a repo-beli `optimal_weights.json`-t csak az NN loss-súlyaihoz
  és a variance_scale-hez olvassák (nem hangolt, statikus defaultok) — a
  fold-izolációt ez nem sérti érdemben.
- Ablakhossz-kérdés (3 vs 4): az aggregátumok empirikusan eldöntik.

### Eredmény (26/26 fold zöld, 2 kör után)

- Menet közben javítva: (1) nflverse korszakok közti dtype-drift
  (jersey_number str↔float → pyarrow-hiba a 2016–2019-es foldokon) —
  `_harmonize_dtypes` + safe-write; (2) a merge a régi `_summary` blokkot
  fold-sorként olvasta vissza → kizárva.
- **Összesítő (13-13 fold):** ablak=3: spread MAE 10,41 | ATS 54,2% (covid
  nélkül 53,6%) | edge ATS 59,1% (covid nélkül 57,4%) | edge ROI +12,9%.
  Ablak=4: MAE 10,41 | ATS 55,4% (53,6%) | edge ATS 59,0% (57,0%) | ROI +12,6%.
- **Ablakhossz-verdikt:** 3 vs 4 szezon gyakorlatilag EGYENÉRTÉKŰ (MAE azonos,
  ATS ±0,5%p) — nincs ok változtatni a jelenlegi 4+folyó ablakon.
- **Korszak-hatás:** a 2013–2015-ös foldok gyengébbek (MAE 10,9–11,8) — kevesebb
  forrás (nincs NGS/PFR/FTN); 2016-tól a MAE jellemzően 9,2–10,6.
- **Óvatossági jegyzet:** az edge ATS ~57% (covid-szűrt) átlaga fold-onként
  34–76% közt szór — a jel pozitívnak tűnik, de a vonalforrás-egységesítés
  (mrcaseb vs schedules) előtt éles pénzre extrapolálni nem szabad; a
  végső bizonyíték az élő, időbélyegzett 2026-os szezon lesz.

## 2026-07-13 — "Runtime round": crash-fix, frozen adatréteg, workflow-darabolás, doksi

**Feladat:** (1) a pipeline nem futott le és/vagy 20+ percig tartott — gyökérok-javítás
és futásidő-csökkentés darabolással; (2) offszezon: a lezárt szezonok adatai a repóba
kerüljenek, ne töltődjenek le minden futásnál; (3) közérthető folyamatleírás.

### Gyökérok (a 07-08-i baseline failure)

- `train.py:99` — a `SimpleImputer` némán KIDOBJA a csupa-NaN oszlopokat (376→363),
  a 376-széles `nan_mask` alkalmazása a 363-széles kimenetre → `IndexError`, a job
  másodpercekkel a "Training on 1139 games" után meghalt. Ugyanez a minta a
  `predict.py`-ban is élesben várta volna ugyanezt.
- A 13 üres oszlop részben abból jött, hogy az nflverse 2025-ben átnevezte a player
  stats release-eket (`player_stats/player_stats_{s}` → `stats_player/stats_player_week_{s}`),
  a 2025-ös fájl azóta 404.

### Elvégezve

1. **Crash-fix:** `SimpleImputer(keep_empty_features=True)` (stabil oszlopszám; üres
   oszlop: 0 a Ridge/NN-nek) + a fáknak `X_tree = X_raw` (az impute→unimpute
   körforgás definíció szerint az X_raw-t rekonstruálta — bugosan). `predict.py`
   tükörjavítás. Az Actions model-cache kulcs `model-saved-v2-` (régi, inkompatibilis
   artifactok sosem állnak vissza).
2. **Halott letöltés törölve:** a 2014–2021-es nyers PBP-t (8×~25 MB + több GB RAM)
   SEMMI nem olvasta — a H2H a schedules eredményeiből számol. A blokk kikerült a
   data_loaderből.
3. **Frozen adatréteg:** `scripts/freeze_data.py` (CI-ban fut) a lezárt szezonok
   (< CURRENT_SEASON) minden tábláját egyszer letölti és `data/frozen/`-be commitolja;
   a nyers PBP-ből csak a szezononkénti csapat-meccs AGGREGÁTUM kerül be
   (`pbp_agg_{s}.parquet`, ~0,2 MB/szezon) — ugyanazzal a kóddal számolva, amit az
   éles út használ (`_aggregate_pbp_core`, egyetlen igazságforrás). `data_loader`
   frozen-first: lezárt szezon = repóból, hálózat nélkül; élő csak a schedules,
   lines és az aktuális szezon fájljai. Player stats URL-fix fallback-lánccal.
   Méret-őr: >90 MB-os fájl = hard fail (git limit előtt).
4. **Workflow-darabolás:** a monolit `baseline_run.yml` (30–50 perc egyben) törölve.
   Helyette: `full_pipeline.yml` (lépcsős jobok: optimize → train+predict →
   backtest-MÁTRIX szezononként párhuzamosan → merge) + önálló `optimize_run` /
   `train_run` / `backtest_run` / `freeze_data` workflow-k saját trigger-fájllal.
   Backtest-fragmentek artifactként utaznak, `scripts/merge_backtests.py` fésüli
   össze (részleges hibánál a régi szezoneredmény megmarad). Minden lépcső
   publikus logot commitol (`data/logs/*_last.txt`). Közös concurrency-group véd
   az egyidejű futások git-versenyétől. `pre_season_train.yml` már csak dispatch-el.
5. **Offszezon-őr:** `scripts/season_guard.py` (stdlib-only, a pip install ELŐTT fut)
   — ha nincs meccs a [-3, +10] napos ablakban, a heti/csütörtöki cron ~30 mp alatt
   kilép a korábbi 15+ perc helyett. Fail-open: ha a menetrend nem elérhető, fut.
6. **Gyorsabb CI-install:** requirements-trim (shap/numba, polars, requests, bs4,
   tqdm, dotenv, joblib, scipy, pytest, optuna-integration — egyiket sem importálta
   semmi) + torch CPU-only wheel indexről (a PyPI-s default CUDA-t bundle-öl, ~4×
   nagyobb). `--backtest-season` CLI arg + `only_season` az evaluate-ben a mátrixhoz.
7. **Doksi:** README teljes újraírás (lépésenkénti folyamat, workflow-táblázat,
   repó-térkép, teljes fogalomtár) + `README.hu.md` magyar változat. A korábbi
   README hamis "weatherapi.com" forráshivatkozása javítva (időjárás a schedules
   feedből jön; a kód sehol nem hív weather API-t).

8. **Spread-előjel bug (a verifikáció során találva):** az első zöld lépcsős futás
   backtest-számai inkonzisztensek voltak — spread MAE őszinte (10,1–10,2), O/U
   őszinte (49–54%), de ATS 76–83%. Ez az előjel-hiba ujjlenyomata: a totalnál
   nincs hazai/vendég orientáció, a spreadnél van. Gyökérok: a kód azt
   feltételezte, hogy `spread_line < 0` = hazai favorit, miközben az nflverse
   schedules konvenciója FORDÍTOTT (pozitív = hazai favorit, cover:
   `margin > spread_line`). Javítva 4 helyen: `evaluate.compute_metrics`
   (threshold + vegas_fav_home), `evaluate.update_performance_from_latest`
   (élő tracker), `predict._add_edges` (a NYILVÁNOS oldal HOME/AWAY value
   címkéi eddig fordítva/torzítva számolódtak!), `feature_h2h.home_covered`.
   A CLV opening-forrása (mrcaseb team-szintű vonalak) továbbra is
   ellenőrizetlen orientációjú — proxynak jelölve.

### Következő lépések

- ✔ Frozen réteg felépült (67 fájl, 27,2 MB), lépcsős teljes futás zöld
  (optimize ~13 p, train+predict ~6 p, backtest-mátrix+merge ~5 p — egyik job
  sem éri el a 20 percet). Előjel-fix utáni backtest-újrafutás folyamatban.
- Vonalforrás-egységesítés: `opening_spread`/`closing_spread` (mrcaseb,
  team-orientált) vs `spread_line` (schedules, home-orientált) — az 1450–1459
  körüli fillna keveri a kettőt; a `vegas_implied_power` előjele is fordított
  (konzisztens feature-ként a modell megtanulja, de félrevezető). Külön kört érdemel.
- Player ratings modul továbbra is kikapcsolva (temporal shift rework vár rá).

**Feladat:** a korábbi chat-körökben azonosított javítások tényleges végrehajtása
(a sandbox-resetek miatt korábban semmi nem került pushra), felesleg törlése,
konzisztencia- és stabilitásjavítások.

### Elvégezve

1. **Kritikus futási hiba:** `ELO_START` / `ELO_K` sehol nem volt definiálva →
   a `_add_elo` minden futásnál `NameError`-t dobott (a pipeline el sem indult
   tiszta környezetben). Konstansok pótolva (1500 / 20).
2. **Szivárgás #1 — same-game statok:** `get_feature_columns` denylist →
   **allowlist** (`_PREGAME_SIDE_BASES`, `_PREGAME_SHARED`, `_r4/_r8/_r16`
   suffix-szabály, `h2h_*`). Minden nem engedélyezett oszlop kiesik és logolódik.
3. **Szivárgás #2 — Elo-sorrend:** `_add_elo` kétfázisú, meccsenkénti feldolgozás —
   előbb MINDKÉT oldal pre-game értéket kap, csak utána frissül a rating.
4. **Szivárgás #3 — player ratings:** alapból kikapcsolva
   (`ENABLE_PLAYER_RATINGS=1` env-vel visszakapcsolható), amíg nincs temporal
   shift a modulban. Az allowlist ettől függetlenül is kizárja az oszlopait.
5. **További megtalált szivárgások lezárva:**
   - NGS statok same-week merge → shift(1) + rolling(4) (`*_r4` nevek);
   - bírói tendenciák teljes karrierből → csak korábbi meccsekből (expanding, min. 10);
   - `season_avg_total` / `scoring_era_adj` → előző szezon átlaga (nem a sajátja);
   - `team_specific_hfa` → csak korábbi szezonokból (expanding prior, 2020 kizárva).
6. **Train/serve skew:** közös `predict.ensemble_predict()` útvonal — az éles
   predikció ÉS a backtest ugyanazt a kódot futtatja (fák: nyers+NaN,
   Ridge/NN: imputált+skálázott, meta-sorrend fix, sanity clamp + variancia-
   kalibráció mindkét úton).
7. **Meta-learner valódi OOF:** foldonként újratanított XGB/LGBM (közös
   `XGB_PARAMS`/`LGBM_PARAMS`) és NN (60 epoch, foldonkénti seed); a régi kód a
   teljes adaton tanított modellekkel "OOF-ozott" (in-sample volt).
8. **Bayesian optimizer:** a 26-ból 18 halott paraméter kivéve a keresésből —
   csak a 8 sample-weight paraméter megy Optunának (csak ezek hatnak a proxy
   objektívre); a 999-es conf-constraint törölve; a nem keresett paraméterek
   research-backed defaultok maradnak. `ref_season` = validációs szezon.
9. **Thursday crash:** `predict_only` fallback tanításra, ha nincsenek
   artifactok + Actions cache (`model-saved-*`) a weekly ↔ thursday között.
10. **Cron-bug:** pre-season workflow `0 12 1-7 8 2` (≈11 futás augusztusban) →
    `0 12 * 8 2` + first-Tuesday guard step.
11. **performance.json bekötve:** `evaluate.update_performance_from_latest()` —
    a pipeline minden futás elején egyezteti a korábban publikált predikciókat
    a lejátszott meccsekkel, MIELŐTT felülírná a fájlt. (Eddig SEMMI nem hívta.)
12. **ROI + CLV:** `compute_metrics` → `edge_roi_flat_110`, `edge_profit_units`,
    `avg_clv_pts`, `clv_positive_pct`, `n_clv_games` (opening vs. closing proxy).
13. **Szezon-konstans egységesítve:** `data_loader.get_current_season()` a
    naptárból — a 2026-os hardcode kikerült a pipeline/predict/train/optimizer
    fájlokból (predictions JSON "season" mezője is dinamikus).
14. **Determinizmus:** `SEED=42` — numpy + torch.manual_seed + DataLoader
    generator; XGB/LGBM random_state a közös param-dictben.
15. **Törölve:** `model/ratings.py` — focis (eloratings.net, gólkülönbség-súlyos)
    Elo-modul volt, semmi nem importálta.
16. **README:** a hamis (leakelt) pontossági táblázat visszavonva, őszinte
    elvárások (spread MAE ~10–11, edge ATS 50–54%) + magyarázat került be.
17. **Workflow-takarítás:** a gitignore-olt fájlokra mutató, `|| true` mögött
    némán elhaló `git add model/saved/*.pkl` sorok eltávolítva; commit-lista
    csak ténylegesen commitolható fájlokat tartalmaz; datetime.utcnow() →
    timezone-aware mindenhol.

### Döntések

| Helyzet | Döntés | Ok |
|---|---|---|
| Denylist vs allowlist | allowlist | új oszlop alapból KIZÁRVA — a korábbi 95%-os "pontosság" a denylist-lyukakból jött |
| Player ratings | kikapcsolva, nem átírva | a memóriában rögzített sorrend szerint: újraintegrálás csak temporal shift után |
| Artifact-perzisztencia CI-ban | Actions cache + fallback-train | repo-bloat nélkül; cache-miss esetén sem crashel |
| Régi (hamis) backtest JSON-ok | a helyükön maradtak | a frontend olvassa őket; a következő backtest-futás felülírja őket őszinte számokkal |
| NN OOF | foldonkénti 60 epoch | teljes 150 epoch × 5 fold aránytalan CI-költség; torch-hiány esetén jelzett fallback |

### Következő lépések (prioritási sorrend)

1. **Őszinte baseline futtatása:** GitHub Actions → "Pre-Season Training" kézi
   indítás (vagy `--mode backtest`) → az új, valós MAE/ATS/ROI/CLV számok
   publikálása. Elvárás: spread MAE ~10–11, edge ATS 50–54%. Ha ennél jobb,
   előbb keress további szivárgást, csak utána örülj.
2. Dynamic season weighting valódi walk-forward validációval (a mostani
   optimize mód egyetlen train/val split — bővíthető expanding-window CV-re).
3. Surprise radar (upset-valószínűség) — csak az őszinte baseline után.
4. Player ratings újraintegrálás shift(1) + rolling formában.
5. `docs/insights.html` frissítése, ha az új backtest-kulcsok (ROI/CLV)
   megjelenítést kapnak.

### Ismert korlátok / figyelmeztetések

- A `docs/assets/backtest_results.json` és a README-n kívüli oldalak még a
  RÉGI (leakelt) számokat mutatják, amíg le nem fut egy friss backtest.
- A CLV proxy konszenzus open/close mediánokból számol, nem valós elérhető
  árakból — irányjelző, nem könyvelési tétel.
- `closing line` típusú feature-ök (book_spread_hist, market_win_prob) élesben
  a lekérdezés pillanatának vonalát látják, nem a tényleges záróvonalat —
  ez pre-game információ (nem outcome-leak), de enyhe train/serve eltérés marad.
