# NFL Eredmény-előrejelző — 2026-os szezon

**🏈 Élő oldal:** [llou77.github.io/NFL](https://llou77.github.io/NFL) · **English version:** [README.md](README.md)

Gépi tanulási modell, amely minden NFL-meccs végeredményét megjósolja a kezdőrúgás előtt. A két csapat becsült pontszámából automatikusan levezeti a várható győztest és a különbséget, a Vegas-i összpontszám-vonalhoz (over/under) képesti eltérést, és egy megbízhatósági besorolást. Minden automatikusan fut GitHub Actions-ön, és a GitHub Pages oldalra publikál.

Minden lentebb használt kifejezés definiálva van a [Fogalomtárban](#fogalomtár). Ha csak két részt olvasol el: [Hogyan zajlik egy futás](#hogyan-zajlik-egy-futás-lépésről-lépésre) és [Mennyire pontos?](#mennyire-pontos).

---

## Mennyire pontos?

**Őszinte helyzetjelentés (2026. július): a korábban publikált pontossági számok érvénytelenek voltak, visszavontuk őket.**

Egy belső audit több *adatszivárgási* csatornát talált — a modell tanításkor látta annak a meccsnek a statisztikáit, amelyet épp megjósolni próbált (a meccs saját box score-ja, meccs utáni Elo-értékek, azonos heti tracking-adatok). Ez teljesen megmagyarázta a hihetetlenül erős régi számokat. A szivárgások le vannak zárva (feature-allowlist, kétfázisú Elo, késleltetett gördülő statisztikák, becsületes out-of-fold stacking, és a backtest pontosan az éles predikciós kódutat futtatja).

Reális elvárások, amelyeket minden új backtest-futás ellenőriz:

| Metrika | Őszinte elvárás | Kontextus |
|---|---|---|
| Átlagos hiba a különbségre (MAE) | ~10–11 pont | Az elméleti maximum NFL-ben ~9,5–10; a Vegas-i záróvonalak ~10,2–10,5-öt hoznak |
| Edge ATS találati arány | 50–54% | -110-es szorzónál 52,4% a nullszaldó; ami tartósan efölött van, az valódi előny |
| ROI / CLV | élőben követve | A fix tétes ROI és a záróvonal-érték elsőrangú backtest-metrikák |

> **A fogadásról:** az oldal *edge jelzéseket* mutat — meccseket, ahol a modell érdemben mást gondol, mint a Vegas-i vonal. Ezek nem fogadási tanácsok. Hogy van-e valódi előny, azt csak egy teljes szezonnyi élő, időbélyegzett predikció dönti el.

---

## Hogyan zajlik egy futás, lépésről lépésre

Egy teljes pipeline-futás (`model/pipeline.py --mode full`) sorrendben ezt csinálja:

1. **Adatbetöltés** (`data_loader.py`). A *lezárt* szezonok adatai (minden, ami a folyó szezon előtt volt) a repóba commitolt **befagyasztott rétegből** (`data/frozen/`) jönnek — nulla hálózati hozzáférés, semmi nem tud eltörni. Csak az *élő* adatok töltődnek le: a menetrend (közelgő meccsek + végeredmények), a fogadási vonalak, és szezon közben az aktuális szezon statfájljai.
2. **Play-by-play aggregálás** (`feature_engineering._aggregate_pbp`). Minden meccs minden játékát egy csapat-meccs sorba sűrítjük (támadó/védekező hatékonyság, tempó, labdavesztések, red zone arányok, …). Lezárt szezonokra ez előre ki van számolva a repóban (`data/frozen/pbp_agg_*.parquet`); csak az aktuális szezon meccsei aggregálódnak frissen.
3. **Feature-építés** (`feature_engineering.build_all_features`). A csapat-meccs sorok kiegészülnek gördülő formaablakokkal (utolsó 4/8/16 meccs, mindig **késleltetve** — egy meccs sosem írhatja le önmagát), Elo-értékekkel (kétfázisú frissítés: mindkét csapat szigorúan meccs *előtti* értéket kap), QB-formával, sérülésekkel, bírói tendenciákkal (csak korábbi meccsekből), Next Gen Stats adatokkal (késleltetve), pihenőnapokkal, a menetrendből jövő időjárással, utazási/primetime kontextussal és előző szezonos priorokkal. Ezután a sorok **meccsenként egy sorrá** fordulnak (hazai vs vendég). Egy szigorú **allowlist** dönti el, mely ~370 oszlopot láthatja a modell — ami a meccs saját kimenetelét szivárogtathatná, az kiesik és logolódik.
4. **Egymás elleni (H2H) feature-ök** (`feature_h2h.py`). A menetrendtábla történelmi *eredményeiből* számolódnak (sosem play-by-play-ből), a párosítás típusától függő visszatekintéssel és keretfolytonossági csillapítással.
5. **Teljesítmény-egyeztetés** (`evaluate.update_performance_from_latest`). Mielőtt bármi felülíródna, a korábban publikált predikciókat összeveti az azóta lejátszott meccsekkel, és frissíti a `performance.json`-t (az oldal élő bizonyítványát).
6. **Ensemble-tanítás** (`train.py`). Lásd [Modell-architektúra](#modell-architektúra). Minden artifact a `model/saved/`-be kerül.
7. **Predikciógenerálás** (`predict.py`). A közelgő meccsek pontosan ugyanazon az `ensemble_predict()` útvonalon mennek át, mint a backtestben (nincs train/serve eltérés), kalibrálva, megbízhatósági besorolással és edge-jelzésekkel, eredmény: `data/predictions/predictions_latest.json`.
8. **Publikálás**. A kimenetek a `docs/assets/`-be másolódnak, ezt szolgálja ki a GitHub Pages.

A **backtest** (`--mode backtest`) különálló: minden tesztszezonhoz újratanítja a teljes ensemble-t a megelőző szezonokon, megjósolja a kihagyott szezont, és MAE / ATS / over-under / ROI / CLV számokat ad. Az **optimalizáló** (`--mode optimize`) szintén különálló, szezon előtti lépés: az Optuna a minta-súly paramétereket hangolja (mennyit számítson az egyes tanítószezonok és meccstípusok). A 2026-07-es walk-forward stabilitás-vizsgálat óta az eredeti 8-ból 2 a fold-ok mediánjára van rögzítve (w_oldest, wt_sb — hangolásuk zajkergetés volt), 2 pedig ±15%-os sávra szűkítve (w_recent, w_current — ezekben az adat folyton egyetért); a `WF_UNCONSTRAINED=1` visszaadja a teljes keresést.

---

## Modell-architektúra

Háromrétegű *stacked ensemble* — több modell, amelyek kimenetét összekeverjük:

```
1. réteg:  Ridge regresszió  +  XGBoost  +  LightGBM     (három független pontszám-előrejelző)
2. réteg:  PyTorch neurális háló (két fej: hazai pontszám, vendég pontszám)
3. réteg:  Ridge meta-learner — a négy predikció optimális keveréke (valódi out-of-fold adaton tanítva)
           → variancia-kalibráció → végső becsült pontszám
```

- **Előfeldolgozás:** a fa-alapú modellek (XGBoost, LightGBM) nyers feature-öket kapnak, a hiányzó értékek megmaradnak — natívan kezelik a NaN-t. A Ridge és a neurális háló medián-imputált, standardizált adatot kap. A megfigyelés nélküli (csupa üres) oszlopok megmaradnak (Ridge/NN-nek nullával töltve, fáknak NaN-ként), így a mátrix szélessége sosem változik tanítás és predikció között.
- **Becsületes stacking:** a meta-learner out-of-fold (OOF) predikciókon tanul, ahol minden 1. rétegbeli modell — a neurális háló is — *foldonként újra van tanítva*, tehát a keverő sosem lát olyan modellt, amely a saját tanítóadatát jósolja.
- **Kalibráció:** a meta-learner a becsült különbségeket az átlag felé húzza; egy adatvezérelt variancia-szorzó ([1, 3] közé vágva) állítja vissza a reális szórást.
- **Tanítóablak:** az utolsó 4 lezárt szezon + a folyó szezon, szezon közben hetente újratanítva. A szezon- és meccstípus-súlyok a Bayes-optimalizálóból jönnek. A szezonév az órából számolódik (`data_loader.get_current_season`), soha nem kézzel írt konstans.

---

## Automatizálás — a workflow-k

Minden automatizálás a `.github/workflows/`-ban él. Minden nehéz lépés **külön, bőven 20 perc alatti job**, az Actions felületről egyenként újrafuttatható. Minden lépés publikus logot commitol a `data/logs/*_last.txt` alá, így az eredmény Actions-hozzáférés nélkül is ellenőrizhető.

| Workflow | Mikor fut | Mit csinál |
|---|---|---|
| `weekly_update.yml` | kedd 10:00 UTC (cron) | Teljes futás (fenti 1–8. lépés). Az **offszezon-őr** ~30 mp alatt kilép, ha nincs NFL-meccs a [-3, +10] napos ablakban. |
| `thursday_update.yml` | szerda 22:00 UTC (cron) | Csak-predikció frissítés a csütörtöki meccs előtt, cache-elt modell-artifactokkal (cache-hiánynál tanítással). Ugyanaz az őr. |
| `full_pipeline.yml` | `.full-trigger` fájlt érintő push, vagy kézi | Lépcsős lánc: optimalizálás → tanítás+predikció → **párhuzamos szezononkénti backtestek** → összefésülés és publikálás. |
| `optimize_run.yml` | `.opt-trigger` push, vagy kézi | Csak az 1. lépcső: Bayes-súlyoptimalizálás (~8–10 perc). |
| `train_run.yml` | `.train-trigger` push, vagy kézi | Csak a 2. lépcső: tanítás + predikció (~8–12 perc). |
| `backtest_run.yml` | `.backtest-trigger` push, vagy kézi | Csak a 3. lépcső: backtest-mátrix (szezononként egy párhuzamos job) + összefésülés. |
| `freeze_data.yml` | `.freeze-trigger` push, kézi, vagy évente ápr. 1. | Újraépíti a `data/frozen/`-t — az összes lezárt szezon adatát egyszer letölti és commitolja (a `FREEZE_FROM` évig vissza, alapból 2009). |
| `walkforward.yml` | `.walkforward-trigger` push, vagy kézi | Mély walk-forward szimuláció: (tesztszezon 2013–2025 × ablak 3/4) párokra egy-egy párhuzamos job, foldonkénti nested hangolással; összesített eredmény + súly-drift ábra a [docs/walkforward.html](https://llou77.github.io/NFL/walkforward.html) oldalon. |
| `pre_season_train.yml` | augusztus első keddje | Elindítja a `full_pipeline.yml`-t az új szezonra. |
| `season_analysis.yml` | kézi | Szezononkénti nehézség/trend elemzés az oldalhoz. |

Bármely push-triggeres workflow indítása az Actions felület nélkül:

```bash
date > .full-trigger && git add .full-trigger && git commit -m "run full pipeline" && git push
```

---

## A befagyasztott adatréteg

Két szezon között — és minden lezárt szezonra — a történelmi statisztikák soha többé nem változnak. Ezért a `scripts/freeze_data.py` **egyszer** letölti és a `data/frozen/`-be commitolja őket:

- szezononkénti stat-táblák (játékos-statisztikák, keretek, snap countok, sérülések, depth chartok, FTN charting, PFR advanced statok) minden lezárt szezonra;
- teljes történetű referenciatáblák (Next Gen Stats, draft, combine, bírók, fogadási vonalak, csapat/játékos azonosítók) offline tartalékként;
- `pbp_agg_{szezon}.parquet` — a nyers play-by-play szezononkénti csapat-meccs aggregátuma, *ugyanazzal a függvénnyel* előállítva, amit az éles pipeline használ. A nyers play-by-play (~25 MB/szezon) szándékosan **nincs** commitolva; az aggregátum ~100× kisebb, és a pipeline-nak pontosan ennyi kell.

A normál futásoknak így csak a menetrendhez, a fogadási vonalakhoz és az aktuális szezon fájljaihoz kell hálózat. Ha az nflverse átnevez vagy töröl egy történelmi fájlt (2025-ben megtörtént), semmi nem törik el. A fagyasztást évente egyszer, a Super Bowl után kell újrafuttatni — az április 1-jei cron ezt automatikusan megteszi.

---

## Repó-térkép

```
model/
  pipeline.py            belépési pont — módok: full / predict_only / backtest / optimize / analyze_seasons
  data_loader.py         minden adat letöltése + cache; frozen-first betöltés; szezon-konstansok
  feature_engineering.py PBP-aggregálás + minden feature-építés + az allowlist
  feature_h2h.py         egymás elleni feature-ök történelmi eredményekből
  train.py               a 3 rétegű ensemble tanítása + artifact mentés/betöltés
  predict.py             az egyetlen közös predikciós útvonal (éles + backtest)
  evaluate.py            backtestek, ATS/ROI/CLV metrikák, élő teljesítmény-egyeztetés
  bayesian_optimizer.py  Optuna-keresés a minta-súlyokon (stabilitás-korlátokkal)
  confidence.py          megbízhatósági besorolás (modell-egyetértés / adatteljesség / H2H minta)
  season_analysis.py     szezononkénti nehézség-elemzés
  player_ratings.py      KIKAPCSOLVA az időbeli eltolás átdolgozásáig (szivárgásveszély)
  walkforward.py         egy walk-forward fold: nested hangolás → tanítás → kihagyott szezon pontozása
scripts/
  freeze_data.py         a data/frozen/ felépítése (CI-ban fut, évente)
  season_guard.py        offszezon-őr — "run"/"skip" az ütemezett workflow-knak
  merge_backtests.py     a szezononkénti backtest-töredékek összefésülése
  merge_walkforward.py   walk-forward foldok összesítése; súly-drift ábra
data/
  frozen/                commitolt, lepecsételt történelmi adat (lásd fent)
  raw/, processed/       gitignore-olt munkacache-ek
  predictions/           commitolt modellkimenetek (az oldal adatforrása)
  logs/                  commitolt utolsó-futás logok minden workflow-lépcsőhöz
docs/                    GitHub Pages oldal (index.html + assets/*.json)
```

---

## Fogalomtár

**Fogadási / NFL-kifejezések**

- **Spread (pontkülönbség-vonal):** a fogadóiroda hendikepje az esélyesnek. A -3,5-ös spread azt jelenti, hogy az esélyestől 3,5 pontnál nagyobb győzelmet várnak.
- **ATS (against the spread):** egy csapat "fedez" (cover), ha a spreadnél jobban teljesít. *Edge ATS találati arány* = milyen gyakran bizonyult igaznak, amikor a modell a spreaddel szemben foglalt állást.
- **Over/Under (total):** a fogadóiroda vonala a két csapat összpontszámára; arra lehet fogadni, hogy a valós összeg e fölé vagy ez alá esik.
- **-110-es szorzó:** standard amerikai árazás — 110-et teszel, hogy 100-at nyerj. Már a nullszaldóhoz 52,4%-os találati arány kell.
- **Záróvonal (closing line):** az utolsó fogadási vonal a kezdés előtt — a piac leginformáltabb ára.
- **CLV (closing line value):** mennyivel volt jobb a vonal, amelyen a modell "lépett", mint a záróvonal. A tartósan pozitív CLV a hosszú távú fogadási profit legerősebb ismert előrejelzője.
- **ROI (fix tétes):** egységnyi tétre jutó profit, ha minden edge-jelzésre azonos tétet tennénk -110-en.
- **Moneyline:** egyszerűen a győztes megtippelése (spread nélkül).

**Adat-kifejezések**

- **PBP (play-by-play):** minden meccs minden játéka egy-egy sor, ~370 oszloppal (forrás: nflverse/nflfastR).
- **EPA (expected points added):** mennyivel mozdította el egyetlen játék a csapat várható pontszámát — az NFL-analitika standard hatékonysági "valutája".
- **WEPA (súlyozott EPA):** EPA, ahol a játékok aszerint kapnak súlyt, mennyire *jelzik előre* a jövőbeli teljesítményt (a lefutott meccsvégi játékok lefelé, a normál passzok felfelé súlyozva, az nfelo kutatása alapján).
- **Success rate:** a pozitív EPA-jú játékok aránya.
- **NGS (Next Gen Stats):** NFL tracking-adatok (elkapó-elválás, passzidő, …). Egy héttel késleltetve használjuk.
- **PFR / FTN:** Pro Football Reference advanced statisztikák; FTN kézi charting-adatok.
- **Snap count / depth chart:** ki hány játékban volt ténylegesen pályán, és hol áll a keretben — a sérülés/folytonosság feature-ök bemenetei.
- **Befagyasztott réteg:** a lezárt szezonok commitolt `data/frozen/` pillanatképe (soha nem töltődik le újra).

**Modellezési kifejezések**

- **Feature:** egy bemeneti oszlop, amit a modell lát (pl. "a hazai csapat támadó EPA/játék értéke az utolsó 8 meccsén").
- **Allowlist:** azoknak a feature-névmintáknak a kifejezett listája, amelyeket a modell *láthat*. Ami nincs rajta, kiesik — a biztonságos fordítottja annak, mint tiltólistázni az ismert szivárgókat.
- **Adatszivárgás (data leakage):** bármely út, amelyen a megjósolandó meccs saját kimenetelének információja eléri a modellt tanításkor vagy predikciókor. A szivárgás felfújja a tesztpontosságot és tönkreteszi az éleset.
- **Késleltetett / gördülő ablak (r4/r8/r16):** az előző 4/8/16 meccs átlagai, egy meccsel eltolva, hogy az aktuális meccs sose szerepeljen a saját feature-eiben.
- **Elo:** meccsről meccsre frissülő erősorrend-érték az eredményből és a különbségből. Kétfázisú frissítés = előbb mindkét csapat *meccs előtti* értéke rögzül, csak utána frissül bármelyik.
- **Imputálás:** a hiányzó értékek pótlása (itt mediánnal) azoknak a modelleknek, amelyek nem kezelik a NaN-t (Ridge, NN). A fák valódi NaN-t kapnak helyette.
- **Ensemble / stacking / meta-learner:** több modell egymástól függetlenül jósol; egy kis záró modell (a meta-learner) megtanulja kimeneteik legjobb keverékét.
- **OOF (out-of-fold):** olyan predikció, amelyet a modell általa nem látott adatra ad, keresztvalidációval előállítva — az egyetlen becsületes bemenet egy meta-learner tanításához.
- **TimeSeriesSplit:** keresztvalidáció, amely mindig a múlton tanít és a jövőn validál (nincs keverés) — ahogy a modellt a valóságban is használjuk.
- **Backtest:** a múlt szimulálása — tanítás csak az X szezon előtti adatokon, X szezon megjóslása, összevetés a tényekkel.
- **Bayes-optimalizálás (Optuna):** irányított keresés, amely ígéretes paraméter-kombinációkat javasol ahelyett, hogy mindet kipróbálná. Itt a minta-súlyokat hangolja tanító/validációs szezonfelosztáson — 6 keresett (ebből 2 stabilitás-szűkített sávban), 2 a walk-forward vizsgálat által rögzített.
- **Minta-súlyok:** mennyit számít egy-egy tanítómeccs (a frissebb szezonok többet; a rájátszás-meccstípusok másképp).
- **Variancia-kalibráció:** a becsült különbségek átskálázása, hogy szórásuk a valóságéhoz igazodjon (a kevert modellek az átlag felé húznak).
- **MAE (átlagos abszolút hiba):** az átlagos tévedés nagysága, pontban.
- **Megbízhatósági besorolás (High/Medium/Low/Weak):** *nem* győzelmi valószínűség — azt méri, mennyire bízik a modell a saját predikciójában: al-modellek egyetértése (55%), bemeneti adatok teljessége (30%), H2H mintanagyság (15%).
- **Edge-jelzés:** csak akkor jelenik meg, ha a megbízhatóság legalább Medium **és** a modell 3+ ponttal tér el a Vegas-i vonaltól.

---

## Adatforrások

Mind ingyenes és nyilvános: [nflverse](https://github.com/nflverse) release-fájlok (play-by-play, játékos-statok, keretek, snap countok, sérülések, depth chartok, FTN, PFR, Next Gen Stats, játékosok, szerződések, draft, combine), [Lee Sharpe / nfldata](https://github.com/nflverse/nfldata) (meccsek és menetrend stadion/időjárás mezőkkel, win totalok, bírók, scoring line-ok), [mrcaseb/nfl-data](https://github.com/mrcaseb/nfl-data) (történelmi fogadási vonalak), [DynastyProcess](https://github.com/dynastyprocess/data) (játékos-azonosító térkép).

## Technológia

Python · pandas · scikit-learn · XGBoost · LightGBM · PyTorch · Optuna · GitHub Actions · GitHub Pages

---

*Készítette: [@llou77](https://github.com/llou77) · A predikciók a szezon alatt minden kedden (és szerda este) automatikusan frissülnek.*
