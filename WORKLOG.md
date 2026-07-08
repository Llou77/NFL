# WORKLOG

Folyamatos munkanapló — bármelyik későbbi session innen tudja folytatni.
(Konvenció: legfrissebb kör felül.)

---

## 2026-07-08 — "Integrity round": szivárgás-zárás, stabilizálás, takarítás

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
