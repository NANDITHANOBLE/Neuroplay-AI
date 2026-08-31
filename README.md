- [ ] Repo pushed to `origin/main` *(confirm this succeeded from earlier)*
- [ ] `CONTRIBUTING.md` added
- [ ] Issue template added
- [ ] CI badge added to README
- [ ] Phase 2 commit pushed to `main`
- [ ] CI workflow runs successfully (check **Actions** tab on GitHub)

---

## 📌 Note for Later

Add to `ROADMAP.md` or mentally flag: **Phase 25 (Final GitHub Polish)** will introduce:
- `develop` branch + feature branch workflow
- Branch protection on `main`
- PR template
- Required status checks

---

Run Steps 1–5 and paste the output — specifically let me know:
1. Did `git push origin main` succeed?
2. Does the **Actions** tab on GitHub show the CI workflow running (green check or red X)?

Once confirmed, Phase 2 is complete and we'll move to **Phase 3: Database Design**.

## 🗄️ Database Schema

NeuroPlay-AI uses a relational schema (SQLite for dev, PostgreSQL for production):

- **users** — player profiles, aggregate stats
- **matches** — one row per game session
- **moves** — every individual round (the core time-series data)
- **predictions** — logged model outputs per round, with explainability data
- **drift_events** — concept drift detections (ADWIN/DDM)
- **psychology_profiles** — behavioral classification per user

See `docs/ARCHITECTURE.md` for the full ER diagram.

## 🎲 Synthetic Dataset Generation

Since no public dataset exists for human RPS psychology, NeuroPlay-AI bootstraps
model training using 6 research-grounded synthetic personas:
Random, Win-Stay/Lose-Shift, Cyclic, Frequency-Biased, Markov-Order-2, and
Drifting (mid-match strategy switch, used to validate concept drift detection).

Run: `python -m neuroplay.data_generation.generate_dataset`
