# Finish-the-project roadmap (7 prompts)

Goal: a **working, submittable** project (not production-grade). Each prompt builds on the previous one.

| # | Prompt | Outcome | Needs from you |
|---|---|---|---|
| 1 | **Environment setup** (DONE) | `.venv` created, core deps installed, heavy torch deps split out, 47 tests pass (4 optional skipped), all 24 saved models predict, backend + frontend run and talk to each other, lint/build clean | nothing |
| 2 | **Groq key + product report end-to-end** (DONE, merged with 3) | Key wired in `.env`, Reddit and file-upload flows tested live, rate-limit / error messages fixed | A free Groq API key (console.groq.com) |
| 3 | **Working-flow polish** (DONE, merged with 2) | Friendly states when the API is down or loading, predictor UX (examples, copy, confidence), report history, no dead buttons | nothing |
| 4 | **UI enhancement, part 1** | Visual redesign of the dashboard: theme, layout, typography, responsive/mobile | Any look you want (colours, reference screenshot) |
| 5 | **UI enhancement, part 2** | Charts, tables and the product-report page polished; demo-friendly empty states | nothing |
| 6 | **Repo cleanup and consistency** (DONE) | Phase 4 transformer work labeled "Future work (started): transformer extension, 3 of 24 runs complete, not part of the frozen baseline" everywhere it's mentioned; `models/` vs `.gitignore` mismatch fixed so the 24 saved `.joblib` models commit; README/docs/viva guide made to agree with the code; CSS tidied against `DESIGN.md`; scroll/loading animations added | nothing |
| 7 | **Deploy + submission package** | Live link (Render + Vercel) if required, final end-to-end check, clean submission ZIP without `.env`, `.venv`, `node_modules` | GitHub/Render/Vercel logins only if a live link is wanted |

> **Phase 4 transformer extension:** Future work (started) — 3 of 24 planned runs complete. Not part of the frozen 24-experiment classical baseline. See `docs/PHASE4_BENCHMARK_EXECUTION.md` for the full protocol and current state.

Notes:
- Prompts 3 and 4 and 5 can be merged if you want fewer prompts (UI + flow in one).
- Reddit often blocks unauthenticated requests from cloud IPs; on a hosted deployment the file-upload path is the reliable one (checked in prompt 2).
