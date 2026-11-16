---
name: Sentiment Study Dashboard
description: Results-night broadcast for a six-model, four-dataset sentiment classification study.
colors:
  studio-indigo: "#080c2a"
  studio-glow: "#2b3a9a"
  studio-mid: "#121a52"
  panel-indigo: "#0e1442"
  panel-indigo-raised: "#141b55"
  hairline-indigo: "#29337f"
  hairline-indigo-soft: "#1c2463"
  studio-ink: "#eef0ff"
  haze-ink: "#bcc4f0"
  dim-ink: "#6f79bd"
  broadcast-yellow: "#ffcc3d"
  broadcast-yellow-ink: "#0b0f33"
  broadcast-yellow-text: "#ffd65c"
  gain-green: "#3dffb0"
  loss-rose: "#ff6b8a"
  signal-blue: "#7fb2ff"
  link-blue: "#9cc2ff"
  svm-cyan: "#3de0ff"
  lr-amber: "#ffb23d"
  nb-lime: "#a6f04a"
  rf-pink: "#ff6fb5"
  dt-violet: "#b294ff"
  knn-coral: "#ff7a7a"
  projector-ground: "#eef0fa"
  projector-glow: "#c9d0fb"
  projector-mid: "#e2e5fb"
  projector-panel: "#ffffff"
  projector-panel-raised: "#f6f7ff"
  projector-hairline: "#d3d8ef"
  projector-hairline-soft: "#e5e8f6"
  projector-ink: "#0b1033"
  projector-ink-2: "#444c7e"
  projector-ink-3: "#a3aad3"
  projector-yellow-text: "#7a5000"
  projector-gain: "#0b8a5b"
  projector-loss: "#c42149"
  projector-link: "#2357c8"
  projector-svm: "#0784a8"
  projector-lr: "#b8620a"
  projector-nb: "#3f7d0c"
  projector-rf: "#c42a76"
  projector-dt: "#6343d6"
  projector-knn: "#cf3434"
typography:
  display:
    fontFamily: "'Archivo Variable', 'Segoe UI', system-ui, sans-serif"
    fontSize: "clamp(3rem, 5.6vw, 4.6rem)"
    fontWeight: 900
    lineHeight: 0.9
    letterSpacing: "0.005em"
    fontVariation: "'wdth' 62"
  headline:
    fontFamily: "'Archivo Variable', 'Segoe UI', system-ui, sans-serif"
    fontSize: "clamp(2.2rem, 4.4vw, 3.4rem)"
    fontWeight: 850
    lineHeight: 1
    letterSpacing: "0.005em"
    fontVariation: "'wdth' 66"
  numeral:
    fontFamily: "'Archivo Variable', 'Segoe UI', system-ui, sans-serif"
    fontSize: "clamp(3rem, 4.4vw, 4rem)"
    fontWeight: 900
    lineHeight: 0.95
    fontFeature: "'tnum'"
    fontVariation: "'wdth' 62"
  model-name:
    fontFamily: "'Archivo Variable', 'Segoe UI', system-ui, sans-serif"
    fontSize: "1.75rem"
    fontWeight: 800
    lineHeight: 1
    fontVariation: "'wdth' 72"
  title:
    fontFamily: "'Archivo Variable', 'Segoe UI', system-ui, sans-serif"
    fontSize: "1.15rem"
    fontWeight: 750
    lineHeight: 1.5
  body:
    fontFamily: "'Archivo Variable', 'Segoe UI', system-ui, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "'Archivo Variable', 'Segoe UI', system-ui, sans-serif"
    fontSize: "0.92rem"
    fontWeight: 650
    lineHeight: 1.5
  stamp:
    fontFamily: "'Archivo Variable', 'Segoe UI', system-ui, sans-serif"
    fontSize: "0.78rem"
    fontWeight: 850
    letterSpacing: "0.08em"
  mono:
    fontFamily: "Consolas, 'Cascadia Mono', monospace"
    fontSize: "0.88em"
rounded:
  tag: "4px"
  stamp: "5px"
  cell: "8px"
  control: "10px"
  table: "12px"
  panel: "14px"
  card: "18px"
  pill: "999px"
spacing:
  gutter: "16px"
  gap: "24px"
  panel: "24px"
  panel-mobile: "18px"
  section: "72px"
  section-mobile: "56px"
  page-max: "1240px"
components:
  button-primary:
    backgroundColor: "{colors.studio-ink}"
    textColor: "{colors.studio-indigo}"
    rounded: "{rounded.control}"
    padding: "10px 18px"
    height: "44px"
  button-ghost:
    backgroundColor: "{colors.panel-indigo-raised}"
    textColor: "{colors.studio-ink}"
    rounded: "{rounded.control}"
    padding: "10px 18px"
    height: "44px"
  button-link:
    textColor: "{colors.link-blue}"
    padding: "10px 8px"
  input:
    backgroundColor: "{colors.panel-indigo-raised}"
    textColor: "{colors.studio-ink}"
    rounded: "{rounded.control}"
    padding: "10px 12px"
    height: "44px"
  nav-tab:
    textColor: "{colors.haze-ink}"
    rounded: "{rounded.pill}"
    padding: "8px 14px"
  nav-tab-active:
    backgroundColor: "{colors.broadcast-yellow}"
    textColor: "{colors.broadcast-yellow-ink}"
    rounded: "{rounded.pill}"
    padding: "8px 14px"
  segmented-option-active:
    backgroundColor: "{colors.broadcast-yellow}"
    textColor: "{colors.broadcast-yellow-ink}"
    rounded: "7px"
    padding: "7px 14px"
  chip:
    backgroundColor: "{colors.panel-indigo-raised}"
    textColor: "{colors.studio-ink}"
    rounded: "{rounded.pill}"
    padding: "7px 13px"
  panel:
    backgroundColor: "{colors.panel-indigo}"
    rounded: "{rounded.panel}"
    padding: "{spacing.panel}"
  winner-card:
    backgroundColor: "{colors.panel-indigo-raised}"
    rounded: "{rounded.card}"
    padding: "24px 24px 22px"
  winner-stamp:
    backgroundColor: "{colors.broadcast-yellow}"
    textColor: "{colors.broadcast-yellow-ink}"
    typography: "{typography.stamp}"
    rounded: "{rounded.stamp}"
    padding: "5px 10px"
  win-pill:
    backgroundColor: "{colors.broadcast-yellow}"
    textColor: "{colors.broadcast-yellow-ink}"
    rounded: "{rounded.tag}"
    padding: "2px 7px"
  results-ticker:
    backgroundColor: "{colors.broadcast-yellow}"
    textColor: "{colors.broadcast-yellow-ink}"
    rounded: "{rounded.table}"
    height: "50px"
---

# Design System: Sentiment Study Dashboard

## Overview

**Creative North Star: "Counting Night"**

The dashboard reads like a results-night broadcast. The study is a fair contest with declared outcomes, so every dataset is announced with its winner, its accuracy and its margin over the runner-up. The stage is a deep studio-indigo ground lit from above by a single soft radial glow; on it sit rounded indigo panels with soft, low depth, condensed heavy numerals, and six fixed model colours that act as each classifier's on-screen identity.

Density is moderate and evidence-led: big declared numbers first (the "6 models. 4 datasets. 24 results." headline, the tally, four WINNER cards, the ticker), then the heatmap, the speed/accuracy trade-off, the full sortable table, the live predictor where six models vote, and the method. Motion is broadcast-graphic: bars grow, stamps land, the ticker crawls, cards tilt slightly in 3D on hover. All of it stops under reduced motion.

There are two themes. Light is the default (best for projector rooms and first-time visitors) and swaps broadcast yellow for a deep amber because raw yellow fails on white; dark ("night") is an optional toggle that keeps the same roles with its original studio-indigo ground and broadcast yellow.

**Key Characteristics:**
- Studio-indigo ground with one overhead radial glow; no page-wide decorative grid.
- Six fixed model colours used only as data series and model identity.
- Broadcast yellow reserved for declared results and the active selection.
- Archivo variable throughout: condensed (62 to 72% width) heavy display and numerals, normal-width body, tabular figures on every number.
- Rounded panels (14px) and cards (18px) with soft, blurred depth; hairlines only inside the heatmap and tables.

## Colors

A cool indigo night with one hot broadcast yellow and six saturated identity hues for the models.

### Primary
- **Broadcast Yellow** (`broadcast-yellow`): the declared-result and active-selection colour. WINNER stamps, WIN pills in the results table, the heatmap winner ring, the results ticker, the active nav tab, the active segmented option, the selected history item, the active stepper dot, the score-meter marker, focus rings and text selection. Text placed on it is always **Broadcast Ink** (`broadcast-yellow-ink`).
- **Broadcast Yellow Text** (`broadcast-yellow-text`; projector: `projector-yellow-text`): yellow as type on the ground, for the "24 results." headline line, the vote-count numeral and the sentiment-score numeral. The light theme swaps to a deep amber because raw yellow fails on white.

### Secondary (model identity)
Each classifier has one fixed hue per theme, defined in `src/lib.js` and handed to charts through the palette context:
- **SVM Cyan** (`svm-cyan` / `projector-svm`): Linear SVM.
- **LR Amber** (`lr-amber` / `projector-lr`): Logistic Regression.
- **NB Lime** (`nb-lime` / `projector-nb`): Multinomial Naive Bayes.
- **RF Pink** (`rf-pink` / `projector-rf`): Random Forest.
- **DT Violet** (`dt-violet` / `projector-dt`): Decision Tree.
- **KNN Coral** (`knn-coral` / `projector-knn`): KNN.

They colour tally bars, winner-card model names and underglow, swatches, scatter points, bars and vote lists. They are injected per element as a `--c` custom property, never hard-coded in component CSS.

### Tertiary (sentiment and status)
- **Gain Green** (`gain-green` / `projector-gain`): positive sentiment, "connected" status dot, success notices, done steps.
- **Loss Rose** (`loss-rose` / `projector-loss`): negative sentiment, errors, backend-down icon.
- **Signal Blue** (`signal-blue`): info notices only.
- **Link Blue** (`link-blue` / `projector-link`): text links, link buttons and the method step icons.

### Neutral
- **Studio Indigo** (`studio-indigo` / `projector-ground`): page ground.
- **Studio Glow / Studio Mid** (`studio-glow`, `studio-mid`): the stops of the single overhead radial light (1200 by 620px, centred above the top edge) and the translucent top-bar tint.
- **Panel Indigo** (`panel-indigo` / `projector-panel`): panels, footer, steps.
- **Panel Indigo Raised** (`panel-indigo-raised` / `projector-panel-raised`): controls, chips, inner list rows, table headers, tooltips, the upper stop of the winner-card gradient.
- **Hairline Indigo** (`hairline-indigo`, `hairline-indigo-soft`): control borders and the soft panel/table/heatmap hairlines.
- **Studio Ink, Haze Ink, Dim Ink** (`studio-ink`, `haze-ink`, `dim-ink`; projector: `projector-ink`, `projector-ink-2`, `projector-ink-3`): primary text, secondary text, and ornament (step numerals, idle placeholder word, empty-state icons, quote marks, brand suffix).

### Heat ramp
Heatmap accuracy maps onto a four-stop ramp per theme (dark: `#161c55` to `#2c3a9c` to `#6f88ea` to `#e6ecff`; light: `#eef1fb` to `#c4cdee` to `#5465c4` to `#151f66`) over 50% to 91%, with the cell ink flipping at 62% (dark) or 50% (light) of the ramp. The ramp stays inside the indigo family so model colours and yellow remain the only loud hues.

### Named Rules
**The Declared Result Rule.** Broadcast yellow marks only a declared result or the current selection (including focus). Never use it for a primary button, a decorative fill, a heading accent, or a chart series.

**The Identity Rule.** A model's colour means that model, everywhere, in both themes. Model hues are never used for decoration, status, or sentiment.

**The Ink Button Rule.** Primary buttons are ink on ground (`studio-ink` fill, `studio-indigo` text), so the brightest call to action never competes with the results.

## Typography

**Display Font:** Archivo Variable, self-hosted with its width axis (with Segoe UI, system-ui fallback)
**Body Font:** Archivo Variable at normal width
**Label/Mono Font:** Consolas / Cascadia Mono for inline code only

**Character:** One family doing two jobs. Squeezed to 62 to 72% width at 800 to 900 weight it becomes broadcast-graphic numerals and uppercase headlines; at normal width and 400 to 750 it is a plain, legible reading face.

### Hierarchy
- **Display** (900, `clamp(3rem, 5.6vw, 4.6rem)`, 0.9, width 62%, uppercase): the first-viewport headline, one line per clause. Report names and the predictor verdict word use the same recipe (64% / 62%) at their own sizes.
- **Headline** (850, `clamp(2.2rem, 4.4vw, 3.4rem)`, 1, width 66%, uppercase): section titles.
- **Numeral** (900, `clamp(3rem, 4.4vw, 4rem)`, 0.95, width 62%, tabular): the declared accuracy on winner cards; the meter score (3rem) and vote count (2.4rem) follow it.
- **Model name** (800, 1.75rem, width 72%): the model name on a winner card, in its identity colour. Tally counts and the overall leader use 850 at 70 to 72%.
- **Title** (750, 1.15rem): panel headings at normal width.
- **Body** (400, 1rem, 1.5): prose; the lede holds to 44ch, findings to 80ch.
- **Label** (650, 0.92rem): form labels, panel sub-labels, credits; secondary text at 0.875rem.
- **Stamp** (850, 0.78rem, 0.08em, uppercase): the WINNER stamp; WIN and sentiment pills follow at 0.7 to 0.82rem with 0.04 to 0.06em tracking.

#### Documented exceptions (intentional, off the steps above)
A handful of sizes recur often enough, or serve a single clearly separate purpose, that they are kept as-is rather than forced onto the nearest step:
- **1.45rem** (850, condensed 70–72%): the brand logotype, tally counts (`.tally b`) and the predictor vote-side counts (`.vote-side b`) — a count/identity size between Model name (1.75rem) and Title (1.15rem).
- **0.95rem**: nav tabs (`.tabs a`), inline notices (`.notice`), the tally title and tally row labels (`.tally-title`, `.tally li`), and the chart tooltip's bold line (`.chart-tip > b`) — a "medium copy" tier between Body (1rem) and Label (0.92rem).
- **0.88rem**: chip-style controls (`.chip-btn`, `.stepper li`, `.tone-chip`, `.dot-key`) — pairs loosely with Mono's `0.88em`.
- **0.85rem**: the heatmap legend, results-table sort buttons, the chart tooltip body and the mobile heatmap row names (`.heat-legend`, `.sort`, `.chart-tip`, `.heatmap tbody th` under 760px).
- **1.05rem**: the overview lede paragraph and the method step body copy (`.lede`, `.step h3`) — a touch larger than Body for a standalone intro line.
- **1.25rem**: the footer title (`.footer-title`), a one-off between Title and Model name.
- **`clamp(2rem, 4vw, 3rem)`**: the "backend is not running" error heading (`.backend-down h2`) — its own, smaller Headline-recipe size for a secondary screen.
- **`clamp(2.4rem, 5vw, 3.8rem)`**: the product-report brand name (`.report-name`), and **`clamp(3.6rem, 7vw, 5.6rem)`**: the predictor verdict word (`.verdict-word`) — the Display recipe (900 weight, 62–64% width) at their own sizes, as noted above.

### Named Rules
**The Tabular Figures Rule.** Every number that can be compared carries tabular figures (`.num`, `tnum`). Numbers are formatted to two decimals with half-up rounding so displayed values match the backend.

**The Two Widths Rule.** Condensed width is for numerals, display headlines and model names only. Body, labels and controls stay at normal width.

## Layout

A single scrolling page under a sticky top bar, centred at up to 1240px with a 16px gutter each side (`min(1240px, 100% - 32px)`). Sections stack with 72px top padding (56px below 760px) and a 24px internal gap; `scroll-padding-top` of 84px keeps anchors clear of the bar.

The first viewport is a two-column grid (a 320 to 430px headline-and-tally column, then a 2 by 2 grid of WINNER cards) followed by a full-width overall strip and the results ticker. Compare, Predict and the product report use two-column grids (roughly 1.1 to 1.5 : 1) that collapse to one column at 1080px; the report grid goes 3, 2, then 1 column. At 760px the tabs wrap to a full-width scrolling row, panels drop to 18px padding, filters become two columns with search spanning both, and the heatmap switches from long names to dataset and model codes. The method pipeline and topic rail are horizontal scroll-snap rails.

Below 600px the speed/accuracy scatter drops its in-plot labels and shows a colour key (dot list) beneath the chart instead.

## Elevation & Depth

Depth is soft and lifted, never hard. Panels sit on one shared blurred offset shadow; winner cards get a deeper shadow plus a 1px inset highlight tinted with their model colour and a 3px glowing underline. Coloured glows (`0 6px 16px -6px` in the element's own hue) appear under tally bars, vote bars, the connection dot, the ticker and the verdict icon. Hover raises: buttons lift 2px, heat cells lift and scale, winner cards tilt in 3D. The tally and vote tracks are tilted back 22 degrees on X for a stage-floor perspective.

### Shadow Vocabulary
- **Panel** (`0 18px 40px -22px rgb(0 0 0 / 80%)`; light: `0 14px 34px -20px rgb(20 28 90 / 35%)`): panels, steps, tooltips.
- **Winner card** (`0 24px 46px -26px rgb(0 0 0 / 85%)` plus model-tinted inset top highlight): the declared cards.
- **Button lift** (`0 10px 24px -14px rgb(0 0 0 / 70%)`, hover `0 16px 30px -14px`): primary button.
- **Winner ring** (`0 0 0 3px panel, 0 0 0 5px broadcast-yellow`): the heatmap cell that won its dataset.

### Named Rules
**The Soft Depth Rule.** Shadows are always blurred and negatively spread. No hard offset shadows, no solid drop blocks.

## Shapes

Rounded and friendly but not bubbly. Panels 14px, winner cards and skeleton cards 18px, controls 10px, tables and blockquotes 12px, heat cells 8px, stamps 5px, WIN pills 4px. Nav tabs, chips, key pills, file pills, sentiment pills and the stepper are full pills. The overall strip uses the panel radius (14px) with a dashed hairline border; everything else is solid 1px. The visible hairline grid exists only inside the heatmap (collapsed 1px cell borders with outer edges removed) and the results table.

**Documented radius exceptions:**
- **6px**: the focus ring corner (`:focus-visible`), the results-table accuracy chip (`.acc-chip`) and the sentiment meter track (`.meter-track`) — a compact rounding between WIN pills (4px) and heat cells (8px), used for small inline chrome rather than cards or controls.
- **3px**: the colour swatch dot (`.swatch`), the winner card's glowing underline (`.winner::after`), the predictor target pill (`.target-pill`) and the per-model vote bars (`.vote-track i`) — proportionally tiny rounding on elements under ~22px tall, where even 1px changes the visible curve.

## Components

### Buttons
Confident and quiet; the result is louder than the button.
- **Shape:** gently rounded (10px), minimum 44px tall, 700 weight.
- **Primary:** ink fill with ground-colour text and a soft lift shadow; hover rises 2px.
- **Ghost:** raised-panel fill with a hairline border; hover shifts the border colour.
- **Link:** link-blue text, underline on hover.
- **Icon button:** 40px circle with hairline border; the theme toggle rotates -12 degrees on hover.
- **Disabled:** 50% opacity, not-allowed cursor.

### Chips
- **Example chips:** full pills on raised panel with hairline border, 600 weight, 0.88rem; hover lifts 1px.
- **Sentiment pills / tone chips:** full pills tinted 16 to 18% with gain-green or loss-rose text, uppercase on the predictor.

### Cards / Containers
- **Corner Style:** 14px panels.
- **Background:** panel indigo with a soft hairline border.
- **Shadow Strategy:** the shared panel shadow.
- **Internal Padding:** 24px (18px on small screens).

### Inputs / Fields
- **Style:** raised-panel fill, hairline border, 10px radius, 44px minimum height; selects use a CSS chevron; the file input is wrapped in a dashed drop zone.
- **Focus:** a 3px broadcast-yellow outline at 2px offset, and a yellow caret.
- **Disabled:** 55% opacity.

### Navigation
A sticky 64px top bar tinted with translucent studio-mid and a 14px backdrop blur. The brand is set condensed (68%, 850) with a dim suffix. Section tabs are pill links in haze ink; hover fills raised panel; the active tab is a broadcast-yellow pill with ink text. A connection dot (green when up, rose when down) and the theme toggle sit at the end. Below 760px the tabs move to their own full-width horizontally scrolling row.

### Winner Card (signature)
A declared result. Gradient from raised panel to panel, 18px radius, deep soft shadow, a model-tinted inset top edge and a 3px glowing underline in the model colour. A tilted broadcast-yellow WINNER stamp lands top-right; below it, dataset name (label), model name (condensed, model colour), accuracy (numeral) and the margin over the runner-up.

### Results Ticker (signature)
A 50px broadcast-yellow band with an ink "RESULTS" tag block, crawling the declared results at a 38s linear loop, edge-faded with a mask, paused on hover and made scrollable under reduced motion.

### Heatmap
Accuracy per dataset and model on the indigo heat ramp inside a visible hairline grid, with contrast-flipped cell ink. The winning cell carries the yellow winner ring; the matching row in the results table gets a WIN pill and a 7% yellow tint.

### Segmented Control
A raised-panel track with 4px padding; options are bare buttons; the active option is a broadcast-yellow block with ink text.

## Do's and Don'ts

### Do:
- **Do** reserve broadcast yellow for declared results (stamps, WIN pills, winner rings, the ticker, result numerals) and the active selection (tab, segmented option, selected item, focus).
- **Do** colour each model with its fixed identity hue from the palette, in both themes, and only for that model.
- **Do** set numbers in tabular figures, two decimals, half-up, exactly as the backend reports them.
- **Do** use condensed Archivo (62 to 72% width, 800 to 900) for numerals, display headlines and model names, and normal width for everything else.
- **Do** keep primary buttons ink-on-ground and links in link blue.
- **Do** keep depth soft: blurred, negatively spread shadows, 14px panels, 18px cards.
- **Do** give every animation a reduced-motion off switch and keep the ticker readable without motion.
- **Do** maintain the light theme with its own contrast-safe model hues and amber yellow text for projector rooms.

### Don't:
- **Don't** use broadcast yellow for primary buttons, decorative fills, chart series or heading accents.
- **Don't** draw a page-wide decorative grid; the hairline grid lives only in the heatmap and tables.
- **Don't** use model colours for status, sentiment, or decoration.
- **Don't** fall back to white KPI tiles with a purple accent.
- **Don't** use hard offset shadows.
- **Don't** set long text or essential information in dim ink; it is for ornament and large decorative numerals.
- **Don't** crowd the speed/accuracy scatter with labels on narrow screens; show a colour key below 600px.
