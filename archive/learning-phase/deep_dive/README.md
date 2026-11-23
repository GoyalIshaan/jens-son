# Keep the Array Busy

A deep dive on AI chip architecture, built from the four PDFs in `claude_research/`, with 3D labs and a tested numeric model.

## Start here

| You want | Open |
| --- | --- |
| The whole thing boiled down | `docs/00_insights.md` |
| Interactive site with 3D labs | `site/dist/index.html` |
| How to tune an array | `docs/03_systolic.md`, `docs/04_shape.md` |
| How designers evolved chips | `docs/12_generations.md` |
| A procedure for your own goals | `docs/13_method.md` |
| Your RTL through this lens | `docs/14_yourchip.md` |
| LUTs, nonlinear units and MoE on your chip | `docs/18_nonlinear.md` |
| What was wrong in the sources | `docs/17_audit.md` |

## Layout

- `docs/`: 19 chapters in Markdown. These are the source of truth.
- `model/model.js`: the analytical model used by every lab.
- `model/test_model.js`: checks the model against worked numbers from the PDFs.
- `model/nonlinear.js`: integer experiments behind the LUT, softmax and MoE numbers.
- `model/sim_array.js`: beat-by-beat array simulators that check the timing, packing and overlap rules with real values.
- `site/src/`: stylesheet, Three.js lab framework, the labs, chapter routing.
- `site/build.py`: builds `site/dist/` from the chapters and sources.

## Commands

Run the model checks:

```bash
node deep_dive/model/test_model.js
```

Rebuild the site (needs Python with the `markdown` package):

```bash
python deep_dive/site/build.py
```

Serve it locally, since a file:// open also works but fonts and Three.js need a network connection either way:

```bash
python3 -m http.server 8765 --directory deep_dive/site/dist
```

## How to read the evidence tags

- **J** Jouppi et al., the Ironwood paper
- **R** Roune, *Designing AI Chip Hardware and Software*
- **IA** Irrational Analysis, an unfinished draft by an author with disclosed Nvidia positions
- **E** The Component Encyclopedia
- **F** Your factbook in `research/`
- **D** Derived here and tested in `model/`
- **O** Outside the four PDFs: general engineering knowledge with ballpark values

Numbers tagged O and the die lab's constants are teaching estimates. Nothing here comes from synthesis, static timing analysis or silicon measurement. Your RTL was not modified.
