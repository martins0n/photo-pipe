# Classic Chrome on the Sony A6700

Sony has no film simulations. The equivalent system is **Creative Look**, and
it has enough adjustment to get most of the way to Classic Chrome — the look
is largely *subtraction* (less saturation, restrained highlights), which is
what these sliders do well.

The values below were **measured, not guessed**: `photo-pipe fit-camera` solves
for the Creative Look sliders that best turn a neutral rendering into the
`classic-chrome` recipe, using your own frames. See
[How this was derived](#how-this-was-derived) — and note it overturned several
plausible-sounding guesses, so it's worth trusting over intuition.

Two things to know first:

- Creative Look applies to **JPEG/HEIF only**. Your ARW is untouched, so this
  only changes the `Camera SOOC` column of the comparison sheet.
- Creative Look and **Picture Profile are mutually exclusive** — if a Picture
  Profile is set it wins. Set PP to `OFF`.

## The recipe

Start from **ST** (Standard). Not FL, and not the **IN** your camera is on now
— the fit measures the look relative to a *neutral* base, and ST is the
closest thing the A6700 has to one.

`MENU → Shooting → Image Quality → Creative Look → ST`, then press right to
open the parameters:

| Parameter | Set to | Why |
|---|---|---|
| Contrast | **+1** | very stable across every shoot tested |
| Highlights | **−7** | the soft, non-clipping shoulder — a big part of the look |
| Shadows | **0** | Classic Chrome's shadows are shaped by the curve, not lifted |
| Fade | **0** | see the note below — this one is counter-intuitive |
| Saturation | **−8** | the muted palette, and it needs to be this strong |
| Clarity | **+1** | where you already are; leave it |
| Sharpness | default | Classic Chrome isn't a sharpness look |

White balance shift (`White Balance → AWB →` press right): **A0, M1** — or
just leave it at 0,0. It barely moves and is doing something slightly
dishonest; see the caveat below.

Set **DRO: OFF**. DRO fights the highlight shoulder you just built.

Save it to a slot so you don't re-enter it: `Creative Look → C1..C6`.

### Two results worth flagging

**Fade should be 0, not lifted.** Lifting the black point is the obvious move
for a "matte film" look and it is wrong here. Across three daylight shoots the
fit asked for Fade 0, 0, +1. Classic Chrome's shadows get their character from
a *dip* in the low-mids followed by a normal black point, not from a raised
floor — and Fade raises the floor uniformly, which reads as washed-out rather
than muted. The `curve:` block in
[`recipes/classic-chrome.yaml`](../recipes/classic-chrome.yaml) has the shape.

**Saturation has to go much further than feels right.** −8 of 9 looks
extreme in the menu, but Classic Chrome is genuinely desaturated and anything
less leaves it looking like a slightly flat Standard.

## How this was derived

```bash
photo-pipe fit-camera ~/Images/2026-07-30 --recipe classic-chrome
```

The command has both ends of the problem already: your camera's SOOC HEIF, and
the pipeline's `classic-chrome` render of the same frame. It models each Sony
slider as a tone/colour operation and solves for the values that best bridge a
neutral render to the recipe, scoring in Lab.

Two corrections were needed to make the fit mean anything, both instructive:

1. **Exposure had to be factored out.** The pipeline's auto-exposure and the
   camera's metering disagree, and no Creative Look slider changes exposure —
   so the optimiser pinned Highlights and Shadows at their limits trying to
   express a brightness offset. Baseline error fell by half once exposure was
   aligned first.
2. **The fit runs against our own neutral render, not against the SOOC.**
   Fitting SOOC → recipe also asks the sliders to absorb the difference
   between Sony's colour science and LibRaw's, which they cannot do. Removing
   that took baseline error from 198 to 20 and the fit from 37% to 75–86%.

Agreement across four shoots in different light (48 daylight frames):

| | 07-27 FL | 07-30 VV | 08-01 VV | 08-14 IN (night) |
|---|---|---|---|---|
| Contrast | +1 | +1 | +1 | +2 |
| Highlights | −8 | −5 | −9 | −9 |
| Shadows | 0 | 0 | −1 | −1 |
| Fade | 0 | 0 | +1 | 0 |
| Saturation | −7 | −9 | −8 | −7 |
| Clarity | +1 | +1 | +1 | 0 |
| gap closed | 75% | 86% | 78% | 83% |

The night column is the odd one out on white balance (it wanted A−6, M−7).
Mixed sodium and LED street lighting makes a global WB fit unstable, so the
daylight consensus is the one to use.

## Caveats worth knowing

**The white balance shift is standing in for something else.** Classic
Chrome's signature is a per-hue rotation — reds toward brick, greens toward
olive, blues held. Creative Look has no per-hue controls, so the fit
approximates "desaturate the greens" with the nearest global move it has, a
slight magenta shift. On camera that tints *everything* magenta, including
skin. That is why the recommendation is A0/M1 or nothing: the honest version
of that adjustment lives in the `hsl:` block of the recipe and is a good
reason to keep shooting raw alongside.

**ST is not exactly our neutral.** The sliders were measured against the
pipeline's `provia` render. Sony's ST is comparable but not identical, so
expect to nudge Saturation and Highlights by a step or two.

**Roughly 20% of the look is out of reach** — that is the per-hue character
above, and it is what the raw pipeline is for.

## Dialling it in

Shoot a few frames with the settings above, then:

```bash
photo-pipe run ~/Images/<date> --recipes classic-chrome --collage --preview
```

The sheet puts your SOOC next to the recipe on the same frame.

| What you see vs the recipe column | Change |
|---|---|
| SOOC still too colourful | Saturation −1 |
| SOOC highlights too bright / clipping | Highlights −1 |
| SOOC too flat and washed | Contrast +1, and check Fade is 0 |
| SOOC too contrasty in the shadows | Shadows +1 |
| SOOC greens too vivid | one step toward M (accepting the caveat above) |
