# Prompt: "Threnody-9B" release trailer (HTML/CSS/JS → MP4)

Copy everything below the line into your video/code-generation tool.

---

Build a 60-second, 1920×1080, 60 fps release trailer for an open-weight AI coding model called **Threnody-9B**, as a single self-contained web page (HTML + CSS + vanilla JavaScript, Canvas 2D/WebGL allowed, no external assets except one Google Font). Then render it to MP4 deterministically.

## The idea (read this first)
A *threnody* is a song of lament for the dead. Threnody-9B is an elegy for every bug it meets. The film is a quiet funeral that turns into a dawn: grief, then precision, then release. It should feel like a requiem written by an engineer: organic and mournful on the surface, exact and mechanical underneath. Restraint over spectacle; every motion must mean something.

**Core symbols (use all, never explain them with on-screen text):**
- **Ink in water**: bugs appear as slow black ink blooms that spread through a pale, paper-like field. They are the "dead" being mourned.
- **The single sustained tone**: one horizontal hairline across the frame that vibrates like a string. It is the model's attention. When it touches ink, the ink is drawn into it and resolved into clean light.
- **Candles → cursors**: small flames at the bottom of the frame that, on a match cut, become blinking terminal cursors. Mourning becomes work.
- **Moth to flame / repetition broken**: a moth circles a flame in the same loop three times (the failure loop). On the fourth pass it changes course. This is the film's emotional turn: determination, not repetition.
- **Nine stones**: nine smooth stones laid in a row by an unseen hand: 9 billion parameters, small enough to hold.
- **Many doorways**: the same corridor seen through different door frames (different harnesses), each with a slightly different typographic style; the tone line passes through all of them unchanged.
- **Dawn**: the palette moves from ash and bone to a single warm ember orange at the end.

## Palette and type
- Ash `#0b0b0c`, bone `#e9e4da`, graphite `#2a2a2e`, ink `#050505`, ember `#c2410c` (used sparingly, only after the turn), cold light `#cfd8dc`.
- Film grain overlay (animated noise, 3–4% opacity), subtle vignette, slight chromatic aberration only on the tone line.
- Typography: one serif for words of grief (e.g. "Cormorant Garamond", light italic) and one monospace for code (e.g. "JetBrains Mono"). Text is sparse, small, letter-spaced, and fades, never slides.

## Sound (generated in-page with Web Audio, original, no samples)
- 0–20 s: a low drone plus a cluster of detuned sine voices a quarter-tone apart, slowly swelling (lament, dissonant, never harsh).
- 20–38 s: the cluster narrows toward a single pitch as bugs resolve; soft keyboard-like ticks (filtered noise bursts) in rhythm with cursor blinks.
- 38–42 s: silence for 1.5 s at the moth's turn, then one clean sine tone.
- 42–60 s: a slow, warm major chord blooms under the tone; ends on a single sustained note that cuts to silence exactly on the logo.
- Export audio by rendering the same graph with OfflineAudioContext to WAV and muxing it in the MP4 step.

## Timeline (seconds)
1. **0–6 · Silence and paper.** Bone-colored field with grain. A single ink drop falls in slow motion and blooms. Text (serif italic, center, fades in/out): *"Every program carries its dead."*
2. **6–14 · The blooms.** More ink blooms appear; inside each, faint monospace fragments surface and dissolve: `TypeError`, `race condition`, `off-by-one`, `deadlock`, `0 tests passed`. Candles flicker along the bottom edge.
3. **14–20 · The tone.** A hairline appears from the left and stretches across the frame, vibrating. Text: *"Threnody: a song for the lost."*
4. **20–30 · The work.** The line moves through the ink; each bloom it touches is pulled into the line and becomes a short clean code line that types itself (`def merge(...)`, `pytest -q`, `14 passed`). Match cut: candle flames become blinking cursors. Brief doorway sequence: the line passes through 5–6 different door frames, each labelled in tiny mono text with a generic harness style (`bash`, `edit`, `apply_patch`, `str_replace`, `search/replace`) and stays steady.
5. **30–38 · The loop.** A moth circles a flame in an identical loop three times (motion trail visible, the same path each time); a faint red `FAILED` pulses with each pass.
6. **38–42 · The turn.** Silence. The moth breaks its orbit, flies out of the loop, and the trail turns ember. Text: *"It does not repeat. It understands."*
7. **42–50 · Nine stones.** Nine stones placed one by one in a row; above them, small mono captions fade in and out: `9B parameters`, `runs on a single consumer GPU`, `open weights · Apache-2.0`, `trained only on verified work`.
8. **50–56 · Dawn.** The palette warms; ink is gone; the tone line becomes a horizon with a low ember sun.
9. **56–60 · Title.** Wordmark **THRENODY-9B** in wide-tracked serif capitals, tone line underneath; then a smaller line: *"An elegy for every bug."* and `coming soon · huggingface · github`. Hold 1 s, hard cut to black with the final note.

## Quality bar
- Production-grade motion: eased curves (cubic-bezier / spring physics), no linear tweens, no stock "glitch" effects, no particle confetti, no neon cyberpunk clichés, no robot or brain imagery.
- The ink must look physical: use layered noise (curl/simplex) advection on a canvas, not simple circles.
- Every scene transition is a match cut or a dissolve driven by a shared shape (ink → line, flame → cursor, line → horizon).
- All text must be legible at 1080p for at least 1.5 s.
- Do not use real company logos, product names other than Threnody-9B, or claims of specific benchmark numbers.

## Rendering to MP4 (required)
- Drive every animation from one deterministic clock `t` (seconds), never `requestAnimationFrame` time or `Date.now()`; expose `window.renderFrame(t)` that draws the exact frame for time `t`.
- Provide a Node script using Puppeteer: open the page at 1920×1080, call `renderFrame(i / 60)` for i = 0..3599, screenshot each frame to PNG (or stream via `page.screenshot({encoding: "binary"})` into ffmpeg's stdin).
- Encode: `ffmpeg -framerate 60 -i frames/%05d.png -i audio.wav -c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p -c:a aac -b:a 192k -movflags +faststart threnody-9b-trailer.mp4`.
- Deliver: `index.html`, `render.js`, `package.json`, the audio export function, and a one-paragraph README with the two commands to produce the MP4.
