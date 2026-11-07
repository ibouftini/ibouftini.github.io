/* Samples of the circle generators of Part 1, widths 16 and 1024 side by side, for 12 or 200 training points
   (one draw, seed 0). Grey dots are samples, open circles (12 points) or short ticks outside the circle (200 points)
   are the training points, the dashed circle is the true one. The samples that count as copies (closer to a training
   point than 99% of fresh points are) lie within 0.0024 of it, far below a pixel, so they are drawn as one dark disc
   per training point with area proportional to their number, and their share is written under each panel. Two framed panels with their titles above, as in the
   static schematics of the post (_dev/figures/no_density_svg.py), and a legend below. Buttons under the drawing
   switch the number of training points. The drawing is redrawn at the width it is given: side by side on wide
   screens, stacked on phones, with text at a constant size.
   Data: memorization-samples.json, written by _dev/figures/memorization_sweeps.py post.
   Static version: figures/memorization-samples.png. */
import { svg, html, scale } from "/assets/js/figures.js";

const SETTINGS = [
	{ n: 12, note: "The wide generator sits on the training points and the chords between them." },
	{ n: 200, note: "With 200 points, both generators follow the circle." },
];
const WIDTHS = [16, 1024];
const LIM = 1.3; // half-width of the data window, the unit circle sits in [-1, 1]^2
const RING = 7.5;
const disc = (count) => 1.15 * Math.sqrt(count); // area proportional to the number of copies (37 copies: radius 7)

// The training points: rings for a few points, ticks outside the circle for many
function points(g, data, x, y) {
	if (data.n <= 12) {
		data.train.forEach(([a, b], i) => {
			const count = data.copies.counts[i];
			if (count > 0) {
				svg("circle", { class: "sch-ink", cx: x(a).toFixed(1), cy: y(b).toFixed(1), r: disc(count).toFixed(2) }, g);
			}
			svg("circle", { class: "sch-ring", cx: x(a).toFixed(1), cy: y(b).toFixed(1), r: RING, "stroke-width": 1.3 }, g);
		});
		return;
	}
	for (const [a, b] of data.train) {
		const t = Math.atan2(b, a), c = Math.cos(t), s = Math.sin(t);
		svg("path", { class: "sch-line sch-thin",
			d: `M${x(1.07 * c).toFixed(1)},${y(1.07 * s).toFixed(1)} L${x(1.15 * c).toFixed(1)},${y(1.15 * s).toFixed(1)}` }, g);
	}
}

// One framed panel of side P at (x0, y0), its title above and the share of copies below
function panel(s, data, x0, y0, P) {
	svg("text", { class: "sch-label", style: "fill: var(--muted)", x: x0 + P / 2, y: y0 + P + 20, "text-anchor": "middle",
		text: `Copies: ${data.copies.percent}% of samples` }, s);
	svg("rect", { class: "sch-frame", x: x0, y: y0, width: P, height: P, rx: 10 }, s);
	svg("text", { class: "sch-label", style: "font-size: 14px", x: x0 + P / 2, y: y0 - 12, "text-anchor": "middle",
		text: `Width ${data.width}` }, s);
	const x = scale(-LIM, LIM, x0, x0 + P), y = scale(-LIM, LIM, y0 + P, y0);
	const g = svg("g", {}, s);
	svg("circle", { class: "sch-line sch-thin", cx: x(0), cy: y(0), r: x(1) - x(0), "stroke-dasharray": "3 4" }, g);
	for (const [a, b] of data.samples) {
		if (Math.abs(a) < LIM && Math.abs(b) < LIM) {
			svg("circle", { class: "sch-muted", cx: x(a).toFixed(1), cy: y(b).toFixed(1), r: 1.5 }, g);
		}
	}
	points(g, data, x, y);
}

// Legend centred under the panels, on two rows when one does not fit in maxWidth. Returns its height.
function legend(s, n, cx, top, maxWidth) {
	const items = [
		["samples", 72, (g, x, y) => svg("circle", { class: "sch-muted", cx: x, cy: y, r: 3 }, g)],
		["training points", 112, (g, x, y) => (n <= 12
			? svg("circle", { class: "sch-ring", cx: x, cy: y, r: 5.5, "stroke-width": 1.3 }, g)
			: svg("path", { class: "sch-line sch-thin", d: `M${x},${y - 5} L${x},${y + 5}` }, g))],
		["true circle", 82, (g, x, y) => svg("path", { class: "sch-line sch-thin", "stroke-dasharray": "3 4",
			d: `M${x - 10},${y} L${x + 10},${y}` }, g)],
	]; // label, width of marker + text at 12px, marker
	if (n <= 12) {
		items.push(["copies", 66, (g, x, y) => svg("circle", { class: "sch-ink", cx: x, cy: y, r: 4 }, g)]);
	}
	const span = (row) => row.reduce((a, item) => a + item[1] + 18, -18);
	const rows = span(items) <= maxWidth ? [items] : [items.slice(0, 2), items.slice(2)];
	const g = svg("g", {}, s);
	rows.forEach((row, r) => {
		const y = top + 22 + 24 * r;
		let x = cx - span(row) / 2;
		for (const [text, width, mark] of row) {
			mark(g, x + 10, y);
			svg("text", { class: "sch-label", x: x + 24, y: y + 4, text }, g);
			x += width + 18;
		}
	});
	return 10 + 24 * rows.length;
}

function drawing(width, byKey, n) {
	const W = Math.round(Math.min(Math.max(width, 300), 720));
	const stacked = W < 520;
	const GAP = 40, TOP = 34, SUB = 30;
	const P = stacked ? Math.min(W - 20, 320) : Math.min((W - GAP - 20) / 2, 300);
	const panels = stacked ? 2 * (TOP + P + SUB) : TOP + P + SUB;
	const s = svg("svg", { class: "schematic", width: W, role: "img",
		"aria-label": `Samples of the generators of width 16 and 1024 trained on ${n} points of the unit circle` });
	WIDTHS.forEach((w, i) => {
		const data = byKey.get(`${w}-${n}`);
		if (stacked) {
			panel(s, data, (W - P) / 2, TOP + i * (TOP + P + SUB), P);
		} else {
			const x0 = (W - 2 * P - GAP) / 2 + i * (P + GAP);
			panel(s, data, x0, TOP, P);
		}
	});
	const H = panels + legend(s, n, W / 2, panels, W - 20) + 8;
	s.setAttribute("viewBox", `0 0 ${W} ${H}`);
	return s;
}

export default async function mount(root, context) {
	const data = await (await fetch(new URL("memorization-samples.json", context.base))).json();
	const byKey = new Map(data.panels.map((p) => [`${p.width}-${p.n}`, p]));
	const id = (context.figure.id || "memorization-samples") + "-panel";

	const holder = html("div", { role: "tabpanel", id, tabindex: "0" });
	const buttons = SETTINGS.map((t, i) => html("button", { type: "button", role: "tab", id: `${id}-tab-${i}`,
		"aria-controls": id, text: String(t.n) }));
	const tablist = html("div", { class: "fig-steps", role: "tablist", "aria-label": "Number of training points" }, buttons);
	const note = html("span", { class: "fig-setting", "aria-live": "polite" });
	const controls = html("div", { class: "fig-controls" },
		[html("span", { class: "fig-word", text: "Training points" }), tablist, note]);

	let current = 0, width = 0;
	function redraw(force = false) {
		const px = holder.clientWidth || 640;
		if (!force && Math.abs(px - width) < 16) {
			return;
		}
		width = px;
		holder.replaceChildren(drawing(px, byKey, SETTINGS[current].n));
	}

	function select(i, focus = false) {
		current = i;
		buttons.forEach((b, j) => {
			b.setAttribute("aria-selected", String(i === j));
			b.tabIndex = i === j ? 0 : -1;
		});
		holder.setAttribute("aria-labelledby", buttons[i].id);
		note.textContent = SETTINGS[i].note;
		redraw(true);
		if (focus) {
			buttons[i].focus();
		}
	}

	buttons.forEach((b, i) => {
		b.addEventListener("click", () => select(i));
		b.addEventListener("keydown", (e) => {
			const last = buttons.length - 1;
			const next = { ArrowRight: i === last ? 0 : i + 1, ArrowLeft: i === 0 ? last : i - 1, Home: 0, End: last }[e.key];
			if (next !== undefined) {
				e.preventDefault();
				select(next, true);
			}
		});
	});

	root.append(holder, controls);
	select(0);
	if ("ResizeObserver" in window) {
		new ResizeObserver(() => redraw()).observe(holder);
	}
}
