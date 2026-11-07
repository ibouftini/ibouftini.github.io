/* Two uniform distributions of width w: P = U[-w/2, w/2] and Q = U[theta - w/2, theta + w/2] (series Part 2).
   With r = min(|theta| / w, 1):  TV = r,  JS = r log 2,  W1 = |theta|,  KL(P || Q) = +inf for every theta != 0.
   While the supports overlap, TV and JS grow with the shift; once they separate, both freeze and only W1 keeps
   measuring the distance. As w -> 0 the uniforms become point masses and TV, JS become steps.
   KL is drawn on a "+inf" level above a break of the vertical axis. Curves are labelled at their right end.
   The drawing is redrawn at the width it is given, so its text keeps its size on phones.
   Static version: _dev/figures/two_uniforms.py. */
import { svg, schematic, scale, slider, readings, texLabel, moveTexLabel } from "/assets/js/figures.js";

const LOG2 = Math.log(2);
const INF = 2.45; // height of the "+inf" level, above the axis break at 2.2

// The distances at shift theta for width w
function distances(theta, w) {
	const r = Math.min(Math.abs(theta) / w, 1);
	return { tv: r, js: r * LOG2, w1: Math.abs(theta), kl: theta === 0 ? 0 : Infinity };
}

// Draw the figure for a given pixel width; returns {element, draw(theta, w)}
function drawing(width) {
	const W = Math.round(Math.min(Math.max(width, 300), 640));
	const narrow = W < 460;
	const X0 = narrow ? 46 : 64, X1 = W - (narrow ? 40 : 80);
	const SB = 92; // baseline of the strip that shows the two distributions
	const YTOP = SB + 44, YBASE = YTOP + (narrow ? 158 : 196), H = YBASE + 36;
	const x = scale(-2, 2, X0, X1);
	const y = scale(0, INF, YBASE, YTOP);
	const s = schematic({ viewBox: `0 0 ${W} ${H}`, width: W, role: "img",
		"aria-label": "Two uniform distributions and the TV, KL, JS and W1 distances between them, as functions of the shift theta" });

	// Strip: the two densities on the same horizontal scale as the plot. Height grows like 1/w, capped.
	svg("path", { class: "sch-grid", d: `M${X0},${SB} L${X1},${SB}` }, s);
	const pRect = svg("rect", { class: "sch-region" }, s);
	const qRect = svg("rect", { fill: "none", stroke: "currentColor", "stroke-width": 1.6, "stroke-dasharray": "5 3" }, s);
	const pLabel = texLabel(s, { x: 0, y: SB - 4, size: "small" }, "\\mathbb{P}");
	const qLabel = texLabel(s, { x: 0, y: SB - 4, size: "small" }, "\\mathbb{Q}");

	// Axes, ticks, and the break between 2 and +inf
	const yBreak = y(2.2);
	svg("path", { class: "sch-frame", d: `M${X0},${y(INF) - 8} L${X0},${yBreak - 4} M${X0},${yBreak + 4} L${X0},${YBASE} L${X1},${YBASE}` }, s);
	svg("path", { class: "sch-line sch-thin", d: `M${X0 - 6},${yBreak} L${X0 + 6},${yBreak - 6} M${X0 - 6},${yBreak + 6} L${X0 + 6},${yBreak}` }, s);
	for (const t of [-2, -1, 0, 1, 2]) {
		svg("path", { class: "sch-grid", d: `M${x(t)},${YBASE} L${x(t)},${YBASE + 5}` }, s);
		texLabel(s, { x: x(t), y: YBASE + 18, anchor: "middle", size: "tiny" }, String(t));
	}
	svg("text", { class: "sch-label sch-tiny", x: (X0 + X1) / 2 - 2, y: YBASE + 32, "text-anchor": "end", text: "Shift" }, s);
	texLabel(s, { x: (X0 + X1) / 2 + 2, y: YBASE + 32, size: "tiny" }, "\\theta");
	for (const [v, t] of [[0, "0"], [LOG2, "\\log 2"], [1, "1"], [2, "2"], [INF, narrow ? "\\infty" : "+\\infty"]]) {
		texLabel(s, { x: X0 - 10, y: y(v) + 4, anchor: "end", size: "tiny" }, t);
	}

	// Curves (they depend on w, so their paths are set in draw), each labelled at its right end
	svg("path", { class: "sch-line", d: `M${x(-2)},${y(2)} L${x(0)},${y(0)} L${x(2)},${y(2)}` }, s);
	svg("path", { class: "sch-line", "stroke-dasharray": "10 4 2 4",
		d: `M${x(-2)},${y(INF)} L${x(0) - 6},${y(INF)} M${x(0) + 6},${y(INF)} L${x(2)},${y(INF)}` }, s);
	svg("circle", { cx: x(0), cy: y(INF), r: 5, fill: "none", stroke: "currentColor", "stroke-width": 1.5 }, s);
	const tvPath = svg("path", { class: "sch-line", "stroke-dasharray": "7 5" }, s);
	const jsPath = svg("path", { class: "sch-line", "stroke-dasharray": "2 4" }, s);
	svg("circle", { class: "sch-ink", cx: x(0), cy: y(0), r: 4 }, s);
	for (const [v, name] of [[2, "W_1"], [INF, "\\mathrm{KL}"], [1, "\\mathrm{TV}"], [LOG2, "\\mathrm{JS}"]]) {
		texLabel(s, { x: X1 + 8, y: y(v) + 4, size: narrow ? "tiny" : "small" }, name);
	}

	// Current theta: a vertical marker and one dot per curve
	const marker = svg("path", { class: "sch-grid", "stroke-dasharray": "3 3" }, s);
	const dots = [0, 1, 2, 3].map(() => svg("circle", { class: "sch-ink", r: 4.5 }, s));

	return {
		element: s.parentNode,
		draw(theta, w) {
			// strip
			const h = Math.min(16 / w, 54);
			const place = (rect, center) => {
				rect.setAttribute("x", x(center - w / 2));
				rect.setAttribute("width", Math.max(x(center + w / 2) - x(center - w / 2), 1.5));
				rect.setAttribute("y", SB - h);
				rect.setAttribute("height", h);
			};
			place(pRect, 0);
			place(qRect, theta);
			// the left distribution is labelled on its left, the right one on its right
			const [left, right] = theta < 0 ? [[qLabel, theta], [pLabel, 0]] : [[pLabel, 0], [qLabel, theta]];
			moveTexLabel(left[0], { x: x(left[1] - w / 2) - 8, y: SB - 4, anchor: "end", size: "small" });
			moveTexLabel(right[0], { x: x(right[1] + w / 2) + 8, y: SB - 4, anchor: "start", size: "small" });

			// TV and JS: linear up to |theta| = w, then flat
			const bend = (top) => `M${x(-2)},${y(top)} L${x(-Math.min(w, 2))},${y(top)} L${x(0)},${y(0)} ` +
				`L${x(Math.min(w, 2))},${y(top)} L${x(2)},${y(top)}`;
			tvPath.setAttribute("d", bend(1));
			jsPath.setAttribute("d", bend(LOG2));

			const d = distances(theta, w);
			marker.setAttribute("d", `M${x(theta)},${y(INF) - 8} L${x(theta)},${YBASE}`);
			[d.w1, d.kl === Infinity ? INF : 0, d.tv, d.js].forEach((v, i) => {
				dots[i].setAttribute("cx", x(theta));
				dots[i].setAttribute("cy", y(v));
			});
		},
	};
}

export default function mount(root) {
	let theta = 0.3;
	let w = 0.5;
	let width = 0;
	let current = null;
	const holder = document.createElement("div");
	const values = readings(["\\mathrm{TV}", "\\mathrm{KL}", "\\mathrm{JS}", "W_1"]);

	function show() {
		current.draw(theta, w);
		const d = distances(theta, w);
		values.update([d.tv.toFixed(3), d.kl === Infinity ? "+\\infty" : "0", d.js.toFixed(3), d.w1.toFixed(2)]);
	}

	function redraw() {
		const px = holder.clientWidth || 640;
		if (current && Math.abs(px - width) < 16) {
			return;
		}
		width = px;
		current = drawing(px);
		holder.replaceChildren(current.element);
		show();
	}

	const shift = slider({
		label: "\\theta", math: true, ariaLabel: "Shift theta", min: -2, max: 2, step: 0.01, value: theta,
		format: (v) => v.toFixed(2),
		onInput: (v) => {
			theta = v;
			if (current) {
				show();
			}
		},
	});
	const spread = slider({
		label: "\\Delta", math: true, ariaLabel: "Width", min: 0.05, max: 1, step: 0.01, value: w,
		format: (v) => v.toFixed(2),
		onInput: (v) => {
			w = v;
			if (current) {
				show();
			}
		},
	});
	const controls = document.createElement("div");
	controls.className = "fig-controls";
	controls.append(shift.element, spread.element);
	root.append(holder, controls, values.element);
	redraw();
	if ("ResizeObserver" in window) {
		new ResizeObserver(redraw).observe(holder);
	}
}
