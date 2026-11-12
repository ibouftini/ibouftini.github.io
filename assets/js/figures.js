/* Helpers for interactive figures (ES module).

An interactive figure lives next to its post, in interactive/<name>.js, and is declared in the post with
  ::: {.interactive src="interactive/<name>.js" fallback="figures/<name>.png"}
Its default export mount(root, context) draws into root; assets/js/post.js loads it when it scrolls near the
screen. Figures are black and white: they draw with currentColor and the post's CSS variables, so they follow
the light and dark themes without extra code. */

export const SVG_NS = "http://www.w3.org/2000/svg";

/** Create an SVG element with attributes (and append it to parent when given). */
export function svg(tag, attrs = {}, parent = null) {
	const el = document.createElementNS(SVG_NS, tag);
	for (const [k, v] of Object.entries(attrs)) {
		if (k === "text") {
			el.textContent = v;
		} else {
			el.setAttribute(k, v);
		}
	}
	if (parent) {
		parent.appendChild(el);
	}
	return el;
}

/** Create an HTML element: html("div", {class: "x"}, [child, "text"]). */
export function html(tag, attrs = {}, children = []) {
	const el = document.createElement(tag);
	for (const [k, v] of Object.entries(attrs)) {
		if (k === "text") {
			el.textContent = v;
		} else {
			el.setAttribute(k, v);
		}
	}
	for (const c of [].concat(children)) {
		el.append(c);
	}
	return el;
}

/** Typeset TeX into an element with the page's KaTeX (loaded by post.js before the figure); plain text without it. */
export function tex(el, source) {
	if (window.katex) {
		window.katex.render(source, el, { throwOnError: false });
	} else {
		el.textContent = source;
	}
	return el;
}

// Font sizes of .sch-tex, .sch-tex-small and .sch-tex-tiny in post.css; _dev/build_blog.py places labels the same way
const TEX_SIZES = { "": 22, small: 18, tiny: 12 };

/** An SVG drawing with an HTML layer over it for its math labels (texLabel): schematic({viewBox: "0 0 W H", ...}).
    Returns the svg; insert svg.parentNode, the div.sch-figure that holds both, into the page. */
export function schematic(attrs) {
	const s = svg("svg", { class: "schematic", ...attrs });
	html("div", { class: "sch-figure", style: `--sch-w: ${s.viewBox.baseVal.width}` },
		[s, html("div", { class: "sch-tex-layer", "aria-hidden": "true" })]);
	return s;
}

/** A math label on a drawing made by schematic(), placed like <text x y text-anchor>:
    texLabel(s, {x, y, anchor, size}, "W_1"). It is HTML laid over the SVG, not a foreignObject, which Safari on iOS
    misplaces. Math in figures is always LaTeX, never Unicode symbols. */
export function texLabel(s, place, source) {
	const label = html("span");
	s.nextElementSibling.appendChild(label);
	moveTexLabel(label, place);
	tex(label, source);
	return label;
}

/** Move a label made by texLabel, for figures that redraw: moveTexLabel(label, {x, y, anchor, size}). */
export function moveTexLabel(label, { x, y, anchor = "start", size = "" }) {
	const box = label.closest(".sch-figure").querySelector("svg").viewBox.baseVal;
	label.style.left = `${(100 * x) / box.width}%`;
	label.style.top = `${(100 * (y - 0.3 * TEX_SIZES[size])) / box.height}%`;
	label.setAttribute("class", ["sch-tex", `sch-tex-${anchor}`, size ? `sch-tex-${size}` : ""].join(" ").trim());
}

/** Linear map from [d0, d1] to [r0, r1]. */
export function scale(d0, d1, r0, r1) {
	return (x) => r0 + ((x - d0) * (r1 - r0)) / (d1 - d0);
}

/** A labelled range slider. Returns {element, input, set}. onInput(value) runs on every change.
    label is a word in the interface font, or TeX when math: true (the name and the value are then typeset). */
export function slider({ label, math = false, ariaLabel = label, min, max, step, value, format = (v) => String(v),
	onInput }) {
	const input = html("input", { type: "range", min, max, step, value: String(value), "aria-label": ariaLabel });
	const readout = html("output", { class: "fig-readout" });
	const name = math ? tex(html("span", { class: "fig-tex" }), label) : html("span", { text: label, class: "fig-word" });
	const element = html("label", { class: "fig-slider" }, [name, input, readout]);
	function set(v) {
		input.value = String(v);
		const shown = format(Number(input.value));
		if (math) {
			tex(readout, shown);
		} else {
			readout.textContent = shown;
		}
		onInput(Number(input.value));
	}
	input.addEventListener("input", () => set(input.value));
	set(value);
	return { element, input, set };
}

/** One ruled row of named values, names and values in TeX: readings(["\\mathrm{TV}", "W_1"]).
    Returns {element, update(values)}, where each value is TeX too ("0.300", "+\\infty"). */
export function readings(names) {
	const cells = names.map((name) => {
		const value = html("strong");
		const term = tex(html("span"), name);
		return { value, el: html("div", {}, [term, value]) };
	});
	return {
		element: html("div", { class: "fig-readings", "aria-live": "polite" }, cells.map((c) => c.el)),
		update(values) {
			values.forEach((v, i) => {
				tex(cells[i].value, v);
			});
		},
	};
}
