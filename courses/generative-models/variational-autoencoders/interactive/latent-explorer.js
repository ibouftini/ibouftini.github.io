/* Part 3: the latent explorer. Drag a point in the 2D latent plane of a VAE; its decoder (784 -> 512 -> 256 -> 2,
   mirrored) runs in the browser and draws the digit. Grey dots: where 2,000 test digits land (encoder means).
   Weights: decoder.bin (int8 with per-row float16 scales), decoder.json, codes.json, all written by
   _dev/figures/vae_lab.py export. Static version: figures/latent-map.png. */
import { svg, scale, readings } from "/assets/js/figures.js";

function half(u) {
	const sign = u & 0x8000 ? -1 : 1, e = (u >> 10) & 0x1f, f = u & 0x3ff;
	if (e === 0) return sign * 2 ** -14 * (f / 1024);
	if (e === 31) return f ? NaN : sign * Infinity;
	return sign * 2 ** (e - 15) * (1 + f / 1024);
}

// Decode the int8 weights into float32 layers {W (out x in, row-major), b, n, m}
function layers(buffer, meta) {
	const view = new DataView(buffer);
	return meta.layers.map((L) => {
		const W = new Float32Array(L.out * L.in), b = new Float32Array(L.out);
		for (let r = 0; r < L.out; r++) {
			const s = half(view.getUint16(L.scales + 2 * r, true));
			for (let c = 0; c < L.in; c++) {
				W[r * L.in + c] = view.getInt8(L.weights + r * L.in + c) * s;
			}
			b[r] = half(view.getUint16(L.bias + 2 * r, true));
		}
		return { W, b, n: L.out, m: L.in };
	});
}

function decode(net, z) {
	let h = Float32Array.from(z);
	net.forEach((L, k) => {
		const out = new Float32Array(L.n);
		for (let r = 0; r < L.n; r++) {
			let acc = L.b[r];
			const row = r * L.m;
			for (let c = 0; c < L.m; c++) {
				acc += L.W[row + c] * h[c];
			}
			out[r] = k < net.length - 1 ? Math.max(acc, 0) : 1 / (1 + Math.exp(-acc));
		}
		h = out;
	});
	return h;
}

export default async function mount(root) {
	const base = import.meta.url;
	const [meta, buffer, codes] = await Promise.all([
		fetch(new URL("decoder.json", base)).then((r) => r.json()),
		fetch(new URL("decoder.bin", base)).then((r) => r.arrayBuffer()),
		fetch(new URL("codes.json", base)).then((r) => r.json()),
	]);
	const net = layers(buffer, meta);

	// Latent plane: [-3.2, 3.2]^2, with the circles of radius 1, 2, 3 of the prior N(0, I)
	const S = 320;
	const x = scale(-3.2, 3.2, 8, S - 8);
	const y = scale(-3.2, 3.2, S - 8, 8);
	const plane = svg("svg", { class: "schematic latent-plane", viewBox: `0 0 ${S} ${S}`, role: "slider", tabindex: 0,
		"aria-label": "Latent code: drag, or use the arrow keys" });
	for (const r of [1, 2, 3]) {
		svg("circle", { class: "sch-grid", cx: x(0), cy: y(0), r: x(r) - x(0), fill: "none", "stroke-dasharray": "3 4" }, plane);
	}
	const dots = svg("g", {}, plane);
	codes.z.forEach(([a, b]) => svg("circle", { class: "sch-muted", cx: x(a).toFixed(1), cy: y(b).toFixed(1), r: 1.4 }, dots));
	const handle = svg("circle", { r: 7, fill: "none", stroke: "currentColor", "stroke-width": 2 }, plane);
	const cross = svg("circle", { class: "sch-ink", r: 2.2 }, plane);

	// Decoded digit, drawn in the text colour so it follows the theme
	const canvas = document.createElement("canvas");
	canvas.width = canvas.height = 28;
	canvas.className = "latent-digit";
	const ctx = canvas.getContext("2d");
	const values = readings(["z_1", "z_2"]);

	let z = [0, 0];
	function draw() {
		handle.setAttribute("cx", x(z[0]));
		handle.setAttribute("cy", y(z[1]));
		cross.setAttribute("cx", x(z[0]));
		cross.setAttribute("cy", y(z[1]));
		const img = decode(net, z);
		const ink = getComputedStyle(canvas).color;
		ctx.clearRect(0, 0, 28, 28);
		ctx.fillStyle = ink;
		for (let i = 0; i < 784; i++) {
			if (img[i] > 0.02) {
				ctx.globalAlpha = Math.min(img[i], 1);
				ctx.fillRect(i % 28, Math.floor(i / 28), 1, 1);
			}
		}
		ctx.globalAlpha = 1;
		values.update([z[0].toFixed(2), z[1].toFixed(2)]);
	}

	function moveTo(evt) {
		const box = plane.getBoundingClientRect();
		const px = ((evt.clientX - box.left) / box.width) * S, py = ((evt.clientY - box.top) / box.height) * S;
		const clamp = (v) => Math.max(-3.2, Math.min(3.2, v));
		z = [clamp(-3.2 + ((px - 8) / (S - 16)) * 6.4), clamp(3.2 - ((py - 8) / (S - 16)) * 6.4)];
		draw();
	}
	let dragging = false;
	plane.addEventListener("pointerdown", (e) => {
		dragging = true;
		plane.setPointerCapture(e.pointerId);
		moveTo(e);
	});
	plane.addEventListener("pointermove", (e) => dragging && moveTo(e));
	plane.addEventListener("pointerup", () => (dragging = false));
	plane.addEventListener("keydown", (e) => {
		const step = e.shiftKey ? 0.5 : 0.1;
		const d = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, step], ArrowDown: [0, -step] }[e.key];
		if (d) {
			e.preventDefault();
			z = [Math.max(-3.2, Math.min(3.2, z[0] + d[0])), Math.max(-3.2, Math.min(3.2, z[1] + d[1]))];
			draw();
		}
	});

	const pair = document.createElement("div");
	pair.className = "fig-pair";
	const right = document.createElement("div");
	right.className = "fig-pair-side";
	right.append(canvas, values.element);
	pair.append(plane, right);
	root.append(pair);
	draw();
}
