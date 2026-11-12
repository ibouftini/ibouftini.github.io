"use strict";

/* 404 page: Conway's Game of Life on a wrapping grid behind the text. A live cell survives with 2 or 3 live
   neighbours, a dead cell is born with exactly 3. Cells are drawn darker when newly born and fade as they age.
   A glider enters every 45 generations, and a random patch is sown when the board dies out or stops changing,
   so the panel never goes still. */
(function () {
	var panel = document.querySelector(".error-panel");
	var canvas = panel.querySelector(".life");
	var ctx = canvas.getContext("2d");
	var counter = panel.querySelector("[data-generation]");
	var CELL = 14;
	var GAP = 2;
	var STEP_MS = 140;
	var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
	var cols = 0, rows = 0, grid, next, age;
	var width = 0, height = 0, generation = 0, lastPop = -1, quiet = 0, last = 0, painting = false;

	function index(x, y) {
		return ((y + rows) % rows) * cols + ((x + cols) % cols);
	}
	function sow(x0, y0, w, h, density) {
		for (var y = y0; y < y0 + h; y++) {
			for (var x = x0; x < x0 + w; x++) {
				if (Math.random() < density) {
					var i = index(x, y);
					grid[i] = 1;
					age[i] = 0;
				}
			}
		}
	}
	function glider() {
		// Enters at a random column from the top or bottom edge, heading diagonally.
		var dx = Math.random() < .5 ? 1 : -1;
		var dy = Math.random() < .5 ? 1 : -1;
		var x0 = Math.floor(Math.random() * cols);
		var y0 = dy > 0 ? 0 : rows - 3;
		[[1, 0], [2, 1], [0, 2], [1, 2], [2, 2]].forEach(function (c) {
			var i = index(x0 + (dx > 0 ? c[0] : 2 - c[0]), y0 + (dy > 0 ? c[1] : 2 - c[1]));
			grid[i] = 1;
			age[i] = 0;
		});
	}
	function resize() {
		var rect = panel.getBoundingClientRect();
		var dpr = window.devicePixelRatio || 1;
		width = rect.width;
		height = rect.height;
		canvas.width = Math.round(width * dpr);
		canvas.height = Math.round(height * dpr);
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
		var c = Math.ceil(width / CELL);
		var r = Math.ceil(height / CELL);
		if (c !== cols || r !== rows) {
			var old = grid, oldCols = cols, oldRows = rows;
			cols = c;
			rows = r;
			grid = new Uint8Array(cols * rows);
			next = new Uint8Array(cols * rows);
			age = new Uint8Array(cols * rows);
			if (old) {
				for (var y = 0; y < Math.min(rows, oldRows); y++) {
					for (var x = 0; x < Math.min(cols, oldCols); x++) {
						grid[y * cols + x] = old[y * oldCols + x];
						age[y * cols + x] = 8;
					}
				}
			} else {
				sow(0, 0, cols, rows, .22);
				age.fill(8);
			}
		}
		draw();
	}
	function step() {
		var pop = 0;
		for (var y = 0; y < rows; y++) {
			var up = ((y - 1 + rows) % rows) * cols, row = y * cols, down = ((y + 1) % rows) * cols;
			for (var x = 0; x < cols; x++) {
				var l = (x - 1 + cols) % cols, r = (x + 1) % cols;
				var n = grid[up + l] + grid[up + x] + grid[up + r] + grid[row + l] + grid[row + r] +
					grid[down + l] + grid[down + x] + grid[down + r];
				var i = row + x;
				var alive = grid[i];
				var live = n === 3 || (alive && n === 2) ? 1 : 0;
				next[i] = live;
				age[i] = live ? (alive ? Math.min(age[i] + 1, 8) : 0) : 0;
				pop += live;
			}
		}
		var swap = grid;
		grid = next;
		next = swap;
		generation++;
		quiet = pop === lastPop ? quiet + 1 : 0;
		lastPop = pop;
		if (pop < cols * rows * .03 || quiet > 40) {
			sow(Math.floor(Math.random() * cols), Math.floor(Math.random() * rows), 14, 10, .45);
			quiet = 0;
		}
		if (generation % 45 === 0) {
			glider();
		}
		counter.textContent = "Generation " + generation;
	}
	function draw() {
		var ink = document.body.classList.contains("theme-dark") ? "255, 255, 255" : "0, 0, 0";
		var styles = [];
		for (var a = 0; a <= 8; a++) {
			// Newborn cells at 34% ink, settling to 10% after eight generations.
			styles.push("rgba(" + ink + ", " + (.34 - .03 * a).toFixed(2) + ")");
		}
		ctx.clearRect(0, 0, width, height);
		for (var y = 0; y < rows; y++) {
			for (var x = 0; x < cols; x++) {
				var i = y * cols + x;
				if (grid[i]) {
					ctx.fillStyle = styles[age[i]];
					ctx.fillRect(x * CELL + GAP / 2, y * CELL + GAP / 2, CELL - GAP, CELL - GAP);
				}
			}
		}
	}
	function frame(t) {
		if (!document.hidden && t - last >= STEP_MS) {
			last = t;
			step();
			draw();
		}
		requestAnimationFrame(frame);
	}
	function paint(e) {
		var rect = panel.getBoundingClientRect();
		var cx = Math.floor((e.clientX - rect.left) / CELL);
		var cy = Math.floor((e.clientY - rect.top) / CELL);
		for (var dy = -1; dy <= 1; dy++) {
			for (var dx = -1; dx <= 1; dx++) {
				if (Math.random() < .6) {
					var i = index(cx + dx, cy + dy);
					grid[i] = 1;
					age[i] = 0;
				}
			}
		}
		draw();
	}

	// Cells are added on the empty parts of the panel; the text and the button keep their
	// usual behaviour. On touch screens a tap adds cells and dragging still scrolls.
	panel.addEventListener("pointerdown", function (e) {
		if (e.target !== panel) {
			return;
		}
		e.preventDefault();
		painting = e.pointerType !== "touch";
		paint(e);
	});
	panel.addEventListener("pointermove", function (e) {
		if (painting) {
			paint(e);
		}
	});
	window.addEventListener("pointerup", function () {
		painting = false;
	});

	resize();
	new ResizeObserver(resize).observe(panel);
	if (!reduce) {
		requestAnimationFrame(frame);
	}
})();
