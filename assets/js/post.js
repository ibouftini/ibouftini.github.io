"use strict";

/* Blog post pages (built by _dev/build_blog.py), loaded after custom.js. */

/* --------------------------------------------------------------------------
   Copy buttons on code blocks, the citation, and the "Copy link" button.
   -------------------------------------------------------------------------- */
(function () {
	function copy(text, button, doneLabel) {
		var icon = button.querySelector("i");
		var label = button.querySelector("span");
		var before = label ? label.textContent : null;
		function done() {
			if (icon) {
				icon.className = "bi bi-check2";
			}
			if (label && doneLabel) {
				label.textContent = doneLabel;
			}
			setTimeout(function () {
				if (icon) {
					icon.className = button.dataset.icon;
				}
				if (label && before !== null) {
					label.textContent = before;
				}
			}, 1600);
		}
		if (navigator.clipboard) {
			navigator.clipboard.writeText(text).then(done);
		}
	}

	document.querySelectorAll(".post-body div.sourceCode, .post-cite .code-block").forEach(function (block) {
		var button = document.createElement("button");
		button.type = "button";
		button.className = "code-copy";
		button.setAttribute("aria-label", "Copy code");
		button.dataset.icon = "bi bi-clipboard";
		button.innerHTML = '<i class="bi bi-clipboard"></i>';
		button.addEventListener("click", function () {
			copy(block.querySelector("pre").innerText, button);
		});
		block.appendChild(button);
	});

	var share = document.querySelector("[data-copy-link]");
	if (share) {
		share.dataset.icon = "bi bi-link-45deg";
		share.addEventListener("click", function () {
			copy(window.location.href.split("#")[0], share, "Copied");
		});
	}

	// System share sheet (phones, some desktops): reaches Instagram, WhatsApp, Bluesky and other apps
	// that have no web share link. The button stays hidden where the browser cannot do it.
	var nativeShare = document.querySelector("[data-native-share]");
	if (nativeShare && navigator.share) {
		nativeShare.hidden = false;
		nativeShare.addEventListener("click", function () {
			navigator.share({ title: document.title, url: window.location.href.split("#")[0] }).catch(function () {});
		});
	}
})();

/* --------------------------------------------------------------------------
   Reading progress bar and current section in the table of contents.
   -------------------------------------------------------------------------- */
(function () {
	var bar = document.querySelector(".reading-progress");
	var article = document.querySelector(".post");
	var tocLinks = document.querySelectorAll(".post-toc-rail a");
	var barLinks = document.querySelectorAll(".post-toc-bar a");
	var barCurrent = document.querySelector(".post-toc-bar-current");
	var headings = Array.prototype.map.call(tocLinks, function (a) {
		return document.getElementById(decodeURIComponent(a.hash.slice(1)));
	});

	function update() {
		if (bar && article) {
			var rect = article.getBoundingClientRect();
			var total = rect.height - window.innerHeight;
			var done = total > 0 ? Math.min(Math.max(-rect.top / total, 0), 1) : 1;
			bar.style.transform = "scaleX(" + done + ")";
		}
		var current = -1;
		headings.forEach(function (h, i) {
			if (h && h.getBoundingClientRect().top < 120) {
				current = i;
			}
		});
		tocLinks.forEach(function (a, i) {
			a.classList.toggle("active", i === current);
		});
		barLinks.forEach(function (a, i) {
			a.classList.toggle("active", i === current);
		});
		if (barCurrent) {
			barCurrent.textContent = current >= 0 ? tocLinks[current].textContent : "";
		}
	}

	// The contents bar (phones and tablets) folds up again after a jump.
	barLinks.forEach(function (a) {
		a.addEventListener("click", function () {
			a.closest("details").open = false;
		});
	});

	// A long jump from the contents can stop short of its target: the interactive figures it passes change
	// height when they start (the static image gives way to the live figure), after the browser has fixed
	// where the scroll ends. When the scroll stops, finish the jump (at most twice). Scrolling by hand cancels.
	var jump = null;
	var settle = 0;
	function finishJump() {
		if (!jump) {
			return;
		}
		var margin = parseFloat(getComputedStyle(jump.target).scrollMarginTop) || 0;
		if (Math.abs(jump.target.getBoundingClientRect().top - margin) > 2 && jump.tries > 0) {
			jump.tries -= 1;
			jump.target.scrollIntoView();
		} else {
			jump = null;
		}
	}
	Array.prototype.forEach.call(document.querySelectorAll(".post-toc-rail a, .post-toc-bar a"), function (a) {
		a.addEventListener("click", function () {
			var target = document.getElementById(decodeURIComponent(a.hash.slice(1)));
			if (target) {
				jump = { target: target, tries: 2 };
				clearTimeout(settle);
				settle = setTimeout(finishJump, 400);
			}
		});
	});
	window.addEventListener("scroll", function () {
		if (jump) {
			clearTimeout(settle);
			settle = setTimeout(finishJump, 150);
		}
	}, { passive: true });
	["wheel", "touchstart", "keydown"].forEach(function (type) {
		window.addEventListener(type, function () {
			jump = null;
		}, { passive: true });
	});

	window.addEventListener("scroll", update, { passive: true });
	window.addEventListener("resize", update);
	update();
})();

/* --------------------------------------------------------------------------
   Folds and links: a link to something inside a closed fold opens it. Printing
   opens every fold, then restores them.
   -------------------------------------------------------------------------- */
(function () {
	function reveal() {
		if (!window.location.hash) {
			return;
		}
		var target = document.getElementById(decodeURIComponent(window.location.hash.slice(1)));
		var opened = false;
		for (var el = target; el; el = el.parentElement) {
			if (el.tagName === "DETAILS" && !el.open) {
				el.open = true;
				opened = true;
			}
		}
		if (opened) {
			target.scrollIntoView();
		}
	}
	window.addEventListener("hashchange", reveal);
	reveal();

	var saved = [];
	window.addEventListener("beforeprint", function () {
		saved = [];
		document.querySelectorAll(".post-body details").forEach(function (d) {
			saved.push([d, d.open]);
			d.open = true;
		});
	});
	window.addEventListener("afterprint", function () {
		saved.forEach(function (pair) {
			pair[0].open = pair[1];
		});
	});
})();

/* --------------------------------------------------------------------------
   Printing (the browser's print, and _dev/build_pdfs.py, which makes the PDF
   that the PDF buttons download). The printed layout is assets/css/print.css;
   this script gives it what CSS cannot know (running head, URL, date) and
   prints in the light theme with the citation and changelog open, then puts
   the page back as it was.
   -------------------------------------------------------------------------- */
(function () {
	var canonical = document.querySelector('link[rel="canonical"]');
	var url = (canonical ? canonical.href : window.location.href.split("#")[0]).replace(/^https?:\/\//, "");
	var heading = document.querySelector(".series-banner-title");
	var title = heading ? heading.textContent : document.title;

	function cssString(text) {
		return '"' + text.trim().replace(/\s+/g, " ").replace(/\\/g, "\\\\").replace(/"/g, '\\"') + '"';
	}
	var page = document.createElement("style");
	page.media = "print";
	// The first page keeps its empty head: restated here because a later @page rule wins over :first in Chromium.
	page.textContent = "@page { @top-left { content: " + cssString(title) + "; } @bottom-left { content: " +
		cssString(url) + "; } } @page :first { @top-left { content: \"\"; } }";
	document.head.appendChild(page);

	var printDate = document.querySelector("[data-print-date]");
	var state = null;
	window.addEventListener("beforeprint", function () {
		state = {
			dark: document.body.classList.contains("theme-dark"),
			scheme: document.documentElement.style.colorScheme,
			folds: []
		};
		document.body.classList.remove("theme-dark");
		document.documentElement.style.colorScheme = "light";
		document.querySelectorAll(".post-cite, .post-changelog").forEach(function (d) {
			state.folds.push([d, d.open]);
			d.open = true;
		});
		if (printDate) {
			printDate.textContent = "Retrieved " + new Date().toLocaleDateString("en-US",
				{ month: "short", day: "numeric", year: "numeric" });
		}
	});
	window.addEventListener("afterprint", function () {
		if (!state) {
			return;
		}
		document.body.classList.toggle("theme-dark", state.dark);
		document.documentElement.style.colorScheme = state.scheme;
		state.folds.forEach(function (pair) {
			pair[0].open = pair[1];
		});
		state = null;
	});
})();

/* --------------------------------------------------------------------------
   Interactive figures (::: interactive in a post). The static image shows until
   the figure scrolls near the screen; then its module (an ES module next to the
   post, default export mount(root, context)) is imported and mounted. If that
   fails, the static image simply stays. The text's formulas are typeset at
   build time; figures typeset theirs live, so KaTeX loads with the first one.
   -------------------------------------------------------------------------- */
(function () {
	var figures = document.querySelectorAll("figure.interactive[data-src]");
	if (!figures.length) {
		return;
	}

	var katexReady;
	function loadKatex() {
		if (!katexReady) {
			katexReady = new Promise(function (done) {
				var script = document.createElement("script");
				script.src = "/assets/plugins/katex/katex.min.js";
				script.onload = script.onerror = done;
				document.head.appendChild(script);
			});
		}
		return katexReady;
	}

	function load(figure) {
		var url = new URL(figure.dataset.src, window.location.href).href;
		loadKatex().then(function () {
			return import(url);
		}).then(function (module) {
			var stage = figure.querySelector(".interactive-stage");
			var root = document.createElement("div");
			root.className = "interactive-live";
			stage.appendChild(root);
			return Promise.resolve(module.default(root, { figure: figure, base: url })).then(function () {
				figure.classList.add("is-live");
			});
		}).catch(function (e) {
			console.warn("Interactive figure failed, keeping the static image:", url, e);
		});
	}

	if ("IntersectionObserver" in window) {
		var observer = new IntersectionObserver(function (entries) {
			entries.forEach(function (entry) {
				if (entry.isIntersecting) {
					observer.unobserve(entry.target);
					load(entry.target);
				}
			});
		}, { rootMargin: "400px 0px" });
		figures.forEach(function (f) {
			observer.observe(f);
		});
	} else {
		figures.forEach(load);
	}
})();
