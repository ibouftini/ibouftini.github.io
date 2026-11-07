"use strict";

/* --------------------------------------------------------------------------
   Theme toggle
   assets/js/theme-init.js, loaded at the top of <body>, applies the theme
   before the first paint. This file wires up the toggle buttons and keeps
   everything in sync.
   Default: follow the visitor's system setting. A click stores an explicit
   choice; picking the system theme again goes back to following the system.
   -------------------------------------------------------------------------- */
(function () {
	var KEY = "theme";
	var body = document.body;
	var system = window.matchMedia("(prefers-color-scheme: dark)");
	var buttons = document.querySelectorAll(".theme-button");
	var themeColor = document.querySelector('meta[name="theme-color"]');

	function systemTheme() {
		return system.matches ? "dark" : "light";
	}

	function savedTheme() {
		try {
			var t = localStorage.getItem(KEY);
			return t === "light" || t === "dark" ? t : null;
		} catch (e) {
			return null;
		}
	}

	function saveTheme(theme) {
		try {
			if (theme === systemTheme()) {
				localStorage.removeItem(KEY);
			} else {
				localStorage.setItem(KEY, theme);
			}
		} catch (e) {}
	}

	function apply(theme) {
		var dark = theme === "dark";
		body.classList.toggle("theme-dark", dark);
		document.documentElement.style.colorScheme = theme;
		if (themeColor) {
			themeColor.content = dark ? "#000000" : "#f2f5f8";
		}
		// Images with a dark-theme variant: <img src="light.png" data-src-dark="dark.png">
		document.querySelectorAll("img[data-src-dark]").forEach(function (img) {
			if (!img.dataset.srcLight) {
				img.dataset.srcLight = img.getAttribute("src");
			}
			img.setAttribute("src", dark ? img.dataset.srcDark : img.dataset.srcLight);
		});
		buttons.forEach(function (button) {
			button.setAttribute("aria-pressed", dark ? "true" : "false");
			button.title = dark ? "Switch to light theme" : "Switch to dark theme";
		});
	}

	// Reveal the new theme as a circle growing from the button that was clicked.
	function switchTheme(theme, origin) {
		var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
		if (!document.startViewTransition || reduceMotion || !origin) {
			apply(theme);
			return;
		}
		var rect = origin.getBoundingClientRect();
		var x = rect.left + rect.width / 2;
		var y = rect.top + rect.height / 2;
		var radius = Math.hypot(Math.max(x, window.innerWidth - x), Math.max(y, window.innerHeight - y));
		document.startViewTransition(function () {
			apply(theme);
		}).ready.then(function () {
			document.documentElement.animate(
				{ clipPath: ["circle(0px at " + x + "px " + y + "px)", "circle(" + radius + "px at " + x + "px " + y + "px)"] },
				{ duration: 550, easing: "cubic-bezier(.4, 0, .2, 1)", pseudoElement: "::view-transition-new(root)" }
			);
		});
	}

	apply(savedTheme() || systemTheme());

	buttons.forEach(function (button) {
		button.addEventListener("click", function () {
			var next = body.classList.contains("theme-dark") ? "light" : "dark";
			saveTheme(next);
			switchTheme(next, button);
		});
	});

	system.addEventListener("change", function () {
		if (!savedTheme()) {
			apply(systemTheme());
		}
	});

	// Keep other open tabs in sync.
	window.addEventListener("storage", function (e) {
		if (e.key === KEY) {
			apply(savedTheme() || systemTheme());
		}
	});
})();

/* --------------------------------------------------------------------------
   Side menu: the three dots open it; the close button or a click anywhere
   outside closes it.
   -------------------------------------------------------------------------- */
(function () {
	var menu = document.querySelector(".toggle-menu");
	var dots = document.querySelector(".menu-dots");
	var close = document.querySelector(".toggle-close");
	if (!menu || !dots) {
		return;
	}
	function setOpen(open) {
		menu.classList.toggle("show", open);
		dots.classList.toggle("active", open);
	}
	dots.addEventListener("click", function () {
		setOpen(!menu.classList.contains("show"));
	});
	if (close) {
		close.addEventListener("click", function () {
			setOpen(false);
		});
	}
	document.addEventListener("click", function (e) {
		if (!e.target.closest(".toggle-menu, .menu-dots")) {
			setOpen(false);
		}
	});
})();

/* --------------------------------------------------------------------------
   Section pages (homepage, CV: [data-spy]): the section nav highlights the
   section being read, the lowest one with a tenth of its height in the top
   three quarters of the screen. Its links jump to 24 px above the section
   and leave the address bar alone.
   -------------------------------------------------------------------------- */
(function () {
	if (!document.querySelector("[data-spy]")) {
		return;
	}
	var links = Array.prototype.slice.call(document.querySelectorAll('.section-nav a[href^="#"]'));
	var sections = links.map(function (link) {
		return document.getElementById(link.hash.slice(1));
	});

	links.forEach(function (link, i) {
		link.addEventListener("click", function (e) {
			e.preventDefault();
			if (sections[i]) {
				window.scrollTo(0, sections[i].getBoundingClientRect().top + window.scrollY - 24);
			}
		});
	});

	var current = -1;
	function spy() {
		var bottom = window.innerHeight * 0.75;
		var active = current;
		sections.forEach(function (section, i) {
			if (!section) {
				return;
			}
			var rect = section.getBoundingClientRect();
			var visible = Math.min(rect.bottom, bottom) - Math.max(rect.top, 0);
			if (visible >= rect.height * 0.1) {
				active = i;
			}
		});
		if (active === current) {
			return;
		}
		current = active;
		links.forEach(function (link, i) {
			link.classList.toggle("active", i === current);
		});
	}
	window.addEventListener("scroll", spy, { passive: true });
	window.addEventListener("resize", spy);
	spy();
})();

/* --------------------------------------------------------------------------
   Typed greeting on the portrait: <span class="typer" data-words="a, b">
   types each word, holds it, deletes it, and loops; a .cursor whose
   data-owner is the typer's id blinks beside it.
   -------------------------------------------------------------------------- */
(function () {
	document.querySelectorAll(".typer").forEach(function (el) {
		var words = (el.dataset.words || "").split(",").filter(Boolean);
		var delay = Number(el.dataset.delay) || 200;
		var hold = Number(el.dataset.deletedelay) || 800;
		var cursor = el.id && document.querySelector('.cursor[data-owner="' + el.id + '"]');
		var word = 0, chars = 0, building = true;
		var blink;
		if (!words.length) {
			return;
		}
		if (cursor) {
			cursor.textContent = "_";
			cursor.style.transition = "all 0.1s";
		}
		function restartBlink() {
			if (!cursor) {
				return;
			}
			clearInterval(blink);
			cursor.style.opacity = "1";
			blink = setInterval(function () {
				cursor.style.opacity = cursor.style.opacity === "0" ? "1" : "0";
			}, 400);
		}
		function step() {
			var full = Array.from(words[word]);
			var done = false;
			restartBlink();
			el.textContent = full.slice(0, chars).join("");
			if (building) {
				done = chars === full.length;
				if (done) {
					building = false;
				} else {
					chars += 1;
				}
			} else if (chars === 0) {
				building = true;
				word = (word + 1) % words.length;
			} else {
				chars -= 1;
			}
			setTimeout(step, done ? hold : delay);
		}
		step();
	});
})();

/* --------------------------------------------------------------------------
   Close the side menu after following one of its in-page links.
   -------------------------------------------------------------------------- */
(function () {
	var menu = document.querySelector(".toggle-menu");
	var dots = document.querySelector(".menu-dots");
	if (!menu) {
		return;
	}
	menu.querySelectorAll('a[href*="#"]').forEach(function (link) {
		link.addEventListener("click", function () {
			menu.classList.remove("show");
			if (dots) {
				dots.classList.remove("active");
			}
		});
	});
})();

/* --------------------------------------------------------------------------
   Contact card: email, phone, LinkedIn and GitHub, always shown together.
   Every [data-contact-card] element is filled from CONTACTS below, so the
   side menu, the homepage and the CV page list the same three rows.
   The email address is public and shown as soon as the page loads. It is
   stored here reversed and base64 encoded, so bots that search the source for
   addresses find nothing.
   -------------------------------------------------------------------------- */
(function () {
	var CONTACTS = [
		{ key: "email", label: "Email", icon: "bi-envelope-fill", data: "bW9jLmxpYW1nQG9ycC5pbml0ZnVvYmVkYW1p", name: "email" },
		{ key: "linkedin", label: "LinkedIn", icon: "bi-linkedin", href: "https://www.linkedin.com/in/imade-bouftini", text: "imade-bouftini" },
		{ key: "github", label: "GitHub", icon: "bi-github", href: "https://github.com/ibouftini", text: "ibouftini" }
	];
	var canCopy = !!(navigator.clipboard && window.isSecureContext);

	function decode(data) {
		return atob(data).split("").reverse().join("");
	}

	function copyButton(value, name) {
		var copy = document.createElement("button");
		copy.type = "button";
		copy.className = "copy-button";
		copy.title = "Copy " + name;
		copy.setAttribute("aria-label", "Copy " + name);
		copy.innerHTML = '<i class="bi bi-copy" aria-hidden="true"></i>';
		copy.addEventListener("click", function () {
			navigator.clipboard.writeText(value).then(function () {
				copy.classList.add("copied");
				copy.title = "Copied";
				copy.firstChild.className = "bi bi-check2";
				setTimeout(function () {
					copy.classList.remove("copied");
					copy.title = "Copy " + name;
					copy.firstChild.className = "bi bi-copy";
				}, 1600);
			});
		});
		return copy;
	}

	function emailLink(contact) {
		var value = decode(contact.data);
		var wrap = document.createElement("span");
		var link = document.createElement("a");
		wrap.className = "contact-email";
		link.href = "mailto:" + value;
		link.textContent = value;
		wrap.appendChild(link);
		if (canCopy) {
			wrap.appendChild(copyButton(value, contact.name));
		}
		return wrap;
	}

	function row(contact) {
		var item = document.createElement("li");
		var value = document.createElement("span");
		item.className = "contact-item";
		item.innerHTML = '<span class="contact-icon" aria-hidden="true"><i class="bi ' + contact.icon + '"></i></span>' +
			'<span class="contact-label">' + contact.label + "</span>";
		value.className = "contact-value";
		if (contact.href) {
			value.innerHTML = '<a href="' + contact.href + '" target="_blank" rel="noopener me">' + contact.text +
				'<i class="bi bi-arrow-up-right" aria-hidden="true"></i></a>';
		} else {
			value.appendChild(emailLink(contact));
		}
		item.appendChild(value);
		return item;
	}

	document.querySelectorAll("[data-contact-card]").forEach(function (card) {
		var list = document.createElement("ul");
		list.className = "contact-list";
		CONTACTS.forEach(function (contact) {
			list.appendChild(row(contact));
		});
		card.appendChild(list);
	});

})();

/* --------------------------------------------------------------------------
   Navigation that follows the reader, on every page and at every width:
   - phones and tablets: the in-page bar (sections, index or contents)
     sticks to the top once the header has scrolled away, and follows
     resizes.
   - everywhere: a copy of the three dots stays in the top right corner so
     the pages menu is always one tap away. On phones and tablets it sits at
     the end of the sticky bar instead of on top of it.
   (Desktop in-page navigation is the sticky left column, see theme.css.)
   -------------------------------------------------------------------------- */
(function () {
	var header = document.getElementById("header");
	var dots = header && header.querySelector(".menu-dots");
	var menu = document.querySelector(".toggle-menu");
	if (!dots || !menu) {
		return;
	}
	var root = document.documentElement;
	var bars = document.querySelectorAll(".section-nav, .post-toc-bar");
	var narrow = window.matchMedia("(max-width: 991.98px)");
	var float = dots.cloneNode(true);
	float.classList.add("menu-dots-float");
	float.classList.remove("active");
	float.setAttribute("tabindex", "-1");
	float.addEventListener("click", function (e) {
		e.stopPropagation();
		dots.click();
	});
	document.body.appendChild(float);

	// Mirror the open state of the menu, whichever control opened or closed it.
	new MutationObserver(function () {
		float.classList.toggle("active", menu.classList.contains("show"));
	}).observe(menu, { attributes: true, attributeFilter: ["class"] });

	function measure() {
		root.style.setProperty("--dots-right", (root.clientWidth - dots.getBoundingClientRect().right) + "px");
		bars.forEach(function (bar) {
			if (bar.offsetHeight) {
				root.style.setProperty("--bar-height", bar.offsetHeight + "px");
			}
		});
		update();
	}
	function update() {
		var floating = window.scrollY >= header.offsetHeight;
		document.body.classList.toggle("menu-floating", floating);
		bars.forEach(function (bar) {
			bar.classList.toggle("fixed", floating && narrow.matches);
		});
		float.setAttribute("tabindex", floating ? "0" : "-1");
		float.setAttribute("aria-hidden", floating ? "false" : "true");
	}
	window.addEventListener("scroll", update, { passive: true });
	window.addEventListener("resize", measure);
	measure();
})();

/* --------------------------------------------------------------------------
   Listing pages (courses.html, blog.html): the index in the left column and
   in the sticky bar highlights where the reader is, and its links (and a
   link from another page, such as courses.html#course-generative-models)
   land below the sticky bar on phones and tablets.
   -------------------------------------------------------------------------- */
(function () {
	var page = document.querySelector(".listing");
	if (!page) {
		return;
	}
	var navs = document.querySelectorAll("[data-index]");
	var narrow = window.matchMedia("(max-width: 991.98px)");

	function scrollToTarget(el) {
		var offset = narrow.matches ? 90 : 24;
		window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY - offset);
	}

	navs.forEach(function (nav) {
		nav.querySelectorAll('a[href^="#"]').forEach(function (link) {
			link.addEventListener("click", function (e) {
				var target = document.getElementById(link.hash.slice(1));
				if (target) {
					e.preventDefault();
					scrollToTarget(target);
					history.replaceState(null, "", link.hash);
				}
			});
		});
	});

	// Highlight the index entry of the part of the page being read.
	function spy() {
		navs.forEach(function (nav) {
			var links = nav.querySelectorAll('a[href^="#"]');
			var active = 0;
			links.forEach(function (link, i) {
				var target = document.getElementById(link.hash.slice(1));
				if (target && target.getBoundingClientRect().top < 140) {
					active = i;
				}
			});
			links.forEach(function (link, i) {
				link.classList.toggle("active", i === active);
			});
		});
	}
	window.addEventListener("scroll", spy, { passive: true });
	spy();

	var hashTarget = location.hash && document.getElementById(decodeURIComponent(location.hash.slice(1)));
	if (hashTarget) {
		requestAnimationFrame(function () {
			scrollToTarget(hashTarget);
		});
	}
})();

/* --------------------------------------------------------------------------
   Page transition, after the template's fade. Each page opens under a layer
   in the page colour that fades out (pure CSS, see custom.css). Following a
   link to another page in the same tab fades the layer back in while the
   next page loads. In-page links, new tabs, downloads, PDFs, mailto: and
   tel: links are left alone.
   -------------------------------------------------------------------------- */
(function () {
	var root = document.documentElement;
	var reduce = window.matchMedia("(prefers-reduced-motion: reduce)");
	var timer;
	document.addEventListener("click", function (e) {
		if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || reduce.matches) {
			return;
		}
		var link = e.target.closest("a[href]");
		if (!link || (link.target && link.target !== "_self") || link.hasAttribute("download")) {
			return;
		}
		var url = new URL(link.href, location.href);
		if (!/^https?:$/.test(url.protocol) || /\.pdf$/i.test(url.pathname)) {
			return;
		}
		if (url.origin === location.origin && url.pathname === location.pathname && url.search === location.search && url.hash) {
			return;
		}
		root.classList.add("page-leaving");
		// If the browser stays on this page after all, uncover it.
		clearTimeout(timer);
		timer = setTimeout(function () {
			root.classList.remove("page-leaving");
		}, 5000);
	});
	// Back/forward cache: the page comes back covered, so fade it in again.
	window.addEventListener("pageshow", function (e) {
		if (e.persisted) {
			clearTimeout(timer);
			root.classList.remove("page-leaving");
		}
	});
})();

/* --------------------------------------------------------------------------
   Back-to-top button, shown after one screen of scrolling (style: custom.css).
   -------------------------------------------------------------------------- */
(function () {
	var button = document.createElement("button");
	button.type = "button";
	button.className = "back-to-top";
	button.title = "Back to top";
	button.setAttribute("aria-label", "Back to top");
	button.innerHTML = '<i class="bi bi-arrow-up" aria-hidden="true"></i>';
	document.body.appendChild(button);

	function update() {
		button.classList.toggle("is-visible", window.scrollY > window.innerHeight);
	}
	button.addEventListener("click", function () {
		var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
		window.scrollTo({ top: 0, behavior: reduce ? "auto" : "smooth" });
	});
	window.addEventListener("scroll", update, { passive: true });
	update();
})();
