"use strict";

/* Apply the saved or system theme before the first paint. Loaded by a plain <script> at the top of <body> on every
   page (the Content-Security-Policy of the pages allows no inline script). Toggle logic: assets/js/custom.js. */
(function () {
	var t;
	try { t = localStorage.getItem("theme"); } catch (e) {}
	if (t !== "light" && t !== "dark") t = matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
	if (t === "dark") document.body.classList.add("theme-dark");
	document.documentElement.style.colorScheme = t;
	document.querySelector('meta[name="theme-color"]').content = t === "dark" ? "#000000" : "#f2f5f8";
})();
