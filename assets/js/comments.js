"use strict";

/* Comments under a chapter: the section at the end of _dev/post-template.html, the API in
   _dev/comments-worker/worker.js. Comments are published at once; readers reply under a comment (one level:
   every reply joins the thread of the first comment). Comments are plain text and enter the page through
   textContent only, never as HTML. Links in them stay text.
   On the local preview (_dev/serve.py) the list also shows held comments, with buttons to approve, answer or
   delete; serve.py does the work through _dev/comments.py and the owner's Cloudflare login. */
(function () {
	var section = document.querySelector("[data-comments]");
	if (!section) {
		return;
	}
	var page = section.dataset.comments;
	var api = section.dataset.api.replace(/\/+$/, "") + "/comments";
	var author = section.dataset.author;
	var local = location.hostname === "127.0.0.1" || location.hostname === "localhost";
	var moderating = false;  // the list came from the preview server (_dev/serve.py), with the held comments
	var list = section.querySelector(".comments-list");
	var empty = section.querySelector(".comments-empty");
	var count = section.querySelector(".comments-count");
	var form = section.querySelector(".comment-form");
	var emptyText = empty.textContent;
	var comments = [];
	var adminStatus = null;
	var DATE = { month: "short", day: "numeric", year: "numeric" };

	function el(tag, className, text) {
		var node = document.createElement(tag);
		if (className) {
			node.className = className;
		}
		if (text !== undefined) {
			node.textContent = text;
		}
		return node;
	}

	function icon(name) {
		var i = el("i", "bi bi-" + name);
		i.setAttribute("aria-hidden", "true");
		return i;
	}

	// Paragraphs at blank lines; single line breaks stay visible (white-space: pre-line in post.css).
	function textBlock(text) {
		var box = el("div", "comment-body");
		text.split(/\n{2,}/).forEach(function (paragraph) {
			box.appendChild(el("p", "", paragraph));
		});
		return box;
	}

	function head(c) {
		var row = el("div", "comment-head");
		row.appendChild(el("span", "comment-name", c.author ? author : (c.name || "Anonymous")));
		if (c.author) {
			row.appendChild(el("span", "comment-badge", "Author"));
		} else if (c.status === "held") {
			row.appendChild(el("span", "comment-badge", "Held"));
		}
		var date = new Date(c.created);
		if (!isNaN(date)) {
			var time = el("time", "comment-date", date.toLocaleDateString("en-US", DATE));
			time.dateTime = c.created;
			row.appendChild(time);
		}
		return row;
	}

	function entry(c) {
		var item = el("li", "comment" + (c.author ? " comment-by-author" : "") + (c.status === "held" ? " comment-held" : ""));
		item.id = "comment-" + c.id;
		item.appendChild(head(c));
		item.appendChild(textBlock(c.body));
		if (moderating) {
			item.appendChild(tools(c));
		}
		return item;
	}

	// Threads: each first comment, then its replies, then a Reply button for the thread.
	function render() {
		list.replaceChildren();
		var replies = {};
		comments.forEach(function (c) {
			if (c.parent_id) {
				(replies[c.parent_id] = replies[c.parent_id] || []).push(c);
			}
		});
		var shown = 0;
		comments.forEach(function (c) {
			if (c.parent_id) {
				return;
			}
			var item = entry(c);
			var thread = replies[c.id] || [];
			if (thread.length) {
				var sub = el("ol", "comment-replies");
				thread.forEach(function (r) {
					sub.appendChild(entry(r));
				});
				item.appendChild(sub);
			}
			if (c.status !== "held") {
				item.appendChild(replyButton(c));
			}
			list.appendChild(item);
		});
		comments.forEach(function (c) {
			shown += c.status === "held" ? 0 : 1;
		});
		count.textContent = shown ? "(" + shown + ")" : "";
		empty.textContent = emptyText;
		empty.hidden = comments.length > 0;
	}

	function load(then) {
		var query = "?page=" + encodeURIComponent(page);
		if (!comments.length) {
			empty.textContent = "Loading comments…";
			empty.hidden = false;
		}
		var reading = local ? fetch("/__comments" + query).then(function (r) {
			moderating = r.status !== 404;  // a plain static server has no moderation route: the public list
			return moderating ? r : fetch(api + query);
		}) : fetch(api + query, { credentials: "omit" });
		reading.then(function (r) {
			if (!r.ok) {
				throw new Error(String(r.status));
			}
			return r.json();
		}).then(function (data) {
			comments = Array.isArray(data.comments) ? data.comments : [];
			render();
			var target = location.hash.indexOf("#comment-") === 0 && document.getElementById(location.hash.slice(1));
			if (target) {
				target.scrollIntoView();
			}
			if (then) {
				then();
			}
		}).catch(function () {
			empty.textContent = "The comments could not be loaded. Please try again later.";
			empty.hidden = false;
		});
	}

	// Load when the reader comes near the section, so readers who stop earlier send no request.
	if ("IntersectionObserver" in window && location.hash.indexOf("#comment") !== 0) {
		var watcher = new IntersectionObserver(function (entries) {
			if (entries.some(function (e) { return e.isIntersecting; })) {
				watcher.disconnect();
				load();
			}
		}, { rootMargin: "800px 0px" });
		watcher.observe(section);
	} else {
		load();
	}

	/* ---------------------------------------------------------------- writing */
	// Sends the fields of a form (the main one, or a reply form) and shows the published comment in place.
	function attach(box, parent) {
		var nameField = box.querySelector("[data-field=name]");
		var bodyField = box.querySelector("[data-field=body]");
		var trapField = box.querySelector("[data-field=website]");
		var status = box.querySelector(".comment-status");
		var send = box.querySelector("button[type=submit]");

		function say(message, isError) {
			status.textContent = message;
			status.classList.toggle("is-error", Boolean(isError));
		}

		box.addEventListener("submit", function (event) {
			event.preventDefault();
			if (bodyField.value.trim().length < 2) {
				say("Please write a comment first.", true);
				bodyField.focus();
				return;
			}
			send.disabled = true;
			say("Sending…");
			fetch(api, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				credentials: "omit",
				body: JSON.stringify({ page: page, parent: parent, name: nameField.value, body: bodyField.value,
					website: trapField.value }),
			}).then(function (r) {
				return r.json().catch(function () { return {}; }).then(function (data) {
					if (!r.ok) {
						var err = new Error(typeof data.error === "string" ? data.error : "");
						err.fromServer = true;
						throw err;
					}
					return data;
				});
			}).then(function (data) {
				bodyField.value = "";
				if (data.status === "published" && data.comment && !moderating) {
					comments.push(data.comment);
					show(data.comment.id, box);
				} else if (data.status === "published") {
					load(function () { show(data.comment.id, box); });
				} else {
					say("Thank you. Many comments arrived in the last hour, so yours will appear after review.");
				}
			}).catch(function (err) {
				say(err.fromServer && err.message ? err.message
					: "The comment could not be sent. Please check your connection and try again.", true);
			}).then(function () {
				send.disabled = false;
			});
		});
	}

	// After sending: the comment in the list, its link in the address bar.
	function show(id, box) {
		var main = box === form;
		render();
		if (main) {
			var status = form.querySelector(".comment-status");
			status.textContent = "Your comment is published.";
			status.classList.remove("is-error");
		}
		var item = document.getElementById("comment-" + id);
		if (item) {
			history.replaceState(null, "", "#comment-" + id);
			item.classList.add("comment-new");
			item.scrollIntoView({ block: "center" });
		}
	}

	var replyCount = 0;
	// The reply form of a thread, built from the fields of the main form.
	function replyButton(c) {
		var button = el("button", "comment-reply-button");
		button.type = "button";
		button.appendChild(icon("reply"));
		button.appendChild(el("span", "", "Reply"));
		button.addEventListener("click", function () {
			var open = section.querySelector(".comment-reply-form");
			if (open) {
				open.remove();
			}
			replyCount += 1;
			var box = el("form", "comment-reply-form");
			box.noValidate = true;
			[["name", "Name", "input", " (optional)"], ["body", "Reply", "textarea", ""]].forEach(function (f) {
				var field = el("div", "comment-field");
				var label = el("label", "", f[1]);
				var input = el(f[2]);
				input.id = "reply-" + f[0] + "-" + replyCount;
				input.dataset.field = f[0];
				label.htmlFor = input.id;
				if (f[3]) {
					label.appendChild(el("span", "comment-optional", f[3]));
				}
				if (f[0] === "name") {
					input.type = "text";
					input.maxLength = 50;
					input.autocomplete = "name";
					input.value = form.querySelector("[data-field=name]").value;
				} else {
					input.rows = 3;
					input.maxLength = 3000;
				}
				field.appendChild(label);
				field.appendChild(input);
				box.appendChild(field);
			});
			var trap = el("div", "comment-trap");
			trap.setAttribute("aria-hidden", "true");
			var trapInput = el("input");
			trapInput.type = "text";
			trapInput.tabIndex = -1;
			trapInput.autocomplete = "off";
			trapInput.dataset.field = "website";
			trap.appendChild(trapInput);
			box.appendChild(trap);
			var actions = el("div", "comment-actions");
			var send = el("button", "action-button");
			send.type = "submit";
			send.appendChild(el("span", "action-button-text", "Send reply"));
			send.appendChild(icon("arrow-right"));
			var cancel = el("button", "paper-button", "Cancel");
			cancel.type = "button";
			cancel.addEventListener("click", function () {
				box.remove();
				button.focus();
			});
			var status = el("p", "comment-status");
			status.setAttribute("role", "status");
			actions.appendChild(send);
			actions.appendChild(cancel);
			actions.appendChild(status);
			box.appendChild(actions);
			attach(box, c.id);
			button.after(box);
			box.querySelector("textarea").focus();
		});
		return button;
	}

	form.hidden = false;
	attach(form, null);

	/* ---------------------------------------------------------------- local preview: moderation */
	function tool(label, iconName, run) {
		var button = el("button", "paper-button comment-tool");
		button.type = "button";
		button.appendChild(icon(iconName));
		button.appendChild(el("span", "", label));
		button.addEventListener("click", function () {
			run(button);
		});
		return button;
	}

	function tools(c) {
		var bar = el("div", "comment-tools");
		if (c.status === "held") {
			bar.appendChild(tool("Approve", "check2", function () {
				act("approve", { id: c.id });
			}));
		}
		bar.appendChild(tool("Answer as author", "reply", function () {
			if (bar.nextElementSibling && bar.nextElementSibling.classList.contains("comment-author-form")) {
				return;
			}
			var box = el("div", "comment-author-form");
			var area = el("textarea");
			area.rows = 4;
			area.maxLength = 5000;
			area.setAttribute("aria-label", "Your answer");
			box.appendChild(area);
			box.appendChild(tool("Publish answer", "check2", function () {
				act("reply", { id: c.id, text: area.value });
			}));
			bar.after(box);
			area.focus();
		}));
		bar.appendChild(tool(c.parent_id ? "Delete" : "Delete with replies", "trash", function (button) {
			if (button.dataset.armed) {
				act("delete", { id: c.id });
			} else {
				button.dataset.armed = "true";
				button.lastChild.textContent = "Delete for good?";
			}
		}));
		return bar;
	}

	function act(action, data) {
		if (!adminStatus) {
			adminStatus = el("p", "comment-status comment-admin-status");
			adminStatus.setAttribute("role", "status");
			list.before(adminStatus);
		}
		adminStatus.classList.remove("is-error");
		adminStatus.textContent = "Working… (Wrangler takes a few seconds)";
		section.querySelectorAll(".comment-tool").forEach(function (b) {
			b.disabled = true;
		});
		fetch("/__comments/" + action, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify(data),
		}).then(function (r) {
			return r.json().catch(function () { return {}; }).then(function (answer) {
				if (!r.ok) {
					throw new Error(answer.error || "Failed (" + r.status + ")");
				}
			});
		}).then(function () {
			adminStatus.textContent = "Done.";
			load();
		}).catch(function (err) {
			adminStatus.textContent = err.message;
			adminStatus.classList.add("is-error");
			section.querySelectorAll(".comment-tool").forEach(function (b) {
				b.disabled = false;
			});
		});
	}
})();
