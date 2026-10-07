/* CogniKernel site — interactions. No dependencies.
   Every diagram here is drawn from data, and every animated state means
   something: a node arriving is a memory captured, gold is a memory recalled,
   a kernel pulse is the store accepting or serving it. */
(function () {
  "use strict";

  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var NS = "http://www.w3.org/2000/svg";
  var reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var hasIO = "IntersectionObserver" in window;
  var esc = function (t) { return String(t).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); };
  var store = {
    get: function (k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set: function (k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
  };
  function svgEl(tag, attrs, parent) {
    var el = document.createElementNS(NS, tag);
    for (var a in attrs) el.setAttribute(a, attrs[a]);
    if (parent) parent.appendChild(el);
    return el;
  }
  function whenVisible(el, fn, threshold) {
    if (!el) return;
    if (!hasIO) { fn(); return; }
    var io = new IntersectionObserver(function (e) {
      if (e[0].isIntersecting) { io.disconnect(); fn(); }
    }, { threshold: threshold || 0.25 });
    io.observe(el);
  }
  var easeOut = function (t) { return 1 - Math.pow(1 - t, 3); };

  /* Memory-type glyphs, drawn into an SVG group at the origin. Shape encodes
     the type; the recalled state recolors it gold. */
  function drawGlyph(g, type, size) {
    var s = size || 5;
    if (type === "dec") return [svgEl("circle", { r: s }, g)];
    if (type === "hard") return [svgEl("path", { d: "M0 " + -s * 1.25 + " L" + s * 1.25 + " 0 L0 " + s * 1.25 + " L" + -s * 1.25 + " 0Z" }, g)];
    if (type === "soft") return [svgEl("path", { d: "M0 " + -s * 1.1 + " L" + s * 1.1 + " 0 L0 " + s * 1.1 + " L" + -s * 1.1 + " 0Z", "stroke-width": 1.5, fill: "none" }, g)];
    return [
      svgEl("circle", { r: s * 0.95, fill: "none", "stroke-width": 1.4 }, g),
      svgEl("path", { d: "M" + -s * 0.7 + " " + s * 0.7 + " L" + s * 0.7 + " " + -s * 0.7, "stroke-width": 1.4 }, g)
    ];
  }
  function paintGlyph(parts, type, color) {
    parts.forEach(function (p) {
      var filled = type === "dec" || type === "hard";
      if (filled) { p.setAttribute("fill", color); p.setAttribute("stroke", "none"); }
      else { p.setAttribute("stroke", color); if (p.tagName === "circle" || type === "soft") p.setAttribute("fill", "none"); }
    });
  }
  var TYPE_NAME = { dec: "Decision", hard: "Hard constraint", soft: "Soft constraint", dead: "Abandoned approach" };
  var GLYPH_ID = { dec: "g-dec", hard: "g-hard", soft: "g-soft", dead: "g-dead" };
  function glyphHTML(type, gold) {
    var cls = gold ? "glyph g-gold" : type === "dead" ? "glyph g-dead" : "glyph";
    return '<svg class="' + cls + '"><use href="#' + (gold ? "g-gold" : GLYPH_ID[type]) + '"/></svg>';
  }

  /* ── theme ─────────────────────────────────────────────────────────── */
  var root = document.documentElement;
  function currentTheme() {
    if (root.dataset.theme) return root.dataset.theme;
    return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  }
  $("#theme-toggle").addEventListener("click", function () {
    var next = currentTheme() === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    store.set("ck-theme", next);
  });

  /* ── nav: background on scroll, mobile menu, scroll-spy ────────────── */
  var nav = $("#nav");
  var onScroll = function () { nav.classList.toggle("scrolled", window.scrollY > 12); };
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();
  var menuBtn = $("#menu-btn"), links = $("#nav-links");
  menuBtn.addEventListener("click", function () {
    var open = links.classList.toggle("open");
    menuBtn.setAttribute("aria-expanded", String(open));
  });
  $$("a", links).forEach(function (a) {
    a.addEventListener("click", function () { links.classList.remove("open"); menuBtn.setAttribute("aria-expanded", "false"); });
  });
  var spyMap = {};
  $$("a", links).forEach(function (a) { spyMap[a.getAttribute("href").slice(1)] = a; });
  // Sections without their own nav entry light up the entry they belong to.
  var spyAlias = { problem: "features", evolution: "memory-model", block: "memory-model", models: "memory-model", internals: "benchmarks", codex: "benchmarks", "get-started": "docs" };
  if (hasIO) {
    var spy = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting) return;
        var id = spyAlias[e.target.id] || e.target.id;
        $$("a", links).forEach(function (a) { a.classList.toggle("active", a === spyMap[id]); });
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    $$("main section[id]").forEach(function (s) { spy.observe(s); });
  }

  /* ── reveal on scroll ──────────────────────────────────────────────── */
  var revealEls = $$(".reveal");
  if (reduced || !hasIO) {
    revealEls.forEach(function (el) { el.classList.add("in"); });
  } else {
    var ro = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add("in"); ro.unobserve(e.target); } });
    }, { rootMargin: "0px 0px -8% 0px" });
    revealEls.forEach(function (el, i) { el.style.transitionDelay = (i % 4) * 90 + "ms"; ro.observe(el); });
  }

  /* ── copy buttons ──────────────────────────────────────────────────── */
  document.addEventListener("click", function (ev) {
    var btn = ev.target.closest(".copy");
    if (!btn) return;
    var text = btn.dataset.copy;
    if (!text && btn.dataset.copyFrom) {
      text = $("#" + btn.dataset.copyFrom).textContent.replace(/^\$\s*/, "").replace(/\s+#.*$/, "");
    }
    var done = function () { btn.classList.add("done"); setTimeout(function () { btn.classList.remove("done"); }, 1400); };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, function () {});
    } else {
      var ta = document.createElement("textarea");
      ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
      document.body.appendChild(ta); ta.select();
      try { document.execCommand("copy"); done(); } catch (e) {}
      document.body.removeChild(ta);
    }
  });

  /* ── live repo facts (best-effort; the page reads fine without them) ─ */
  if (window.fetch) {
    fetch("https://api.github.com/repos/KanishkNoir/cognikernel", { headers: { Accept: "application/vnd.github+json" } })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (!d || typeof d.stargazers_count !== "number") return;
        var n = d.stargazers_count;
        var s = n >= 1000 ? (n / 1000).toFixed(1).replace(/\.0$/, "") + "k" : String(n);
        $("#stars").textContent = s;
        $("#stat-stars").textContent = "★ " + s + " on GitHub · live";
      })
      .catch(function () {});
    fetch("https://pypi.org/pypi/cognikernel/json")
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (!d || !d.info || !d.info.version) return;
        $("#ver").textContent = "v" + d.info.version;
        $("#stat-ver").textContent = d.info.version;
      })
      .catch(function () {});
  }

  /* ── footer banner: the README's ASCII art, redrawn as SVG ───────────
     Box-drawing glyphs don't tile seamlessly in web fonts, so the <pre> is
     parsed into cells: █ becomes a block, the shadow characters strokes. */
  (function () {
    var pre = $(".ascii");
    if (!pre) return;
    var rows = pre.textContent.replace(/^\n+|\n+$/g, "").split("\n");
    var cw = 10, ch = 20, cols = 0;
    rows.forEach(function (r) { cols = Math.max(cols, r.length); });
    var segs = { "═": "lr", "║": "ud", "╔": "rd", "╗": "ld", "╚": "ru", "╝": "lu" };
    var blocks = "", lines = "";
    rows.forEach(function (r, y) {
      for (var x = 0; x < r.length; x++) {
        var c = r[x], X = x * cw, Y = y * ch;
        if (c === "█") { blocks += "M" + X + "," + Y + "h" + cw + "v" + ch + "h-" + cw + "z"; continue; }
        var s = segs[c]; if (!s) continue;
        var mx = X + cw / 2, my = Y + ch / 2;
        for (var k = 0; k < s.length; k++) {
          var d = s[k];
          lines += "M" + mx + "," + my + (d === "l" ? "H" + X : d === "r" ? "H" + (X + cw) : d === "u" ? "V" + Y : "V" + (Y + ch));
        }
      }
    });
    var W = cols * cw, H = rows.length * ch;
    pre.insertAdjacentHTML("afterend",
      '<svg class="banner" viewBox="-4 -4 ' + (W + 8) + " " + (H + 8) + '" aria-hidden="true">' +
      '<path d="' + lines + '" fill="none" stroke="var(--blue)" stroke-opacity=".5" stroke-width="2.2" stroke-linecap="square"/>' +
      '<path d="' + blocks + '" fill="var(--ink-3)" shape-rendering="crispEdges"/></svg>');
  })();

  /* ── HERO: the orbital memory map ──────────────────────────────────── */
  // Radius encodes authority: hard constraints orbit closest to the kernel,
  // decisions in the middle, abandoned approaches on the outer ring.
  var HERO = [
    { id: "h1", ring: 0, a: -64, type: "hard", label: "Never log webhook payloads in full", meta: "Stated in sessions 1, 2 and 3 · consolidated (×3)" },
    { id: "h2", ring: 0, a: 118, type: "hard", label: "All retries go through relay/backoff.py", meta: "Session 2 · no ad-hoc sleep loops" },
    { id: "d1", ring: 1, a: -14, type: "dec", label: "Use Redis for the rate limiter", meta: "Session 2 · key ratelimit.backend · superseded the in-process LRU", recall: true },
    { id: "d2", ring: 1, a: 74, type: "dec", label: "Route on tenant_id, not hostname", meta: "Session 3 · custom domains broke host routing", recall: true },
    { id: "d3", ring: 1, a: 166, type: "dec", label: "Dead letters get their own queue, 7-day retention", meta: "Session 1" },
    { id: "s1", ring: 1, a: 244, type: "soft", label: "Prefer dataclasses over dicts", meta: "Session 1 · a convention, stated softly" },
    { id: "x1", ring: 2, a: 26, type: "dead", label: "In-process LRU cache for rate limits", meta: "Abandoned in session 2 · counts drifted across the 4 workers", recall: true },
    { id: "x2", ring: 2, a: 204, type: "dead", label: "Polling upstream every 5s", meta: "Abandoned · hit their 429 ceiling within an hour" }
  ];
  // Semantic links: the LRU and Redis share a decision key; polling and the
  // retry rule are about the same upstream.
  var HERO_LINKS = [["d1", "x1"], ["h2", "x2"], ["d2", "d1"]];
  var QUERY = "add a per-tenant rate limit";

  (function hero() {
    var svg = $("#orbit-svg"), readout = $("#readout");
    if (!svg) return;
    var C = 280, RINGS = [104, 164, 224], SPEED = [1.6, -1.05, 0.6];
    var RING_NAME = ["Hard constraints", "Decisions", "Graveyard"];

    var defs = svgEl("defs", {}, svg);
    var kg = svgEl("radialGradient", { id: "kernel-g" }, defs);
    svgEl("stop", { offset: "0%", "stop-color": "var(--gold)", "stop-opacity": "0.9" }, kg);
    svgEl("stop", { offset: "55%", "stop-color": "var(--gold)", "stop-opacity": "0.3" }, kg);
    svgEl("stop", { offset: "100%", "stop-color": "var(--gold)", "stop-opacity": "0" }, kg);
    ["blue", "gold"].forEach(function (c) {
      var h = svgEl("radialGradient", { id: "halo-" + c }, defs);
      svgEl("stop", { offset: "0%", "stop-color": "var(--" + c + ")", "stop-opacity": "0.55" }, h);
      svgEl("stop", { offset: "100%", "stop-color": "var(--" + c + ")", "stop-opacity": "0" }, h);
    });

    // Instrument frame: outer graduated circle, crosshair, labelled rings.
    svgEl("circle", { cx: C, cy: C, r: 262, class: "o-faint" }, svg);
    for (var deg = 0; deg < 360; deg += 5) {
      var rad = deg * Math.PI / 180, long = deg % 30 === 0;
      var r1 = 262, r2 = long ? 252 : 257;
      svgEl("line", { x1: C + r1 * Math.cos(rad), y1: C + r1 * Math.sin(rad), x2: C + r2 * Math.cos(rad), y2: C + r2 * Math.sin(rad), class: "o-tick", "stroke-opacity": long ? 1 : 0.5 }, svg);
    }
    svgEl("line", { x1: C - 262, y1: C, x2: C + 262, y2: C, class: "o-faint" }, svg);
    svgEl("line", { x1: C, y1: C - 262, x2: C, y2: C + 262, class: "o-faint" }, svg);
    RINGS.forEach(function (r, i) {
      svgEl("circle", { cx: C, cy: C, r: r, class: "o-ring" + (i === 1 ? " dash" : "") }, svg);
      // ring label set along the arc, upper left
      var a0 = -178 * Math.PI / 180, a1 = -96 * Math.PI / 180;
      var rr = r + 7;
      svgEl("path", { id: "ring-p" + i, d: "M" + (C + rr * Math.cos(a0)) + " " + (C + rr * Math.sin(a0)) + " A" + rr + " " + rr + " 0 0 1 " + (C + rr * Math.cos(a1)) + " " + (C + rr * Math.sin(a1)), fill: "none" }, svg);
      var t = svgEl("text", { class: "o-label" }, svg);
      var tp = svgEl("textPath", { href: "#ring-p" + i }, t);
      tp.textContent = RING_NAME[i];
    });

    var linkLayer = svgEl("g", {}, svg);
    var traceLayer = svgEl("g", {}, svg);

    // Kernel
    var kGlow = svgEl("circle", { cx: C, cy: C, r: 64, fill: "url(#kernel-g)", opacity: 0.45 }, svg);
    svgEl("circle", { cx: C, cy: C, r: 26, fill: "var(--gold)", "fill-opacity": 0.12, stroke: "var(--gold)", "stroke-opacity": 0.5 }, svg);
    svgEl("circle", { cx: C, cy: C, r: 11, fill: "var(--gold)" }, svg);
    var kt = svgEl("text", { x: C, y: C + 50, "text-anchor": "middle", class: "o-kernel-text" }, svg);
    kt.textContent = "KERNEL";

    var byId = {};
    var nodes = HERO.map(function (n, i) {
      var g = svgEl("g", { class: "o-node", tabindex: "0", role: "button", "aria-label": TYPE_NAME[n.type] + ": " + n.label }, svg);
      svgEl("circle", { r: 18, class: "hit" }, g);
      var halo = svgEl("circle", { r: 22, fill: "url(#halo-blue)", class: "halo", opacity: 0.22 }, g);
      var ring = svgEl("circle", { r: 12, class: "ring", stroke: "var(--blue)", "stroke-opacity": 0.22 }, g);
      var parts = drawGlyph(svgEl("g", {}, g), n.type, 5);
      var node = { d: n, g: g, halo: halo, ring: ring, parts: parts, angle: n.a, x: 0, y: 0, born: null, alpha: 0, i: i };
      byId[n.id] = node;
      g.setAttribute("opacity", "0");
      g.addEventListener("focus", function () { setHover(node); });
      g.addEventListener("blur", function () { if (hovered === node) setHover(null); });
      g.addEventListener("keydown", function (e) { if (e.key === "Escape") { setHover(null); g.blur(); } });
      return node;
    });
    var links = HERO_LINKS.map(function (l) {
      return { a: byId[l[0]], b: byId[l[1]], el: svgEl("path", { class: "o-arc", "stroke-opacity": 0 }, linkLayer) };
    });
    var traces = nodes.filter(function (n) { return n.d.recall; }).map(function (n) {
      return { n: n, el: svgEl("line", { class: "o-trace", "stroke-opacity": 0 }, traceLayer) };
    });

    var hovered = null, recalling = false, pulse = 0, arrived = 0, start = null, last = null;

    // Hit-testing is done once on the SVG rather than per node: the nodes
    // move, and enter/leave events on moving targets flicker. The nearest
    // memory within reach wins, which also gives touch a generous target.
    function setHover(node) {
      if (hovered === node) return;
      hovered = node;
      svg.style.cursor = node ? "pointer" : "";
      paintAll();
      if (node) showNode(node); else showIdle();
    }
    function nearest(ev) {
      var m = svg.getScreenCTM();
      if (!m) return null;
      var pt = new DOMPoint(ev.clientX, ev.clientY).matrixTransform(m.inverse());
      var best = null, bd = 24 * 24;
      nodes.forEach(function (n) {
        if (n.born === null) return;
        var dx = n.x - pt.x, dy = n.y - pt.y, d = dx * dx + dy * dy;
        if (d < bd) { bd = d; best = n; }
      });
      return best;
    }
    svg.addEventListener("pointermove", function (ev) { if (ev.pointerType === "mouse") setHover(nearest(ev)); });
    svg.addEventListener("pointerleave", function (ev) { if (ev.pointerType === "mouse") setHover(null); });
    // Touch and pen have no hover: a tap selects, a second tap clears.
    var lastType = "mouse";
    svg.addEventListener("pointerdown", function (ev) { lastType = ev.pointerType; });
    svg.addEventListener("click", function (ev) {
      if (lastType === "mouse") return;
      var n = nearest(ev);
      setHover(n === hovered ? null : n);
    });

    function paint(node) {
      var gold = recalling && node.d.recall;
      var active = hovered === node;
      var col = gold ? "var(--gold)" : node.d.type === "dead" ? "var(--ink-3)" : "var(--blue)";
      paintGlyph(node.parts, node.d.type, col);
      node.ring.setAttribute("stroke", gold ? "var(--gold)" : "var(--blue)");
      node.ring.setAttribute("stroke-opacity", active ? 0.8 : gold ? 0.55 : 0.2);
      node.halo.setAttribute("fill", gold ? "url(#halo-gold)" : "url(#halo-blue)");
      node.halo.setAttribute("opacity", active ? 0.85 : gold ? 0.7 : 0.18);
    }
    function paintAll() {
      nodes.forEach(paint);
      traces.forEach(function (t) { t.el.setAttribute("stroke-opacity", recalling || hovered === t.n ? (hovered === t.n ? 0.9 : 0.6) : 0); });
      links.forEach(function (l) {
        var on = hovered && (hovered === l.a || hovered === l.b);
        l.el.setAttribute("stroke-opacity", on ? 0.55 : arrived >= nodes.length ? 0.16 : 0);
      });
    }
    function pos(node) {
      var r = RINGS[node.d.ring];
      var rad = node.angle * Math.PI / 180;
      return { x: C + r * Math.cos(rad), y: C + r * Math.sin(rad), r: r, rad: rad };
    }
    function render() {
      nodes.forEach(function (n) {
        var p = pos(n), x = p.x, y = p.y;
        if (n.born !== null && n.alpha < 1) {
          // spiral in from beyond the outer ring
          var t = easeOut(n.alpha);
          var rr = p.r + (1 - t) * 90, ang = p.rad - (1 - t) * 0.9;
          x = C + rr * Math.cos(ang); y = C + rr * Math.sin(ang);
        }
        n.x = x; n.y = y;
        n.g.setAttribute("transform", "translate(" + x.toFixed(2) + " " + y.toFixed(2) + ")");
        n.g.setAttribute("opacity", n.born === null ? 0 : Math.min(1, n.alpha * 1.6).toFixed(3));
      });
      traces.forEach(function (t) { t.el.setAttribute("x1", t.n.x); t.el.setAttribute("y1", t.n.y); t.el.setAttribute("x2", C); t.el.setAttribute("y2", C); });
      links.forEach(function (l) {
        var mx = (l.a.x + l.b.x) / 2, my = (l.a.y + l.b.y) / 2;
        var cx = C + (mx - C) * 0.35, cy = C + (my - C) * 0.35;
        l.el.setAttribute("d", "M" + l.a.x + " " + l.a.y + " Q" + cx + " " + cy + " " + l.b.x + " " + l.b.y);
      });
      kGlow.setAttribute("opacity", (0.4 + pulse * 0.45).toFixed(3));
      kGlow.setAttribute("r", (64 + pulse * 10).toFixed(2));
    }

    function showIdle() {
      readout.classList.remove("is-gold");
      if (recalling) return showRecall();
      readout.innerHTML = '<span class="eyebrow">Kernel readout · relay</span>' +
        "<p><b>" + arrived + " of " + nodes.length + "</b> memories in orbit, grouped by type: hard constraints closest to the kernel, then decisions, then the graveyard. Hover or tap one to read it.</p>";
    }
    function showRecall() {
      readout.classList.add("is-gold");
      var got = nodes.filter(function (n) { return n.d.recall; });
      readout.innerHTML = '<div class="row"><span class="eyebrow gold">CK-1 recall</span><span class="q">&gt; ' + esc(QUERY) + "</span></div>" +
        "<p>" + got.map(function (n) { return esc(n.d.label); }).join(" · ") + "</p>";
    }
    function showNode(n) {
      var gold = recalling && n.d.recall;
      readout.classList.toggle("is-gold", gold);
      readout.innerHTML = '<div class="row">' + glyphHTML(n.d.type, gold) + '<span class="eyebrow' + (gold ? " gold" : "") + '">' + TYPE_NAME[n.d.type] + (gold ? " · recalled" : "") + "</span></div>" +
        "<p><b>" + esc(n.d.label) + "</b><br>" + esc(n.d.meta) + "</p>";
    }

    // Recall cycle: a prompt arrives, the relevant memories light gold and
    // trace into the kernel, then the orbit settles again.
    var cycleTimer = null;
    function scheduleRecall(delay) {
      clearTimeout(cycleTimer);
      cycleTimer = setTimeout(function () {
        if (hovered) { scheduleRecall(3000); return; }
        recalling = true; pulse = 1; paintAll(); showIdle();
        cycleTimer = setTimeout(function () {
          recalling = false; paintAll(); if (!hovered) showIdle();
          scheduleRecall(9000);
        }, 5200);
      }, delay);
    }

    var running = false, rafId = 0, visible = true;
    function frame(ts) {
      if (start === null) start = ts;
      var dt = Math.min(0.05, (ts - last) / 1000); last = ts;
      var t = ts - start;
      nodes.forEach(function (n, i) {
        var due = 300 + i * 420;
        if (n.born === null && t >= due) { n.born = t; }
        if (n.born !== null && n.alpha < 1) {
          n.alpha = Math.min(1, (t - n.born) / 1500);
          if (n.alpha >= 1) { arrived++; pulse = Math.max(pulse, 0.7); if (!hovered) showIdle(); if (arrived === nodes.length) { paintAll(); scheduleRecall(1800); } }
        }
        if (!hovered) n.angle += SPEED[n.d.ring] * dt;
      });
      pulse = Math.max(0, pulse - dt * 0.6);
      render();
      if (running) rafId = requestAnimationFrame(frame);
    }
    function play() {
      if (running || reduced) return;
      running = true;
      rafId = requestAnimationFrame(function (ts) { last = ts; frame(ts); });
    }
    function stop() { running = false; cancelAnimationFrame(rafId); }

    paintAll();
    if (reduced) {
      // Static: every memory in place, the recalled ones marked.
      nodes.forEach(function (n) { n.born = 0; n.alpha = 1; });
      arrived = nodes.length; recalling = true; render(); paintAll(); showIdle();
      return;
    }
    showIdle();
    if (hasIO) {
      new IntersectionObserver(function (e) {
        visible = e[0].isIntersecting;
        if (visible && !document.hidden) play(); else stop();
      }, { threshold: 0.05 }).observe(svg);
      document.addEventListener("visibilitychange", function () { if (document.hidden) stop(); else if (visible) play(); });
    } else { play(); }
  })();

  /* ── §01 three sessions ────────────────────────────────────────────── */
  var AMNESIA = {
    1: {
      bad: { reads: 12, cum: 12, html: '<div class="q">&gt; explain the auth flow</div><div class="files">%F</div>reads 12 files and works it out.<br><span class="lost">session ends → all of it is lost</span>' },
      good: { reads: 12, cum: 12, html: '<div class="q">&gt; explain the auth flow</div><div class="files">%F</div>reads 12 files and works it out.<br><span class="kept">session ends → decisions captured and stored</span>' }
    },
    2: {
      bad: { reads: 12, cum: 24, html: '<div class="q">&gt; explain the auth flow</div><div class="files">%F</div>reads the same 12 files again and re-derives the same thing.<br><span class="lost">session ends → nothing kept</span>' },
      good: { reads: 2, cum: 14, html: '<div class="q">&gt; explain the auth flow</div><div class="files">%F</div>already knows the flow, why Redis beat an in-process cache, and the approach that was tried and abandoned.<br><span class="kept">→ picks up where it left off</span>' }
    },
    3: {
      bad: { reads: 12, cum: 36, html: '<div class="q">&gt; explain the auth flow</div><div class="files">%F</div>reads 12 files a third time.<br><span class="lost">context resets again</span>' },
      good: { reads: 1, cum: 15, html: '<div class="q">&gt; explain the auth flow</div><div class="files">%F</div>already knows everything sessions 1 and 2 settled.<br><span class="kept">→ starts from where you are, not from zero</span>' }
    }
  };
  function files(n, kind) { var s = ""; for (var i = 0; i < n; i++) s += '<i class="' + kind + '"></i>'; return s; }
  (function () {
    if (!$("#vs-bad")) return;
    var btns = $$("#problem .seg button");
    function set(n) {
      var d = AMNESIA[n];
      $("#vs-bad").innerHTML = d.bad.html.replace("%F", files(d.bad.reads, n === 1 ? "" : "r"));
      $("#vs-good").innerHTML = d.good.html.replace("%F", files(d.good.reads, n === 1 ? "" : "k"));
      $("#m-bad").textContent = d.bad.cum;
      $("#m-good").textContent = d.good.cum;
      btns.forEach(function (b) { b.setAttribute("aria-pressed", String(+b.dataset.s === n)); });
    }
    set(1);
    var timer = null, play = $("#amnesia-play");
    function halt() { clearInterval(timer); timer = null; play.textContent = "Auto-play"; }
    btns.forEach(function (b) { b.addEventListener("click", function () { halt(); set(+b.dataset.s); }); });
    play.addEventListener("click", function () {
      if (timer) { halt(); return; }
      var n = 1; set(n); play.textContent = "Stop";
      timer = setInterval(function () { n = n % 3 + 1; set(n); }, 2600);
    });
  })();

  /* ── §03 the memory loop ───────────────────────────────────────────── */
  var STAGES = [
    { k: "observe", t: "Observe", d: "CogniKernel watches the session through its hook surfaces. Nothing to call, nothing to remember to save.",
      b: ["The Stop hook hands over the transcript when a Claude Code session ends", "codex-sync pulls Codex rollouts whose working directory maps to the project", "Delta cursors mean a transcript is never processed twice"], m: "integration/hooks.py" },
    { k: "extract", t: "Extract", d: "A deterministic pipeline turns the raw transcript into typed candidate claims.",
      b: ["sanitize → classify → salience (ONNX) → decision key", "salience_v2 types each sentence or drops it as noise", "Step narration and answer lines rank low or are dropped"], m: "extraction/pipeline.py" },
    { k: "consolidate", t: "Consolidate", d: "New claims merge into what's already known, so the store converges instead of accumulating contradictions.",
      b: ["claim → delta-merge → supersede (latest wins) → project", "supersession_xenc catches paraphrased reversals", "Idempotent: replaying a job can't double-count"], m: "delta/supersede.py" },
    { k: "store", t: "Store", d: "An event-sourced SQLite database (WAL), one per project, shared by every agent that works there.",
      b: ["Typed events with evidence and provenance", "An FTS5 index, optional embeddings, a render ledger", "Belief history powers show --as-of and why"], m: "storage/" },
    { k: "retrieve", t: "Retrieve", d: "Lexical-first hybrid retrieval, with dense vectors as a fused signal, never the only one.",
      b: ["FTS5 BM25 ∪ dense embeddings → Reciprocal Rank Fusion", "prohibition_search keeps \"don't do X\" rules from being crowded out", "AST skeleton graph ranked by PageRank"], m: "retrieval/hybrid.py" },
    { k: "assemble", t: "Assemble", d: "Ranked claims are packed into a fixed token budget, weighted by authority.",
      b: ["Authority-weighted drop-to-fit", "Hard constraints are prioritized ahead of lower-value content", "Deterministic order, so the prompt cache keeps hitting"], m: "compression/greedy.py" },
    { k: "inject", t: "Inject", d: "The block lands in the agent's context at the moments it can change what happens next.",
      b: ["SessionStart: the full session block", "UserPromptSubmit: prompt-relevant recall", "PreToolUse: a past prohibition, just before an edit breaks it"], m: "injection/template.py" }
  ];
  (function loop() {
    var svg = $("#loop-svg"), card = $("#stage-card");
    if (!svg) return;
    var cx = 260, cy = 260, R = 178, n = STAGES.length, idx = 0, timer = null, touched = false;
    var defs = svgEl("defs", {}, svg);
    var kg = svgEl("radialGradient", { id: "loop-k" }, defs);
    svgEl("stop", { offset: "0%", "stop-color": "var(--gold)", "stop-opacity": "0.8" }, kg);
    svgEl("stop", { offset: "100%", "stop-color": "var(--gold)", "stop-opacity": "0" }, kg);
    svgEl("circle", { cx: cx, cy: cy, r: R + 62, class: "o-faint" }, svg);
    svgEl("circle", { cx: cx, cy: cy, r: R, class: "o-ring dash" }, svg);
    svgEl("circle", { cx: cx, cy: cy, r: 84, class: "o-faint" }, svg);
    var circ = 2 * Math.PI * R;
    var prog = svgEl("circle", { cx: cx, cy: cy, r: R, class: "prog", transform: "rotate(-90 " + cx + " " + cy + ")", "stroke-dasharray": circ, "stroke-dashoffset": circ }, svg);
    // half labels set along the outer ring
    [["CAPTURE", -70, 60], ["RECALL", 120, 250]].forEach(function (s, i) {
      var rr = R + 62, a0 = s[1] * Math.PI / 180, a1 = s[2] * Math.PI / 180;
      svgEl("path", { id: "loop-side" + i, d: "M" + (cx + rr * Math.cos(a0)) + " " + (cy + rr * Math.sin(a0)) + " A" + rr + " " + rr + " 0 0 1 " + (cx + rr * Math.cos(a1)) + " " + (cy + rr * Math.sin(a1)), fill: "none" }, svg);
      var t = svgEl("text", { class: "side", dy: -6 }, svg);
      var tp = svgEl("textPath", { href: "#loop-side" + i, startOffset: "50%" }, t);
      tp.textContent = s[0];
    });
    svgEl("circle", { cx: cx, cy: cy, r: 60, fill: "url(#loop-k)", class: "k-glow" }, svg);
    svgEl("circle", { cx: cx, cy: cy, r: 24, fill: "var(--gold)", "fill-opacity": 0.12, stroke: "var(--gold)", "stroke-opacity": 0.5 }, svg);
    svgEl("circle", { cx: cx, cy: cy, r: 10, fill: "var(--gold)" }, svg);
    var t1 = svgEl("text", { x: cx, y: cy + 48, "text-anchor": "middle", class: "o-kernel-text" }, svg); t1.textContent = "KERNEL";
    var t2 = svgEl("text", { x: cx, y: cy + 64, "text-anchor": "middle", class: "o-label" }, svg); t2.textContent = "event-sourced store";

    var nodes = STAGES.map(function (s, i) {
      var ang = -Math.PI / 2 + (i / n) * 2 * Math.PI;
      var x = cx + R * Math.cos(ang), y = cy + R * Math.sin(ang);
      var g = svgEl("g", { class: "node", tabindex: "0", role: "button", "aria-label": "Stage " + (i + 1) + ": " + s.t }, svg);
      svgEl("circle", { cx: x, cy: y, r: 22, fill: "transparent" }, g);
      svgEl("circle", { cx: x, cy: y, r: 12, class: "dot" }, g);
      svgEl("circle", { cx: x, cy: y, r: 4.5, class: "core" }, g);
      var lx = cx + (R + 30) * Math.cos(ang), ly = cy + (R + 30) * Math.sin(ang);
      var anchor = Math.abs(Math.cos(ang)) < 0.2 ? "middle" : Math.cos(ang) > 0 ? "start" : "end";
      var tx = svgEl("text", { x: lx, y: ly + 4 + (Math.sin(ang) < -0.9 ? -6 : Math.sin(ang) > 0.9 ? 8 : 0), "text-anchor": anchor }, g);
      tx.textContent = "0" + (i + 1) + " " + s.k;
      var pick = function () { touched = true; stopCycle(); show(i); };
      g.addEventListener("click", pick);
      g.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
      return g;
    });
    function show(i) {
      idx = i;
      nodes.forEach(function (g, j) { g.classList.toggle("on", j === i); });
      prog.setAttribute("stroke-dashoffset", circ * (1 - (i + 0.0001) / n));
      var s = STAGES[i];
      card.innerHTML = '<span class="eyebrow">Stage 0' + (i + 1) + " of 0" + n + "</span><h3>" + s.t + "</h3><p>" + esc(s.d) + "</p>" +
        "<ul>" + s.b.map(function (b) { return "<li><span>" + esc(b) + "</span></li>"; }).join("") + "</ul>" +
        '<div class="mod"><span>source: <a href="https://github.com/KanishkNoir/cognikernel/tree/main/src/cognikernel/' + s.m + '">src/cognikernel/' + s.m + "</a></span>" +
        '<span class="nav2"><button class="linkbtn" type="button" data-d="-1">Prev</button><button class="linkbtn" type="button" data-d="1">Next</button></span></div>';
      $$(".nav2 button", card).forEach(function (b) {
        b.addEventListener("click", function () { touched = true; stopCycle(); show((idx + +b.dataset.d + n) % n); });
      });
    }
    function stopCycle() { clearInterval(timer); timer = null; }
    show(0);
    if (!reduced && hasIO) {
      new IntersectionObserver(function (e) {
        if (e[0].isIntersecting && !timer && !touched) timer = setInterval(function () { show((idx + 1) % n); }, 3600);
        else if (!e[0].isIntersecting) stopCycle();
      }, { threshold: 0.4 }).observe(svg);
    }
  })();

  /* ── §04 pipeline mini-orbits: the memory travels around one orbit,
     turning gold once it reaches the kernel stage ───────────────────── */
  $$("#chain svg").forEach(function (svg) {
    var i = +svg.dataset.i, r = 26, a = (-90 + i * 62) * Math.PI / 180;
    var x = 36 + r * Math.cos(a), y = 36 + r * Math.sin(a), on = i >= 3;
    var col = on ? "var(--gold)" : "var(--blue)";
    svgEl("circle", { cx: 36, cy: 36, r: r, fill: "none", stroke: "var(--orbit)" }, svg);
    if (i === 2) svgEl("circle", { cx: 36 + r * Math.cos(a - 0.5), cy: 36 + r * Math.sin(a - 0.5), r: 3.5, fill: "none", stroke: "var(--blue)", "stroke-opacity": 0.6 }, svg);
    svgEl("line", { x1: x, y1: y, x2: 36, y2: 36, stroke: col, "stroke-opacity": on ? 0.7 : 0.25, "stroke-dasharray": "2 4" }, svg);
    svgEl("circle", { cx: 36, cy: 36, r: 6, fill: "var(--gold)", "fill-opacity": on ? 1 : 0.35 }, svg);
    if (i === 4) svgEl("circle", { cx: x, cy: y, r: 9, fill: "var(--gold)", "fill-opacity": 0.18 }, svg);
    svgEl("circle", { cx: x, cy: y, r: 4.5, fill: col }, svg);
  });

  /* ── §05 time is an orbit: memory as of each session ───────────────── */
  var SESSIONS = [
    { s: "S1", kind: "Decision", title: "A first choice", body: "Rate-limit counts go in an in-process LRU cache. Dead letters get their own queue." },
    { s: "S2", kind: "Supersession", title: "The decision changes", body: "Counts must be shared across 4 workers, so: Redis. Same decision key, so session 1's claim is marked superseded and the LRU goes to the graveyard with the reason it failed. Retries become a hard rule." },
    { s: "S3", kind: "Constraint", title: "Rules accumulate", body: "Routing moves to tenant_id. \"Never log webhook payloads in full\" is stated for the third time and consolidated into one record (×3)." },
    { s: "S4", kind: "Recalled", title: "Memory reused", body: "Asked to add a per-tenant limit, the agent starts knowing the backend is Redis, that the LRU failed, and that routing keys on tenant_id. Nothing is re-derived.", gold: true }
  ];
  // m.from: session it appears in. m.dead: session from which it's abandoned.
  var EVO = [
    { id: "lru", from: 0, a: -128, type: "dec", label: "LRU for rate limits", dead: 1 },
    { id: "dlq", from: 0, a: 44, type: "dec", label: "Own dead-letter queue" },
    { id: "redis", from: 1, a: -96, type: "dec", label: "Redis for rate limits", recall: true },
    { id: "retry", from: 1, a: 66, type: "hard", label: "Retries via backoff.py" },
    { id: "tenant", from: 2, a: -52, type: "dec", label: "Route on tenant_id", recall: true },
    { id: "pii", from: 2, a: 150, type: "hard", label: "Never log payloads (×3)" }
  ];
  (function evolution() {
    var svg = $("#evo-svg"), list = $("#sessions"), state = $("#evo-state");
    if (!svg) return;
    var C = 220, R = [62, 104, 146, 188], sel = 3, auto = null;
    var kg = svgEl("radialGradient", { id: "evo-k" }, svgEl("defs", {}, svg));
    svgEl("stop", { offset: "0%", "stop-color": "var(--gold)", "stop-opacity": "0.75" }, kg);
    svgEl("stop", { offset: "100%", "stop-color": "var(--gold)", "stop-opacity": "0" }, kg);
    var rings = R.map(function (r, i) {
      var c = svgEl("circle", { cx: C, cy: C, r: r, class: "ring" }, svg);
      var t = svgEl("text", { x: C + 6, y: C - r - 5, class: "lbl" }, svg);
      t.textContent = SESSIONS[i].s;
      return c;
    });
    var traceLayer = svgEl("g", {}, svg);
    svgEl("circle", { cx: C, cy: C, r: 40, fill: "url(#evo-k)", class: "k-glow" }, svg);
    svgEl("circle", { cx: C, cy: C, r: 9, fill: "var(--gold)" }, svg);

    var mems = EVO.map(function (m) {
      // A memory sits on the ring of the session that created it.
      var r = R[m.from], a = m.a * Math.PI / 180;
      var x = C + r * Math.cos(a), y = C + r * Math.sin(a);
      var g = svgEl("g", { class: "mem" }, svg);
      var trace = svgEl("line", { x1: x, y1: y, x2: C, y2: C, stroke: "var(--gold)", "stroke-dasharray": "3 5", "stroke-opacity": 0 }, traceLayer);
      var halo = svgEl("circle", { cx: x, cy: y, r: 12, fill: "var(--gold)", "fill-opacity": 0 }, g);
      var gg = svgEl("g", { transform: "translate(" + x + " " + y + ")" }, g);
      var right = Math.cos(a) >= 0;
      var txt = svgEl("text", { x: x + (right ? 12 : -12), y: y + 3.5, "text-anchor": right ? "start" : "end", class: "txt" }, g);
      txt.textContent = m.label;
      return { m: m, g: g, gg: gg, halo: halo, trace: trace, txt: txt, parts: null, type: null };
    });
    function draw(k) {
      sel = k;
      rings.forEach(function (c, i) {
        c.classList.toggle("on", i === k);
        c.classList.toggle("gold", i === k && !!SESSIONS[i].gold);
        c.style.opacity = i <= k ? 1 : 0.35;
      });
      mems.forEach(function (o) {
        var m = o.m, shown = m.from <= k;
        var dead = m.dead !== undefined && k >= m.dead;
        var type = dead ? "dead" : m.type;
        var gold = !!SESSIONS[k].gold && (m.recall || dead);
        if (o.type !== type) { while (o.gg.firstChild) o.gg.removeChild(o.gg.firstChild); o.parts = drawGlyph(o.gg, type, 4.5); o.type = type; }
        paintGlyph(o.parts, type, gold ? "var(--gold)" : dead ? "var(--ink-3)" : "var(--blue)");
        o.g.style.opacity = shown ? 1 : 0;
        o.halo.setAttribute("fill-opacity", gold ? 0.2 : 0);
        o.trace.setAttribute("stroke-opacity", shown && gold ? 0.7 : 0);
        o.txt.style.textDecoration = dead ? "line-through" : "none";
      });
      $$("button", list).forEach(function (b, i) { b.setAttribute("aria-pressed", String(i === k)); });
      var backend = k === 0 ? "in-process LRU" : "Redis";
      state.innerHTML = '<div class="row"><span class="eyebrow">show --as-of ' + SESSIONS[k].s + '</span><span class="eyebrow">ratelimit.backend</span></div>' +
        '<div class="val">' + (k === 0 ? "in-process LRU" : backend + "  <s>in-process LRU</s>") + "</div>" +
        '<p class="note" style="margin:6px 0 0">' + (k === 0 ? "One decision on the key." : k < 3 ? "Superseded, not deleted: <code>why</code> shows the chain." : "Recalled at session start, with the dead end beside it.") + "</p>";
      state.classList.toggle("is-gold", !!SESSIONS[k].gold);
      state.style.borderColor = SESSIONS[k].gold ? "var(--gold-line)" : "";
    }
    list.innerHTML = SESSIONS.map(function (s, i) {
      return '<button type="button" aria-pressed="false" class="' + (s.gold ? "is-gold" : "") + '" data-k="' + i + '">' +
        '<span class="h"><i></i><span class="eyebrow">Session ' + (i + 1) + " · " + s.kind + "</span></span>" +
        '<span class="ti">' + s.title + '</span><span class="body">' + esc(s.body) + "</span></button>";
    }).join("");
    $$("button", list).forEach(function (b) {
      var k = +b.dataset.k;
      b.addEventListener("click", function () { clearInterval(auto); auto = null; draw(k); });
      b.addEventListener("focus", function () { clearInterval(auto); auto = null; draw(k); });
    });
    draw(3);
    if (!reduced) {
      // Play the project's history once when it first scrolls into view.
      whenVisible(svg, function () {
        var k = 0; draw(0);
        auto = setInterval(function () { k++; draw(k); if (k >= 3) { clearInterval(auto); auto = null; } }, 2200);
      }, 0.45);
    }
  })();

  /* ── §06 the injected block (also typed out in the session demo) ───── */
  // [section, class, text], in injection/template.py's section order.
  var BLOCK = [
    ["hdr", "c-h2", "## Session context [auto-generated — do not edit]"],
    ["hdr", "", "project: relay · session 4 of 5 · state v37"],
    ["hdr", "c-m", "Sn · MM-DD: the session the active thread was carried over from (S1 first)."],
    ["hdr", "c-m", "Codebase skeleton below lists public classes, methods and imports — use it to orient."],
    ["", "", ""],
    ["hard", "c-hard", "### Hard constraints — never violate"],
    ["hard", "", "- Webhook payloads are never logged in full — they carry customer PII (×3)"],
    ["hard", "", "- All retries go through relay/backoff.py — no ad-hoc sleep loops"],
    ["", "", ""],
    ["thread", "c-h3", "### Active thread"],
    ["thread", "", "Working on: replay endpoint for dead-lettered events (S3 · 09-12)"],
    ["thread", "", "Current state: handler and tests written; tenant-scope check missing"],
    ["thread", "", "Next: add the tenant-scope check, then wire the --replay CLI flag"],
    ["", "", ""],
    ["files", "c-h3", "### Most active files — structure in Codebase skeleton below"],
    ["files", "", "- relay/router.py · 14x"],
    ["files", "", "- relay/ratelimit.py · 9x"],
    ["", "", ""],
    ["dead", "c-gy", "### Do not retry — confirmed failures"],
    ["dead", "", "- In-process LRU cache for rate limits -> counts drift across the 4 workers"],
    ["dead", "", "- Polling upstream every 5s -> hit their 429 ceiling within an hour"],
    ["", "", ""],
    ["dec", "c-h3", "### Key decisions"],
    ["dec", "", "1. Use Redis for the rate limiter — shared across workers, TTL expires keys (session 2)"],
    ["dec", "", "2. Route on tenant_id, not hostname — custom domains broke host routing (session 3)"],
    ["dec", "", "3. Dead letters go to their own queue with 7-day retention (session 1)"],
    ["", "", ""],
    ["files", "c-h3", "### Codebase skeleton"],
    ["files", "c-m", "Coverage: 38 files scanned · 31 with public symbols listed."],
    ["files", "", "relay/ratelimit.py → redis, relay.config"],
    ["files", "", "  RedisLimiter: client, window_s"],
    ["files", "", "    allow(key: str)→bool | reset(key: str)"],
    ["files", "c-m", "  …"]
  ];
  (function block() {
    var pre = $("#block-pre");
    if (!pre) return;
    pre.innerHTML = BLOCK.map(function (l) {
      return '<span class="ln' + (l[1] ? " " + l[1] : "") + '" data-sec="' + l[0] + '">' + (esc(l[2]) || " ") + "</span>";
    }).join("\n");
    var lit = function (sec) {
      $$(".ln", pre).forEach(function (ln) { ln.classList.toggle("hl", !!sec && ln.dataset.sec === sec); });
      $$("#annots .annot").forEach(function (a) { a.classList.toggle("on", a.dataset.sec === sec); });
    };
    $$("#annots .annot").forEach(function (a) {
      a.tabIndex = 0;
      a.addEventListener("mouseenter", function () { lit(a.dataset.sec); });
      a.addEventListener("mouseleave", function () { lit(null); });
      a.addEventListener("focus", function () { lit(a.dataset.sec); });
      a.addEventListener("blur", function () { lit(null); });
    });
    $$(".ln", pre).forEach(function (ln) {
      ln.addEventListener("mouseenter", function () { if (ln.dataset.sec) lit(ln.dataset.sec); });
      ln.addEventListener("mouseleave", function () { lit(null); });
    });
  })();

  (function terminal() {
    var term = $("#term");
    if (!term) return;
    var run = 0, timer = null;
    function play() {
      var me = ++run;
      clearTimeout(timer);
      var script = [
        { type: "cmd", text: "claude" },
        { type: "out", html: '<span class="c-m">SessionStart › </span><span class="c-b">cognikernel</span><span class="c-m"> · drained 2 Codex rollouts</span>' },
        { type: "out", html: '<span class="c-m">SessionStart › </span><span class="c-g">memory block injected</span><span class="c-m"> · session 4, fitted to the token budget</span>' },
        { type: "out", html: "" }
      ];
      BLOCK.forEach(function (l) { script.push({ type: "blk", html: l[1] ? '<span class="' + l[1] + '">' + esc(l[2]) + "</span>" : esc(l[2]) }); });
      script.push({ type: "out", html: "" });
      script.push({ type: "prompt", text: "pick up the replay endpoint" });
      script.push({ type: "out", html: '<span class="c-g">●</span> Picking up the replay endpoint from session 3. The handler and tests exist;' });
      script.push({ type: "out", html: "  the tenant-scope check is what's missing. Adding it in relay/replay.py, then" });
      script.push({ type: "out", html: "  wiring the --replay flag. Rate limiting stays on Redis, per decision 1." });
      if (reduced) {
        term.innerHTML = script.map(function (s) {
          if (s.type === "cmd") return '<span class="p">$ </span>' + esc(s.text);
          if (s.type === "prompt") return '<span class="p">&gt; </span>' + esc(s.text);
          return s.html;
        }).join("\n");
        return;
      }
      var i = 0, html = "", cursor = '<span class="cursor"></span>';
      var down = function () { term.scrollTop = term.scrollHeight; };
      (function step() {
        if (me !== run) return;
        if (i >= script.length) { term.innerHTML = html + cursor; down(); return; }
        var s = script[i++];
        if (s.type === "cmd" || s.type === "prompt") {
          var prefix = s.type === "cmd" ? '<span class="p">$ </span>' : '<span class="p">&gt; </span>', k = 0;
          (function typeChar() {
            if (me !== run) return;
            term.innerHTML = html + prefix + esc(s.text.slice(0, k)) + cursor; down();
            if (k++ < s.text.length) timer = setTimeout(typeChar, 45 + Math.random() * 55);
            else { html += prefix + esc(s.text) + "\n"; timer = setTimeout(step, 420); }
          })();
        } else {
          html += s.html + "\n"; term.innerHTML = html + cursor; down();
          timer = setTimeout(step, s.type === "blk" ? 38 : 300);
        }
      })();
    }
    whenVisible(term, play, 0.3);
    $("#term-replay").addEventListener("click", play);
  })();

  /* ── §08 charts ────────────────────────────────────────────────────── */
  var tip = $("#tip");
  function showTip(html, ev) {
    tip.innerHTML = html; tip.classList.add("on");
    var x = ev.clientX + 14, y = ev.clientY + 14, w = tip.offsetWidth, h = tip.offsetHeight;
    if (x + w > window.innerWidth - 8) x = ev.clientX - w - 14;
    if (y + h > window.innerHeight - 8) y = ev.clientY - h - 14;
    tip.style.left = x + "px"; tip.style.top = y + "px";
  }
  function hideTip() { tip.classList.remove("on"); }
  // Horizontal bar: 4px rounded data end, square at the baseline.
  function hbar(x0, x1, y, h, r) {
    var len = Math.abs(x1 - x0);
    r = Math.min(r, len, h / 2);
    var s = x1 >= x0 ? -1 : 1;
    return "M" + x0 + "," + y + " H" + (x1 + s * r) + " Q" + x1 + "," + y + " " + x1 + "," + (y + r) +
      " V" + (y + h - r) + " Q" + x1 + "," + (y + h) + " " + (x1 + s * r) + "," + (y + h) + " H" + x0 + " Z";
  }
  var ORIENT = [
    { p: "Relay", s: "evolving decisions", ck: 27, auto: 39 },
    { p: "Taskflow", s: "small, re-readable", ck: 17, auto: 20 },
    { p: "Conductor", s: "quality invariants", ck: 97, auto: 104 },
    { p: "Toolbelt", s: "self-authored API", ck: 90, auto: 92 }
  ];
  var COST = [
    { p: "Relay", reads: -38.1, raw: -28.3, w: -25.0 },
    { p: "Toolbelt", reads: 2.9, raw: -23.0, w: -19.3 },
    { p: "Conductor", reads: 4.5, raw: -8.0, w: 3.0 },
    { p: "Taskflow", reads: 6.7, raw: 0.4, w: 4.2 }
  ];
  var pct = function (v) { return (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(1) + "%"; };
  var sw = function (c) { return '<i style="background:' + c + '"></i>'; };
  function orientChart() {
    var el = $("#c-orient"); if (!el) return;
    var W = Math.max(300, Math.round(el.clientWidth) || 560), narrow = W < 460;
    var L = narrow ? 114 : 134, Rr = narrow ? 58 : 72, rowH = 66, barH = 13, gap = 3, top = 6;
    var H = top + ORIENT.length * rowH + 26, max = 120;
    var sx = function (v) { return L + (v / max) * (W - L - Rr); };
    var s = '<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="Orientation reads per project, CogniKernel versus native auto-memory. CogniKernel is lower on all four.">';
    [0, 40, 80, 120].forEach(function (t) {
      s += '<line class="gridl" x1="' + sx(t) + '" x2="' + sx(t) + '" y1="' + top + '" y2="' + (H - 22) + '"/>';
      s += '<text class="tick" x="' + sx(t) + '" y="' + (H - 6) + '" text-anchor="middle">' + t + "</text>";
    });
    s += '<text class="tick" x="' + (W - 4) + '" y="' + (top + 10) + '" text-anchor="end">Δ</text>';
    ORIENT.forEach(function (d, i) {
      var y = top + i * rowH + 14;
      s += '<text class="proj" x="0" y="' + (y + 12) + '">' + d.p + "</text>";
      s += '<text class="sub" x="0" y="' + (y + 28) + '">' + d.s + "</text>";
      s += '<path class="grow from-left" fill="var(--s-ck)" d="' + hbar(L, sx(d.ck), y, barH, 4) + '"/>';
      s += '<path class="grow from-left" fill="var(--s-auto)" d="' + hbar(L, sx(d.auto), y + barH + gap, barH, 4) + '"/>';
      s += '<text class="val" x="' + (sx(d.ck) + 6) + '" y="' + (y + 10.5) + '">' + d.ck + "</text>";
      s += '<text class="val" x="' + (sx(d.auto) + 6) + '" y="' + (y + barH + gap + 10.5) + '" style="fill:var(--ink-2)">' + d.auto + "</text>";
      s += '<text class="val" x="' + (W - 4) + '" y="' + (y + 18) + '" text-anchor="end">' + pct((d.ck - d.auto) / d.auto * 100) + "</text>";
      s += '<rect class="hit" data-i="' + i + '" x="0" y="' + (y - 8) + '" width="' + W + '" height="' + (rowH - 4) + '"/>';
    });
    el.innerHTML = s + "</svg>";
    $$(".hit", el).forEach(function (h) {
      var d = ORIENT[+h.dataset.i];
      var html = "<b>" + d.p + " · " + d.s + "</b>" +
        '<div class="r"><span>' + sw("var(--s-ck)") + "CogniKernel</span><em>" + d.ck + "</em></div>" +
        '<div class="r"><span>' + sw("var(--s-auto)") + "Auto-memory</span><em>" + d.auto + "</em></div>" +
        '<div class="r"><span>Difference</span><em>' + pct((d.ck - d.auto) / d.auto * 100) + "</em></div>";
      h.addEventListener("mousemove", function (ev) { showTip(html, ev); });
      h.addEventListener("mouseleave", hideTip);
    });
    $("#t-orient tbody").innerHTML = ORIENT.map(function (d) {
      return "<tr><td>" + d.p + "</td><td>" + d.ck + "</td><td>" + d.auto + "</td><td>" + pct((d.ck - d.auto) / d.auto * 100) + "</td></tr>";
    }).join("");
  }
  function costChart() {
    var el = $("#c-cost"); if (!el) return;
    var W = Math.max(300, Math.round(el.clientWidth) || 560), narrow = W < 460;
    var L = narrow ? 84 : 100, Rr = narrow ? 14 : 20, rowH = 56, barH = 20, top = 6;
    var H = top + COST.length * rowH + 26, lo = -30, hi = 10;
    var sx = function (v) { return L + ((v - lo) / (hi - lo)) * (W - L - Rr); };
    var s = '<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="Weighted cost versus auto-memory: Relay minus 25 percent, Toolbelt minus 19.3, Conductor plus 3.0, Taskflow plus 4.2.">';
    [-30, -20, -10, 0, 10].forEach(function (t) {
      s += '<line class="' + (t === 0 ? "axis" : "gridl") + '" x1="' + sx(t) + '" x2="' + sx(t) + '" y1="' + top + '" y2="' + (H - 22) + '"/>';
      s += '<text class="tick" x="' + sx(t) + '" y="' + (H - 6) + '" text-anchor="middle">' + (t > 0 ? "+" : t < 0 ? "−" : "") + Math.abs(t) + "%</text>";
    });
    COST.forEach(function (d, i) {
      var y = top + i * rowH + 12, neg = d.w < 0;
      s += '<text class="proj" x="0" y="' + (y + 15) + '">' + d.p + "</text>";
      s += '<path class="grow ' + (neg ? "from-right" : "from-left") + '" fill="' + (neg ? "var(--s-ck)" : "var(--ink-3)") + '" d="' + hbar(sx(0), sx(d.w), y, barH, 4) + '"/>';
      s += '<text class="val" x="' + (neg ? sx(d.w) - 6 : sx(d.w) + 6) + '" y="' + (y + 14) + '" text-anchor="' + (neg ? "end" : "start") + '">' + pct(d.w) + "</text>";
      s += '<rect class="hit" data-i="' + i + '" x="0" y="' + (y - 10) + '" width="' + W + '" height="' + (rowH - 4) + '"/>';
    });
    el.innerHTML = s + "</svg>";
    $$(".hit", el).forEach(function (h) {
      var d = COST[+h.dataset.i];
      var html = "<b>" + d.p + " · CK vs auto-memory</b>" +
        '<div class="r"><span>Weighted cost</span><em>' + pct(d.w) + "</em></div>" +
        '<div class="r"><span>Raw tokens</span><em>' + pct(d.raw) + "</em></div>" +
        '<div class="r"><span>Total reads</span><em>' + pct(d.reads) + "</em></div>";
      h.addEventListener("mousemove", function (ev) { showTip(html, ev); });
      h.addEventListener("mouseleave", hideTip);
    });
    $("#t-cost tbody").innerHTML = COST.map(function (d) {
      return "<tr><td>" + d.p + "</td><td>" + pct(d.reads) + "</td><td>" + pct(d.raw) + "</td><td>" + pct(d.w) + "</td></tr>";
    }).join("");
  }
  orientChart();
  costChart();
  // Charts are drawn at their container's width so text stays at 1:1 scale.
  var lastW = window.innerWidth, rz = null;
  window.addEventListener("resize", function () {
    if (window.innerWidth === lastW) return;
    lastW = window.innerWidth;
    clearTimeout(rz);
    rz = setTimeout(function () { orientChart(); costChart(); }, 150);
  });
  $$(".chart").forEach(function (c) {
    if (reduced) { c.classList.add("in"); return; }
    whenVisible(c, function () { c.classList.add("in"); }, 0.3);
  });
  $$("[data-table-toggle]").forEach(function (b) {
    b.addEventListener("click", function () {
      var t = $("#" + b.dataset.tableToggle), chart = t.previousElementSibling, show = t.hidden;
      t.hidden = !show; chart.hidden = show;
      b.textContent = show ? "Chart" : "Table";
    });
  });

  /* ── §11 two agents, one kernel ────────────────────────────────────── */
  (function xp() {
    var svg = $("#xp-svg");
    if (!svg) return;
    var K = { x: 320, y: 200 };
    var defs = svgEl("defs", {}, svg);
    var kg = svgEl("radialGradient", { id: "xp-k" }, defs);
    svgEl("stop", { offset: "0%", "stop-color": "var(--gold)", "stop-opacity": "0.8" }, kg);
    svgEl("stop", { offset: "100%", "stop-color": "var(--gold)", "stop-opacity": "0" }, kg);
    var CL = { cx: 236, cy: 200, rx: 176, ry: 128 }, CX = { cx: 404, cy: 200, rx: 176, ry: 128 };
    svgEl("ellipse", { cx: CL.cx, cy: CL.cy, rx: CL.rx, ry: CL.ry, class: "orb c" }, svg);
    svgEl("ellipse", { cx: CX.cx, cy: CX.cy, rx: CX.rx, ry: CX.ry, class: "orb x" }, svg);
    var on = function (e, deg) { var a = deg * Math.PI / 180; return { x: e.cx + e.rx * Math.cos(a), y: e.cy + e.ry * Math.sin(a) }; };
    var tt = svgEl("text", { x: 70, y: 52, class: "ttl" }, svg); tt.textContent = "Claude Code";
    var tc = svgEl("text", { x: 570, y: 52, class: "ttl", "text-anchor": "end" }, svg); tc.textContent = "Codex";
    var claude = [["SessionStart", 232], ["UserPromptSubmit", 196], ["PreToolUse", 160], ["Stop", 124]];
    var codex = [["MCP recall", -40], ["codex-sync", 36]];
    var pts = {};
    claude.forEach(function (c) {
      var p = on(CL, c[1]); pts[c[0]] = p;
      svgEl("circle", { cx: p.x, cy: p.y, r: 10, fill: "var(--blue)", "fill-opacity": 0.12 }, svg);
      svgEl("circle", { cx: p.x, cy: p.y, r: 4.5, fill: "var(--blue)" }, svg);
      // labels sit inside the orbit, so they never run off a narrow screen
      var t = svgEl("text", { x: p.x + 14, y: p.y + 3.5, class: "lbl" }, svg); t.textContent = c[0];
    });
    codex.forEach(function (c) {
      var p = on(CX, c[1]); pts[c[0]] = p;
      svgEl("circle", { cx: p.x, cy: p.y, r: 10, fill: "var(--blue)", "fill-opacity": 0.12 }, svg);
      svgEl("circle", { cx: p.x, cy: p.y, r: 4.5, fill: "var(--blue)" }, svg);
      var t = svgEl("text", { x: p.x + 14, y: p.y + 3.5, class: "lbl" }, svg); t.textContent = c[0];
    });
    // A decision captured in Codex travels through the kernel to Claude's
    // next session block.
    svgEl("path", { d: "M" + pts["codex-sync"].x + " " + pts["codex-sync"].y + " Q" + 430 + " " + 270 + " " + K.x + " " + K.y, class: "flow" }, svg);
    svgEl("path", { d: "M" + K.x + " " + K.y + " Q" + 230 + " " + 110 + " " + pts.SessionStart.x + " " + pts.SessionStart.y, class: "flow" }, svg);
    svgEl("circle", { cx: K.x, cy: K.y, r: 46, fill: "url(#xp-k)", class: "k-glow" }, svg);
    svgEl("circle", { cx: K.x, cy: K.y, r: 20, fill: "var(--gold)", "fill-opacity": 0.12, stroke: "var(--gold)", "stroke-opacity": 0.5 }, svg);
    svgEl("circle", { cx: K.x, cy: K.y, r: 9, fill: "var(--gold)" }, svg);
    var kt = svgEl("text", { x: K.x, y: K.y + 42, "text-anchor": "middle", class: "o-kernel-text" }, svg); kt.textContent = "ONE STORE";
  })();

  /* ── §12 installer ─────────────────────────────────────────────────── */
  var INSTALL = {
    pip: '<span class="pr">$ </span>pip install "cognikernel[embedding]"',
    pipx: '<span class="pr">$ </span>pipx install "cognikernel[embedding]"',
    uv: '<span class="pr">$ </span>uv tool install "cognikernel[embedding]"',
    src: '<span class="pr">$ </span>git clone https://github.com/KanishkNoir/cognikernel &amp;&amp; cd cognikernel &amp;&amp; uv tool install ".[embedding]"'
  };
  (function installer() {
    var cmd = $("#inst-cmd");
    if (!cmd) return;
    function set(k) {
      cmd.innerHTML = INSTALL[k];
      $$("[data-inst]").forEach(function (b) { b.setAttribute("aria-pressed", String(b.dataset.inst === k)); });
      store.set("ck-inst", k);
    }
    var saved = store.get("ck-inst");
    set(INSTALL[saved] ? saved : "pip");
    $$("[data-inst]").forEach(function (b) { b.addEventListener("click", function () { set(b.dataset.inst); }); });
  })();

  /* ── §13 reference ─────────────────────────────────────────────────── */
  var REF = [
    ["cli", "init", "<project>", "Register the project and install the session hooks and MCP config for Claude Code and Codex. Idempotent."],
    ["cli", "doctor", "[--strict] <project>", "Per-subsystem health report. --strict exits non-zero when anything is degraded."],
    ["cli", "install-heads", "[--source <dir>] [--no-download]", "Install the fine-tuned salience and supersession encoders from the heads-v1 release, sha256-verified."],
    ["cli", "show", "<project> [--as-of <date|commit>]", "Print stored memory, or what memory believed at a past date or commit."],
    ["cli", "why", "<project> <#id or words>", "Where a claim came from, why it ranks where it does, and what it replaced or was replaced by."],
    ["cli", "explain-recall", "<project> <query> [--claim #id]", "Why recall and the per-prompt push retrieved what they did, or didn't."],
    ["cli", "codex-sync", "<project>", "Capture Codex CLI sessions for this project from ~/.codex/sessions through the same pipeline."],
    ["cli", "telemetry", "<project>", "Count Claude Code usage once per API response, for doctor's cache and round-trip figures."],
    ["cli", "reset", "<project>", "Delete all stored memory for a project."],
    ["mcp", "get_session_state", "(project_path)", "The current session block on demand. How Codex reads memory at session start."],
    ["mcp", "recall", "(project_path, query, limit)", "Hybrid BM25 + dense retrieval over stored memory, consolidated to one answer per key."],
    ["mcp", "find_related", "(project_path, query, limit)", "Semantic neighbours unioned with import-graph-adjacent events: what a change touches."],
    ["mcp", "skeleton", "(project_path, file_path?)", "The AST skeleton (classes, methods, imports) for the project or one file."]
  ];
  (function reference() {
    var el = $("#ref"), q = $("#ref-q"), kind = "all";
    if (!el) return;
    el.innerHTML = REF.map(function (r) {
      var sig = r[0] === "cli"
        ? '<span class="kw">cognikernel</span> ' + esc(r[1]) + ' <span class="arg">' + esc(r[2]) + "</span>"
        : esc(r[1]) + '<span class="arg">' + esc(r[2]) + "</span>";
      return '<div class="item" data-kind="' + r[0] + '" data-text="' + esc((r[1] + " " + r[2] + " " + r[3]).toLowerCase()) + '">' +
        '<span class="tg kind">' + r[0] + '</span><div class="sig">' + sig + "</div><p>" + esc(r[3]) + "</p></div>";
    }).join("");
    var tools = $("#docs .ref-tools");
    function filter() {
      var term = q.value.trim().toLowerCase(), shown = 0;
      $$(".item", el).forEach(function (it) {
        var ok = (kind === "all" || it.dataset.kind === kind) && (!term || it.dataset.text.indexOf(term) !== -1);
        it.hidden = !ok; if (ok) shown++;
      });
      $("#ref-empty").hidden = shown > 0;
    }
    q.addEventListener("input", filter);
    $$("[data-kind]", tools).forEach(function (b) {
      b.addEventListener("click", function () {
        kind = b.dataset.kind;
        $$("[data-kind]", tools).forEach(function (x) { x.setAttribute("aria-pressed", String(x === b)); });
        filter();
      });
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "/" && document.activeElement && !/INPUT|TEXTAREA/.test(document.activeElement.tagName)) {
        e.preventDefault(); q.focus();
        q.scrollIntoView({ block: "center", behavior: reduced ? "auto" : "smooth" });
      }
    });
  })();
})();
