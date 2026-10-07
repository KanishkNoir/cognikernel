/* CogniKernel site — interactions. No dependencies. */
(function () {
  "use strict";

  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var esc = function (t) { return String(t).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); };
  var store = {
    get: function (k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set: function (k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
  };

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

  /* ── banner: the README's ASCII art, redrawn as SVG ─────────────────
     Box-drawing glyphs don't tile seamlessly in most web fonts, so the
     <pre> is parsed into cells: █ becomes a solid block, the double-line
     shadow characters become strokes. The <pre> stays as the no-JS view. */
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
    var svg = '<svg class="banner" viewBox="-4 -4 ' + (W + 8) + " " + (H + 8) + '" role="img" aria-label="CogniKernel">' +
      '<defs><linearGradient id="bg-grad" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="var(--accent-ink)"/><stop offset=".6" stop-color="var(--accent)"/><stop offset="1" stop-color="var(--banner-end)"/></linearGradient></defs>' +
      '<path d="' + lines + '" fill="none" stroke="var(--accent)" stroke-opacity=".38" stroke-width="2.2" stroke-linecap="square"/>' +
      '<path d="' + blocks + '" fill="url(#bg-grad)" shape-rendering="crispEdges"/></svg>';
    pre.insertAdjacentHTML("afterend", svg);
    pre.hidden = true;
  })();

  /* ── nav: shadow, mobile menu, scroll-spy ──────────────────────────── */
  var nav = $("#nav");
  var onScroll = function () { nav.classList.toggle("scrolled", window.scrollY > 8); };
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
  // Sections without a nav entry light up their nearest parent entry.
  var spyAlias = { hooks: "loop", block: "memory", internals: "benchmarks", codex: "benchmarks", faq: "reference" };
  if ("IntersectionObserver" in window) {
    var spy = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting) return;
        var id = spyAlias[e.target.id] || e.target.id;
        $$("a", links).forEach(function (a) { a.classList.remove("active"); });
        if (spyMap[id]) spyMap[id].classList.add("active");
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    $$("main section[id]").forEach(function (s) { spy.observe(s); });
  }

  /* ── reveal on scroll ──────────────────────────────────────────────── */
  var revealEls = $$(".reveal");
  if (reduced || !("IntersectionObserver" in window)) {
    revealEls.forEach(function (el) { el.classList.add("in"); });
  } else {
    var ro = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { e.target.classList.add("in"); ro.unobserve(e.target); }
      });
    }, { rootMargin: "0px 0px -8% 0px" });
    revealEls.forEach(function (el, i) {
      el.style.transitionDelay = (i % 4) * 70 + "ms";
      ro.observe(el);
    });
  }

  /* ── copy buttons ──────────────────────────────────────────────────── */
  document.addEventListener("click", function (ev) {
    var btn = ev.target.closest(".copy");
    if (!btn) return;
    var text = btn.dataset.copy;
    if (!text && btn.dataset.copyFrom) {
      text = $("#" + btn.dataset.copyFrom).textContent.replace(/^\$\s*/, "").replace(/\s+#.*$/, "");
    }
    var done = function () {
      btn.classList.add("done");
      setTimeout(function () { btn.classList.remove("done"); }, 1400);
    };
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

  /* ── live repo stats (best-effort; the page reads fine without them) ─ */
  if (window.fetch) {
    fetch("https://api.github.com/repos/KanishkNoir/cognikernel", { headers: { Accept: "application/vnd.github+json" } })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (!d || typeof d.stargazers_count !== "number") return;
        var n = d.stargazers_count;
        $("#stars").textContent = n >= 1000 ? (n / 1000).toFixed(1).replace(/\.0$/, "") + "k" : String(n);
      })
      .catch(function () {});
    fetch("https://pypi.org/pypi/cognikernel/json")
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) { if (d && d.info && d.info.version) $("#ver").textContent = "v" + d.info.version; })
      .catch(function () {});
  }

  /* ── the injected block (shared by the hero terminal and §05) ──────── */
  // [section, css class, text]. Rendered in the same section order as
  // injection/template.py: header, hard, thread, hot files, graveyard,
  // decisions, skeleton.
  var BLOCK = [
    ["hdr", "h2", "## Session context [auto-generated — do not edit]"],
    ["hdr", "", "project: relay · session 4 of 5 · state v37"],
    ["hdr", "m", "Sn · MM-DD: the session the active thread was carried over from (S1 first)."],
    ["hdr", "m", "Codebase skeleton below lists public classes, methods and imports — use it to orient."],
    ["", "", ""],
    ["hard", "hc", "### Hard constraints — never violate"],
    ["hard", "", "- Webhook payloads are never logged in full — they carry customer PII"],
    ["hard", "", "- All retries go through relay/backoff.py — no ad-hoc sleep loops (×3)"],
    ["", "", ""],
    ["thread", "h3", "### Active thread"],
    ["thread", "", "Working on: replay endpoint for dead-lettered events (S3 · 09-12)"],
    ["thread", "", "Current state: handler and tests written; tenant-scope check missing"],
    ["thread", "", "Next: add the tenant-scope check, then wire the --replay CLI flag"],
    ["", "", ""],
    ["files", "h3", "### Most active files — structure in Codebase skeleton below"],
    ["files", "", "- relay/router.py · 14x"],
    ["files", "", "- relay/ratelimit.py · 9x"],
    ["", "", ""],
    ["dead", "gy", "### Do not retry — confirmed failures"],
    ["dead", "", "- In-process LRU cache for rate limits -> counts drift across the 4 workers"],
    ["dead", "", "- Polling upstream every 5s -> hit their 429 ceiling within an hour"],
    ["", "", ""],
    ["dec", "h3", "### Key decisions"],
    ["dec", "", "1. Use Redis for the rate limiter — shared across workers, TTL expires keys (×2) (session 2)"],
    ["dec", "", "2. Route on tenant_id, not hostname — custom domains broke host routing (session 3)"],
    ["dec", "", "3. Dead letters go to their own queue with 7-day retention (session 1)"],
    ["", "", ""],
    ["files", "h3", "### Codebase skeleton"],
    ["files", "m", "Coverage: 38 files scanned · 31 with public symbols listed."],
    ["files", "", "relay/ratelimit.py → redis, relay.config"],
    ["files", "", "  RedisLimiter: client, window_s"],
    ["files", "", "    allow(key: str)→bool | reset(key: str)"],
    ["files", "m", "  …"]
  ];

  // §05 static block with hoverable sections
  var pre = $("#block-pre");
  if (pre) {
    pre.innerHTML = BLOCK.map(function (l) {
      var cls = l[1] ? ' style="color:' + ({ h2: "#e8f0ec;font-weight:700", h3: "#5be3aa;font-weight:700", hc: "#f4a093;font-weight:700", gy: "#f2c46c;font-weight:700", m: "#6f807a" }[l[1]]) + '"' : "";
      return '<span class="ln" data-sec="' + l[0] + '"' + cls + ">" + (esc(l[2]) || " ") + "</span>";
    }).join("\n");
    var lit = function (sec) {
      $$(".ln", pre).forEach(function (ln) { ln.classList.toggle("hl", !!sec && ln.dataset.sec === sec); });
      $$("#annots .annot").forEach(function (a) { a.classList.toggle("on", a.dataset.sec === sec); });
    };
    $$("#annots .annot").forEach(function (a) {
      a.addEventListener("mouseenter", function () { lit(a.dataset.sec); });
      a.addEventListener("mouseleave", function () { lit(null); });
      a.tabIndex = 0;
      a.addEventListener("focus", function () { lit(a.dataset.sec); });
      a.addEventListener("blur", function () { lit(null); });
    });
    $$(".ln", pre).forEach(function (ln) {
      ln.addEventListener("mouseenter", function () { if (ln.dataset.sec) lit(ln.dataset.sec); });
      ln.addEventListener("mouseleave", function () { lit(null); });
    });
  }

  /* ── hero terminal ─────────────────────────────────────────────────── */
  var term = $("#term");
  var termTimer = null, termRun = 0;
  function runTerminal() {
    if (!term) return;
    var run = ++termRun;
    clearTimeout(termTimer);
    term.innerHTML = "";
    var script = [
      { type: "cmd", text: "claude" },
      { type: "out", html: '<span class="m">SessionStart hook › </span><span class="tag">cognikernel</span><span class="m"> · drained 2 Codex rollouts · ranked 41 claims</span>' },
      { type: "out", html: '<span class="m">SessionStart hook › </span><span class="ok">injected memory block</span><span class="m"> (session 4, fitted to the token budget)</span>' },
      { type: "out", html: "" }
    ];
    BLOCK.forEach(function (l) {
      script.push({ type: "blk", html: l[1] ? '<span class="' + l[1] + '">' + esc(l[2]) + "</span>" : esc(l[2]) });
    });
    script.push({ type: "out", html: "" });
    script.push({ type: "prompt", text: "pick up the replay endpoint" });
    script.push({ type: "out", html: '<span class="m">● </span>Picking up the replay endpoint from session 3. The handler and tests exist;' });
    script.push({ type: "out", html: "  the tenant-scope check is what's missing. Adding it in relay/replay.py, then" });
    script.push({ type: "out", html: "  wiring the --replay flag. Rate limiting stays on Redis per decision 1." });

    if (reduced) {
      term.innerHTML = script.map(function (s) {
        if (s.type === "cmd") return '<span class="p">$ </span>' + esc(s.text);
        if (s.type === "prompt") return '<span class="p">&gt; </span>' + esc(s.text);
        return s.html;
      }).join("\n");
      return;
    }

    var i = 0;
    var cursor = '<span class="cursor"></span>';
    var html = "";
    function scrollDown() { term.scrollTop = term.scrollHeight; }
    function step() {
      if (run !== termRun) return;
      if (i >= script.length) { term.innerHTML = html + cursor; scrollDown(); return; }
      var s = script[i++];
      if (s.type === "cmd" || s.type === "prompt") {
        var prefix = s.type === "cmd" ? '<span class="p">$ </span>' : '<span class="p">&gt; </span>';
        var k = 0;
        (function typeChar() {
          if (run !== termRun) return;
          term.innerHTML = html + prefix + esc(s.text.slice(0, k)) + cursor;
          scrollDown();
          if (k++ < s.text.length) { termTimer = setTimeout(typeChar, 38 + Math.random() * 50); }
          else { html += prefix + esc(s.text) + "\n"; termTimer = setTimeout(step, 380); }
        })();
      } else {
        html += s.html + "\n";
        term.innerHTML = html + cursor;
        scrollDown();
        termTimer = setTimeout(step, s.type === "blk" ? 34 : 260);
      }
    }
    termTimer = setTimeout(step, 600);
  }
  if (term) {
    // Start when visible so the animation isn't spent off-screen.
    if ("IntersectionObserver" in window) {
      var started = false;
      new IntersectionObserver(function (e, obs) {
        if (e[0].isIntersecting && !started) { started = true; runTerminal(); obs.disconnect(); }
      }).observe(term);
    } else { runTerminal(); }
    $("#term-replay").addEventListener("click", runTerminal);
  }

  /* ── §01 amnesia stepper ───────────────────────────────────────────── */
  var AMNESIA = {
    1: {
      bad: { reads: 12, cum: 12, html: '<div class="q">&gt; explain the auth flow</div><div class="files">%F</div>reads 12 files, works it out.<br><span class="lost">session ends → all of it is lost</span>' },
      good: { reads: 12, cum: 12, kind: "r", html: '<div class="q">&gt; explain the auth flow</div><div class="files">%F</div>reads 12 files, works it out.<br><span class="kept">session ends → decisions captured &amp; stored</span>' }
    },
    2: {
      bad: { reads: 12, cum: 24, html: '<div class="q">&gt; explain the auth flow</div><div class="files">%F</div>reads the same 12 files again and re-derives the same thing.<br><span class="lost">session ends → nothing kept</span>' },
      good: { reads: 2, cum: 14, kind: "k", html: '<div class="q">&gt; explain the auth flow</div><div class="files">%F</div>already knows the flow, why Redis beat an in-process cache, and the approach that was tried and abandoned.<br><span class="kept">→ picks up where it left off</span>' }
    },
    3: {
      bad: { reads: 12, cum: 36, html: '<div class="q">&gt; explain the auth flow</div><div class="files">%F</div>reads 12 files a third time.<br><span class="lost">context resets again</span>' },
      good: { reads: 1, cum: 15, kind: "k", html: '<div class="q">&gt; explain the auth flow</div><div class="files">%F</div>already knows everything sessions 1 and 2 settled.<br><span class="kept">→ starts from where <b>you</b> are, not from zero</span>' }
    }
  };
  function files(n, kind) { var s = ""; for (var i = 0; i < n; i++) s += '<i class="' + kind + '"></i>'; return s; }
  function setSession(n) {
    var d = AMNESIA[n];
    $("#vs-bad").innerHTML = d.bad.html.replace("%F", files(d.bad.reads, n === 1 ? "" : "r"));
    $("#vs-good").innerHTML = d.good.html.replace("%F", files(d.good.reads, n === 1 ? "" : "k"));
    $("#m-bad").textContent = d.bad.cum;
    $("#m-good").textContent = d.good.cum;
    $$("#why .seg button").forEach(function (b) { b.setAttribute("aria-selected", String(+b.dataset.s === n)); });
  }
  if ($("#vs-bad")) {
    setSession(1);
    $$("#why .seg button").forEach(function (b) {
      b.addEventListener("click", function () { stopAmnesia(); setSession(+b.dataset.s); });
    });
    var amTimer = null;
    var playBtn = $("#amnesia-play");
    function stopAmnesia() { clearInterval(amTimer); amTimer = null; playBtn.innerHTML = playBtn.innerHTML.replace("stop", "auto-play"); }
    playBtn.addEventListener("click", function () {
      if (amTimer) { stopAmnesia(); return; }
      var n = 1; setSession(n);
      playBtn.innerHTML = playBtn.innerHTML.replace("auto-play", "stop");
      amTimer = setInterval(function () {
        n = n % 3 + 1; setSession(n);
      }, 2200);
    });
  }

  /* ── §02 memory loop ───────────────────────────────────────────────── */
  var STAGES = [
    { k: "observe", t: "Observe", d: "CogniKernel watches the session through its hook surfaces. Nothing to call, nothing to remember to save.",
      b: ["The Stop hook hands over the transcript when a Claude Code session ends", "codex-sync pulls Codex rollouts whose cwd maps to the project", "Delta cursors mean a transcript is never processed twice"],
      m: "integration/hooks.py" },
    { k: "extract", t: "Extract", d: "A deterministic pipeline turns raw transcript into typed candidate claims.",
      b: ["sanitize → classify → salience (ONNX) → decision key", "salience_v2 types each sentence or drops it as noise", "Step narration and answer lines rank low or are dropped"],
      m: "extraction/pipeline.py" },
    { k: "consolidate", t: "Consolidate", d: "New claims are merged into what's already known, so the store converges instead of accumulating contradictions.",
      b: ["claim → delta-merge → supersede (latest-wins) → project", "supersession_xenc catches paraphrased reversals", "Idempotent: replaying a job can't double-count"],
      m: "delta/supersede.py" },
    { k: "store", t: "Store", d: "An event-sourced SQLite database (WAL), one per logical project, shared by every agent that works there.",
      b: ["Typed events with evidence and provenance", "FTS5 index, optional embeddings, a render ledger", "Belief history powers show --as-of and why"],
      m: "storage/" },
    { k: "retrieve", t: "Retrieve", d: "Lexical-primary hybrid retrieval, with dense vectors as a fused signal, never the only one.",
      b: ["FTS5 BM25 ∪ dense embeddings → Reciprocal Rank Fusion", "prohibition_search keeps \"don't do X\" rules from being crowded out", "AST skeleton graph ranked by PageRank"],
      m: "retrieval/hybrid.py" },
    { k: "assemble", t: "Assemble", d: "Ranked claims are packed into a fixed token budget, weighted by authority.",
      b: ["Authority-weighted drop-to-fit", "Hard constraints are prioritized ahead of lower-value content", "Deterministic order so the prompt cache keeps hitting"],
      m: "compression/greedy.py" },
    { k: "inject", t: "Inject", d: "The block lands in the agent's context at the moments it can change what happens next.",
      b: ["SessionStart: the full session block", "UserPromptSubmit: prompt-relevant recall", "PreToolUse: a past prohibition, just before an edit breaks it"],
      m: "injection/template.py" }
  ];
  var loopSvg = $("#loop-svg");
  var stageIdx = 0, loopTimer = null;
  if (loopSvg) {
    var NS = "http://www.w3.org/2000/svg";
    var cx = 260, cy = 260, R = 190, n = STAGES.length;
    var mk = function (tag, attrs, parent) {
      var el = document.createElementNS(NS, tag);
      for (var a in attrs) el.setAttribute(a, attrs[a]);
      (parent || loopSvg).appendChild(el);
      return el;
    };
    mk("circle", { cx: cx, cy: cy, r: R, class: "ring" });
    // Progress arc: a circle starting at 12 o'clock, drawn clockwise.
    var circ = 2 * Math.PI * R;
    var arc = mk("circle", { cx: cx, cy: cy, r: R, class: "arc", transform: "rotate(-90 " + cx + " " + cy + ")", "stroke-dasharray": circ, "stroke-dashoffset": circ });
    var side1 = mk("text", { x: cx + R + 76, y: cy, class: "side", transform: "rotate(90 " + (cx + R + 76) + " " + cy + ")" }); side1.textContent = "CAPTURE";
    var side2 = mk("text", { x: cx - R - 76, y: cy, class: "side", transform: "rotate(-90 " + (cx - R - 76) + " " + cy + ")" }); side2.textContent = "RECALL";
    var core = mk("g", { class: "core" });
    mk("rect", { x: cx - 98, y: cy - 62, width: 196, height: 124, rx: 18 }, core);
    var ct = mk("text", { x: cx, y: cy - 16 }, core); ct.textContent = "Event-sourced store";
    var cs = mk("text", { x: cx, y: cy + 8, class: "sub" }, core); cs.textContent = "SQLite · WAL · FTS5";
    var cs2 = mk("text", { x: cx, y: cy + 32, class: "sub" }, core); cs2.textContent = "+ reliability spine";
    var nodes = STAGES.map(function (s, i) {
      var ang = -Math.PI / 2 + (i / n) * 2 * Math.PI;
      var x = cx + R * Math.cos(ang), y = cy + R * Math.sin(ang);
      var g = mk("g", { class: "node", tabindex: "0", role: "button", "aria-label": "Stage " + (i + 1) + ": " + s.t });
      mk("circle", { cx: x, cy: y, r: 46 }, g);
      var num = mk("text", { x: x, y: y - 11, class: "num" }, g); num.textContent = "0" + (i + 1);
      var lab = mk("text", { x: x, y: y + 7 }, g); lab.textContent = s.k;
      var pick = function () { stopLoop(); showStage(i); };
      g.addEventListener("click", pick);
      g.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
      return g;
    });
    var card = $("#stage-card");
    function showStage(i) {
      stageIdx = i;
      nodes.forEach(function (g, j) { g.classList.toggle("on", j === i); });
      arc.setAttribute("stroke-dashoffset", circ * (1 - (i + 0.0001) / n));
      var s = STAGES[i];
      card.innerHTML =
        '<div class="stage-n">STAGE 0' + (i + 1) + " / 0" + n + "</div>" +
        "<h3>" + s.t + "</h3><p>" + esc(s.d) + "</p>" +
        "<ul>" + s.b.map(function (b) { return "<li>" + esc(b) + "</li>"; }).join("") + "</ul>" +
        '<div class="mod">source: <a href="https://github.com/KanishkNoir/cognikernel/tree/main/src/cognikernel/' + s.m + '">src/cognikernel/' + s.m + "</a></div>" +
        '<div class="stage-nav"><button class="linkbtn" type="button" data-d="-1">← prev</button><button class="linkbtn" type="button" data-d="1">next →</button></div>';
      $$(".stage-nav button", card).forEach(function (b) {
        b.addEventListener("click", function () { stopLoop(); showStage((stageIdx + +b.dataset.d + n) % n); });
      });
    }
    function stopLoop() { clearInterval(loopTimer); loopTimer = null; }
    showStage(0);
    if (!reduced && "IntersectionObserver" in window) {
      new IntersectionObserver(function (e) {
        if (e[0].isIntersecting && loopTimer === null && !loopSvg.dataset.touched) {
          loopTimer = setInterval(function () { showStage((stageIdx + 1) % n); }, 3200);
        } else if (!e[0].isIntersecting) { stopLoop(); }
      }, { threshold: 0.4 }).observe(loopSvg);
      loopSvg.addEventListener("pointerdown", function () { loopSvg.dataset.touched = "1"; });
      card.addEventListener("pointerdown", function () { loopSvg.dataset.touched = "1"; stopLoop(); });
    }
  }

  /* ── §04 supersession demo ─────────────────────────────────────────── */
  var lane = $("#sup-lane"), supLog = $("#sup-log"), supBtn = $("#sup-play");
  var supTimers = [];
  function claimEl(id, s, txt, st, stCls, extra) {
    var d = document.createElement("div");
    d.className = "claim " + (extra || "");
    d.id = id;
    d.innerHTML = '<span class="s">' + s + '</span><span class="txt">' + esc(txt) + '</span><span class="st ' + stCls + '">' + st + "</span>";
    return d;
  }
  function addClaim(el) {
    el.classList.add("enter");
    lane.appendChild(el);
    requestAnimationFrame(function () { requestAnimationFrame(function () { el.classList.remove("enter"); }); });
  }
  function resetSup() {
    supTimers.forEach(clearTimeout); supTimers = [];
    lane.innerHTML = "";
  }
  function runSup() {
    resetSup();
    supBtn.setAttribute("aria-pressed", "true");
    var t = reduced ? 0 : 1;
    var at = function (ms, fn) { supTimers.push(setTimeout(fn, ms * t)); };
    at(0, function () {
      addClaim(claimEl("c1", "S1", "Cache rate-limit counts in an in-process LRU", "live", "live"));
      supLog.innerHTML = "S1 · salience_v2 → <b>DECISION</b> · key ratelimit.backend";
    });
    at(1800, function () {
      addClaim(claimEl("c2", "S2", "Use Redis for the rate limiter: counts must be shared across workers", "live", "live", "new"));
      supLog.innerHTML = "S2 · new claim on the same key · supersession_xenc: P(supersede) <b>above threshold</b>";
    });
    at(3000, function () {
      var c1 = $("#c1");
      if (!c1) return;
      c1.classList.add("superseded");
      c1.querySelector(".st").className = "st sup";
      c1.querySelector(".st").textContent = "superseded by S2";
    });
    at(4300, function () {
      addClaim(claimEl("c3", "S2", "In-process LRU → counts drift across the 4 workers", "do not retry", "dead", "grave"));
      supLog.innerHTML = "S2 · salience_v2 → <b>APPROACH_ABANDONED</b> · moved to the graveyard";
    });
    at(6000, function () {
      supLog.innerHTML = "S3 · \"keep Redis for rate limits\" · same key, same value → <b>consolidated</b> into one golden record";
      var c2 = $("#c2");
      if (c2) c2.querySelector(".st").textContent = "live (×2)";
    });
    at(7600, function () {
      supLog.innerHTML = "recall(\"rate limiter\") → <b>one answer</b>: Redis (S2, ×2) · plus the dead end, so it isn't retried";
      supBtn.setAttribute("aria-pressed", "false");
    });
  }
  if (lane) {
    supBtn.addEventListener("click", runSup);
    if (!reduced && "IntersectionObserver" in window) {
      new IntersectionObserver(function (e, obs) {
        if (e[0].isIntersecting) { runSup(); obs.disconnect(); }
      }, { threshold: 0.6 }).observe(lane.parentNode);
    }
  }

  /* ── §07 charts ────────────────────────────────────────────────────── */
  var tip = $("#tip");
  function showTip(html, ev) {
    tip.innerHTML = html;
    tip.classList.add("on");
    var x = ev.clientX + 14, y = ev.clientY + 14;
    var w = tip.offsetWidth, h = tip.offsetHeight;
    if (x + w > window.innerWidth - 8) x = ev.clientX - w - 14;
    if (y + h > window.innerHeight - 8) y = ev.clientY - h - 14;
    tip.style.left = x + "px"; tip.style.top = y + "px";
  }
  function hideTip() { tip.classList.remove("on"); }

  // Horizontal bar with a 4px rounded data-end and a square baseline end.
  function hbar(x0, x1, y, h, r) {
    var dir = x1 >= x0 ? 1 : -1;
    var len = Math.abs(x1 - x0);
    r = Math.min(r, len, h / 2);
    if (dir === 1) {
      return "M" + x0 + "," + y + " H" + (x1 - r) + " Q" + x1 + "," + y + " " + x1 + "," + (y + r) +
        " V" + (y + h - r) + " Q" + x1 + "," + (y + h) + " " + (x1 - r) + "," + (y + h) + " H" + x0 + " Z";
    }
    return "M" + x0 + "," + y + " H" + (x1 + r) + " Q" + x1 + "," + y + " " + x1 + "," + (y + r) +
      " V" + (y + h - r) + " Q" + x1 + "," + (y + h) + " " + (x1 + r) + "," + (y + h) + " H" + x0 + " Z";
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
  var swatch = function (c) { return '<i style="background:' + c + '"></i>'; };

  function orientChart() {
    var el = $("#c-orient"); if (!el) return;
    var W = Math.max(300, Math.round(el.clientWidth) || 560), narrow = W < 460;
    var L = narrow ? 114 : 132, Rr = narrow ? 58 : 70, rowH = 66, barH = 14, gap = 3, top = 6;
    var H = top + ORIENT.length * rowH + 26;
    var max = 120, sx = function (v) { return L + (v / max) * (W - L - Rr); };
    var s = '<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="Orientation reads per project, CogniKernel versus native auto-memory. CogniKernel is lower on all four.">';
    [0, 40, 80, 120].forEach(function (t) {
      s += '<line class="gridl" x1="' + sx(t) + '" x2="' + sx(t) + '" y1="' + top + '" y2="' + (H - 22) + '"/>';
      s += '<text class="tick" x="' + sx(t) + '" y="' + (H - 6) + '" text-anchor="middle">' + t + "</text>";
    });
    s += '<text class="tick" x="' + (W - 4) + '" y="' + (top + 10) + '" text-anchor="end">Δ</text>';
    ORIENT.forEach(function (d, i) {
      var y = top + i * rowH + 14;
      s += '<text class="proj" x="0" y="' + (y + 12) + '">' + d.p + "</text>";
      s += '<text class="sub" x="0" y="' + (y + 29) + '">' + d.s + "</text>";
      s += '<path class="bar grow from-left" fill="var(--s-ck)" d="' + hbar(L, sx(d.ck), y, barH, 4) + '"/>';
      s += '<path class="bar grow from-left" fill="var(--s-auto)" d="' + hbar(L, sx(d.auto), y + barH + gap, barH, 4) + '"/>';
      s += '<text class="val" x="' + (sx(d.ck) + 6) + '" y="' + (y + 11) + '">' + d.ck + "</text>";
      s += '<text class="val" x="' + (sx(d.auto) + 6) + '" y="' + (y + barH + gap + 11) + '" style="fill:var(--ink-2)">' + d.auto + "</text>";
      var delta = (d.ck - d.auto) / d.auto * 100;
      s += '<text class="val" x="' + (W - 4) + '" y="' + (y + 19) + '" text-anchor="end">' + pct(delta) + "</text>";
      s += '<rect class="hit" data-i="' + i + '" x="0" y="' + (y - 8) + '" width="' + W + '" height="' + (rowH - 4) + '"/>';
    });
    s += "</svg>";
    el.innerHTML = s;
    $$(".hit", el).forEach(function (h) {
      var d = ORIENT[+h.dataset.i];
      var html = "<b>" + d.p + " · " + d.s + "</b>" +
        '<div class="r"><span>' + swatch("var(--s-ck)") + "CogniKernel</span><em>" + d.ck + "</em></div>" +
        '<div class="r"><span>' + swatch("var(--s-auto)") + "Auto-memory</span><em>" + d.auto + "</em></div>" +
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
    var L = narrow ? 84 : 100, Rr = narrow ? 14 : 20, rowH = 56, barH = 22, top = 6;
    var H = top + COST.length * rowH + 26;
    var lo = -30, hi = 10, sx = function (v) { return L + ((v - lo) / (hi - lo)) * (W - L - Rr); };
    var s = '<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="Price-weighted cost difference versus auto-memory: Relay minus 25 percent, Toolbelt minus 19.3, Conductor plus 3.0, Taskflow plus 4.2.">';
    [-30, -20, -10, 0, 10].forEach(function (t) {
      s += '<line class="' + (t === 0 ? "axis" : "gridl") + '" x1="' + sx(t) + '" x2="' + sx(t) + '" y1="' + top + '" y2="' + (H - 22) + '"' + (t === 0 ? ' style="stroke:var(--ink-3)"' : "") + "/>";
      s += '<text class="tick" x="' + sx(t) + '" y="' + (H - 6) + '" text-anchor="middle">' + (t > 0 ? "+" : t < 0 ? "−" : "") + Math.abs(t) + "%</text>";
    });
    COST.forEach(function (d, i) {
      var y = top + i * rowH + 12;
      var neg = d.w < 0;
      s += '<text class="proj" x="0" y="' + (y + 16) + '">' + d.p + "</text>";
      s += '<path class="bar grow ' + (neg ? "from-right" : "from-left") + '" fill="' + (neg ? "var(--s-ck)" : "var(--s-worse)") + '" d="' + hbar(sx(0), sx(d.w), y, barH, 4) + '"/>';
      var lx = neg ? sx(d.w) - 6 : sx(d.w) + 6;
      s += '<text class="val" x="' + lx + '" y="' + (y + 15) + '" text-anchor="' + (neg ? "end" : "start") + '">' + pct(d.w) + "</text>";
      s += '<rect class="hit" data-i="' + i + '" x="0" y="' + (y - 10) + '" width="' + W + '" height="' + (rowH - 4) + '"/>';
    });
    s += "</svg>";
    el.innerHTML = s;
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
    if (reduced || !("IntersectionObserver" in window)) { c.classList.add("in"); return; }
    new IntersectionObserver(function (e, obs) {
      if (e[0].isIntersecting) { c.classList.add("in"); obs.disconnect(); }
    }, { threshold: 0.3 }).observe(c);
  });
  $$("[data-table-toggle]").forEach(function (b) {
    b.addEventListener("click", function () {
      var t = $("#" + b.dataset.tableToggle);
      var chart = t.previousElementSibling;
      var show = t.hidden;
      t.hidden = !show; chart.hidden = show;
      b.textContent = show ? "Chart" : "Table";
    });
  });

  /* ── §11 install tabs ──────────────────────────────────────────────── */
  var INSTALL = {
    pip: '<span class="pr">$ </span>pip install "cognikernel[embedding]"',
    pipx: '<span class="pr">$ </span>pipx install "cognikernel[embedding]"',
    uv: '<span class="pr">$ </span>uv tool install "cognikernel[embedding]"',
    src: '<span class="pr">$ </span>git clone https://github.com/KanishkNoir/cognikernel &amp;&amp; cd cognikernel &amp;&amp; uv tool install ".[embedding]"'
  };
  var instCmd = $("#inst-cmd");
  function setInst(k) {
    instCmd.innerHTML = INSTALL[k];
    $$("[data-inst]").forEach(function (b) { b.setAttribute("aria-pressed", String(b.dataset.inst === k)); });
    store.set("ck-inst", k);
  }
  if (instCmd) {
    var savedInst = store.get("ck-inst");
    setInst(INSTALL[savedInst] ? savedInst : "pip");
    $$("[data-inst]").forEach(function (b) { b.addEventListener("click", function () { setInst(b.dataset.inst); }); });
  }

  /* ── §12 reference ─────────────────────────────────────────────────── */
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
  var refEl = $("#ref"), refQ = $("#ref-q"), refKind = "all";
  if (refEl) {
    refEl.innerHTML = REF.map(function (r) {
      var sig = r[0] === "cli"
        ? '<span class="kw">cognikernel</span> ' + esc(r[1]) + ' <span class="arg">' + esc(r[2]) + "</span>"
        : esc(r[1]) + '<span class="arg">' + esc(r[2]) + "</span>";
      return '<div class="item" data-kind="' + r[0] + '" data-text="' + esc((r[1] + " " + r[2] + " " + r[3]).toLowerCase()) + '">' +
        '<span class="kind ' + r[0] + '">' + r[0].toUpperCase() + '</span><div class="sig">' + sig + "</div><p>" + esc(r[3]) + "</p></div>";
    }).join("");
    var filter = function () {
      var q = refQ.value.trim().toLowerCase(), shown = 0;
      $$(".item", refEl).forEach(function (it) {
        var ok = (refKind === "all" || it.dataset.kind === refKind) && (!q || it.dataset.text.indexOf(q) !== -1);
        it.hidden = !ok; if (ok) shown++;
      });
      $("#ref-empty").hidden = shown > 0;
    };
    refQ.addEventListener("input", filter);
    $$("[data-kind]", $("#reference .ref-tools")).forEach(function (b) {
      b.addEventListener("click", function () {
        refKind = b.dataset.kind;
        $$("[data-kind]", $("#reference .ref-tools")).forEach(function (x) { x.setAttribute("aria-pressed", String(x === b)); });
        filter();
      });
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "/" && document.activeElement && !/INPUT|TEXTAREA/.test(document.activeElement.tagName)) {
        e.preventDefault(); refQ.focus();
        refQ.scrollIntoView({ block: "center", behavior: reduced ? "auto" : "smooth" });
      }
    });
  }
})();
