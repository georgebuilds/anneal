/* ============================================================================
   anneal tutorial / learn  ·  behaviour
   ----------------------------------------------------------------------------
   The page is built to GROW: new examples land continuously. To add a lesson,
   touch these places (all small, all documented):

     1. MODELS below ............ add { id, label }. This is the source of
                                  truth for which lessons exist + their names.
     2. index.html modelbar ..... add a <button role="tab" data-model="ID">.
     3. index.html #lessons ..... add an <a class="lesson-card" data-model="ID">.
     4. index.html steps ........ add <div class="variant" data-model="ID"> blocks
                                  to the steps that need model-specific content.
     5. tutorial.css ............ add ID to the `.variant[data-model=...]` show
                                  rule (one line in the enumerated list).

   The anti-FOUC script in <head> resolves the selected model BEFORE paint and
   sets data-model on <html>; this file wires the interactivity on top.
   ========================================================================== */
(function () {
    "use strict";

    /* ── Model registry: the single source of truth ─────────────────────── */
    var MODELS = [
        { id: "mlp",     label: "MLP" },
        { id: "conv",    label: "ConvNet" },
        { id: "nanogpt", label: "nanoGPT" },
        { id: "llama",   label: "Llama" },
        { id: "vit",     label: "ViT" },
        { id: "resnet9", label: "ResNet-9" },
        { id: "gpt2",    label: "GPT-2-small" },
        { id: "dit",     label: "DiT" },
        { id: "meanflow", label: "MeanFlow" }
    ];
    var IDS = MODELS.map(function (m) { return m.id; });
    var LABEL = {};
    MODELS.forEach(function (m) { LABEL[m.id] = m.label; });
    function isModel(x) { return IDS.indexOf(x) !== -1; }

    var $ = function (sel, root) { return (root || document).querySelector(sel); };
    var $$ = function (sel, root) {
        return Array.prototype.slice.call((root || document).querySelectorAll(sel));
    };

    /* ── Theme toggle (shares localStorage["anneal-theme"] with the site) ── */
    var themeBtn = $("#js-theme");
    function applyTheme(t) {
        document.documentElement.setAttribute("data-theme", t);
        if (themeBtn) {
            themeBtn.textContent = t === "light" ? "☾" : "☀";
            themeBtn.setAttribute("aria-label", "switch to " + (t === "light" ? "dark" : "light") + " mode");
        }
    }
    applyTheme(document.documentElement.getAttribute("data-theme") || "dark");
    if (themeBtn) {
        themeBtn.addEventListener("click", function () {
            var cur = document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light";
            try { localStorage.setItem("anneal-theme", cur); } catch (e) {}
            applyTheme(cur);
        });
    }

    /* ── Sticky scroll progress bar ─────────────────────────────────────── */
    var fill = $("#js-progress");
    if (fill) {
        var updateProgress = function () {
            var doc = document.documentElement;
            var max = doc.scrollHeight - doc.clientHeight;
            var pct = max > 0 ? (window.scrollY / max) * 100 : 0;
            fill.style.width = pct.toFixed(1) + "%";
        };
        window.addEventListener("scroll", updateProgress, { passive: true });
        window.addEventListener("resize", updateProgress);
        updateProgress();
    }

    /* ── Active rail link via IntersectionObserver ──────────────────────── */
    var railLinks = $$("#js-rail a");
    var byHash = {};
    railLinks.forEach(function (a) { byHash[a.getAttribute("href")] = a; });
    if ("IntersectionObserver" in window && railLinks.length) {
        var io = new IntersectionObserver(function (entries) {
            entries.forEach(function (e) {
                if (e.isIntersecting) {
                    var id = "#" + e.target.id;
                    if (!byHash[id]) return;
                    railLinks.forEach(function (a) { a.classList.remove("active"); });
                    byHash[id].classList.add("active");
                }
            });
        }, { rootMargin: "-35% 0px -55% 0px" });
        Object.keys(byHash).forEach(function (hash) {
            var el = document.getElementById(hash.slice(1));
            if (el) io.observe(el);
        });
    }

    /* ── Terminal copy buttons ──────────────────────────────────────────── */
    $$(".term-copy").forEach(function (btn) {
        btn.addEventListener("click", function () {
            var cmd = btn.getAttribute("data-copy") || "";
            if (!cmd || !navigator.clipboard) return;
            navigator.clipboard.writeText(cmd).then(function () {
                var prev = btn.textContent;
                btn.textContent = "copied";
                btn.classList.add("copied");
                setTimeout(function () {
                    btn.textContent = prev;
                    btn.classList.remove("copied");
                }, 1600);
            });
        });
    });

    /* ── Model selection: picker tabs + catalog cards stay in sync ──────── */
    var tabs = $$("#js-modeltabs button");
    var cards = $$(".lesson-card[data-model]");
    var announce = $("#js-modelannounce");

    function paint(current) {
        tabs.forEach(function (btn) {
            var on = btn.getAttribute("data-model") === current;
            btn.setAttribute("aria-selected", on ? "true" : "false");
            btn.setAttribute("tabindex", on ? "0" : "-1");
        });
        cards.forEach(function (card) {
            var on = card.getAttribute("data-model") === current;
            if (on) card.setAttribute("aria-current", "true");
            else card.removeAttribute("aria-current");
        });
    }

    function setHashModel(name) {
        try {
            var newHash = "model=" + name;
            if (("#" + newHash) !== window.location.hash) {
                history.replaceState(null, "",
                    window.location.pathname + window.location.search + "#" + newHash);
            }
        } catch (e) {}
    }

    function applyModel(name, opts) {
        opts = opts || {};
        if (!isModel(name)) name = IDS[0];
        document.documentElement.setAttribute("data-model", name);
        paint(name);
        try { localStorage.setItem("anneal-tutorial-model", name); } catch (e) {}
        if (opts.writeHash !== false) setHashModel(name);
        if (announce && opts.announce !== false) {
            announce.textContent = "Now showing the " + (LABEL[name] || name) + " lesson.";
        }
    }

    /* initial paint mirrors the pre-paint <html data-model> */
    var initial = document.documentElement.getAttribute("data-model") || IDS[0];
    if (!isModel(initial)) initial = IDS[0];
    paint(initial);

    /* tablist: click + roving-tabindex keyboard nav */
    tabs.forEach(function (btn, i) {
        btn.addEventListener("click", function () {
            applyModel(btn.getAttribute("data-model"));
            btn.focus();
        });
        btn.addEventListener("keydown", function (e) {
            var k = e.key, idx = i;
            if (k === "ArrowRight") idx = (i + 1) % tabs.length;
            else if (k === "ArrowLeft") idx = (i - 1 + tabs.length) % tabs.length;
            else if (k === "Home") idx = 0;
            else if (k === "End") idx = tabs.length - 1;
            else if (k === "Enter" || k === " ") {
                e.preventDefault();
                applyModel(btn.getAttribute("data-model"));
                btn.focus();
                return;
            } else { return; }
            e.preventDefault();
            var next = tabs[idx];
            applyModel(next.getAttribute("data-model"));
            next.focus();
        });
    });

    /* catalog cards: choose a model AND drop into the guided path.
       Without JS the card is a plain anchor to the path start and every
       variant is revealed, so nothing is gated behind script. */
    var pathStart = document.getElementById("prereqs");
    cards.forEach(function (card) {
        card.addEventListener("click", function (e) {
            var id = card.getAttribute("data-model");
            if (!isModel(id)) return; /* e.g. the "more coming" card */
            e.preventDefault();
            applyModel(id);
            if (pathStart) {
                pathStart.scrollIntoView({
                    behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
                    block: "start"
                });
            }
        });
    });

    /* browser back/forward across #model=… deep links */
    window.addEventListener("hashchange", function () {
        var hash = (window.location.hash || "").replace(/^#/, "");
        var m = (hash.match(/(?:^|[&;])model=([a-z0-9]+)/) || [])[1];
        if (m && isModel(m)) applyModel(m, { writeHash: false });
    });
})();
