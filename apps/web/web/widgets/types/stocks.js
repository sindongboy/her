// Stocks widget — watchlist of tickers; routed server-side to the right
// provider (finnhub/polygon for global, kiwoom for KR). Tickers are
// configured per-instance via the inline editor.
import { register } from "/static/widgets/index.js";
import { iconHTML } from "/static/icons.js";

const DEFAULT_TICKERS = ["AAPL", "005930.KS"];

function pct(v) {
  if (v == null || isNaN(v)) return "-";
  const sign = v > 0 ? "+" : "";
  return `${sign}${v.toFixed(2)}%`;
}

function pctClass(v) {
  if (v == null || isNaN(v) || v === 0) return "neutral";
  return v > 0 ? "up" : "down";
}

function fmtPrice(v, currency) {
  if (v == null) return "-";
  if (currency === "KRW") {
    return new Intl.NumberFormat("ko-KR").format(Math.round(v)) + "원";
  }
  return v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

register({
  type: "stocks",
  title: "주식",
  icon: "trend",
  description: "관심 종목 시세",

  mount(container, ctx) {
    const tickers = (ctx.settings?.tickers && ctx.settings.tickers.length)
      ? [...ctx.settings.tickers]
      : [...DEFAULT_TICKERS];

    container.innerHTML = `
      <ul class="stocks-list"></ul>
      <div class="stocks-edit">
        <div class="stocks-search">
          <input type="text" class="stocks-input" placeholder="종목명 또는 티커 (예: 삼성전자, AAPL)" autocomplete="off" />
          <ul class="stocks-suggest" hidden></ul>
        </div>
        <button type="button" class="stocks-add"><span class="icon">${iconHTML("plus")}</span></button>
      </div>
    `;

    const listEl = container.querySelector(".stocks-list");
    const inputEl = container.querySelector(".stocks-input");
    const addBtn = container.querySelector(".stocks-add");
    const suggestEl = container.querySelector(".stocks-suggest");

    function persist() {
      ctx.updateSettings({ tickers });
    }

    function renderEmpty() {
      listEl.innerHTML = "";
      const li = document.createElement("li");
      li.className = "empty";
      li.textContent = "관심 종목을 추가하세요";
      listEl.append(li);
    }

    function renderRows(quotes) {
      listEl.innerHTML = "";
      const byTicker = new Map(quotes.map((q) => [q.ticker, q]));
      for (const ticker of tickers) {
        const q = byTicker.get(ticker);
        const li = document.createElement("li");
        li.className = "stock-row";
        if (q?.error) li.classList.add("err");
        li.dataset.ticker = ticker;
        const sym = document.createElement("div");
        sym.className = "stock-sym";
        sym.textContent = q?.name ? `${ticker}` : ticker;
        const name = document.createElement("div");
        name.className = "stock-name";
        name.textContent = q?.name || (q?.error ? "조회 실패" : "");
        const price = document.createElement("div");
        price.className = "stock-price";
        price.textContent = q ? fmtPrice(q.price, q.currency) : "-";
        const change = document.createElement("div");
        change.className = `stock-change ${pctClass(q?.change_pct)}`;
        change.textContent = q ? pct(q.change_pct) : "-";
        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "stock-remove";
        remove.title = `${ticker} 제거`;
        remove.innerHTML = `<span class="icon">${iconHTML("x")}</span>`;
        remove.addEventListener("click", () => {
          const idx = tickers.indexOf(ticker);
          if (idx >= 0) tickers.splice(idx, 1);
          persist();
          fetchQuotes();
        });
        li.append(sym, name, price, change, remove);
        listEl.append(li);
      }
    }

    async function fetchQuotes() {
      if (tickers.length === 0) { renderEmpty(); return; }
      try {
        const url = `/api/widgets/stocks?tickers=${encodeURIComponent(tickers.join(","))}`;
        const r = await fetch(url);
        if (!r.ok) {
          renderRows(tickers.map((t) => ({ ticker: t, error: true })));
          return;
        }
        const quotes = await r.json();
        renderRows(quotes);
      } catch (err) {
        console.warn("stocks refresh failed", err);
        renderRows(tickers.map((t) => ({ ticker: t, error: true })));
      }
    }

    function escape(s) {
      return String(s ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }

    function addTicker(symbol) {
      const sym = String(symbol || "").trim().toUpperCase();
      if (!sym) return;
      if (tickers.includes(sym)) {
        inputEl.value = "";
        hideSuggestions();
        return;
      }
      tickers.push(sym);
      inputEl.value = "";
      hideSuggestions();
      persist();
      fetchQuotes();
    }

    function tryAddFromInput() {
      const v = inputEl.value.trim();
      if (!v) return;
      // If suggestions are open and have a focused item, prefer that.
      const focused = suggestEl.querySelector(".stocks-suggest-item.focused");
      if (focused) { addTicker(focused.dataset.symbol); return; }
      // Else add the literal text as a ticker (uppercased).
      addTicker(v);
    }

    let suggestSeq = 0;
    let lastQuery = "";
    let debounceTimer = null;

    function hideSuggestions() {
      suggestEl.hidden = true;
      suggestEl.innerHTML = "";
    }

    function renderSuggestions(matches) {
      if (!matches.length) { hideSuggestions(); return; }
      suggestEl.innerHTML = matches
        .map(
          (m) => `
            <li class="stocks-suggest-item" data-symbol="${escape(m.symbol)}" tabindex="-1">
              <span class="ss-sym">${escape(m.display_symbol || m.symbol)}</span>
              <span class="ss-name">${escape(m.name)}</span>
            </li>`,
        )
        .join("");
      suggestEl.hidden = false;
      suggestEl.querySelectorAll(".stocks-suggest-item").forEach((li) => {
        li.addEventListener("mousedown", (ev) => {
          ev.preventDefault();   // keep focus on input so blur doesn't fire first
          addTicker(li.dataset.symbol);
        });
      });
    }

    async function fetchSuggestions(query) {
      const seq = ++suggestSeq;
      try {
        const r = await fetch(`/api/widgets/stocks/search?q=${encodeURIComponent(query)}`);
        if (!r.ok) return;
        const matches = await r.json();
        if (seq !== suggestSeq) return;  // stale response — newer query in-flight
        renderSuggestions(matches);
      } catch (err) {
        console.warn("stocks suggest failed", err);
      }
    }

    function scheduleSuggest() {
      const v = inputEl.value.trim();
      if (v === lastQuery) return;
      lastQuery = v;
      clearTimeout(debounceTimer);
      if (v.length < 2) { hideSuggestions(); return; }
      debounceTimer = setTimeout(() => fetchSuggestions(v), 250);
    }

    function moveFocus(dir) {
      const items = [...suggestEl.querySelectorAll(".stocks-suggest-item")];
      if (!items.length) return;
      const cur = items.findIndex((it) => it.classList.contains("focused"));
      const next = (cur + dir + items.length) % items.length;
      items.forEach((it) => it.classList.remove("focused"));
      const target = cur < 0 && dir < 0 ? items[items.length - 1] : items[next < 0 ? 0 : next];
      target.classList.add("focused");
      target.scrollIntoView({ block: "nearest" });
    }

    addBtn.addEventListener("click", tryAddFromInput);
    inputEl.addEventListener("input", scheduleSuggest);
    inputEl.addEventListener("blur", () => {
      // Hide after a short delay so mousedown on a suggestion still wins.
      setTimeout(hideSuggestions, 120);
    });
    inputEl.addEventListener("keydown", (ev) => {
      if (ev.isComposing || ev.keyCode === 229) return;
      if (ev.key === "ArrowDown") { ev.preventDefault(); moveFocus(+1); return; }
      if (ev.key === "ArrowUp")   { ev.preventDefault(); moveFocus(-1); return; }
      if (ev.key === "Escape")    { hideSuggestions(); return; }
      if (ev.key === "Enter") {
        ev.preventDefault();
        tryAddFromInput();
      }
    });

    fetchQuotes();
    const interval = setInterval(fetchQuotes, 60 * 1000);
    return () => {
      clearInterval(interval);
      clearTimeout(debounceTimer);
    };
  },
});
