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
        <input type="text" class="stocks-input" placeholder="티커 추가 (예: TSLA, 005930.KS)" />
        <button type="button" class="stocks-add"><span class="icon">${iconHTML("plus")}</span></button>
      </div>
    `;

    const listEl = container.querySelector(".stocks-list");
    const inputEl = container.querySelector(".stocks-input");
    const addBtn = container.querySelector(".stocks-add");

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

    function tryAdd() {
      const v = inputEl.value.trim().toUpperCase();
      if (!v) return;
      if (tickers.includes(v)) {
        inputEl.value = "";
        return;
      }
      tickers.push(v);
      inputEl.value = "";
      persist();
      fetchQuotes();
    }

    addBtn.addEventListener("click", tryAdd);
    inputEl.addEventListener("keydown", (ev) => {
      if (ev.isComposing || ev.keyCode === 229) return;
      if (ev.key === "Enter") {
        ev.preventDefault();
        tryAdd();
      }
    });

    fetchQuotes();
    const interval = setInterval(fetchQuotes, 60 * 1000);
    return () => clearInterval(interval);
  },
});
