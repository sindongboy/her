// Stock-news widget — pulls Tavily news for the user's stocks watchlist.
// Shares the ticker list with the stocks widget (her.widgets.settings.stocks.tickers)
// so adding a ticker over there automatically populates here.
import { register, loadConfig } from "/static/widgets/index.js";
import { iconHTML } from "/static/icons.js";

const PER_TICKER_OPTIONS = [1, 2, 3, 5];
const DEFAULT_PER_TICKER = 3;

function fmtAge(iso) {
  if (!iso) return "";
  const t = new Date(iso);
  if (isNaN(t.valueOf())) return "";
  const diffMs = Date.now() - t.valueOf();
  const days = Math.floor(diffMs / (24 * 3600 * 1000));
  if (days === 0) return "오늘";
  if (days === 1) return "어제";
  if (days < 7) return `${days}일 전`;
  return t.toLocaleDateString("ko-KR", { month: "numeric", day: "numeric" });
}

function escape(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

register({
  type: "stock-news",
  title: "주식 뉴스",
  icon: "trend",
  description: "관심 종목 최신 뉴스 (Tavily)",

  mount(container, ctx) {
    let perTicker = Number(ctx.settings?.per_ticker) || DEFAULT_PER_TICKER;

    container.innerHTML = `
      <div class="news-toolbar">
        <label>종목당 <select class="news-per-ticker"></select> 개</label>
      </div>
      <div class="news-body"></div>
    `;
    const sel = container.querySelector(".news-per-ticker");
    const body = container.querySelector(".news-body");

    const opts = new Set([...PER_TICKER_OPTIONS, perTicker]);
    for (const n of [...opts].sort((a, b) => a - b)) {
      const o = document.createElement("option");
      o.value = String(n); o.textContent = String(n);
      if (n === perTicker) o.selected = true;
      sel.append(o);
    }
    sel.addEventListener("change", () => {
      perTicker = Number(sel.value) || DEFAULT_PER_TICKER;
      ctx.updateSettings({ ...(ctx.settings || {}), per_ticker: perTicker });
      refresh();
    });

    function getTickers() {
      const cfg = loadConfig();
      const stocksSettings = cfg.settings?.stocks || {};
      const list = Array.isArray(stocksSettings.tickers) ? stocksSettings.tickers : [];
      return list.filter(Boolean);
    }

    async function refresh() {
      const tickers = getTickers();
      if (tickers.length === 0) {
        body.innerHTML = `<p class="empty">주식 위젯에서 ticker 를 먼저 추가하세요.</p>`;
        return;
      }
      body.innerHTML = `<p class="empty">불러오는 중…</p>`;
      try {
        const url = `/api/widgets/stock-news?tickers=${encodeURIComponent(tickers.join(","))}&per_ticker=${perTicker}`;
        const r = await fetch(url);
        if (!r.ok) {
          body.innerHTML = `<p class="empty">뉴스 로드 실패</p>`;
          return;
        }
        const data = await r.json();
        body.innerHTML = "";
        let any = false;
        for (const ticker of tickers) {
          const items = data[ticker] || [];
          if (items.length === 0) continue;
          any = true;
          const group = document.createElement("section");
          group.className = "news-group";
          group.innerHTML = `<h3 class="news-ticker">${escape(ticker)}</h3>`;
          for (const it of items) {
            const card = document.createElement("a");
            card.className = "news-card";
            card.href = it.url;
            card.target = "_blank";
            card.rel = "noopener noreferrer";
            card.innerHTML = `
              <div class="news-title">${escape(it.title)}</div>
              <div class="news-meta">
                ${it.source ? `<span class="news-src">${escape(it.source)}</span>` : ""}
                ${it.published_date ? `<span class="news-age">${escape(fmtAge(it.published_date))}</span>` : ""}
              </div>
            `;
            group.append(card);
          }
          body.append(group);
        }
        if (!any) {
          body.innerHTML = `<p class="empty">최근 뉴스가 없어요</p>`;
        }
      } catch (err) {
        console.warn("stock-news refresh failed", err);
        body.innerHTML = `<p class="empty">오프라인</p>`;
      }
    }

    refresh();
    const interval = setInterval(refresh, 10 * 60 * 1000);
    return () => clearInterval(interval);
  },
});
