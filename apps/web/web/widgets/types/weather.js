// Weather widget — current weather at the configured location.
import { register } from "/static/widgets/index.js";

register({
  type: "weather",
  title: "날씨",
  icon: "sun",
  description: "현재 위치의 날씨",

  mount(container) {
    container.classList.add("widget-body--weather");
    container.innerHTML = `
      <div class="wx-row">
        <div class="wx-temp">--°</div>
        <div class="wx-meta">
          <div class="wx-cond">로딩 중...</div>
          <div class="wx-loc"></div>
        </div>
      </div>
      <div class="wx-extras">
        <div><span class="label">체감</span> <span class="apparent">-</span></div>
        <div><span class="label">습도</span> <span class="humidity">-</span></div>
        <div><span class="label">바람</span> <span class="wind">-</span></div>
      </div>
    `;
    const tempEl = container.querySelector(".wx-temp");
    const condEl = container.querySelector(".wx-cond");
    const locEl  = container.querySelector(".wx-loc");
    const apparentEl = container.querySelector(".apparent");
    const humidityEl = container.querySelector(".humidity");
    const windEl     = container.querySelector(".wind");

    async function refresh() {
      try {
        const r = await fetch("/api/widgets/weather");
        if (!r.ok) {
          condEl.textContent = "정보를 가져올 수 없어요";
          return;
        }
        const w = await r.json();
        tempEl.textContent = `${Math.round(w.temperature_c)}°`;
        condEl.textContent = w.condition_ko || w.condition || "";
        locEl.textContent = w.location_name || "";
        apparentEl.textContent = w.apparent_c != null ? `${Math.round(w.apparent_c)}°` : "-";
        humidityEl.textContent = w.humidity_pct != null ? `${w.humidity_pct}%` : "-";
        windEl.textContent = w.wind_ms != null ? `${w.wind_ms.toFixed(1)} m/s` : "-";
      } catch (err) {
        console.warn("weather refresh failed", err);
        condEl.textContent = "오프라인";
      }
    }

    refresh();
    const interval = setInterval(refresh, 10 * 60 * 1000); // 10 min
    return () => clearInterval(interval);
  },
});
