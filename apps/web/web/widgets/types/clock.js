// Clock widget — current local time and date. No backend.
import { register } from "/static/widgets/index.js";

register({
  type: "clock",
  title: "시계",
  icon: "clock",
  description: "현재 시각",

  mount(container) {
    container.classList.add("widget-body--clock");
    container.innerHTML = `
      <div class="clock-time"></div>
      <div class="clock-date"></div>
    `;
    const timeEl = container.querySelector(".clock-time");
    const dateEl = container.querySelector(".clock-date");

    function tick() {
      const now = new Date();
      timeEl.textContent = now.toLocaleTimeString("ko-KR", {
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
      });
      dateEl.textContent = now.toLocaleDateString("ko-KR", {
        year: "numeric",
        month: "long",
        day: "numeric",
        weekday: "long",
      });
    }

    tick();
    const interval = setInterval(tick, 1000 * 30);
    // Align to next minute boundary so :01-:29 don't update silently
    const ms = 60000 - (Date.now() % 60000);
    const align = setTimeout(tick, ms);

    return () => {
      clearInterval(interval);
      clearTimeout(align);
    };
  },
});
