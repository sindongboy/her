// Widget bootstrap — import each widget definition (which calls `register`)
// then mount everything once into the rails.
import { boot } from "/static/widgets/index.js";
import "/static/widgets/types/sessions.js";
import "/static/widgets/types/recall.js";
import "/static/widgets/types/facts.js";
import "/static/widgets/types/notes.js";
import "/static/widgets/types/people.js";
import "/static/widgets/types/clock.js";
import "/static/widgets/types/weather.js";
import "/static/widgets/types/calendar.js";
import "/static/widgets/types/stocks.js";
import "/static/widgets/types/stock-news.js";

boot();
