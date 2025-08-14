
import {
    rememberAndRestoreQueryParams,
    registerTablistHandler,
    registerSearchBarHandler,
} from './helpers.js';

function adjustDate(deltaDays) {
    const input = document.getElementById("date");
    if (!input || !input.value) return;
    const current = new Date(input.value);
    current.setDate(current.getDate() + deltaDays);
    const next = current.toISOString().split("T")[0];
    input.value = next;

    const form = document.getElementById('chart-controls');
    if (form) {
        htmx.trigger(form, 'change');
    }
}

function setYesterdayDateString() {
    const input = document.getElementById("date");
    if (!input || !input.value) return;

    const d = new Date();
    d.setDate(d.getDate() - 1);
    const dateStr = d.toISOString().split("T")[0];
    input.value = dateStr;

    const form = document.getElementById('chart-controls');
    if (form) {
        htmx.trigger(form, 'change');
    }
}


export function initDayPage() {

    document.addEventListener("DOMContentLoaded", () => {

        // Trigger form change to force htmx to load elements.
        const form = document.getElementById('chart-controls');
        if (form) {
            rememberAndRestoreQueryParams(form);
            htmx.trigger(form, 'change');
        }

        // Listen for prev/next button clicks.
        document.getElementById("prev-date")?.addEventListener("click", () => adjustDate(-1));
        document.getElementById("next-date")?.addEventListener("click", () => adjustDate(+1));
        document.getElementById("latest-date")?.addEventListener("click", () => setYesterdayDateString());

    });

    // Handle clicks on tablist buttons.
    registerTablistHandler();

    registerSearchBarHandler();
}
