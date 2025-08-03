
import { rememberAndRestoreQueryParams } from './helpers.js';

function adjustYear(delta) {
    const input = document.getElementById("year");
    if (!input || !input.value) return;

    if (delta == "latest") {
        // Get year from 7 days ago:
        const d = new Date();
        d.setDate(d.getDate() - 7);
        input.value = d.getFullYear();
    } else {
        // Relative change
        const current = parseInt(input.value);
        const next = current + delta;
        input.value = next;
    }

    const form = document.getElementById('chart-controls');
    if (form) {
        htmx.trigger(form, 'change');
    }
}

function toggleLegend(button, contentId) {
    const content = document.getElementById(contentId);
    const isHidden = content.classList.contains("hidden");

    content.classList.toggle("hidden");
    button.innerHTML = isHidden ? "Hide legend &#9660;" : "Show legend &#9658;";
}


export function initYearPage() {

    document.addEventListener("DOMContentLoaded", () => {

        // Trigger form change to force htmx to load elements.
        const form = document.getElementById('chart-controls');
        if (form) {
            rememberAndRestoreQueryParams(form);
            htmx.trigger(form, 'change');
        }

        // Listen for prev/next button clicks.
        document.getElementById("prev-year")?.addEventListener("click", () => adjustYear(-1));
        document.getElementById("next-year")?.addEventListener("click", () => adjustYear(+1));
        document.getElementById("current-year")?.addEventListener("click", () => adjustYear("latest"));

        // Expand / collapse legend
        document.getElementById("drywet-legend-toggle")?.addEventListener("click", (e) =>
            toggleLegend(e.target, "drywet-legend-content")
        );
    });
}
