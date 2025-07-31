import './style.css'; // Make sure Tailwind CSS gets included by vite.

import { rememberAndRestoreQueryParams, htmxSwapStationSummaryContainer } from './helpers.js';

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById('chart-controls');
    if (form) {
        rememberAndRestoreQueryParams(form);
        htmx.trigger(form, 'change');
    }

    // Add handlers for postprocessing after htmx swaps.
    document.body.addEventListener('htmx:afterSwap', (evt) => {
        if (evt.target.id === 'station-summary-container') {
            htmxSwapStationSummaryContainer(evt.target);
        }
    });

});
