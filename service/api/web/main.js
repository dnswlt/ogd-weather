import './style.css'; // Make sure Tailwind CSS gets included by vite.

import {
    htmxSwapStationSummaryContainer,
    rememberAndRestoreQueryParams,
} from './helpers.js';

import { initMapPage } from './map.js';
import { initDayPage } from './day.js';

// Default initialization for pages with chart controls.
function initPage() {

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

}

const page = document.body.dataset.page;

switch (page) {
    case 'timeline':
        initPage();
        break;
    case 'sun_rain':
        initPage();
        break;
    case 'day':
        initDayPage();
        break;
    case 'map':
        initMapPage();
        break;
    default:
        console.error(`Unhandled page in init: ${page}`);
        break;
}
