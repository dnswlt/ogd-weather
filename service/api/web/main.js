import './style.css'; // Make sure Tailwind CSS gets included by vite.

import {
    htmxSwapStationSummaryContainer,
    rememberAndRestoreQueryParams,
} from './helpers.js';

import { initMapPage } from './map.js';
import { initDayPage } from './day.js';
import { initYearPage } from './year.js';

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

function embedVegaScript(script) {
    const parent = script.closest("div[hx-get]");
    // Determine ID of element (<div>) to embed Vega chart.
    let targetId;
    if (parent.dataset.vegaTargets) {
        const targets = JSON.parse(parent.dataset.vegaTargets);
        targetId = targets[script.dataset.vegaSpec];
    } else if (parent.dataset.vegaTarget) {
        targetId = parent.dataset.vegaTarget;
    } else {
        // Default behaviour: Use data-vega-spec attribute to find target.
        targetId = `${script.dataset.vegaSpec}-chart`;
    }
    const spec = JSON.parse(script.textContent);
    vegaEmbed(`#${targetId}`, spec, { actions: false });
}

// Initialization that applies to all pages.
function commonInit() {
    // Run vegaEmbed when receiving new snippets containing Vega-Lite JSON.
    htmx.onLoad(function (content) {
        content.querySelectorAll('script[type="application/json"]')
            .forEach(script => {
                try {
                    embedVegaScript(script);
                } catch (e) {
                    console.error('Failed to embed Vega script:', e);
                } finally {
                    // avoid reprocessing and cluttering the DOM.
                    script.remove();
                }
            });
    });
}

commonInit();

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
    case 'year':
        initYearPage();
        break;
    case 'map':
        initMapPage();
        break;
    default:
        console.error(`Unhandled page in init: ${page}`);
        break;
}
