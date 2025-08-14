import './style.css'; // Make sure Tailwind CSS gets included by vite.

import {
    htmxSwapStationSummaryContainer,
    rememberAndRestoreQueryParams,
    registerSearchBarHandler,
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

    registerSearchBarHandler();
}

function getVegaTargets(el) {
    if (el.dataset.vegaTargets) {
        const ts = JSON.parse(el.dataset.vegaTargets);
        return Object.values(ts);
    }
    if (el.dataset.vegaTarget) {
        return [el.dataset.vegaTarget];
    }
    return [];
}

function embedVegaScript(parent, script) {
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
    htmx.onLoad(function (content) {
        // Find <div hx-get> parent that loaded this content.
        // NOTE: This assumes that snippets have "this" as the htmx target.
        const hxParent = content.closest("div[hx-get]");
        // Run embedVegaScript on all Vega-Lite JSON <script> elements.
        const scripts = content.querySelectorAll('script[type="application/json"]');
        scripts.forEach(script => {
            try {
                embedVegaScript(hxParent, script);
            } catch (e) {
                console.error('Failed to embed Vega script:', e);
            } finally {
                // avoid reprocessing and cluttering the DOM.
                script.remove();
            }
        });
        // Error handling: 
        // Remove all vega-target(s) if no corresponding vega scripts were found.
        if (hxParent) {
            for (const tgt of getVegaTargets(hxParent)) {
                const el = document.getElementById(tgt);
                // Remove all content (charts in particular).
                if (el) {
                    while (el.firstChild) {
                        el.removeChild(el.firstChild);
                    }
                }
            }
        }
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
