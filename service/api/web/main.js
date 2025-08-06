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

// Initialization that applies to all pages.
function commonInit() {
    // Run vegaEmbed when receiving new snippets containing Vega-Lite JSON.
    htmx.onLoad(function (content) {
        content.querySelectorAll('script[type="application/json"][data-vega-target]')
            .forEach(s => {
                const targetId = s.dataset.vegaTarget;
                let spec;
                try {
                    spec = JSON.parse(s.textContent);
                } catch (e) {
                    console.error('Invalid Vega JSON for', targetId, e);
                    s.remove();
                    return;
                }
                vegaEmbed(`#${targetId}`, spec, { actions: false })
                // 
                // .then(
                //     res => {
                //         res.view.addEventListener("click", function (event, item) {
                //             if (!item?.datum) {
                //                 return;
                //             }
                //             console.log(`Clicked on something in ${targetId}`, event, item.datum);
                //         });
                //     })
                s.remove(); // avoid reprocessing and cluttering the DOM.
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
