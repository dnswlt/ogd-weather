
import {
    rememberAndRestoreQueryParams,
    updateNavbarLinks,
} from './helpers.js';


function registerSearchBarHandler() {
    const results = document.getElementById('search-results');
    if (!results) return;
    const form = document.getElementById('compare-controls');
    if (!form) return;

    // Delegate clicks from buttons inside the results box.
    results.addEventListener('click', (e) => {
        const btn = e.target.closest('button[data-station]');
        if (!btn || !results.contains(btn)) return;


        // Update the hidden stations= <input>: add clicked station, unless it is already present.
        const stationId = btn.dataset.station;
        const stationsInput = form.querySelector('input[name="stations"]');
        if (!stationsInput) return;

        // If stationId already exists: return, else append it to .value.
        const stations = stationsInput.value ? stationsInput.value.split(",") : [];
        if (stations.includes(stationId)) {
            return;
        } else if (stations.length >= 4) {
            stations[stations.length-1] = stationId;
        } else {
            stations.push(stationId);
        }
        stationsInput.value = stations.join(',');

        // Trigger a 'change' event on the form so htmx can reload.
        htmx.trigger(form, 'change');

        // Clear the dropdown.
        results.innerHTML = '';
    });
}

// Add listeners to remove a column if its "X" button is clicked.
function registerStationColumnRemoval() {
    const tableContainer = document.getElementById('comparison-table-container');
    if (!tableContainer) return;
    const form = document.getElementById('compare-controls');
    if (!form) return;

    // Add click listener to container that is not reloaded by htmx.
    tableContainer.addEventListener('click', (e) => {
        // Target a remove button within a table header.
        const removeBtn = e.target.closest('.remove-station-btn');
        if (!removeBtn) return;

        // Find the parent header to get the station ID.
        const th = removeBtn.closest('th[data-station]');
        if (!th) return;

        // Remove the clicked station from the stations= hidden input.
        const stationToRemove = th.dataset.station;
        const stationsInput = form.querySelector('input[name="stations"]');
        if (!stationsInput) return;

        const stations = stationsInput.value ? stationsInput.value.split(",") : [];
        const newStations = stations.filter(s => s != stationToRemove);
        if (stations.length == newStations.length) {
            return; // No station removed.
        }
        stationsInput.value = newStations.join(",");

        // Trigger htmx to reload the table.
        htmx.trigger(form, 'change');
    });
}

export function initComparePage() {

    document.addEventListener("DOMContentLoaded", () => {

        // Trigger form change to force htmx to load elements.
        const form = document.getElementById('compare-controls');
        if (form) {
            rememberAndRestoreQueryParams(form);
            htmx.trigger(form, 'change');
        }
        // On page load: update navbar using current + saved state
        updateNavbarLinks(new URLSearchParams(window.location.search));

        registerSearchBarHandler();
        registerStationColumnRemoval();
    });
}
