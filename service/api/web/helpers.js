// Helper functions

// safeUpdateYearInputs updates the form <input> fields from_year and to_year
function safeUpdateYearInputs(firstYear, lastYear, minYear, maxYear) {
    const fy = document.getElementById('from_year');
    const ty = document.getElementById('to_year');
    if (!fy || !ty) return;

    fy.value = firstYear;
    ty.value = lastYear;
    fy.min = minYear;
    ty.min = minYear;
    fy.max = maxYear;
    ty.max = maxYear;
};

export function htmxSwapStationSummaryContainer(container) {
    // htmx swap of the station summary => Update from/to year inputs.
    const ss = container.querySelector("#station-summary");
    if (!ss) {
        console.log(`container ${container.id} was htmx swapped without station-summary.`);
        return;
    }
    const first = parseInt(ss.dataset.firstYear, 10);
    const last = parseInt(ss.dataset.lastYear, 10);
    const min = parseInt(ss.dataset.minYear, 10);
    const max = parseInt(ss.dataset.maxYear, 10);

    // Update form input fields if we have new data.
    const valuesAreValid = [first, last, min, max].every(v => !isNaN(v));
    if (valuesAreValid) {
        safeUpdateYearInputs(first, last, min, max);
    }
}

export function rememberAndRestoreQueryParams(form) {
    const path = window.location.pathname;
    const currentParams = new URLSearchParams(window.location.search);

    // On form change: update sessionStorage, URL, and navbar.
    form.addEventListener('change', () => {
        const formData = new FormData(form);
        const params = new URLSearchParams();

        for (const [key, value] of formData.entries()) {
            if (value !== "") {
                params.set(key, value);
            }
        }

        sessionStorage.setItem(path, params.toString());

        const newUrl = `${path}?${params.toString()}`;
        window.history.replaceState({}, '', newUrl);

        updateNavbarLinks(params);
    });

    // On page load: update navbar using current + saved state
    updateNavbarLinks(currentParams);
}


// Updates #navbar-links a[href] according to their declared data-params.
export function updateNavbarLinks(currentParams) {
    for (const a of document.querySelectorAll("#navbar-links a")) {
        const linkUrl = new URL(a.href, window.location.origin);
        const linkPath = linkUrl.pathname;

        // Get saved params for the target page
        const saved = sessionStorage.getItem(linkPath);
        const savedParams = saved ? new URLSearchParams(saved) : new URLSearchParams();
        // Get list of relevant parameters from data-params attribute
        const paramList = (a.dataset.params || "").split(',').map(p => p.trim()).filter(Boolean);

        // Merge new values from currentParams and existing savedParams.
        for (const param of paramList) {
            if (currentParams.has(param)) {
                linkUrl.searchParams.set(param, currentParams.get(param));
            } else if (savedParams.has(param)) {
                linkUrl.searchParams.set(param, savedParams.get(param));
            } else {
                linkUrl.searchParams.delete(param);
            }
        }

        a.href = linkUrl.toString();
    }
}
