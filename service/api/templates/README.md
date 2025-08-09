# Go HTML templates

This directory contains the HTML templates for pages and snippets loaded via htmx.

## Embedding Vega charts

The frontend uses htmx to query individual charts from the backends.

The Go `api` backend receives these htmx GET requests and requests the chart from
the Python `charts` backend.

The `charts` backend generates Vega and Vega-Lite charts and returns them in a `VegaSpecResponse`,
which is a map of Vega spec names to specs. Typically only one spec is returned For example:

```bash
$ curl -H 'Accept: application/json' 'http://localhost:8000/stations/BER/charts/year/2020/temperature:month'

{
    "vega_specs": {
        "temperature:month": {
            # Vega or Vega-Lite spec
            "config": {
                # ...
            },
            "autosize": {
                "type": "fit",
                "contains": "padding"
            }
            # ...
        }
    ]
}
```

The `api` backend wraps these specs in individual `<script>` elements in [vega_embed.html](./vega_embed.html).
Example:

```html
    <script type="application/json" data-vega-spec="{{vegaSpecHTMLID $name}}">
        {{ $spec }}
    </script>
```

(`vegaSpecHTMLID` ensures that spec names are valid HTML element IDs by replacing ':' by '-'.)

Now htmx will embed these script elements into the DOM. An `htmx:onLoad` handler defined
in [main.js](../web/main.js) will trigger and call `embedVegaScript` to identify the target
div for the chart canvas and make the appropriate `vegaEmbed` calls.

### Embedding logic

There are three cases to consider:

1. **Single chart**: There is one `div` that the single chart returned from the backend should
   be displayed in.
2. **Tabbed charts**: One of multiple charts can be displayed in the same `<div>`,
   depending on which tab is selected.
3. **Multiple charts**: The backend returns multiple charts, and all should be displayed
   at the same time.

#### Case 1: Single chart

For the simplest case of single charts, you can use the implicit mapping from
the chart type (as specified in the URL path, e.g. `humidity:month`)
to (HTML-sanitized) div ID ("-chart" appended to the chart type; e.g. `humidity-month-chart`).

```html
    <div>
        <div hx-get="/stations/{station}/charts/year/{year}/humidity:month">
            <!-- Vega JSON goes here -->
        </div>
        <div id="humidity-month-chart" class="w-full overflow-hidden"></div>
    </div>
```

Alternatively, we can specify the target div ID explicitly using `data-vega-target`, as explained below.

#### Case 2: Tabbed charts

For tabbed charts, we use a `tab-widget` that contains one button per tab,
a `vega-spec-loader` div containing the `hx-get` and
a `data-vega-target` data attribute to identify the target `<div>` holding the
chart, and the target div itself:

```html
    <div class="tab-widget">
        <div role="tablist">
            <!-- Sunshine hours -->
            <button data-facet="sunshine" role="tab" aria-selected="true">
                Sunshine
            </button>
            <!-- Clear (sunny) days -->
            <button data-facet="sunny_days" role="tab" aria-selected="false">
                Clear days
            </button>
        </div>

        <div class="vega-spec-loader" 
            hx-get="/stations/{station}/charts/year/{year}/{facet}:month" hx-ext="path-params"
            data-vega-target="sunshine-vega-chart">
            <!-- Vega JSON goes here -->
        </div>
        <div id="sunshine-vega-chart"></div>
    </div>
```

The Vega spec name is irrelevant in this case.

#### Case 3: Multiple charts

For multple charts, we set the `data-vega-targets` data attribute on the div containing the
`hx-get` to a JSON object that maps each (HTML sanitized) spec name to the target `<div>` ID:

```html
    <div>
        <div hx-get="/stations/{station}/charts/year/{year}/drywet"
            data-vega-targets='{"drywet":"drywet-chart","drywet-spells":"drywet-spells-chart"}'>
            <!-- Vega JSON goes here -->
        </div>
        <!-- Two charts, stacked: one for the dry/wet grid and one for the dry/wet spell bars. -->
        <div id="drywet-chart"></div>
        <div id="drywet-spells-chart"></div>
    </div>
```
