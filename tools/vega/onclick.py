import altair as alt
import pandas as pd


def create_and_save_circles(filename: str):
    # Sample data
    data = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [10, 12, 8, 14, 9],
            "category": ["A", "B", "A", "C", "B"],
        }
    )

    # Create a selection parameter that triggers on click
    click_param = alt.selection_point(on="click", empty="none", nearest=True)

    # Base chart
    base = (
        alt.Chart(data)
        .mark_circle(size=200)
        .encode(
            x="x",
            y="y",
            # Show category's color if clicked, else lightgray.
            color=alt.condition(click_param, "category:N", alt.value("lightgray")),
        )
        .add_params(click_param)
    )

    # Text layer to display information on click
    text = base.mark_text(align="left", dx=0, dy=-15).encode(
        # Show text only if selected, else show an empty text
        text=alt.condition(click_param, "category:N", alt.value(""))
    )

    # Layer the charts
    chart = (base + text).properties(title="Click on a point to see its category")

    chart.save(filename)


def create_and_save_bars(filename: str):

    # Sample data
    df = pd.DataFrame({"category": ["A", "B", "C", "D"], "value": [10, 23, 45, 8]})

    # Selection triggered by clicking on a bar
    click_param = alt.selection_point(fields=["category"], empty="none")

    # Bar chart with click selection
    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x="category:N",
            y="value:Q",
            color=alt.condition(
                click_param, alt.value("steelblue"), alt.value("lightgray")
            ),
        )
        .add_params(click_param)
    )

    # Fixed text box showing details of the clicked bar
    details = (
        bars.mark_text(align="center", dx=0, dy=-15)
        .encode(
            text=alt.condition(
                click_param,
                alt.format("value:Q", "%d %%"),
                alt.value("Tap a bar to see value"),
            )
        )
        .transform_filter(click_param)
    )

    # Vertical layout
    chart = alt.layer(bars, details)

    chart.save(filename)


if __name__ == "__main__":
    create_and_save_circles("./examples/onclick_circle.html")
    create_and_save_bars("./examples/onclick_bars.html")
