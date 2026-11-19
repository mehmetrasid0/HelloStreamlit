from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Titanic Passenger Insights",
    page_icon="🚢",
    layout="wide",
)
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    [data-testid="stMetric"] {
        background: #f1f5f9;
        border-left: 4px solid #1d4e89;
        padding: 0.8rem 1rem;
        border-radius: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

NAVY = "#1d4e89"
BLUE = "#4f86c6"
TEAL = "#2a9d8f"
CORAL = "#d66a75"
SLATE = "#94a3b8"
CLASS_ORDER = ["1st class", "2nd class", "3rd class"]
CLASS_COLORS = [NAVY, BLUE, TEAL]
CLASS_LABELS = {1: "1st class", 2: "2nd class", 3: "3rd class"}
AGE_GROUPS = ["Child", "Teen", "Young Adult", "Adult", "Older Adult"]
FIELDS = ["survived", "pclass", "sex", "age", "fare", "embarked"]


def categorical_bar_chart(
    data: pd.DataFrame,
    category_column: str,
    value_column: str,
    category_order: list[str],
    colors: list[str],
    value_title: str,
    value_format: str | None = None,
) -> alt.Chart:
    value_axis = alt.Axis(format=value_format) if value_format else alt.Axis()
    value_tooltip = (
        alt.Tooltip(
            f"{value_column}:Q",
            title=value_title,
            format=value_format,
        )
        if value_format
        else alt.Tooltip(f"{value_column}:Q", title=value_title)
    )
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X(f"{value_column}:Q", title=value_title, axis=value_axis),
            y=alt.Y(f"{category_column}:N", title=None, sort=category_order),
            color=alt.Color(
                f"{category_column}:N",
                scale=alt.Scale(domain=category_order, range=colors),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip(f"{category_column}:N", title=category_column),
                value_tooltip,
            ],
        )
        .properties(height=max(150, len(data) * 42))
    )


df = pd.read_csv(Path(__file__).with_name("titanic.csv"))
analysis = df[FIELDS].copy()
missing_before = analysis.isna().sum()
median_age = analysis["age"].median()
most_common_embarked = analysis["embarked"].mode().iloc[0]

analysis["age"] = analysis["age"].fillna(median_age)
analysis["embarked"] = analysis["embarked"].fillna(most_common_embarked)
missing_after = analysis.isna().sum()

analysis["age_group"] = pd.cut(
    analysis["age"],
    bins=[0, 13, 19, 36, 61, float("inf")],
    right=False,
    labels=AGE_GROUPS,
)

for column in ("age", "fare"):
    minimum = analysis[column].min()
    maximum = analysis[column].max()
    analysis[f"{column}_norm"] = (analysis[column] - minimum) / (
        maximum - minimum
    )

dummy_columns = pd.get_dummies(
    analysis[["sex", "embarked"]],
    prefix=["sex", "embarked"],
    dtype=int,
)

st.title("Titanic Passenger Insights")
st.subheader("Explore survival patterns across passenger groups")
st.caption(
    "Use the shared sidebar filters to compare passengers. "
    "All rates describe this dataset and do not imply causation."
)

st.sidebar.header("Passenger filters")
classes = sorted(analysis["pclass"].unique().tolist())
sexes = sorted(analysis["sex"].unique().tolist())
selected_classes = st.sidebar.multiselect(
    "Passenger class",
    options=classes,
    default=classes,
    format_func=lambda value: CLASS_LABELS[value],
)
selected_sexes = st.sidebar.multiselect(
    "Sex",
    options=sexes,
    default=sexes,
    format_func=str.title,
)
selected_age_groups = st.sidebar.multiselect(
    "Age group",
    options=AGE_GROUPS,
    default=AGE_GROUPS,
)

filtered = analysis[
    analysis["pclass"].isin(selected_classes)
    & analysis["sex"].isin(selected_sexes)
    & analysis["age_group"].isin(selected_age_groups)
]

passengers = len(filtered)
survivors = int(filtered["survived"].sum())
survival_rate = filtered["survived"].mean()
average_fare = filtered["fare"].mean()

overview_tab, explore_tab, preprocessing_tab, explorer_tab = st.tabs(
    ["Overview", "Explore", "Preprocessing", "Data Explorer"]
)

with overview_tab:
    st.header("Overview")
    st.caption("Summary of the passengers selected in the sidebar.")
    kpi_columns = st.columns(4)
    kpi_columns[0].metric("Passengers in selection", f"{passengers:,}")
    kpi_columns[1].metric("Survivors", f"{survivors:,}")
    kpi_columns[2].metric(
        "Survival rate",
        f"{survival_rate:.1%}" if pd.notna(survival_rate) else "—",
    )
    kpi_columns[3].metric(
        "Average fare",
        f"${average_fare:,.2f}" if pd.notna(average_fare) else "—",
    )

    st.subheader("Survival summary")
    if filtered.empty:
        st.info("No passengers match these filters. Select at least one option in each filter.")
    else:
        survival_counts = (
            filtered["survived"]
            .map({0: "Did not survive", 1: "Survived"})
            .value_counts()
            .reindex(["Did not survive", "Survived"], fill_value=0)
            .rename_axis("outcome")
            .rename("passengers")
            .reset_index()
        )
        survival_chart, class_chart = st.columns(2)
        with survival_chart:
            st.markdown("**Survival counts**")
            st.altair_chart(
                categorical_bar_chart(
                    survival_counts,
                    "outcome",
                    "passengers",
                    ["Did not survive", "Survived"],
                    [SLATE, TEAL],
                    "Passengers",
                ),
                width="stretch",
            )
        with class_chart:
            st.markdown("**Survival rate by passenger class**")
            class_rates = (
                filtered.groupby("pclass")["survived"]
                .mean()
                .rename("survival_rate")
                .rename_axis("passenger_class")
                .reset_index()
            )
            class_rates["passenger_class"] = class_rates["passenger_class"].map(
                CLASS_LABELS
            )
            st.altair_chart(
                categorical_bar_chart(
                    class_rates,
                    "passenger_class",
                    "survival_rate",
                    CLASS_ORDER,
                    CLASS_COLORS,
                    "Survival rate",
                    ".0%",
                ),
                width="stretch",
            )
        st.caption("Survival rates are descriptive summaries of this dataset only.")

with explore_tab:
    st.header("Compare passenger groups")
    st.caption("Charts below all use the current sidebar selection.")
    if filtered.empty:
        st.info("No passengers match these filters. Select at least one option in each filter.")
    else:
        class_column, sex_column = st.columns(2)
        with class_column:
            st.markdown("**Survival rate by passenger class**")
            class_rates = (
                filtered.groupby("pclass")["survived"]
                .mean()
                .rename("survival_rate")
                .rename_axis("passenger_class")
                .reset_index()
            )
            class_rates["passenger_class"] = class_rates["passenger_class"].map(
                CLASS_LABELS
            )
            st.altair_chart(
                categorical_bar_chart(
                    class_rates,
                    "passenger_class",
                    "survival_rate",
                    CLASS_ORDER,
                    CLASS_COLORS,
                    "Survival rate",
                    ".0%",
                ),
                width="stretch",
            )
        with sex_column:
            st.markdown("**Survival rate by sex**")
            sex_rates = (
                filtered.groupby("sex")["survived"]
                .mean()
                .rename("survival_rate")
                .rename_axis("sex")
                .reset_index()
            )
            st.altair_chart(
                categorical_bar_chart(
                    sex_rates,
                    "sex",
                    "survival_rate",
                    ["female", "male"],
                    [CORAL, NAVY],
                    "Survival rate",
                    ".0%",
                ),
                width="stretch",
            )

        age_column, fare_column = st.columns(2)
        with age_column:
            st.markdown("**Age distribution**")
            age_histogram = (
                alt.Chart(filtered)
                .mark_bar(color=BLUE)
                .encode(
                    x=alt.X("age:Q", bin=alt.Bin(maxbins=20), title="Age (years)"),
                    y=alt.Y("count():Q", title="Passengers"),
                    tooltip=[
                        alt.Tooltip("count():Q", title="Passengers"),
                    ],
                )
                .properties(height=280)
            )
            st.altair_chart(age_histogram, width="stretch")
        with fare_column:
            st.markdown("**Average fare by passenger class**")
            fare_by_class = (
                filtered.groupby("pclass")["fare"]
                .mean()
                .rename("average_fare")
                .rename_axis("passenger_class")
                .reset_index()
            )
            fare_by_class["passenger_class"] = fare_by_class[
                "passenger_class"
            ].map(CLASS_LABELS)
            st.altair_chart(
                categorical_bar_chart(
                    fare_by_class,
                    "passenger_class",
                    "average_fare",
                    CLASS_ORDER,
                    CLASS_COLORS,
                    "Average fare ($)",
                    ".2f",
                ),
                width="stretch",
            )

        filtered_class_rates = filtered.groupby("pclass")["survived"].mean()
        if len(filtered_class_rates) > 1:
            highest = filtered_class_rates.idxmax()
            lowest = filtered_class_rates.idxmin()
            finding = (
                f"Among the selected passengers, the observed survival rate "
                f"was {filtered_class_rates[highest]:.1%} in "
                f"{CLASS_LABELS[highest]} and {filtered_class_rates[lowest]:.1%} "
                f"in {CLASS_LABELS[lowest]}. This is an association, not evidence "
                "that passenger class caused the difference."
            )
        else:
            selected_class = filtered_class_rates.index[0]
            finding = (
                f"Among the selected passengers in {CLASS_LABELS[selected_class]}, "
                f"the observed survival rate was "
                f"{filtered_class_rates[selected_class]:.1%}. This is descriptive "
                "and does not imply causation."
            )
        st.info(f"Finding: {finding}")

with preprocessing_tab:
    st.header("Preprocessing")
    st.write(
        "The analysis uses survived, pclass, sex, age, fare, and embarked. "
        "The deck column is omitted because it contains many missing values."
    )
    st.subheader("1. Missing values")
    st.caption(
        f"Missing ages are filled with the median ({median_age:g}); "
        f"missing embarked values are filled with the most frequent category "
        f"({most_common_embarked})."
    )
    st.dataframe(
        pd.DataFrame({"Before": missing_before, "After": missing_after}).loc[
            ["age", "embarked"]
        ],
        width="stretch",
    )

    st.subheader("2. Normalization")
    st.caption(
        "Min-Max normalization maps age and fare to the 0–1 range; "
        "the original age and fare columns are kept beside age_norm and fare_norm."
    )
    st.dataframe(
        analysis.loc[:4, ["age", "fare", "age_norm", "fare_norm"]],
        hide_index=True,
        width="stretch",
    )

    st.subheader("3. Age discretization")
    st.caption(
        "age_group is assigned from the original, unnormalized age, using "
        "Child (0–12), Teen (13–18), Young Adult (19–35), Adult (36–60), "
        "and Older Adult (61+)."
    )
    st.dataframe(
        analysis.loc[:4, ["age", "age_group"]],
        hide_index=True,
        width="stretch",
    )

    st.subheader("4. Dummy coding")
    st.caption(
        "Sex and embarked are represented by 0/1 indicator columns; "
        "survived is not encoded and the dummy columns are not normalized."
    )
    dummy_before, dummy_after = st.tabs(["Before", "After"])
    with dummy_before:
        st.dataframe(
            analysis[["sex", "embarked"]].iloc[:5],
            hide_index=True,
            width="stretch",
        )
    with dummy_after:
        dummy_preview = pd.concat(
            [
                analysis[["sex", "embarked"]].iloc[:5].reset_index(drop=True),
                dummy_columns.iloc[:5].reset_index(drop=True),
            ],
            axis=1,
        )
        st.dataframe(dummy_preview, hide_index=True, width="stretch")

with explorer_tab:
    st.header("Data Explorer")
    data_view = st.radio(
        "Select view",
        ["Raw", "Cleaned", "Transformed"],
        horizontal=True,
    )
    if data_view == "Raw":
        view_data = df.loc[filtered.index, FIELDS]
        default_columns = FIELDS
    elif data_view == "Cleaned":
        cleaned_columns = FIELDS + ["age_group"]
        view_data = analysis.loc[filtered.index, cleaned_columns]
        default_columns = cleaned_columns
    else:
        transformed_columns = FIELDS + ["age_group", "age_norm", "fare_norm"]
        view_data = analysis.loc[filtered.index, transformed_columns]
        default_columns = transformed_columns

    shape_columns = st.columns(2)
    shape_columns[0].metric("Source rows", f"{df.shape[0]:,}")
    shape_columns[1].metric("Source columns", f"{df.shape[1]:,}")

    with st.expander("Source columns and data types"):
        st.dataframe(
            pd.DataFrame(
                {"Column": df.columns, "Data type": df.dtypes.astype(str).values}
            ),
            hide_index=True,
            width="stretch",
        )
    with st.expander("First five source rows"):
        st.dataframe(df.iloc[:5], hide_index=True, width="stretch")
    with st.expander("Missing values in source data"):
        st.dataframe(
            df.isna().sum().rename("Missing values").to_frame(),
            width="stretch",
        )

    st.subheader(f"{data_view} passenger records")
    st.metric("Rows after filters", f"{len(view_data):,}")
    explorer_columns = st.multiselect(
        "Columns to display",
        options=view_data.columns.tolist(),
        default=default_columns,
        key=f"explorer_columns_{data_view.lower()}",
    )
    if explorer_columns:
        st.dataframe(
            view_data.loc[:, explorer_columns],
            hide_index=True,
            height=420,
            width="stretch",
        )
    else:
        st.info("Select at least one column to display.")
