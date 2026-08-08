# ==============================================================================
# CELL 1: Import Dependencies
# ==============================================================================
# Install dependencies from requirements.txt before running this application.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
import gradio as gr

print("Libraries imported successfully.")

# ==============================================================================
# CELL 2: Load & Validate Dataset
# ==============================================================================
# Upload Airbnb_db.csv to the Colab workspace before running this cell
# (Files panel on the left -> Upload, or use files.upload()).

CSV_FILENAME = "Airbnb_db.csv"

REQUIRED_COLUMNS = [
    "id",
    "neighbourhood_group_c",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "availability_365",
    "price",
]

# Load the CSV
df_raw = pd.read_csv(CSV_FILENAME)
print(f"Loaded {CSV_FILENAME}: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

# Confirm required columns exist
missing_cols = [c for c in REQUIRED_COLUMNS if c not in df_raw.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")
else:
    print("All required columns are present.")

# Check missing values
print("\nMissing values per column:")
print(df_raw[REQUIRED_COLUMNS].isnull().sum())

# Check duplicates
n_dupes = df_raw.duplicated().sum()
print(f"\nDuplicate rows: {n_dupes}")

# Check data types
print("\nData types:")
print(df_raw[REQUIRED_COLUMNS].dtypes)

# ==============================================================================
# CELL 3: Data Preparation & Cleaning
# ==============================================================================

df = df_raw.copy()

# Remove rows with missing values only when necessary (only in required columns)
rows_before = df.shape[0]
df = df.dropna(subset=REQUIRED_COLUMNS)
rows_after = df.shape[0]
if rows_before != rows_after:
    print(f"Removed {rows_before - rows_after} rows with missing values.")
else:
    print("No missing values found - no rows removed.")

# Remove exact duplicate rows if any
df = df.drop_duplicates()

# Validate borough codes are within 1-5; drop anything outside that range
valid_mask = df["neighbourhood_group_c"].isin([1, 2, 3, 4, 5])
invalid_count = (~valid_mask).sum()
if invalid_count > 0:
    print(f"Warning: removing {invalid_count} rows with invalid borough codes.")
df = df[valid_mask].copy()

# Map numeric borough codes to readable names (for charts and UI only)
BOROUGH_MAP = {
    1: "Manhattan",
    2: "Brooklyn",
    3: "Queens",
    4: "Staten Island",
    5: "Bronx",
}
BOROUGH_ORDER = ["Manhattan", "Brooklyn", "Queens", "Staten Island", "Bronx"]
REFERENCE_BOROUGH = "Manhattan"

df["borough_name"] = pd.Categorical(
    df["neighbourhood_group_c"].map(BOROUGH_MAP),
    categories=BOROUGH_ORDER,
)

# The numeric borough code is retained only for validation/mapping.
# The regression model below treats borough as a categorical variable.
# Add a high-value flag used throughout the dashboard
PRICE_THRESHOLD = 120.0
df["above_120"] = df["price"] > PRICE_THRESHOLD

print(f"\nFinal cleaned dataset: {df.shape[0]} rows")
print(df[["neighbourhood_group_c", "borough_name"]].drop_duplicates().sort_values("neighbourhood_group_c"))

# ==============================================================================
# CELL 4: Multiple Linear Regression Model
# ==============================================================================

CONTINUOUS_FEATURES = [
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "availability_365",
]
TARGET = "price"

# One-hot encode borough with Manhattan as the reference category. This avoids
# treating borough codes 1-5 as if they formed a continuous numeric scale.
borough_dummies = pd.get_dummies(
    df["borough_name"],
    prefix="borough",
    drop_first=True,
    dtype=float,
)

X = pd.concat(
    [
        df[CONTINUOUS_FEATURES].astype(float),
        borough_dummies,
    ],
    axis=1,
)
FEATURES = X.columns.tolist()
y = df[TARGET].astype(float)

# 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


def adjusted_r2(r2, n, p):
    """Adjusted R-squared = 1 - ((1 - R2) * (n - 1) / (n - p - 1))"""
    denominator = n - p - 1
    if denominator <= 0:
        return np.nan
    return 1 - ((1 - r2) * (n - 1) / denominator)


n_train, p_train = X_train.shape[0], X_train.shape[1]
n_test, p_test = X_test.shape[0], X_test.shape[1]

# Training metrics
train_r2 = r2_score(y_train, y_train_pred)
train_adj_r2 = adjusted_r2(train_r2, n_train, p_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)

# Test metrics
test_r2 = r2_score(y_test, y_test_pred)
test_adj_r2 = adjusted_r2(test_r2, n_test, p_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)

print("=== Training Metrics ===")
print(f"R-squared:          {train_r2:.4f}")
print(f"Adjusted R-squared: {train_adj_r2:.4f}")
print(f"RMSE:               {train_rmse:.4f}")
print(f"MAE:                {train_mae:.4f}")

print("\n=== Test Metrics ===")
print(f"R-squared:          {test_r2:.4f}")
print(f"Adjusted R-squared: {test_adj_r2:.4f}")
print(f"RMSE:               {test_rmse:.4f}")
print(f"MAE:                {test_mae:.4f}")

# Overfitting assessment based on the absolute gap between train and test Adjusted R2
adj_r2_gap = abs(train_adj_r2 - test_adj_r2)
if adj_r2_gap > 0.10:
    RELIABILITY_MSG = (
        f"The gap between training Adjusted R-squared ({train_adj_r2:.3f}) and test "
        f"Adjusted R-squared ({test_adj_r2:.3f}) is {adj_r2_gap:.3f}, which suggests the "
        "model may be overfitting to the training data and could perform less reliably "
        "on new listings."
    )
elif adj_r2_gap > 0.05:
    RELIABILITY_MSG = (
        f"The gap between training Adjusted R-squared ({train_adj_r2:.3f}) and test "
        f"Adjusted R-squared ({test_adj_r2:.3f}) is {adj_r2_gap:.3f}, a moderate difference. "
        "The model appears reasonably stable but should be monitored as new data arrives."
    )
else:
    RELIABILITY_MSG = (
        f"Training Adjusted R-squared ({train_adj_r2:.3f}) and test Adjusted R-squared "
        f"({test_adj_r2:.3f}) are close (gap of {adj_r2_gap:.3f}), suggesting the model "
        "generalizes consistently between the training and test data and is not strongly overfit."
    )

print("\n" + RELIABILITY_MSG)

# Store coefficients for interpretation and reuse in the app
COEF_MAP = dict(zip(FEATURES, model.coef_))
INTERCEPT = model.intercept_

# ==============================================================================
# CELL 5: Coefficient Interpretation (business-friendly text)
# ==============================================================================

def build_coefficient_interpretation():
    """Build a business-friendly explanation of the regression coefficients."""
    lines = []
    lines.append(f"**Reference borough:** {REFERENCE_BOROUGH}")
    lines.append(
        f"**Intercept:** ${INTERCEPT:.2f}. This is the model's baseline prediction for "
        f"{REFERENCE_BOROUGH} when the numeric predictors are zero; it is a mathematical "
        "reference point and may not describe a realistic listing."
    )
    lines.append("")
    lines.append("**Coefficients:**")
    for feat in FEATURES:
        lines.append(f"- {feat}: {COEF_MAP[feat]:.4f}")

    lines.append("")

    continuous_labels = {
        "reviews_per_month": ("Reviews per month", "one additional review per month"),
        "availability_365": ("Availability", "one additional available day per year"),
        "minimum_nights": ("Minimum nights", "one additional required minimum night"),
        "number_of_reviews": ("Number of reviews", "one additional total review"),
    }

    for feature, (label, unit_text) in continuous_labels.items():
        coef = COEF_MAP[feature]
        direction = "an increase" if coef >= 0 else "a decrease"
        lines.append(
            f"**{label}:** Holding other variables constant, {unit_text} is associated "
            f"with {direction} of about ${abs(coef):.2f} in nightly price."
        )

    lines.append("")
    lines.append(f"**Borough effects relative to {REFERENCE_BOROUGH}:**")
    for borough_name in BOROUGH_ORDER[1:]:
        feature_name = f"borough_{borough_name}"
        if feature_name in COEF_MAP:
            coef = COEF_MAP[feature_name]
            direction = "higher" if coef >= 0 else "lower"
            lines.append(
                f"- **{borough_name}:** about ${abs(coef):.2f} {direction} expected nightly "
                f"price than {REFERENCE_BOROUGH}, holding other variables constant."
            )

    largest_feat = max(COEF_MAP, key=lambda k: abs(COEF_MAP[k]))
    lines.append("")
    lines.append(
        f"**Largest raw coefficient:** {largest_feat} has the largest absolute raw "
        f"coefficient ({COEF_MAP[largest_feat]:.4f}). Raw coefficients should still be "
        "compared cautiously because predictors use different units."
    )

    lines.append("")
    lines.append(
        "**Important:** Borough is now modeled as a categorical variable using dummy "
        f"variables, with {REFERENCE_BOROUGH} as the reference category. These results "
        "describe associations in historical data and do not establish causal relationships."
    )

    return "\n\n".join(lines)


COEFFICIENT_TEXT = build_coefficient_interpretation()
print(COEFFICIENT_TEXT)

# ==============================================================================
# CELL 6: Helper Functions for KPIs, Filtering, and Charts
# ==============================================================================

def compute_kpis(data: pd.DataFrame):
    """Returns a markdown string of KPI values for the given (filtered) dataframe."""
    if data.empty:
        return "No listings match the current filters."

    n_listings = len(data)
    avg_price = data["price"].mean()
    median_price = data["price"].median()
    pct_above_120 = (data["price"] > PRICE_THRESHOLD).mean() * 100
    max_price = data["price"].max()

    kpi_md = (
        f"### Key Metrics\n\n"
        f"| Metric | Value |\n"
        f"|---|---|\n"
        f"| Number of Listings | {n_listings:,} |\n"
        f"| Average Nightly Price | ${avg_price:,.2f} |\n"
        f"| Median Nightly Price | ${median_price:,.2f} |\n"
        f"| % Priced Above $120 | {pct_above_120:.1f}% |\n"
        f"| Highest Nightly Price | ${max_price:,.2f} |\n"
    )
    return kpi_md


def apply_filters(
    borough_choice,
    min_nights_range,
    num_reviews_range,
    reviews_per_month_range,
    availability_range,
    price_range,
    high_value_choice,
):
    """Filters the master dataframe based on all Gradio inputs."""
    data = df.copy()

    if borough_choice and borough_choice != "All":
        data = data[data["borough_name"] == borough_choice]

    data = data[
        (data["minimum_nights"] >= min_nights_range[0])
        & (data["minimum_nights"] <= min_nights_range[1])
    ]
    data = data[
        (data["number_of_reviews"] >= num_reviews_range[0])
        & (data["number_of_reviews"] <= num_reviews_range[1])
    ]
    data = data[
        (data["reviews_per_month"] >= reviews_per_month_range[0])
        & (data["reviews_per_month"] <= reviews_per_month_range[1])
    ]
    data = data[
        (data["availability_365"] >= availability_range[0])
        & (data["availability_365"] <= availability_range[1])
    ]
    data = data[
        (data["price"] >= price_range[0]) & (data["price"] <= price_range[1])
    ]

    if high_value_choice == "Above $120":
        data = data[data["price"] > PRICE_THRESHOLD]
    elif high_value_choice == "$120 or below":
        data = data[data["price"] <= PRICE_THRESHOLD]

    return data


def empty_figure(message="No data available for the current filters"):
    """Returns a blank Plotly figure with a centered message, used when filters return no rows."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16),
    )
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        height=400,
    )
    return fig


def make_charts(data: pd.DataFrame):
    """Builds all six Plotly charts from the (filtered) dataframe."""
    if data.empty:
        blank = empty_figure()
        return blank, blank, blank, blank, blank, blank

    borough_order = ["Manhattan", "Brooklyn", "Queens", "Staten Island", "Bronx"]
    present_order = [b for b in borough_order if b in data["borough_name"].unique()]

    # 1. Nightly price by borough (box plot)
    fig_borough = px.box(
        data,
        x="borough_name",
        y="price",
        category_orders={"borough_name": present_order},
        title="Nightly Price by Borough",
        labels={"borough_name": "Borough", "price": "Nightly Price ($)"},
        color="borough_name",
    )
    fig_borough.add_hline(
        y=PRICE_THRESHOLD, line_dash="dash", line_color="red",
        annotation_text="$120 reference", annotation_position="top left",
    )
    fig_borough.update_layout(showlegend=False)

    def add_manual_trendline(fig, x_vals, y_vals):
        """Adds a linear trendline computed with numpy (avoids the statsmodels
        dependency required by Plotly's built-in trendline='ols')."""
        x_arr = np.asarray(x_vals, dtype=float)
        y_arr = np.asarray(y_vals, dtype=float)
        if len(x_arr) >= 2 and np.unique(x_arr).size >= 2:
            slope, intercept = np.polyfit(x_arr, y_arr, 1)
            x_line = np.linspace(x_arr.min(), x_arr.max(), 100)
            y_line = slope * x_line + intercept
            fig.add_trace(
                go.Scatter(
                    x=x_line, y=y_line, mode="lines",
                    name="Trend", line=dict(color="black", width=2),
                )
            )
        return fig

    # 2. Reviews per month vs nightly price
    fig_reviews_month = px.scatter(
        data,
        x="reviews_per_month",
        y="price",
        title="Reviews per Month vs Nightly Price",
        labels={"reviews_per_month": "Reviews per Month", "price": "Nightly Price ($)"},
        opacity=0.5,
    )
    fig_reviews_month = add_manual_trendline(fig_reviews_month, data["reviews_per_month"], data["price"])
    fig_reviews_month.add_hline(
        y=PRICE_THRESHOLD, line_dash="dash", line_color="red",
        annotation_text="$120 reference", annotation_position="top left",
    )

    # 3. Number of reviews vs nightly price
    fig_reviews_total = px.scatter(
        data,
        x="number_of_reviews",
        y="price",
        title="Number of Reviews vs Nightly Price",
        labels={"number_of_reviews": "Number of Reviews", "price": "Nightly Price ($)"},
        opacity=0.5,
    )
    fig_reviews_total = add_manual_trendline(fig_reviews_total, data["number_of_reviews"], data["price"])
    fig_reviews_total.add_hline(
        y=PRICE_THRESHOLD, line_dash="dash", line_color="red",
        annotation_text="$120 reference", annotation_position="top left",
    )

    # 4. Minimum nights vs nightly price
    fig_min_nights = px.scatter(
        data,
        x="minimum_nights",
        y="price",
        title="Minimum Nights vs Nightly Price",
        labels={"minimum_nights": "Minimum Nights", "price": "Nightly Price ($)"},
        opacity=0.5,
    )
    fig_min_nights = add_manual_trendline(fig_min_nights, data["minimum_nights"], data["price"])
    fig_min_nights.add_hline(
        y=PRICE_THRESHOLD, line_dash="dash", line_color="red",
        annotation_text="$120 reference", annotation_position="top left",
    )

    # 5. Availability vs nightly price
    fig_availability = px.scatter(
        data,
        x="availability_365",
        y="price",
        title="Availability (365) vs Nightly Price",
        labels={"availability_365": "Availability (days/year)", "price": "Nightly Price ($)"},
        opacity=0.5,
    )
    fig_availability = add_manual_trendline(fig_availability, data["availability_365"], data["price"])
    fig_availability.add_hline(
        y=PRICE_THRESHOLD, line_dash="dash", line_color="red",
        annotation_text="$120 reference", annotation_position="top left",
    )

    # 6. Percentage of listings above $120 by borough
    pct_by_borough = (
        data.groupby("borough_name")["above_120"]
        .mean()
        .mul(100)
        .reindex(present_order)
        .reset_index()
    )
    pct_by_borough.columns = ["borough_name", "pct_above_120"]
    fig_pct_120 = px.bar(
        pct_by_borough,
        x="borough_name",
        y="pct_above_120",
        title="Percentage of Listings Above $120 by Borough",
        labels={"borough_name": "Borough", "pct_above_120": "% Above $120"},
        color="borough_name",
        text=pct_by_borough["pct_above_120"].round(1).astype(str) + "%",
    )
    fig_pct_120.update_layout(showlegend=False)

    return (
        fig_borough,
        fig_reviews_month,
        fig_reviews_total,
        fig_min_nights,
        fig_availability,
        fig_pct_120,
    )


# ==============================================================================
# CELL 7: Dataset Bounds (used for filter sliders and simulator inputs)
# ==============================================================================

BOROUGH_OPTIONS = ["All"] + BOROUGH_ORDER

MIN_NIGHTS_BOUNDS = (int(df["minimum_nights"].min()), int(df["minimum_nights"].max()))
NUM_REVIEWS_BOUNDS = (int(df["number_of_reviews"].min()), int(df["number_of_reviews"].max()))
REVIEWS_PM_BOUNDS = (float(df["reviews_per_month"].min()), float(df["reviews_per_month"].max()))
AVAILABILITY_BOUNDS = (int(df["availability_365"].min()), int(df["availability_365"].max()))
PRICE_BOUNDS = (float(df["price"].min()), float(df["price"].max()))

# ==============================================================================
# CELL 8: Price Simulator Logic
# ==============================================================================

def predict_price(borough_name, minimum_nights, number_of_reviews, reviews_per_month, availability_365):
    """Predict nightly price for a hypothetical listing and return a formatted result."""
    try:
        minimum_nights = float(minimum_nights)
        number_of_reviews = float(number_of_reviews)
        reviews_per_month = float(reviews_per_month)
        availability_365 = float(availability_365)
    except (TypeError, ValueError):
        return "**Error:** Please enter valid numeric values for all fields."

    if borough_name not in BOROUGH_ORDER:
        return "**Error:** Please select a valid borough."

    input_base = pd.DataFrame(
        [{
            "borough_name": borough_name,
            "minimum_nights": minimum_nights,
            "number_of_reviews": number_of_reviews,
            "reviews_per_month": reviews_per_month,
            "availability_365": availability_365,
        }]
    )
    input_base["borough_name"] = pd.Categorical(
        input_base["borough_name"],
        categories=BOROUGH_ORDER,
    )
    input_borough_dummies = pd.get_dummies(
        input_base["borough_name"],
        prefix="borough",
        drop_first=True,
        dtype=float,
    )
    input_row = pd.concat(
        [
            input_base[CONTINUOUS_FEATURES].astype(float),
            input_borough_dummies,
        ],
        axis=1,
    ).reindex(columns=FEATURES, fill_value=0.0)

    predicted_price = model.predict(input_row)[0]
    predicted_price = max(predicted_price, 0)

    diff_from_benchmark = predicted_price - PRICE_THRESHOLD

    if predicted_price > PRICE_THRESHOLD:
        status_label = "Potentially High-Value Opportunity"
        diff_text = f"${diff_from_benchmark:,.2f} above the $120 benchmark"
    else:
        status_label = "Predicted Price Does Not Exceed $120"
        diff_text = f"${abs(diff_from_benchmark):,.2f} below the $120 benchmark"

    result_md = (
        f"### Estimated Nightly Price: ${predicted_price:,.2f}\n\n"
        f"**Status:** {status_label}\n\n"
        f"**Difference from $120 benchmark:** {diff_text}\n\n"
        f"**Interpretation:** Based on the selected characteristics (borough: {borough_name}, "
        f"minimum nights: {int(minimum_nights)}, number of reviews: {int(number_of_reviews)}, "
        f"reviews per month: {reviews_per_month}, availability: {int(availability_365)} days/year), "
        f"the model estimates a nightly price of ${predicted_price:,.2f}. Borough is treated "
        f"categorically with {REFERENCE_BOROUGH} as the reference category. This is an estimate, "
        "not a guarantee, and reflects patterns in historical listing data.\n\n"
        "*This estimate is based on historical listing patterns and should support, "
        "not replace, business judgment.*"
    )
    return result_md


# ==============================================================================
# CELL 9: Build the Gradio Blocks Dashboard
# ==============================================================================

with gr.Blocks(title="Airbnb Rental Arbitrage Dashboard - NYC") as demo:

    gr.Markdown("# Airbnb Rental Arbitrage Dashboard - New York City")
    gr.Markdown(
        "Explore which listing characteristics are associated with higher nightly "
        "prices, and estimate the nightly price of a potential listing before "
        "signing a lease."
    )

    # ------------------- KPI SECTION -------------------
    kpi_display = gr.Markdown(compute_kpis(df))

    gr.Markdown("---")
    gr.Markdown("## Filters")

    with gr.Row():
        borough_filter = gr.Dropdown(
            choices=BOROUGH_OPTIONS, value="All", label="Borough"
        )
        high_value_filter = gr.Radio(
            choices=["All", "Above $120", "$120 or below"],
            value="All",
            label="High-Value Status",
        )

    with gr.Row():
        min_nights_filter = gr.Slider(
            minimum=MIN_NIGHTS_BOUNDS[0], maximum=MIN_NIGHTS_BOUNDS[1],
            value=MIN_NIGHTS_BOUNDS[0], step=1, label="Minimum Nights (lower bound)"
        )
        min_nights_filter_high = gr.Slider(
            minimum=MIN_NIGHTS_BOUNDS[0], maximum=MIN_NIGHTS_BOUNDS[1],
            value=MIN_NIGHTS_BOUNDS[1], step=1, label="Minimum Nights (upper bound)"
        )

    with gr.Row():
        num_reviews_filter_low = gr.Slider(
            minimum=NUM_REVIEWS_BOUNDS[0], maximum=NUM_REVIEWS_BOUNDS[1],
            value=NUM_REVIEWS_BOUNDS[0], step=1, label="Number of Reviews (lower bound)"
        )
        num_reviews_filter_high = gr.Slider(
            minimum=NUM_REVIEWS_BOUNDS[0], maximum=NUM_REVIEWS_BOUNDS[1],
            value=NUM_REVIEWS_BOUNDS[1], step=1, label="Number of Reviews (upper bound)"
        )

    with gr.Row():
        reviews_pm_filter_low = gr.Slider(
            minimum=REVIEWS_PM_BOUNDS[0], maximum=REVIEWS_PM_BOUNDS[1],
            value=REVIEWS_PM_BOUNDS[0], step=0.01, label="Reviews per Month (lower bound)"
        )
        reviews_pm_filter_high = gr.Slider(
            minimum=REVIEWS_PM_BOUNDS[0], maximum=REVIEWS_PM_BOUNDS[1],
            value=REVIEWS_PM_BOUNDS[1], step=0.01, label="Reviews per Month (upper bound)"
        )

    with gr.Row():
        availability_filter_low = gr.Slider(
            minimum=AVAILABILITY_BOUNDS[0], maximum=AVAILABILITY_BOUNDS[1],
            value=AVAILABILITY_BOUNDS[0], step=1, label="Availability 365 (lower bound)"
        )
        availability_filter_high = gr.Slider(
            minimum=AVAILABILITY_BOUNDS[0], maximum=AVAILABILITY_BOUNDS[1],
            value=AVAILABILITY_BOUNDS[1], step=1, label="Availability 365 (upper bound)"
        )

    with gr.Row():
        price_filter_low = gr.Slider(
            minimum=PRICE_BOUNDS[0], maximum=PRICE_BOUNDS[1],
            value=PRICE_BOUNDS[0], step=1, label="Price (lower bound)"
        )
        price_filter_high = gr.Slider(
            minimum=PRICE_BOUNDS[0], maximum=PRICE_BOUNDS[1],
            value=PRICE_BOUNDS[1], step=1, label="Price (upper bound)"
        )

    with gr.Row():
        apply_btn = gr.Button("Apply Filters", variant="primary")
        reset_btn = gr.Button("Reset Filters")

    filter_message = gr.Markdown("")

    gr.Markdown("---")
    gr.Markdown("## Visualizations")

    with gr.Row():
        chart_borough = gr.Plot(label="Nightly Price by Borough")
        chart_pct_120 = gr.Plot(label="% Above $120 by Borough")

    with gr.Row():
        chart_reviews_month = gr.Plot(label="Reviews per Month vs Price")
        chart_reviews_total = gr.Plot(label="Number of Reviews vs Price")

    with gr.Row():
        chart_min_nights = gr.Plot(label="Minimum Nights vs Price")
        chart_availability = gr.Plot(label="Availability vs Price")

    gr.Markdown("---")
    gr.Markdown("## Model Performance")

    model_performance_md = gr.Markdown(
        f"### Training Data\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| R-squared | {train_r2:.4f} |\n"
        f"| Adjusted R-squared | {train_adj_r2:.4f} |\n"
        f"| RMSE | {train_rmse:.4f} |\n"
        f"| MAE | {train_mae:.4f} |\n\n"
        f"### Test Data\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| R-squared | {test_r2:.4f} |\n"
        f"| Adjusted R-squared | {test_adj_r2:.4f} |\n"
        f"| RMSE | {test_rmse:.4f} |\n"
        f"| MAE | {test_mae:.4f} |\n\n"
        f"### Reliability & Overfitting Assessment\n\n{RELIABILITY_MSG}"
    )

    gr.Markdown("---")
    gr.Markdown("## Coefficient Interpretation")
    gr.Markdown(COEFFICIENT_TEXT)

    gr.Markdown("---")
    gr.Markdown("## Nightly Price Simulator")

    with gr.Row():
        sim_borough = gr.Dropdown(
            choices=list(BOROUGH_MAP.values()), value="Manhattan", label="Borough"
        )
        sim_min_nights = gr.Slider(
            minimum=MIN_NIGHTS_BOUNDS[0], maximum=MIN_NIGHTS_BOUNDS[1],
            value=int(df["minimum_nights"].median()), step=1, label="Minimum Nights"
        )
        sim_num_reviews = gr.Slider(
            minimum=NUM_REVIEWS_BOUNDS[0], maximum=NUM_REVIEWS_BOUNDS[1],
            value=int(df["number_of_reviews"].median()), step=1, label="Number of Reviews"
        )

    with gr.Row():
        sim_reviews_pm = gr.Slider(
            minimum=REVIEWS_PM_BOUNDS[0], maximum=REVIEWS_PM_BOUNDS[1],
            value=round(float(df["reviews_per_month"].median()), 2), step=0.01,
            label="Reviews per Month"
        )
        sim_availability = gr.Slider(
            minimum=AVAILABILITY_BOUNDS[0], maximum=AVAILABILITY_BOUNDS[1],
            value=int(df["availability_365"].median()), step=1, label="Availability (days/year)"
        )

    with gr.Row():
        simulate_btn = gr.Button("Estimate Nightly Price", variant="primary")
        reset_sim_btn = gr.Button("Reset Simulator")

    simulator_output = gr.Markdown("")

    gr.Markdown(
        "*This estimate is based on historical listing patterns and should support, "
        "not replace, business judgment.*"
    )

    # ------------------- CALLBACK: Apply Filters -------------------
    def on_apply_filters(
        borough_choice, mn_low, mn_high, nr_low, nr_high,
        rpm_low, rpm_high, av_low, av_high, pr_low, pr_high, hv_choice
    ):
        filtered = apply_filters(
            borough_choice,
            (mn_low, mn_high),
            (nr_low, nr_high),
            (rpm_low, rpm_high),
            (av_low, av_high),
            (pr_low, pr_high),
            hv_choice,
        )

        if filtered.empty:
            msg = "**No listings match the current filters. Please adjust your filter selections.**"
            blank_charts = make_charts(filtered)
            return (compute_kpis(filtered), msg, *blank_charts)

        charts = make_charts(filtered)
        return (compute_kpis(filtered), "", *charts)

    filter_inputs = [
        borough_filter,
        min_nights_filter, min_nights_filter_high,
        num_reviews_filter_low, num_reviews_filter_high,
        reviews_pm_filter_low, reviews_pm_filter_high,
        availability_filter_low, availability_filter_high,
        price_filter_low, price_filter_high,
        high_value_filter,
    ]

    filter_outputs = [
        kpi_display,
        filter_message,
        chart_borough,
        chart_reviews_month,
        chart_reviews_total,
        chart_min_nights,
        chart_availability,
        chart_pct_120,
    ]

    apply_btn.click(fn=on_apply_filters, inputs=filter_inputs, outputs=filter_outputs)

    # ------------------- CALLBACK: Reset Filters -------------------
    def on_reset_filters():
        default_charts = make_charts(df)
        return (
            "All",  # borough
            MIN_NIGHTS_BOUNDS[0], MIN_NIGHTS_BOUNDS[1],
            NUM_REVIEWS_BOUNDS[0], NUM_REVIEWS_BOUNDS[1],
            REVIEWS_PM_BOUNDS[0], REVIEWS_PM_BOUNDS[1],
            AVAILABILITY_BOUNDS[0], AVAILABILITY_BOUNDS[1],
            PRICE_BOUNDS[0], PRICE_BOUNDS[1],
            "All",  # high-value status
            compute_kpis(df),
            "",
            *default_charts,
        )

    reset_btn.click(
        fn=on_reset_filters,
        inputs=[],
        outputs=[
            borough_filter,
            min_nights_filter, min_nights_filter_high,
            num_reviews_filter_low, num_reviews_filter_high,
            reviews_pm_filter_low, reviews_pm_filter_high,
            availability_filter_low, availability_filter_high,
            price_filter_low, price_filter_high,
            high_value_filter,
            kpi_display,
            filter_message,
            chart_borough,
            chart_reviews_month,
            chart_reviews_total,
            chart_min_nights,
            chart_availability,
            chart_pct_120,
        ],
    )

    # ------------------- CALLBACK: Price Simulator -------------------
    simulate_btn.click(
        fn=predict_price,
        inputs=[sim_borough, sim_min_nights, sim_num_reviews, sim_reviews_pm, sim_availability],
        outputs=simulator_output,
    )

    def on_reset_simulator():
        return (
            "Manhattan",
            int(df["minimum_nights"].median()),
            int(df["number_of_reviews"].median()),
            round(float(df["reviews_per_month"].median()), 2),
            int(df["availability_365"].median()),
            "",
        )

    reset_sim_btn.click(
        fn=on_reset_simulator,
        inputs=[],
        outputs=[sim_borough, sim_min_nights, sim_num_reviews, sim_reviews_pm, sim_availability, simulator_output],
    )

    # ------------------- LOAD DEFAULT CHARTS ON STARTUP -------------------
    def on_load():
        return make_charts(df)

    demo.load(
        fn=on_load,
        inputs=[],
        outputs=[
            chart_borough,
            chart_reviews_month,
            chart_reviews_total,
            chart_min_nights,
            chart_availability,
            chart_pct_120,
        ],
    )

# ==============================================================================
# CELL 10: Launch the App
# ==============================================================================

demo.launch(share=True, debug=False)
