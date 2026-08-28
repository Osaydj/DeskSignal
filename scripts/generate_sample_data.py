"""Generate a realistic, entirely fictional sales dataset for InsightIQ.

The generator intentionally creates patterns that an analytics dashboard can
discover, including seasonality, regional differences, customer-type behavior,
a product launch, a promotion, and a temporary supply-cost increase.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


DEFAULT_ROW_COUNT = 5_000
DEFAULT_YEAR = 2025
DEFAULT_SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "sample_data" / "sample_sales_data.csv"


@dataclass(frozen=True)
class Product:
    """Business rules for one fictional product."""

    name: str
    category: str
    base_price: float
    cost_ratio: float
    popularity: float
    seasonal_profile: str
    retail_affinity: float
    small_business_affinity: float
    enterprise_affinity: float
    launch_month: int = 1


@dataclass(frozen=True)
class Region:
    """Sales and cost behavior for one fictional region."""

    name: str
    demand_weight: float
    price_multiplier: float
    cost_multiplier: float


PRODUCTS = (
    Product(
        "Wireless Mouse",
        "Electronics",
        34.99,
        0.45,
        1.35,
        "steady",
        1.25,
        1.25,
        0.90,
    ),
    Product(
        "Mechanical Keyboard",
        "Electronics",
        94.99,
        0.53,
        0.95,
        "holiday",
        1.20,
        1.15,
        0.85,
    ),
    Product(
        "USB-C Hub",
        "Electronics",
        54.99,
        0.50,
        1.10,
        "business",
        1.00,
        1.35,
        1.30,
    ),
    Product(
        "Noise-Canceling Headphones",
        "Electronics",
        189.99,
        0.57,
        0.78,
        "holiday",
        1.35,
        0.85,
        0.70,
    ),
    Product(
        "HD Webcam",
        "Electronics",
        79.99,
        0.48,
        0.78,
        "promotion",
        1.00,
        1.20,
        1.05,
    ),
    Product(
        "Portable Monitor",
        "Electronics",
        259.99,
        0.62,
        0.62,
        "launch",
        0.90,
        1.20,
        1.35,
        launch_month=7,
    ),
    Product(
        "Laptop Stand",
        "Workspace Accessories",
        64.99,
        0.40,
        1.05,
        "business",
        1.05,
        1.30,
        1.00,
    ),
    Product(
        "Desk Lamp",
        "Workspace Accessories",
        69.99,
        0.43,
        0.72,
        "business",
        1.10,
        1.10,
        0.75,
    ),
    Product(
        "Cable Organizer",
        "Workspace Accessories",
        19.99,
        0.28,
        1.20,
        "steady",
        1.25,
        1.20,
        0.75,
    ),
    Product(
        "Travel Tech Case",
        "Workspace Accessories",
        42.99,
        0.36,
        0.78,
        "holiday",
        1.30,
        0.90,
        0.60,
    ),
    Product(
        "Ergonomic Chair",
        "Office Furniture",
        349.99,
        0.59,
        0.48,
        "budget_cycle",
        0.55,
        1.15,
        1.70,
    ),
    Product(
        "Standing Desk",
        "Office Furniture",
        579.99,
        0.61,
        0.34,
        "budget_cycle",
        0.45,
        1.10,
        1.85,
    ),
)

REGIONS = (
    Region("East", 1.12, 1.02, 1.00),
    Region("West", 1.22, 1.03, 1.05),
    Region("South", 1.03, 0.99, 0.98),
    Region("North", 0.92, 1.00, 1.04),
    Region("Central", 0.86, 0.98, 0.99),
)

CUSTOMER_TYPES = ("Retail", "Small Business", "Enterprise")
CUSTOMER_TYPE_WEIGHTS = (0.58, 0.29, 0.13)
CUSTOMER_DISCOUNTS = {
    "Retail": 0.00,
    "Small Business": 0.05,
    "Enterprise": 0.12,
}

# These factors produce a quieter first quarter, a late-summer recovery, and a
# strong November/December holiday period. A small underlying growth trend is
# added separately in date_weight().
MONTHLY_DEMAND = {
    1: 0.78,
    2: 0.82,
    3: 0.92,
    4: 0.97,
    5: 1.02,
    6: 0.95,
    7: 0.88,
    8: 0.98,
    9: 1.08,
    10: 1.16,
    11: 1.52,
    12: 1.38,
}

OUTPUT_COLUMNS = [
    "transaction_id",
    "date",
    "product",
    "category",
    "region",
    "customer_type",
    "units_sold",
    "unit_price",
    "revenue",
    "cost",
    "profit",
]


def build_calendar(year: int) -> list[date]:
    """Return every calendar date in the requested year."""

    start = date(year, 1, 1)
    end = date(year + 1, 1, 1)
    return [start + timedelta(days=offset) for offset in range((end - start).days)]


def date_weight(transaction_date: date, customer_type: str) -> float:
    """Return the relative likelihood of a transaction occurring on a date."""

    month = transaction_date.month
    weekday = transaction_date.weekday()
    is_weekend = weekday >= 5

    # The fictional business grows about 1.2% per month before seasonality.
    gradual_growth = 1 + (month - 1) * 0.012

    if customer_type == "Retail":
        weekday_factor = 1.18 if is_weekend else 1.00
    elif customer_type == "Small Business":
        weekday_factor = 0.68 if is_weekend else 1.12
    else:
        weekday_factor = 0.40 if is_weekend else 1.20

    promotion_factor = 1.0
    if month == 11 and 20 <= transaction_date.day <= 30:
        promotion_factor = 1.65
    elif month == 12 and transaction_date.day <= 15:
        promotion_factor = 1.25

    return (
        MONTHLY_DEMAND[month]
        * gradual_growth
        * weekday_factor
        * promotion_factor
    )


def product_season_multiplier(product: Product, month: int) -> float:
    """Return a product-specific seasonal multiplier."""

    if month < product.launch_month:
        return 0.0

    if product.seasonal_profile == "holiday":
        return {10: 1.20, 11: 1.85, 12: 1.60}.get(month, 0.92)
    if product.seasonal_profile == "business":
        return {1: 1.28, 3: 1.15, 8: 1.18, 9: 1.45}.get(month, 1.00)
    if product.seasonal_profile == "budget_cycle":
        return {1: 1.50, 3: 1.25, 9: 1.35, 10: 1.25}.get(month, 0.90)
    if product.seasonal_profile == "promotion":
        return 1.55 if month in {4, 5, 6} else 0.92
    if product.seasonal_profile == "launch":
        return {7: 2.10, 8: 1.85, 9: 1.55, 10: 1.25}.get(month, 1.00)
    return 1.0


def customer_product_affinity(product: Product, customer_type: str) -> float:
    """Return how strongly a customer type prefers a product."""

    if customer_type == "Retail":
        return product.retail_affinity
    if customer_type == "Small Business":
        return product.small_business_affinity
    return product.enterprise_affinity


def region_customer_multiplier(region: Region, customer_type: str) -> float:
    """Create modest regional differences in customer mix."""

    if customer_type == "Enterprise":
        return {"East": 1.30, "West": 1.22, "Central": 0.85}.get(
            region.name, 1.00
        )
    if customer_type == "Retail":
        return {"West": 1.10, "South": 1.08, "North": 0.93}.get(
            region.name, 1.00
        )
    return {"East": 1.10, "Central": 1.05}.get(region.name, 1.00)


def choose_units_sold(
    rng: random.Random, customer_type: str, category: str
) -> int:
    """Choose a realistic line-item quantity."""

    if customer_type == "Retail":
        units = rng.choices((1, 2, 3, 4), weights=(68, 23, 7, 2), k=1)[0]
    elif customer_type == "Small Business":
        units = rng.choices(
            (2, 3, 4, 5, 6, 8, 10, 12),
            weights=(14, 18, 20, 16, 13, 9, 6, 4),
            k=1,
        )[0]
    else:
        units = rng.choices(
            (5, 8, 10, 15, 20, 25, 30, 40, 50),
            weights=(8, 11, 18, 18, 16, 11, 8, 6, 4),
            k=1,
        )[0]

    # Furniture orders normally contain fewer units; inexpensive accessories
    # are often bought in larger packs.
    if category == "Office Furniture":
        units = max(1, round(units * 0.40))
    elif category == "Workspace Accessories":
        units = max(1, round(units * 1.15))

    return units


def calculate_unit_price(
    rng: random.Random,
    product: Product,
    region: Region,
    customer_type: str,
    month: int,
) -> float:
    """Calculate the final per-unit selling price after realistic variation."""

    unit_price = product.base_price * region.price_multiplier
    unit_price *= rng.uniform(0.985, 1.025)
    unit_price *= 1 - CUSTOMER_DISCOUNTS[customer_type]

    # A deliberate Q2 webcam promotion increases demand but reduces margin.
    if product.name == "HD Webcam" and month in {4, 5, 6}:
        unit_price *= 0.90

    # Holiday promotions trade some margin for additional sales volume.
    if month == 11:
        unit_price *= 0.94
    elif month == 12:
        unit_price *= 0.96

    return round(unit_price, 2)


def calculate_unit_cost(
    rng: random.Random, product: Product, region: Region, month: int
) -> float:
    """Calculate a fictional per-unit business cost."""

    unit_cost = product.base_price * product.cost_ratio
    unit_cost *= region.cost_multiplier
    unit_cost *= rng.uniform(0.985, 1.025)

    # A temporary late-summer supplier increase creates a visible margin dip.
    if (
        product.name in {"USB-C Hub", "Noise-Canceling Headphones"}
        and month in {8, 9}
    ):
        unit_cost *= 1.12

    return round(unit_cost, 2)


def generate_sales_data(
    row_count: int = DEFAULT_ROW_COUNT,
    year: int = DEFAULT_YEAR,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Generate and return a fictional sales DataFrame."""

    if row_count < 12:
        raise ValueError("row_count must be at least 12 so every month can appear.")

    rng = random.Random(seed)
    calendar = build_calendar(year)
    date_weights = {
        customer_type: [date_weight(day, customer_type) for day in calendar]
        for customer_type in CUSTOMER_TYPES
    }

    records: list[dict[str, object]] = []

    for _ in range(row_count):
        customer_type = rng.choices(
            CUSTOMER_TYPES, weights=CUSTOMER_TYPE_WEIGHTS, k=1
        )[0]
        transaction_date = rng.choices(
            calendar, weights=date_weights[customer_type], k=1
        )[0]

        region_weights = [
            region.demand_weight
            * region_customer_multiplier(region, customer_type)
            for region in REGIONS
        ]
        region = rng.choices(REGIONS, weights=region_weights, k=1)[0]

        product_weights = [
            product.popularity
            * product_season_multiplier(product, transaction_date.month)
            * customer_product_affinity(product, customer_type)
            for product in PRODUCTS
        ]
        product = rng.choices(PRODUCTS, weights=product_weights, k=1)[0]

        units_sold = choose_units_sold(rng, customer_type, product.category)
        unit_price = calculate_unit_price(
            rng, product, region, customer_type, transaction_date.month
        )
        unit_cost = calculate_unit_cost(
            rng, product, region, transaction_date.month
        )

        revenue = round(units_sold * unit_price, 2)
        total_cost = round(units_sold * unit_cost, 2)
        profit = round(revenue - total_cost, 2)

        records.append(
            {
                "date": transaction_date,
                "product": product.name,
                "category": product.category,
                "region": region.name,
                "customer_type": customer_type,
                "units_sold": units_sold,
                "unit_price": unit_price,
                "revenue": revenue,
                "cost": total_cost,
                "profit": profit,
            }
        )

    data = pd.DataFrame(records).sort_values(
        ["date", "region", "product"], ignore_index=True
    )
    data.insert(
        0,
        "transaction_id",
        [f"TXN-{year}-{number:06d}" for number in range(1, row_count + 1)],
    )
    data["date"] = pd.to_datetime(data["date"]).dt.strftime("%Y-%m-%d")

    return data[OUTPUT_COLUMNS]


def validate_dataset(data: pd.DataFrame, expected_rows: int, year: int) -> None:
    """Fail clearly if generated data does not meet the expected contract."""

    if list(data.columns) != OUTPUT_COLUMNS:
        raise ValueError("Generated columns do not match the required schema.")
    if len(data) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, generated {len(data)}.")
    if not data["transaction_id"].is_unique:
        raise ValueError("transaction_id values must be unique.")
    if data.isna().any().any():
        raise ValueError("The generated dataset contains missing values.")
    if (data[["units_sold", "unit_price", "revenue", "cost"]] <= 0).any().any():
        raise ValueError("Sales quantities and financial values must be positive.")

    parsed_dates = pd.to_datetime(data["date"], errors="raise")
    expected_months = set(range(1, 13))
    actual_months = set(parsed_dates.dt.month.unique())
    if actual_months != expected_months:
        raise ValueError("The dataset must contain transactions in all 12 months.")
    if set(parsed_dates.dt.year.unique()) != {year}:
        raise ValueError("Every transaction must be inside the requested year.")

    expected_revenue = (data["units_sold"] * data["unit_price"]).round(2)
    expected_profit = (data["revenue"] - data["cost"]).round(2)
    if not expected_revenue.equals(data["revenue"]):
        raise ValueError("Revenue does not equal units_sold multiplied by unit_price.")
    if not expected_profit.equals(data["profit"]):
        raise ValueError("Profit does not equal revenue minus cost.")


def save_dataset(data: pd.DataFrame, output_path: Path) -> None:
    """Save the DataFrame to CSV, creating its parent folder if needed."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    """Read optional command-line settings."""

    parser = argparse.ArgumentParser(
        description="Generate fictional sales data for InsightIQ."
    )
    parser.add_argument("--rows", type=int, default=DEFAULT_ROW_COUNT)
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    """Generate, validate, and save the requested dataset."""

    args = parse_args()
    data = generate_sales_data(args.rows, args.year, args.seed)
    validate_dataset(data, args.rows, args.year)
    save_dataset(data, args.output)

    print(f"Created {len(data):,} fictional transactions.")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"Saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
