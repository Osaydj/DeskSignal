"""The required DeskSignal sales-data schema."""

# A portfolio dashboard should reject unexpectedly large datasets before they
# consume excessive memory. The bundled sample contains only 5,000 rows.
MAX_SALES_ROWS = 100_000

# Uploaded currency fields are expected to agree to the nearest cent. The tiny
# extra allowance prevents normal floating-point rounding from rejecting $0.01.
FINANCIAL_TOLERANCE = 0.010_001

REQUIRED_COLUMNS = frozenset(
    {
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
    }
)

TEXT_COLUMNS = (
    "transaction_id",
    "product",
    "category",
    "region",
    "customer_type",
)

NUMERIC_COLUMNS = (
    "units_sold",
    "unit_price",
    "revenue",
    "cost",
    "profit",
)
