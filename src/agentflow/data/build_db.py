"""Build the sample sales database. Fully synthetic, seeded, rebuilt on demand."""

from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

SCHEMA = """
CREATE TABLE regions   (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL,
                        region_id INTEGER NOT NULL REFERENCES regions(id), signup_date TEXT NOT NULL);
CREATE TABLE products  (id INTEGER PRIMARY KEY, name TEXT NOT NULL, unit_price REAL NOT NULL);
CREATE TABLE orders    (id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customers(id),
                        order_date TEXT NOT NULL, status TEXT NOT NULL);
CREATE TABLE order_items (order_id INTEGER NOT NULL REFERENCES orders(id),
                          product_id INTEGER NOT NULL REFERENCES products(id),
                          quantity INTEGER NOT NULL);
"""

# Shown to the model. Kept short on purpose: every token costs CPU time.
SCHEMA_HINT = (
    "Tables: regions(id, name); customers(id, name, email, region_id, signup_date); "
    "products(id, name, unit_price); orders(id, customer_id, order_date, status); "
    "order_items(order_id, product_id, quantity). "
    "Dates are text 'YYYY-MM-DD'. Order status is one of: shipped, delivered, returned, cancelled. "
    "Regions: East, West, South, North. Products: Atlas, Beacon, Cadence, Delta. "
    "Order revenue = SUM(order_items.quantity * products.unit_price)."
)

REGIONS = ["East", "West", "South", "North"]
PRODUCTS = [("Atlas", 59.0), ("Beacon", 129.0), ("Cadence", 89.0), ("Delta", 249.0)]
STATUSES = ["shipped", "delivered", "returned", "cancelled"]
_FIRST = ["Ava", "Noah", "Mia", "Liam", "Zoe", "Ethan", "Ivy", "Lucas", "Nora", "Omar", "Priya",
          "Ravi", "Sara", "Tom", "Uma", "Victor", "Wen", "Yara", "Zane", "Elena"]
_LAST = ["Shah", "Meyer", "Okafor", "Silva", "Tanaka", "Haddad", "Novak", "Reyes", "Kaur", "Lund"]


def build_db(path: str | Path, seed: int = 7) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    rng = random.Random(seed)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO regions VALUES (?, ?)", list(enumerate(REGIONS, 1)))
    con.executemany(
        "INSERT INTO products VALUES (?, ?, ?)", [(i, n, p) for i, (n, p) in enumerate(PRODUCTS, 1)]
    )
    start = date(2025, 11, 1)
    customers = []
    for i in range(1, 81):
        name = f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"
        region = rng.choices([1, 2, 3, 4], weights=[3, 4, 2, 2])[0]
        signup = start + timedelta(days=rng.randrange(0, 330))
        customers.append((i, name, f"user{i}@example.com", region, signup.isoformat()))
    con.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?)", customers)
    for order_id in range(1, 641):
        cust = rng.choice(customers)
        day = date(2026, 1, 1) + timedelta(days=rng.randrange(0, 273))
        status = rng.choices(STATUSES, weights=[25, 60, 9, 6])[0]
        con.execute("INSERT INTO orders VALUES (?, ?, ?, ?)", (order_id, cust[0], day.isoformat(), status))
        for pid in rng.sample([1, 2, 3, 4], k=rng.choice([1, 1, 2])):
            con.execute(
                "INSERT INTO order_items VALUES (?, ?, ?)", (order_id, pid, rng.choice([1, 1, 2, 3]))
            )
    con.commit()
    con.close()
    return path
