import pytest
from pyspark.sql import SparkSession
from pyspark_job import clean_data


@pytest.fixture(scope="session")
def spark():
    session = (
        SparkSession.builder
        .master("local[1]")
        .appName("tests")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


def make_df(spark, rows):
    """Build a DataFrame in the JVM via SQL (no Python worker needed)."""
    values = ", ".join(
        f"({'NULL' if name is None else repr(name)}, {amount})"
        for name, amount in rows
    )
    return spark.sql(
        "SELECT CAST(col1 AS STRING) AS name, CAST(col2 AS DOUBLE) AS amount "
        f"FROM VALUES {values}"
    )


def test_valid_records_kept(spark):
    df = make_df(spark, [("Ali", 100.0), ("Sara", 50.0)])
    assert clean_data(df).count() == 2


def test_amount_less_or_equal_zero_removed(spark):
    df = make_df(spark, [("A", 0.0), ("B", -5.0), ("C", 10.0)])
    result = clean_data(df).collect()
    assert [r["name"] for r in result] == ["C"]


def test_null_names_removed(spark):
    df = make_df(spark, [(None, 10.0), ("Ali", 20.0)])
    result = clean_data(df).collect()
    assert [r["name"] for r in result] == ["Ali"]


def test_amount_with_tax_calculated(spark):
    df = make_df(spark, [("Ali", 100.0)])
    row = clean_data(df).collect()[0]
    assert row["amount_with_tax"] == pytest.approx(120.0)