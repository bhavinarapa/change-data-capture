import argparse
from typing import List, Optional

from pyspark.sql import DataFrame, SparkSession, functions as F
from delta.tables import DeltaTable


class CDCUtility:
    """
    A utility class for performing Change Data Capture (CDC) operations
    using Apache Spark and Delta Lake.

    This class:
    - Identifies new, updated, or unchanged records based on selected fields.
    - Adds CDC metadata fields (e.g., `is_active`, `validity_date`, `cdc_key`).
    - Generates a hash-based key (`cdc_key`) for tracking changes.
    - Performs Delta merge to mark old rows inactive and insert new/updated ones.

    Note:
        The target Delta table must already contain all columns present in the source DataFrame
        as well as the CDC tracking fields:
        - `cdc_key`
        - `is_active`
        - `active_date`
        - `validity_date`
        - `cdc_upd_dttm`

    Example:
        cdc_util = CDCUtility(spark)
        cdc_util.apply_scd_type2(
            target_table_name="my_delta_table",
            source_df=new_data_df,
            change_fields=["id", "name", "email"],
            change_flag="include"
        )
    """
    def __init__(self, spark: SparkSession):
        """
        Initialize the CDCUtility with a Spark session.

        Args:
            spark (SparkSession): The Spark session to use for all operations.
        """
        self.spark = spark
        self._tgt = "target"
        self._src = "source"

    def add_cdc_fields(self, df: DataFrame) -> DataFrame:
        """
        Adds standard CDC tracking fields to a DataFrame.

        Returns:
            DataFrame: The enriched DataFrame with CDC metadata fields added.
        """
        return (
            df.withColumn("cdc_key", F.lit(None))
              .withColumn("is_active", F.lit(True))
              .withColumn("active_date", F.current_date())
              .withColumn("validity_date", F.to_date(F.lit("2099-12-31")))
              .withColumn("cdc_upd_dttm", F.current_timestamp())
        )

    def get_change_tracking_cols(
        self,
        source_df: DataFrame,
        change_fields: Optional[List[str]] = None,
        change_flag: Optional[str] = None
    ) -> List[str]:
        """
        Determines the columns to use for change tracking.

        Args:
            source_df (DataFrame): Source DataFrame.
            change_fields (List[str], optional): List of field names to include/exclude.
            change_flag (str, optional): One of 'include', 'exclude', or None.

        Returns:
            List[str]: Columns to use for change detection.

        Raises:
            ValueError: If change_flag is not 'include', 'exclude', or None.
        """
        if change_flag is not None and change_flag not in ("include", "exclude"):
            raise ValueError(
                f"change_flag must be 'include', 'exclude', or None; got {change_flag!r}"
            )

        if change_fields is None:
            change_fields = []

        change_cols = set(change_fields)
        schema_cols = source_df.schema.fields

        if change_flag == "exclude":
            return [i.name for i in schema_cols if i.name not in change_cols]
        elif change_flag == "include":
            return [i.name for i in schema_cols if i.name in change_cols]
        else:
            return [i.name for i in schema_cols]

    def generate_merge_key(self, df: DataFrame, change_tracking_cols: List[str]) -> DataFrame:
        """
        Generates an md5-based `cdc_key` from selected columns.

        Args:
            df (DataFrame): Input DataFrame with CDC fields.
            change_tracking_cols (List[str]): Columns to include in the key.

        Returns:
            DataFrame: DataFrame with `cdc_key` populated.
        """
        # ignoreNullFields=false keeps null columns in the JSON, so a value
        # flipping between null and non-null always changes the hash input.
        json_col = F.to_json(F.struct(*change_tracking_cols), {"ignoreNullFields": "false"})
        merge_key = F.md5(json_col)
        return df.withColumn("cdc_key", merge_key)

    def apply_scd_type2(
        self,
        target_table_name: str,
        source_df: DataFrame,
        change_fields: Optional[List[str]] = None,
        change_flag: Optional[str] = None,
    ) -> None:
        """
        Performs the CDC merge operation into the Delta table.

        - Deactivates active rows whose `cdc_key` is no longer present in the source.
        - Inserts new or changed rows as active.
        - Rows that are unchanged between source and target are left untouched.

        Args:
            target_table_name (str): Name of the target Delta table.
            source_df (DataFrame): The new data to merge.
            change_fields (List[str], optional): Fields to use for change detection.
            change_flag (str, optional): 'include', 'exclude', or None.
        """

        target_table_name = DeltaTable.forName(self.spark, target_table_name)

        change_tracking_cols = self.get_change_tracking_cols(source_df, change_fields, change_flag)
        print(f"All change_tracking_cols : \n {change_tracking_cols}\n")

        new_df = source_df.dropDuplicates(subset=change_tracking_cols)
        print(f"cdc: new_df count: {new_df.count()}")

        enriched_df = self.add_cdc_fields(new_df)
        source_with_key_df = self.generate_merge_key(enriched_df, change_tracking_cols)

        merge_condition = f"{self._tgt}.cdc_key = {self._src}.cdc_key AND {self._tgt}.is_active = true"
        cdc_upd_dttm = F.current_timestamp()

        update_set = {
            "is_active": F.lit(False),
            "validity_date": F.current_date(),
            "cdc_upd_dttm": cdc_upd_dttm,
        }

        (
            target_table_name.alias(self._tgt)
                .merge(source_with_key_df.alias(self._src), merge_condition)
                .whenNotMatchedBySourceUpdate(
                    condition=f"{self._tgt}.is_active = true",
                    set=update_set
                )
                .whenNotMatchedInsertAll()
                .execute()
        )

        print(f"cdc: merge completed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply SCD Type 2 CDC merge to a Delta table.")
    parser.add_argument("--cdc-table", required=True, help="Target Delta table name")
    parser.add_argument("--source-table", required=True, help="Source table or view holding the new data")
    parser.add_argument("--change-fields", default="", help="Comma-separated column names to include/exclude")
    parser.add_argument(
        "--change-flag",
        choices=["include", "exclude"],
        default=None,
        help="Whether change-fields are included in or excluded from change detection",
    )
    return parser.parse_args()


def get_spark() -> SparkSession:
    """Returns the active Spark session (Databricks) or builds one with Delta support (elsewhere)."""
    active = SparkSession.getActiveSession()
    if active is not None:
        return active
    return (
        SparkSession.builder
        .appName("scd_type2_cdc")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )


def main() -> None:
    args = parse_args()
    change_fields = [e.strip() for e in args.change_fields.split(",") if e.strip()]
    print(f"cdc_table: {args.cdc_table}, change_fields: {change_fields}, change_flag: {args.change_flag}")

    spark = get_spark()
    source_df = spark.table(args.source_table)

    cdc_util = CDCUtility(spark)
    cdc_util.apply_scd_type2(
        target_table_name=args.cdc_table,
        source_df=source_df,
        change_fields=change_fields or None,
        change_flag=args.change_flag,
    )


if __name__ == "__main__":
    main()
