# Databricks notebook source
dbutils.widgets.text("cdc_table", "" , "target table")
dbutils.widgets.text("change_fields", "", "comma separated column names")
dbutils.widgets.text("change_flag", "", "include/ exclude")

# COMMAND ----------

try:
    cdc_table = dbutils.widgets.get("cdc_table")
except Exception as ex:
    print(f"Exception occurred while retrieving cdc table from input arguments", ex)
    raise ex
print(f"cdc_table value is: {cdc_table}")

# COMMAND ----------

try:
    change_fields = [e.strip() for e in dbutils.widgets.get('change_fields').split(',')]
except Exception as ex:
    print(f"Exception occurred while retrieving fields to exclude from input arguments", ex)
    raise ex
print(f"fields to exclude from cdc: {change_fields}")

# COMMAND ----------

try:
    change_flag = dbutils.widgets.get("change_flag").strip().lower()
except Exception as ex:
    print(f"Exception occurred while retrieving change flag from input arguments", ex)
    raise ex
print(f"change flag value is: {change_flag}")

# COMMAND ----------

from pyspark.sql import DataFrame, SparkSession, functions as F
from delta.tables import DeltaTable
from typing import List, Optional


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
        - `dbr_upd_dttm`

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
              .withColumn("dbr_upd_dttm", F.current_timestamp())
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
        """
        if change_fields is None:
            change_fields = []

        __change_cols = set(change_fields)
        schema_cols = source_df.schema.fields

        if change_flag == "exclude":
            return [i.name for i in schema_cols if i.name not in __change_cols]
        elif change_flag == "include":
            return [i.name for i in schema_cols if i.name in __change_cols]
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
        json_col = F.to_json(F.struct(*change_tracking_cols))
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

        - Deactivates existing active rows with the same or missing `cdc_key`.
        - Inserts new or changed rows as active.
        - Updates `dbr_upd_dttm` for tracking.

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
        dbr_upd_dttm = F.current_timestamp()

        update_set = {
            "is_active": F.lit(False),
            "validity_date": F.current_date(),
            "dbr_upd_dttm": dbr_upd_dttm,
        }

        (
            target_table_name.alias(self._tgt)
                .merge(source_with_key_df.alias(self._src), merge_condition)
                .whenMatchedUpdate(set={"dbr_upd_dttm": dbr_upd_dttm})
                .whenNotMatchedBySourceUpdate(
                    condition=f"{self._tgt}.is_active = true",
                    set=update_set
                )
                .whenNotMatchedBySourceUpdate(
                    set={"dbr_upd_dttm": dbr_upd_dttm}
                )
                .whenNotMatchedInsertAll()
                .execute()
        )

        print(f"cdc: merge completed")
