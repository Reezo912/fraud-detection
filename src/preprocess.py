from config import settings

from pyspark.sql import SparkSession
import pyspark.sql.functions as sf
from pyspark.ml.feature import StringIndexer


class FraudETL:
    """
    TODO Docstring for FraudETL
    """

    def __init__(self):
        print("Starting ETL...  ")
        print("Creating Spark Session...   ")
        self.spark = (
            SparkSession.builder.appName("Fraud_Detection")
            .master("local[*]")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.driver.memory", "12g")
            .getOrCreate()
        )

    def extract(self):
        """
        Data loading and join of dataframes
        """
        print("Reading Raw data...  ")

        def load_join_data(prefix):
            """
            :param prefix: train/test
            """
            trans = self.spark.read.csv(
                str(settings.paths.data_raw / f"{prefix}_transaction.csv"),
                header=True,
                inferSchema=True,
            )
            ident = self.spark.read.csv(
                str(settings.paths.data_raw / f"{prefix}_identity.csv"),
                header=True,
                inferSchema=True,
            )

            df_joined = trans.join(ident, on="TransactionID", how="left")
            return df_joined

        self.raw_train = load_join_data("train")
        self.raw_test = load_join_data("test")

        print(
            f"Joined train count: {self.raw_train.count()} || Joined test count: {self.raw_test.count()}"
        )

    def _process_dataset(self, df):
        seconds_per_day = sf.lit(24 * 3600)
        seconds_per_hour = sf.lit(3600)
        df = df.withColumns(
            {
                "hour": (sf.col("TransactionDT") % seconds_per_day / seconds_per_hour),
                "Day_of_week": sf.floor(
                    sf.col("TransactionDT") / seconds_per_day % sf.lit(7)
                ),
                "Day_of_month": sf.floor(
                    sf.col("TransactionDT") / seconds_per_day % sf.lit(30)
                ),
            }
        )

        if "P_emaildomain" in df.columns:
            df = df.withColumn("P_split", sf.split(sf.col("P_emaildomain"), "\\."))
            df = df.withColumns(
                {
                    "P_emaildomain_1": sf.col("P_split").getItem(0),
                    "P_emaildomain_2": sf.when(
                        sf.size(sf.col("P_split")) > 1, sf.col("P_split").getItem(1)
                    ).otherwise(sf.lit(None)),
                    # TRUCO: getItem(2) devuelve null si no existe, PERO si Spark está en modo ANSI falla.
                    # Lo solucionamos forzando un cast o usando try_element_at si tu spark es >3.3
                    # Solución universal: size check
                    "P_emaildomain_3": sf.when(
                        sf.size(sf.col("P_split")) > 2, sf.col("P_split").getItem(2)
                    ).otherwise(sf.lit(None)),
                }
            ).drop("P_split")

        if "R_emaildomain" in df.columns:
            df = df.withColumn("R_split", sf.split(sf.col("R_emaildomain"), "\\."))
            df = df.withColumns(
                {
                    "R_emaildomain_1": sf.col("R_split").getItem(0),
                    "R_emaildomain_2": sf.when(
                        sf.size(sf.col("R_split")) > 1, sf.col("R_split").getItem(1)
                    ).otherwise(sf.lit(None)),
                    "R_emaildomain_3": sf.when(
                        sf.size(sf.col("R_split")) > 2, sf.col("R_split").getItem(2)
                    ).otherwise(sf.lit(None)),
                }
            ).drop("R_split")

        num_cols = [
            col_name
            for col_name, col_type in df.dtypes
            if col_type in ["int", "bigint", "float", "double"]
            and col_name not in ["isFraud", "TransactionID"]
        ]
        string_cols = [
            col_name for col_name, col_type in df.dtypes if col_type == "string"
        ]
        df = df.fillna(-999, num_cols)
        df = df.fillna("UNKNOWN", string_cols)

        return df

    def transform(self):
        """ "Orquestator for feature engineering and encoding"""
        print("Processing datasets... ")

        df_train_fe = self._process_dataset(self.raw_train)
        df_test_fe = self._process_dataset(self.raw_test)

        cat_cols = [
            f.name
            for f in df_train_fe.schema.fields
            if isinstance(f.dataType, sf.StringType)
        ]
        outputcols = [c + "_idx" for c in cat_cols]

        if cat_cols:
            print(f"      indexing {len(cat_cols)} categoric columns...")
            # marcador
            df_train_fe = df_train_fe.withColumn("is_train", sf.lit(1))
            df_test_fe = df_test_fe.withColumn("is_train", sf.lit(0))

            full_df = df_train_fe.unionByName(df_test_fe, allowMissingColumns=True)

            indexer = StringIndexer(
                inputCols=cat_cols,
                outputCols=outputcols,
                handleInvalid="keep",
            )
            full_encoded = indexer.fit(full_df).transform(full_df)
            full_encoded = full_encoded.drop(*cat_cols)

            # separar
            self.final_train = full_encoded.filter(sf.col("is_train") == 1).drop(
                "is_train"
            )
            self.final_test = full_encoded.filter(sf.col("is_train") == 0).drop(
                "is_train", "isfraud"
            )
        else:
            self.final_train = df_train_fe
            self.final_test = df_test_fe

    def load(self):
        """ "Final save"""
        print("Saving processed datasets... ")
        settings.paths.data_processed.mkdir(parents=True, exist_ok=True)

        cutoff_time = self.final_train.approxQuantile("TransactionDT", [0.8], 0.001)[0]
        train_df = self.final_train.filter(sf.col("TransactionDT") <= cutoff_time)
        val_df = self.final_train.filter(sf.col("TransactionDT") > cutoff_time)

        train_df = train_df.drop("TransactionDT", "P_emaildomain", "R_emaildomain")
        val_df = val_df.drop("TransactionDT", "P_emaildomain", "R_emaildomain")
        test_df = self.final_test.drop(
            "TransactionDT", "P_emaildomain", "R_emaildomain"
        )

        train_df.write.mode("overwrite").parquet(
            str(settings.paths.data_processed / "train_df.parquet")
        )
        val_df.write.mode("overwrite").parquet(
            str(settings.paths.data_processed / "val_df.parquet")
        )
        test_df.write.mode("overwrite").parquet(
            str(settings.paths.data_processed / "test_df.parquet")
        )
        print("ETL done... \nDatasets saved as parquet...")

    def run(self):
        try:
            self.extract()
            self.transform()
            self.load()
        finally:
            self.spark.stop()


def run_ETL():
    FraudETL().run()


if __name__ == "__main__":
    # Esto permite ejecutar el archivo directamente
    run_ETL()
