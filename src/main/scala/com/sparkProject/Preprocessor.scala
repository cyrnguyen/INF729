package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql._
import org.apache.spark.sql.functions._

object Preprocessor {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()

    val fileName = "train.csv"
    val filePath = "/home/cyril/Data/INF729/"

    import spark.implicits._


    /** *****************************************************************************
      *
      * TP 2-3
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      * if problems with unimported modules => sbt plugins update
      *
      * *******************************************************************************/

    /** 1 - CHARGEMENT DES DONNEES **/

    val importedCsv = spark.read.format("csv").option("header", true)
      .option("nullValue", "false")
      .load(filePath + fileName)

    println("-------- DATA DESCRIPTION ----------")
    println("# Lines = " + importedCsv.count())
    println("# Columns = " + importedCsv.columns.length)
    importedCsv.show()
    importedCsv.printSchema()

    /** cast column goal, deadline, state_changed_at, created_at, launched_at, backers_count as int **/
    var castedDf = castColumnToType("goal", "Int", importedCsv)
    castedDf = castColumnToType("deadline", "Int", castedDf)
    castedDf = castColumnToType("state_changed_at", "Int", castedDf)
    castedDf = castColumnToType("created_at", "Int", castedDf)
    castedDf = castColumnToType("launched_at", "Int", castedDf)
    castedDf = castColumnToType("backers_count", "Int", castedDf)
    castedDf = castColumnToType("final_status", "Int", castedDf)

    castedDf.printSchema()

    /** 2 - CLEANING **/
    //    castedDf.filter($"final_status" != "0")
    castedDf.describe().show()
    println("Group by final_status")
    castedDf.groupBy($"final_status").count().sort($"count".desc).head(20).foreach(println)
    println("Group by country")
    castedDf.groupBy($"country").count().sort($"count".desc).head(20).foreach(println)
    println("Group by currency")
    castedDf.groupBy($"currency").count().sort($"count".desc).head(20).foreach(println)
    println("Group by disable_communication")
    castedDf.groupBy($"disable_communication").count().sort($"count".desc).head(20).foreach(println)

    println("Total row numbers : " + castedDf.count().toString)
    println("Row numbers after removing duplicates projects id : " + castedDf.dropDuplicates("project_id").count().toString)

    /** Drop disable_communication **/


    def udf_country = udf { (country: String, currency: String) =>
      if (country != null)
        country
      else
        currency
    }

    def udf_currency = udf { currency: String =>
      if (currency != null && currency.length == 3)
        currency
      else
        null
    }

    val cleanedDf = castedDf.drop("disable_communication", "backers_count", "state_changed_at").
      withColumn("country2", udf_country(col("country"), col("currency"))).
      withColumn("currency2", udf_currency(col("currency"))).
      drop("country", "currency").
      filter($"final_status".isin(0, 1))

    /** FEATURE ENGINEERING: Ajouter et manipuler des colonnes **/
    def udf_replace_null_values = udf { (value: AnyVal) => if (value == null) -1 }

    val engineeredDf = cleanedDf.withColumn("days_campaign", datediff(from_unixtime($"deadline"), from_unixtime($"launched_at"))).
      withColumn("hours_prepa", bround(($"launched_at" - $"created_at") / (60 * 60), 3)).
      filter($"days_campaign" >= 0 && $"hours_prepa" >= 0).
      drop("launched_at", "created_at", "deadline").
      withColumn("text", concat_ws(" ", $"name", $"desc", $"keywords")).
      filter($"goal" > 0).na.fill(Map(
      "days_campaign" -> -1,
      "hours_prepa" -> -1
    ))
    //      withColumn("days_campaign", udf_replace_null_values($"days_campaign")).
    //      withColumn("hours_prepa", udf_replace_null_values($"hours_prepa")).
    //      withColumn("goal", udf_replace_null_values($"goal"))
    engineeredDf.show(50)
    engineeredDf.printSchema()

    engineeredDf.write.mode(SaveMode.Overwrite).parquet(filePath + "cleaned_train")
  }

  def castColumnToType(columnName: String, targetType: String, dataset: DataFrame): DataFrame = {
    val tmpColumnName = "tmp" + columnName
    dataset.withColumnRenamed(columnName, tmpColumnName).withColumn(columnName, col(tmpColumnName)
      .cast(targetType)).drop(tmpColumnName)
  }

}
