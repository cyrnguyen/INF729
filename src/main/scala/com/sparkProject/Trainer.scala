package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.SparkSession


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()

    val fileName = "cleaned_train"
    val filePath = "/home/cyril/Data/INF729/"


    /** *****************************************************************************
      *
      * TP 4-5
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      * if problems with unimported modules => sbt plugins update
      *
      * *******************************************************************************/

    /** CHARGER LE DATASET **/
    val data_init = spark.read.parquet(filePath + fileName)
    data_init.show()

    /** TF-IDF **/
    val tokenizer = new RegexTokenizer().setPattern("\\W+").setGaps(true).setInputCol("text").setOutputCol("tokens")
    val stopWordsRemover = new StopWordsRemover().setInputCol("tokens").setOutputCol("withoutStopWords")
    val countVectorizer = new CountVectorizer().setInputCol("withoutStopWords").setOutputCol("rawCount")
    val idf = new IDF().setInputCol("rawCount").setOutputCol("tfidf")


    /** PreProcess categorical columns **/
    val countryIndexer = new StringIndexer().setInputCol("country2").setOutputCol("country_indexed").
      setHandleInvalid("keep")
    val currencyIndexer = new StringIndexer().setInputCol("currency2").setOutputCol("currency_indexed").
      setHandleInvalid("keep")

    /** VECTOR ASSEMBLER **/
    val vectorAssembler = new VectorAssembler().setInputCols(
      Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_indexed", "currency_indexed")).
      setOutputCol("features")

    /** MODEL **/
    val lr = new LogisticRegression().
      setElasticNetParam(0.0).
      setFitIntercept(true).
      setFeaturesCol("features").
      setLabelCol("final_status").
      setStandardization(true).
      setPredictionCol("predictions").
      setRawPredictionCol("raw_predictions").
      setThresholds(Array(0.7, 0.3)).
      setTol(1.0e-6).
      setMaxIter(300)

    /** PIPELINE **/

    val pipeline = new Pipeline().setStages(
      Array(tokenizer, stopWordsRemover, countVectorizer, idf, countryIndexer, currencyIndexer, vectorAssembler, lr))

    /** TRAINING AND GRID-SEARCH **/
    val Array(train, test) = data_init.randomSplit(Array(0.9, 0.1))

    val paramGrid = new ParamGridBuilder().
      addGrid(lr.regParam, List.range(-8, -1, 2).map(i => scala.math.pow(10, i))).
      addGrid(countVectorizer.minDF, List.range(55, 96, 20).map(i => i.toDouble)).
      build()

    val trainValidationSplit = new TrainValidationSplit().
      setEstimator(pipeline).
      setEvaluator(new BinaryClassificationEvaluator().setLabelCol("final_status").setRawPredictionCol("raw_predictions")).
      setEstimatorParamMaps(paramGrid).
      setTrainRatio(0.7)

    val model = trainValidationSplit.fit(train)

    val predictions = model.transform(test)
    predictions.show()
    val evaluator = new BinaryClassificationEvaluator().setLabelCol("final_status").setRawPredictionCol("raw_predictions")
    println("Evaluation du modèle : " + evaluator.evaluate(predictions))
    predictions.groupBy("final_status", "predictions").count().show()

    /** Save model **/
    model.write.overwrite().save(filePath + "model")
  }
}
