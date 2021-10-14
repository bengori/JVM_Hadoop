import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, StopWordsRemover, Tokenizer}
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator


object ImdbML {
  def main(args: Array[String]): Unit = {
    val input_train = args(0) // /user/hduser/IMDB/Train.csv
    val input_test = args(1) // /user/hduser/IMDB/Valid.csv
    // val output = args(2) // /user/hduser/IMDB-Mllib-out

    val spark = SparkSession.builder().getOrCreate()

    import spark.sqlContext.implicits._

    // Prepare training data
    val training = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(input_train)

    // Configure an ML pipeline, which consists of four stages: tokenizer, StopWordRemover(swr), hashingTF, and lr.

    val tokenizer = new Tokenizer()
      .setInputCol("text").
      setOutputCol("words")

    val swt = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered")

    val hashingTF = new HashingTF()
      .setNumFeatures(1000)
      .setInputCol(swt.getOutputCol)
      .setOutputCol("features")

    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.001)

    val pipeline = new Pipeline()
      .setStages(Array[PipelineStage](tokenizer, swt, hashingTF, lr))

    // Fit the pipeline to training documents.
    val model = pipeline.fit(training)

    // save this unfit pipeline to disk
    pipeline.write.overwrite().save("/tmp/unfit-lr-model")
    // save the fitted pipeline to disk
    model.write.overwrite().save("/tmp/logistic-regression-model")

    // Prepare test documents, which are unlabeled (text).
    val test = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(input_test)


    // Make predictions on test documents.
    val predictions = model.transform(test)

    // save this predictions to disk
    predictions.write.mode("overwrite").csv("/tmp/logistic-regression-model")

    // show predictions
    predictions.select($"text", $"label", $"features", $"probability", $"prediction")
      .printSchema()

    predictions.show(5)

    // model accuracy
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("prediction")
      .setMetricName("Accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Accuracy: $accuracy")
    spark.stop()
  }
}

