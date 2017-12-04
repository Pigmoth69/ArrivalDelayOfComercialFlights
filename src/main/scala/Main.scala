import explorer.FileExplorer
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.attribute.Attribute
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.mllib.tree.RandomForest


object Main {
  Logger.getLogger("org").setLevel(Level.ERROR)


  def main(args: Array[String]) {


    //Spark context/session setup
    val spark = SparkSession.builder
      .master("local")
      .appName("ArrivalDelayOfComercialFlights")
      .getOrCreate;

    //create some udf functions
    val toInt    = udf[Int, String](data => if(data.isInstanceOf[Int]) data.toInt else 1)

    //CSV data loading for dataframe
    val flights_df = spark.read
      .format("csv")
      .option("header", "true")
      .load(FileExplorer.flights_2007)
    val flights_df_test = spark.read
      .format("csv")
      .option("header", "true")
      .load(FileExplorer.flights_2008)


    import spark.implicits._
    //Dataframe data pruned
    val flights_df_pruned = flights_df
      .withColumn("Month", toInt(flights_df("Month")))
      .withColumn("DayOfMonth", toInt(flights_df("DayOfMonth")))
      .withColumn("DayOfWeek", toInt(flights_df("DayOfWeek")))
      .withColumn("DepTime", toInt(flights_df("DepTime")))
      .withColumn("CRSDepTime", toInt(flights_df("CRSDepTime")))
      .withColumn("CRSArrTime", toInt(flights_df("CRSArrTime")))
      .withColumn("CRSElapsedTime", toInt(flights_df("CRSElapsedTime")))
      .withColumn("ArrDelay", toInt(flights_df("ArrDelay")))
      .withColumn("DepDelay", toInt(flights_df("DepDelay")))
      .withColumn("Distance", toInt(flights_df("Distance")))
      .withColumn("TaxiOut", toInt(flights_df("TaxiOut")))
      .filter($"Cancelled".equalTo(0))
      .drop("Year"
        ,"ArrTime"
        ,"FlightNum"
        ,"TailNum"
        ,"ActualElapsedTime"
        ,"AirTime"
        ,"TaxiIn"
        ,"Cancelled"
        ,"CancellationCode"
        ,"Diverted"
        ,"CarrierDelay"
        ,"WeatherDelay"
        ,"NASDelay"
        ,"SecurityDelay"
        ,"LateAircraftDelay")

    flights_df_pruned.cache()
    /*val flights_df_pruned_test = flights_df_test
      .withColumn("Month", toInt(flights_df_test("Month")))
      .withColumn("DayOfMonth", toInt(flights_df_test("DayOfMonth")))
      .withColumn("DayOfWeek", toInt(flights_df_test("DayOfWeek")))
      .withColumn("DepTime", toInt(flights_df_test("DepTime")))
      .withColumn("CRSDepTime", toInt(flights_df_test("CRSDepTime")))
      .withColumn("CRSArrTime", toInt(flights_df_test("CRSArrTime")))
      .withColumn("CRSElapsedTime", toInt(flights_df_test("CRSElapsedTime")))
      .withColumn("ArrDelay", toInt(flights_df_test("ArrDelay")))
      .withColumn("DepDelay", toInt(flights_df_test("DepDelay")))
      .withColumn("Distance", toInt(flights_df_test("Distance")))
      .withColumn("TaxiOut", toInt(flights_df_test("TaxiOut")))
      .filter($"Cancelled".equalTo(0))
      .drop("Year"
        ,"ArrTime"
        ,"FlightNum"
        ,"TailNum"
        ,"ActualElapsedTime"
        ,"AirTime"
        ,"TaxiIn"
        ,"Cancelled"
        ,"CancellationCode"
        ,"Diverted"
        ,"CarrierDelay"
        ,"WeatherDelay"
        ,"NASDelay"
        ,"SecurityDelay"
        ,"LateAircraftDelay")

    flights_df_pruned_test.cache()*/

    flights_df_pruned.printSchema()

    //Create Indexers
    val OriginIndx = new StringIndexer().setInputCol("Origin").setOutputCol("OriginIndx")
    val DestIndx = new StringIndexer().setInputCol("Dest").setOutputCol("DestIndx")
    val UniqueCarrierIndx = new StringIndexer().setInputCol("UniqueCarrier").setOutputCol("UniqueCarrierIndx")

    val df1 = OriginIndx.fit(flights_df_pruned).transform(flights_df_pruned)
    val df2 = DestIndx.fit(df1).transform(df1)
    val df3 = UniqueCarrierIndx.fit(df2).transform(df2)


    //Add features column
    val assembler = new VectorAssembler()
      .setInputCols(Array("Month","DayOfMonth","DayOfWeek","DepTime","CRSDepTime","CRSArrTime","UniqueCarrierIndx","CRSElapsedTime","DepDelay","OriginIndx","DestIndx","Distance","TaxiOut"))
      .setOutputCol("rawFeatures")


    val df_with_features = assembler.transform(df3)

   df_with_features.select("rawFeatures").show(truncate=false)


    val splitSeed = 5043
    val Array(trainingData, testData) = df_with_features.randomSplit(Array(0.7, 0.3), splitSeed)

    // Train a RandomForest model.
    //  Empty categoricalFeaturesInfo indicates all features are continuous.
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 3 // Use more in practice.
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "variance"
    val maxDepth = 4
    val maxBins = 32


   // val model = RandomForest.trainClassifier(df3,featureSubsetStrategy,numTrees,13,21)
    /*val model = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelsAndPredictions = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testMSE = labelsAndPredictions.map{ case(v, p) => math.pow((v - p), 2)}.mean()
    println("Test Mean Squared Error = " + testMSE)
    println("Learned regression forest model:\n" + model.toDebugString)


*/


    //Columns ordering
   /* val columns: Array[String] = flights_df_pruned.columns
    val reorderedColumnNames: Array[String] = Array(
      "Month",
      "DayOfMonth",
      "DayOfWeek",
      "DepTime",
      "CRSDepTime",
      "CRSArrTime",
      "UniqueCarrier",
      "CRSElapsedTime",
      "DepDelay",
      "Origin",
      "Dest",
      "Distance",
      "TaxiOut",
      "ArrDelay"
    )
    val final_pruned_flights = flights_df_pruned.select(reorderedColumnNames.head, reorderedColumnNames.tail: _*)*/



    //Add Indexers

    /*

    //Add features column
    // Or if you want to exclude columns
    val ignored = List("foo", "target", "x2")
    val featInd = final_pruned_flights.columns.diff(ignored).map(final_pruned_flights.columns.indexOf(_))

    val assembler = new VectorAssembler()
      .setInputCols(Array("Month","DayOfMonth","DayOfWeek","DepTime","CRSDepTime","CRSArrTime","UniqueCarrier","CRSElapsedTime","DepDelay","Origin","Dest","Distance","TaxiOut"))
      .setOutputCol("features")

    val df_with_features = assembler.transform(final_pruned_flights)
    df_with_features.show(truncate=false)
*/




    //Enable for performance
    //flights_df_pruned.cache()
    //flights_df_pruned.printSchema()
    //flights_df_pruned.show()
    //flights_df_pruned.show(5)

    //val splits = flights_df_pruned.randomSplit(Array(0.7, 0.3))
    //val (trainingData, testData) = (splits(0), splits(1))



    /*val data = MLUtils.loadLibSVMFile(spark.sparkContext, "data/mllib/sample_libsvm_data.txt")

    // Discretize data in 16 equal bins since ChiSqSelector requires categorical features
    // Even though features are doubles, the ChiSqSelector treats each unique value as a category
    val discretizedData = flights_df_pruned.map { lp =>
      LabeledPoint(lp.label, Vectors.dense(lp.features.toArray.map { x => (x / 16).floor }))
    }*/


    //flights_df.createOrReplaceTempView("flights")

    //spark.sql("SELECT * from flights where TailNum = \'N685\'").show()


    //n_df.take(5).foreach(println)

  }



}