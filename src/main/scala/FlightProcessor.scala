import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame,  SparkSession}
import org.apache.spark.ml.feature.{ StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation.{RegressionMetrics}
import org.apache.spark.rdd.RDD
import org.jfree.chart.{ChartFactory, ChartFrame}
import org.jfree.data.xy.DefaultXYDataset

class FlightProcessor(spark: SparkSession, targetVariable: String){

  //Datasets
  var flight_dataframe_training : DataFrame = null
  var training_percentage: Double = 0.7
  var test_percentage: Double = 0.3
  var seed: Int = 5149
  private var labelAndPreds: RDD[(Double,Double)] = null
  //create some udf functions
  val toInt = udf[Int, String](_.toInt)
  val toHours = udf[Int, String](time => (time.toInt/100))

  //@TODO meter pipelines

  //Load all data from CSV file and creates the dataframes
  def load(file :String, training_percentage :Double,test_percentage :Double, seed :Int): Unit ={

    //
    this.test_percentage=test_percentage
    this.training_percentage=training_percentage
    this.seed=seed
    val flights_df = spark.read
      .format("csv")
      .option("header", "true")
      .load(file)
    flight_dataframe_training = transformCSVFile(flights_df)
    flight_dataframe_training.na.drop()
  }

  //Parses the columns and gives them all the necessary arragements of datatype conversion
  private def transformCSVFile(df: DataFrame): DataFrame ={

    df
      .filter(df("Cancelled").equalTo(0))
      .filter(df("DepDelay").notEqual(("NA")))
      .filter(df("DepTime").notEqual(("NA")))
      .filter(df("TaxiOut").notEqual(("NA")))
      .filter(df("ArrDelay").notEqual(("NA")))
      .filter(df("CRSElapsedTime").notEqual(("NA")))
      .withColumn("Month", toInt(df("Month")))
      .withColumn("DayOfMonth", toInt(df("DayOfMonth")))
      .withColumn("DayOfWeek", toInt(df("DayOfWeek")))
      .withColumn("DepTime", toHours(df("DepTime")))
      .withColumn("CRSDepTime", toHours(df("CRSDepTime")))
      .withColumn("CRSElapsedTime", toInt(df("CRSElapsedTime")))
      .withColumn("ArrDelay", toInt(df("ArrDelay")))
      .withColumn("DepDelay", toInt(df("DepDelay")))
      .withColumn("Distance", toInt(df("Distance")))
      .withColumn("TaxiOut", toInt(df("TaxiOut")))
      .drop(
        "Year"
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
        ,"CRSArrTime"
        ,"LateAircraftDelay")
  }

  //Transforms the dataframe(types and indexes) for the machine learning algorithms and add features columns
  def transformDataframe(): Unit ={

    flight_dataframe_training.createOrReplaceTempView("flights")

    val df_tmp = spark.sql("select CRSDepTime as CRSDepTime_tmp , MEAN(ArrDelay) as MeanArrDelay from flights as Tmp GROUP BY CRSDepTime")
    flight_dataframe_training = flight_dataframe_training.join(df_tmp,flight_dataframe_training("CRSDepTime") === df_tmp("CRSDepTime_tmp")).drop("CRSDepTime_tmp")


    val MonthIndx = new StringIndexer().setInputCol("Month").setOutputCol("MonthIndx")
    val DayOfMonthIndx = new StringIndexer().setInputCol("DayOfMonth").setOutputCol("DayOfMonthIndx")
    val DayOfWeekIndx = new StringIndexer().setInputCol("DayOfWeek").setOutputCol("DayOfWeekIndx")
    val OriginIndx = new StringIndexer().setInputCol("Origin").setOutputCol("OriginIndx")
    val DestIndx = new StringIndexer().setInputCol("Dest").setOutputCol("DestIndx")
    val UniqueCarrierIndx = new StringIndexer().setInputCol("UniqueCarrier").setOutputCol("UniqueCarrierIndx")
    val CRSDepTimeIndx = new StringIndexer().setInputCol("CRSDepTime").setOutputCol("CRSDepTimeIndx")


    flight_dataframe_training = MonthIndx.fit(flight_dataframe_training).transform(flight_dataframe_training)
    flight_dataframe_training = DayOfMonthIndx.fit(flight_dataframe_training).transform(flight_dataframe_training)
    flight_dataframe_training = DayOfWeekIndx.fit(flight_dataframe_training).transform(flight_dataframe_training)
    flight_dataframe_training = OriginIndx.fit(flight_dataframe_training).transform(flight_dataframe_training)
    flight_dataframe_training = DestIndx.fit(flight_dataframe_training).transform(flight_dataframe_training)
    flight_dataframe_training = UniqueCarrierIndx.fit(flight_dataframe_training).transform(flight_dataframe_training)
    flight_dataframe_training = CRSDepTimeIndx.fit(flight_dataframe_training).transform(flight_dataframe_training)

  }

  def RandomForest(): Unit ={

    //Normalize continuous values
    val assembler1 = new VectorAssembler()
      .setInputCols(Array(
        "DepDelay",
        "MeanArrDelay",
        "Distance",
        "TaxiOut"
      ))
      .setOutputCol("contValues")

    flight_dataframe_training = assembler1.transform(flight_dataframe_training)

    val scaler1 = new StandardScaler()
      .setInputCol("contValues")
      .setOutputCol("contScaledValues")
      .setWithStd(true)

    val tmp1 = scaler1.fit(flight_dataframe_training)
    flight_dataframe_training = tmp1.transform(flight_dataframe_training)
    flight_dataframe_training.printSchema()


    val assembler2 = new VectorAssembler()
      .setInputCols(Array(
        "MonthIndx",
        "DayOfMonthIndx",
        "DayOfWeekIndx",
        "UniqueCarrierIndx",
        "OriginIndx",
        "DestIndx",
        "CRSDepTimeIndx",
        "CRSElapsedTime",
        "contScaledValues"))
      .setOutputCol("finalfeatures")
      flight_dataframe_training = assembler2.transform(flight_dataframe_training)

    //This is commented in order to decrease the execution time of the application
    //val MonthIndx = flight_dataframe_training.select("MonthIndx").distinct().count().toInt
    //val DayOfMonthIndx = flight_dataframe_training.select("DayOfMonthIndx").distinct().count().toInt
    //val DayOfWeekIndx = flight_dataframe_training.select("DayOfWeekIndx").distinct().count().toInt
    val numUniqueCarrierIndx = flight_dataframe_training.select("UniqueCarrierIndx").distinct().count().toInt
    val numOriginIndx = flight_dataframe_training.select("OriginIndx").distinct().count().toInt
    val numDestIndx = flight_dataframe_training.select("DestIndx").distinct().count().toInt
    val CRSDepTimeIndx = flight_dataframe_training.select("CRSDepTimeIndx").distinct().count().toInt

    val labeled = flight_dataframe_training.rdd.map(row =>
      if(row.getAs[Any]("finalfeatures").isInstanceOf[org.apache.spark.ml.linalg.DenseVector])
        LabeledPoint(row.getAs[Integer]("ArrDelay").toDouble,Vectors.dense(row.getAs[org.apache.spark.ml.linalg.DenseVector]("finalfeatures").values))
      else
        LabeledPoint(row.getAs[Integer]("ArrDelay").toDouble,Vectors.dense(row.getAs[org.apache.spark.ml.linalg.SparseVector]("finalfeatures").toDense.values)))

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = labeled.randomSplit(Array(training_percentage, test_percentage))

    // Train a RandomForest model.
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    val categoricalFeaturesInfo = Map[Int, Int]((0,12/*MonthIndx*/),(1,31/*DayOfMonthIndx*/),(2,7/*DayOfWeekIndx*/),(3,numUniqueCarrierIndx),(4,numOriginIndx),(5,numDestIndx),(6,CRSDepTimeIndx))
    val numTrees = 100
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "variance"
    val maxDepth = 9
    val maxBins = 2700


    val model = org.apache.spark.mllib.tree.RandomForest.trainRegressor(trainingData,categoricalFeaturesInfo,numTrees,featureSubsetStrategy,impurity,maxDepth,maxBins,seed)

    // Evaluate model on test instances and save it to labelAndPreds variable
    labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    println("-------------------------------------------------------START PREDICTIONS------------------------------------------------------------")
    labelAndPreds.take(50).foreach(println)
    println("--------------------------------------------------------END PREDICTIONS------------------------------------------------------------")
  }

  def computeTestError(): Unit={
    val metrics = new RegressionMetrics(labelAndPreds)
    /// Mean Squared Error
    println(s"MSE = ${metrics.meanSquaredError}")
    // Root Mean Squared Error
    println(s"RMSE = ${metrics.rootMeanSquaredError}")
    // R-squared
    println(s"R-squared = ${metrics.r2}")
    // Explained variance
    println(s"Explained variance = ${metrics.explainedVariance}")
  }

  private def toIntPlot(num: String): Int ={
    try{
      num.toInt
    }catch {
      case e: Exception => 50
    }
  }

  def plotPredictions(num_rows: String): Unit ={
    val num_values = toIntPlot(num_rows)

    val plot_values = labelAndPreds.collect()

    if(plot_values.length < num_values)
      println("Total number of rows inferior to the input number")

    val x = (0 to num_values-1 by 1 toArray).map(_.toDouble)

    val y_real = plot_values.take(num_values).map(_._1)
    val y_prediction = plot_values.take(num_values).map(_._2)

    val dataset = new DefaultXYDataset
    dataset.addSeries("RealValues",Array(x,y_real))
    dataset.addSeries("PredictionValues",Array(x,y_prediction))

    val frame = new ChartFrame(
      "Real value and Prediction Value",
      ChartFactory.createXYLineChart(
        "Real value and Prediction Value",
        "Row Number",
        "Value",
        dataset,
        org.jfree.chart.plot.PlotOrientation.VERTICAL,
        true,false,false
      )
    )
    frame.pack()
    frame.setVisible(true)
  }

}