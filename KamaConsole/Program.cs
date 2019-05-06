using System;
using System.IO;
using Microsoft.ML;

namespace KamaConsole
{
    class Program
    {
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static void Main(string[] args)
        {
            /*
             * https://docs.microsoft.com/ru-ru/dotnet/machine-learning/tutorials/taxi-fare
             */

            MLContext mlContext = new MLContext(seed: 0);

            
            var model = Train(mlContext, _trainDataPath);

            Evaluate(mlContext, model);

            TestSinglePrediction(mlContext);

            Console.ReadLine();
        }

        public static ITransformer Train(MLContext mlContext, string dataPath)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(dataPath, hasHeader: true, separatorChar: ',');

            /*
             * При обучении и оценке модели значения в столбце Label по умолчанию рассматриваются как правильные значения для прогноза. 
             * Поскольку нам необходимо спрогнозировать плату за поездку на такси, скопируйте столбец FareAmount в столбец Label. 
             * Для этого используйте класс преобразования CopyColumnsEstimator
             */
            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
                /*
                 * Алгоритм, который обучает модель, принимает числовые признаки, поэтому значения категориальных данных (VendorId, RateCode и PaymentType)
                 * нужно преобразовать в числа (VendorIdEncoded, RateCodeEncoded и PaymentTypeEncoded). 
                 * Для этого используйте класс преобразования Microsoft.ML.Transforms.OneHotEncodingTransformer>, 
                 * который присваивает разные числовые значения ключа разным значениям в каждом из столбцов
                 */
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
                /*
                 * Последний шаг на этапе подготовки данных заключается в объединении всех столбцов признаков в столбце Features 
                 * с помощью класса преобразования mlContext.Transforms.Concatenate. 
                 * По умолчанию алгоритм обучения обрабатывает только признаки, представленные в столбце Features.
                 */
                .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripTime", "TripDistance", "PaymentTypeEncoded"))
                /*
                 * Алгоритм обучения
                 */
                 //TODO ?
                ;//.Append(mlContext.Regression.Trainers.());

            var model = pipeline.Fit(dataView);

            SaveModelAsFile(mlContext, dataView.Schema, model);
            return model;
        }

        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(_testDataPath, hasHeader: true, separatorChar: ',');

            var predictions = model.Transform(dataView);

            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");
        }

        private static void TestSinglePrediction(MLContext mlContext)
        {
            ITransformer loadedModel = mlContext.Model.Load(_modelPath, out var modelInputSchema);
            var predictionFunction = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(loadedModel);

            var taxiTripSample = new TaxiTrip()
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1140,
                TripDistance = 3.75f,
                PaymentType = "CRD",
                FareAmount = 0 // To predict. Actual/Observed = 15.5
            };
            var prediction = predictionFunction.Predict(taxiTripSample);

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
            Console.WriteLine($"**********************************************************************");
        }

        private static void SaveModelAsFile(MLContext mlContext, DataViewSchema dataViewSchema, ITransformer model)
        {
            mlContext.Model.Save(model, dataViewSchema, _modelPath);
            Console.WriteLine("The model is saved to {0}", _modelPath);
        }
    }
}
