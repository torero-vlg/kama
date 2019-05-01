using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace KamaConsole
{
    class Program
    {
        static readonly string _Traindatapath = Path.Combine(Environment.CurrentDirectory, "Data", "StockTrain.csv");
        static readonly string _Evaluatedatapath = Path.Combine(Environment.CurrentDirectory, "Data", "StockTest.csv");
        static readonly string _modelpath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static async Task Main(string[] args)
        {
            PredictionModel<ItemStock, itemStockQtyPrediction> model = await TrainourModel();

            Evaluate(model);

            Console.ReadLine();
        }

        public static async Task<PredictionModel<ItemStock, itemStockQtyPrediction>> TrainourModel()
        {
            var pipeline = new LearningPipeline
            {
                new TextLoader(_Traindatapath).CreateFrom<ItemStock>(useHeader:true, separator:','),
                new ColumnCopier(("TotalStockQty", "Label")),
                new CategoricalOneHotVectorizer("ItemID", "ItemType"),
                new ColumnConcatenator("Features","ItemID","Loccode","InQty","OutQty","ItemType"),
                new FastTreeRegressor()
            };

            PredictionModel<ItemStock, itemStockQtyPrediction> model = pipeline.Train<ItemStock, itemStockQtyPrediction>();

            await model.WriteAsync(_modelpath);
            return model;
        }

        private static void Evaluate(PredictionModel<ItemStock, itemStockQtyPrediction> model)
        {
            var testData = new TextLoader(_Evaluatedatapath).CreateFrom<ItemStock>(useHeader: true, separator: ',');
            var evaluator = new RegressionEvaluator();
            RegressionMetrics metrics = evaluator.Evaluate(model, testData);

            Console.WriteLine($"Rms = {metrics.Rms}");
            Console.WriteLine($"RSquared = {metrics.RSquared}");


        }
    }
}
