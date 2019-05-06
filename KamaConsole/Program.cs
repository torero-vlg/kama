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
            Console.ReadLine();
        }
    }
}
