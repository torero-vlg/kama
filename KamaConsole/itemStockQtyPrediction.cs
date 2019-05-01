using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace KamaConsole
{
    public class itemStockQtyPrediction
    {
        [ColumnName("Score")]
        public float TotalStockQty;
    }
}
