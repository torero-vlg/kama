using Microsoft.ML.Runtime.Api;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace KamaConsole
{
    public static class ItemStocks
    {
        internal static readonly ItemStock stock1 = new ItemStock
        {
            ItemID = "Item001",
            Loccode = 1,
            InQty = 100,
            OutQty = 10,
            ItemType = "IN",
            TotalStockQty = 0 // predict it. Actual Total Stock Quantity is = 90
        };

        internal static readonly ItemStock stock2 = new ItemStock
        {
            ItemID = "Item003",
            Loccode = 4,
            InQty = 6000,
            OutQty = 1200,
            ItemType = "IN",
            TotalStockQty = 0 // predict it. Actual Total Stock Quantity is = 4800
        };
    }
}
