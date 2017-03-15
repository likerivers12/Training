using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CSWinformSimple
{
    class SimpleData
    {
        private float input1;
        private float input2;
        private float output;

        public float Input1
        {
            get { return input1;  }
            set { input1 = value; }
        }

        public float Input2
        {
            get { return input2; }
            set { input2 = value; }
        }

        public void Compute()
        {
            output = input1 + input2;
        }

        public float GetOutput()
        {
            return output;
        }

        public void Clear()
        {
            input1 = 0;
            input2 = 0;
            output = 0;
        }
    }
}
