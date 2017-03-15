using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace CSWinformSimple
{
    public partial class frmMain : Form
    {
        private SimpleData data;
        public frmMain()
        {
            data = new SimpleData();

            InitializeComponent();
        }

        private void txtInput1_TextChanged(object sender, EventArgs e)
        {
            String val = txtInput1.Text;
            if (val == "") return;

            data.Input1 = (float)Convert.ToDouble(val);
        }

        private void txtInput2_TextChanged(object sender, EventArgs e)
        {
            String val = txtInput2.Text;
            if (val == "") return;

            data.Input2 = (float)Convert.ToDouble(val);
        }

        private void btnDo_Click(object sender, EventArgs e)
        {
            data.Compute();
            txtOutput.Text = data.GetOutput().ToString();
        }

        private void btnClear_Click(object sender, EventArgs e)
        {
            txtInput1.Text = "";
            txtInput2.Text = "";
            txtOutput.Text = "";

            data.Clear();
        }
    }
}
