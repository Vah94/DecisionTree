using System;
using System.Collections.Generic;
using System.Data;
using System.Drawing;
using System.Globalization;
using System.Linq;

namespace DecisionTree
{
    [Serializable]
    public class DecisionTreeId3
    {
        public int FeaturesCount => ColumnsName.Count - 1;
        public List<string> ColumnsName;
        public DecisionNode Root;

        public DecisionTreeId3()
        {
        }

        public DecisionTreeId3(DataTable trainingData)
        {
            ColumnsName = new List<string>();
            foreach (DataColumn column in trainingData.Columns)
            {
                ColumnsName.Add(column.ColumnName);
            }

            Root = BuildTree(trainingData);
        }


        private static void DebugLog(string log, ConsoleColor color = ConsoleColor.White)
        {
            Console.ForegroundColor = color;
            Console.WriteLine(log);
            Console.ResetColor();
        }
        
        private static void DebugLogWarning(string log)
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine(log);
            Console.ResetColor();
        }
        
        private static void DebugLogError(string log)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine(log);
            Console.ResetColor();
        }

        public Answer GetAnswer(Dictionary<string, string> dataRow)
        {
            if (dataRow == null)
            {
                DebugLogWarning("dataRow is null!");
                return null;
            }

            if (dataRow.Any(item => !ColumnsName.Contains(item.Key)))
            {
                DebugLogError("Wrong Data input!");
                return null;
            }

            if (dataRow.Count != FeaturesCount)
            {
                DebugLogError("FeaturesCount is incorrect: " + FeaturesCount + "/" + dataRow.Count);
                return null;
            }

            var leaf = Classify(dataRow, Root);
            var predictions = leaf.Predictions;
            switch (predictions.Count)
            {
                case 0:
                    return null;
                case 1:
                    return new Answer(predictions.First().Key, 100);
                default:
                    DebugLogWarning("Leaf: " + PrintLeaf(leaf));
                    var temp = predictions.Select(item => new Result(item.Key, item.Value)).ToList();
                    var orderByDescending = temp.OrderByDescending(result => result.Count);
                    var finalResult = orderByDescending.First();

                    var sumOfReasons = temp.Sum(item => item.Count);
                    var resultPercent = (int) (((double) finalResult.Count / (double) sumOfReasons) * 100);
                    DebugLogWarning("Answer success rate " + resultPercent + "%!");

                    return new Answer(finalResult.ResultName, resultPercent);
            }
        }

        public class Answer
        {
            public readonly string Reason;
            public readonly int Percent;

            public Answer(string reason, int percent)
            {
                Reason = reason;
                Percent = percent;
            }
        }

        private class Result
        {
            public readonly string ResultName;
            public readonly int Count;

            public Result(string resultName, int count)
            {
                ResultName = resultName;
                Count = count;
            }
        }

        /// <summary>
        /// Counts the number of each type of example in a dataset.
        /// </summary>
        private static Dictionary<string, int> GetResultsMap(DataTable dataTable)
        {
            var classCounts = new Dictionary<string, int>();
            foreach (DataRow row in dataTable.Rows)
            {
                var result = row[dataTable.Columns[dataTable.Columns.Count - 1]].ToString();
                if (!classCounts.ContainsKey(result))
                {
                    classCounts.Add(result, 1);
                }
                else
                {
                    classCounts[result] += 1;
                }
            }

            return classCounts;
        }

        /// <summary>
        /// Test if a value is numeric.
        /// </summary>
        private static bool IsNumeric(object value)
        {
            return GetNumeric(value.ToString()).Item1;
        }

        private static Tuple<bool, double> GetNumeric(string value)
        {
            var isNumber = double.TryParse(
                value,
                NumberStyles.Number,
                CultureInfo.InvariantCulture,
                out var number);
            return new Tuple<bool, double>(isNumber, number);
        }

        /// <summary>
        /// A Question is used to partition a dataset.
        /// This class just records a 'column number' (e.g., 0 for Color) and a
        /// 'column value' (e.g., Green). The 'match' method is used to compare
        /// the feature value in an example to the feature value stored in the
        /// question. See the demo below.
        /// </summary>
        [Serializable]
        public class Question
        {
            public string ColumnName;
            public string Value;

            public Question(string columnName, string value)
            {
                ColumnName = columnName;
                Value = value;
            }

            public bool IsMatch(DataRow dataRow)
            {
                return IsMatch(dataRow[ColumnName].ToString());
            }

            public bool IsMatch(Dictionary<string, string> dataRow)
            {
                return IsMatch(dataRow[ColumnName]);
            }

            public bool IsMatch(string val)
            {
                if (IsNumeric(val))
                {
                    return GetNumeric(val.ToString()).Item2 >= GetNumeric(Value).Item2;
                }
                else
                {
                    return val.ToString() == Value;
                }
            }

            public override string ToString()
            {
                var condition = IsNumeric(Value) ? ">=" : "==";


                return "Is: " + ColumnName + " " + condition + " " + Value;
            }
        }

        /// <summary>
        /// Partitions a dataset.
        /// For each row in the dataset, check if it matches the question. If
        /// so, add it to 'true rows', otherwise, add it to 'false rows'.
        /// </summary>
        private Tuple<DataTable, DataTable> GetPartitions(DataTable dataTable, Question question)
        {
            var trueRows = dataTable.Clone();
            var falseRows = dataTable.Clone();

            foreach (DataRow row in dataTable.Rows)
            {
                if (question.IsMatch(row))
                {
                    trueRows.Rows.Add(row.ItemArray);
                }
                else
                {
                    falseRows.Rows.Add(row.ItemArray);
                }
            }

            return new Tuple<DataTable, DataTable>(trueRows, falseRows);
        }

        /// <summary>
        /// Calculate the Gini Impurity for a list of rows
        /// There are a few different ways to do this, I thought this one was
        /// the most concise. See:
        /// https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
        /// </summary>
        private double GetGiniImpurity(DataTable dataTable)
        {
            var counts = GetResultsMap(dataTable);
            var impurity = 1.0;
            foreach (var result in counts)
            {
                var prob_of_lbl = (double) result.Value / (double) dataTable.Rows.Count;
                impurity -= Math.Pow(prob_of_lbl, 2);
            }

            return impurity;
        }

        /// <summary>
        /// The uncertainty of the starting node, minus the weighted impurity of two child nodes.
        /// </summary>
        private double GetInfoGain(DataTable left, DataTable right, double current_uncertainty)
        {
            var p = (double) (left.Rows.Count) / (left.Rows.Count + right.Rows.Count);
            return current_uncertainty - p * GetGiniImpurity(left) - (1 - p) * GetGiniImpurity(right);
        }

        /// <summary>
        /// Find the best question to ask by iterating over every feature / value and calculating the information gain.
        /// </summary>
        private Tuple<double, Question> GetBestSplit(DataTable dataTable)
        {
            var bestGain = 0.0;
            Question bestQuestion = null;
            var currentUncertainty = GetGiniImpurity(dataTable);
            var featuresCount = dataTable.Columns.Count - 1;
            for (int i = 0; i < featuresCount; i++)
            {
                var currentColumnName = dataTable.Columns[i].ColumnName;
                var values = new List<string>();
                foreach (DataRow row in dataTable.Rows)
                {
                    var temp = row[currentColumnName].ToString();
                    if (!values.Contains(temp))
                    {
                        values.Add(temp);
                    }
                }

                foreach (var val in values)
                {
                    var question = new Question(currentColumnName, val);
                    var partitions = GetPartitions(dataTable, question);
                    var trueRows = partitions.Item1;
                    var falseRows = partitions.Item2;

                    if (trueRows.Rows.Count == 0 || falseRows.Rows.Count == 0)
                    {
                        continue;
                    }

                    var gain = GetInfoGain(trueRows, falseRows, currentUncertainty);
                    if (gain >= bestGain)
                    {
                        bestGain = gain;
                        bestQuestion = question;
                    }
                }
            }

            return new Tuple<double, Question>(bestGain, bestQuestion);
        }

        /// <summary>
        /// A Decision Node asks a question.
        /// This holds a reference to the question, and to the two child nodes.
        /// </summary>
        [Serializable]
        public class DecisionNode
        {
            public Question Question;
            public DecisionNode TrueBranch;
            public DecisionNode FalseBranch;
            public Dictionary<string, int> Predictions;

            public DecisionNode()
            {
            }

            public DecisionNode(
                Question question = null,
                DecisionNode trueBranch = null,
                DecisionNode falseBranch = null)
            {
                Question = question;
                TrueBranch = trueBranch;
                FalseBranch = falseBranch;
                Predictions = null;
            }

            /// <summary>
            /// A Leaf node classifies data.
            /// This holds a dictionary of class (e.g., "Apple") -> number of times
            /// it appears in the rows from the training data that reach this leaf.
            /// </summary>
            public DecisionNode(DataTable dataTable)
            {
                Predictions = GetResultsMap(dataTable);
            }

            public bool IsLeaf()
            {
                return Predictions != null;
            }

            public override string ToString()
            {
                var startSymbol = "{";
                var finalText = startSymbol;
                foreach (var prediction in Predictions)
                {
                    if (finalText != startSymbol)
                    {
                        finalText += ",";
                    }

                    finalText += "\'" + prediction.Key + ": " + prediction.Value + " ";
                }

                finalText += "}";

                return "Predict " + finalText;
            }
        }

        /// <summary>
        /// Builds the tree.
        /// Rules of recursion: 1) Believe that it works. 2) Start by checking
        /// for the base case (no further information gain). 3) Prepare for
        /// giant stack traces.
        /// </summary>
        private DecisionNode BuildTree(DataTable dataTable)
        {
            var bestSplit = GetBestSplit(dataTable);
            var gain = bestSplit.Item1;
            var question = bestSplit.Item2;
            if (gain == 0)
            {
                return new DecisionNode(dataTable);
            }

            var partitions = GetPartitions(dataTable, question);
            var trueRows = partitions.Item1;
            var falseRows = partitions.Item2;

            var trueBranch = BuildTree(trueRows);
            var falseBranch = BuildTree(falseRows);
            return new DecisionNode(question, trueBranch, falseBranch);
        }

        /// <summary>
        /// World's most elegant tree printing function.
        /// </summary>
        public static void PrintTree(DecisionNode node, string spacing = "")
        {
            if (node.IsLeaf())
            {
                DebugLog(spacing + node, ConsoleColor.Green);
                return;
            }

            DebugLog(spacing + node.Question);

            DebugLog(spacing + "--> True:");
            PrintTree(node.TrueBranch, spacing + "  ");

            DebugLog(spacing + "--> False:");
            PrintTree(node.FalseBranch, spacing + "  ");
        }

        /// <summary>
        /// Decide whether to follow the true-branch or the false-branch.
        /// Compare the feature / value stored in the node,
        /// to the example we're considering.
        /// </summary>
        private static DecisionNode Classify(Dictionary<string, string> dataRow, DecisionNode node)
        {
            if (node.IsLeaf())
            {
                return node;
            }

            if (node.Question.IsMatch(dataRow))
            {
                return Classify(dataRow, node.TrueBranch);
            }
            else
            {
                return Classify(dataRow, node.FalseBranch);
            }
        }

        /// <summary>
        /// A nicer way to print the predictions at a leaf.
        /// </summary>
        private static string PrintLeaf(DecisionNode leaf)
        {
            var predictions = leaf.Predictions;
            var total = predictions.Values.Sum() * 1.0;

            var startSymbol = "{";
            var finalText = startSymbol;
            foreach (var prediction in predictions)
            {
                if (finalText != startSymbol)
                {
                    finalText += ",";
                }

                finalText += "\'" + prediction.Key + ": " + (((double) prediction.Value / total) * 100) + "% ";
            }

            finalText += "}";
            return finalText;
        }
    }
}