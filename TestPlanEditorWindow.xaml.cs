using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Text.Json;
using System.Windows;

namespace jpmc_genai
{
    public class TestPlanRow
    {
        public string TestCaseId { get; set; }
        public string Step { get; set; }
        public string TestData { get; set; }

        public int GlobalRowNumber { get; set; }
        public int StepNumber { get; set; }  // resets per TestCaseId
    }

    public partial class TestPlanEditorWindow : Window
    {
        private readonly ObservableCollection<TestPlanRow> _rows =
            new ObservableCollection<TestPlanRow>();

        public string EditedTestPlanJson { get; private set; } = null;

        public TestPlanEditorWindow(string testplanJson)
        {
            InitializeComponent();
            TestPlanGrid.ItemsSource = _rows;
            LoadTestPlan(testplanJson);
        }

        private void LoadTestPlan(string json)
        {
            try
            {
                using JsonDocument doc = JsonDocument.Parse(json);
                var root = doc.RootElement;

                _rows.Clear(); // ensure clean start

                var tempList = new List<TestPlanRow>();
                int globalCounter = 1;

                // Load prereq steps
                if (root.TryGetProperty("pretestid_steps", out var pre))
                {
                    foreach (var tc in pre.EnumerateObject())
                    {
                        string tcid = tc.Name;
                        int stepCounter = 1;

                        foreach (var stepEntry in tc.Value.EnumerateObject())
                        {
                            tempList.Add(new TestPlanRow
                            {
                                GlobalRowNumber = globalCounter++,
                                TestCaseId = tcid,
                                StepNumber = stepCounter++,
                                Step = stepEntry.Name,
                                TestData = stepEntry.Value.GetString() ?? ""
                            });
                        }
                    }
                }

                // Load current testcase steps
                string currentId = root.GetProperty("current_testid").GetString();
                int currentStepCounter = 1;

                if (root.TryGetProperty("current_bdd_steps", out var cur))
                {
                    foreach (var stepEntry in cur.EnumerateObject())
                    {
                        tempList.Add(new TestPlanRow
                        {
                            GlobalRowNumber = globalCounter++,
                            TestCaseId = currentId,
                            StepNumber = currentStepCounter++,
                            Step = stepEntry.Name,
                            TestData = stepEntry.Value.GetString() ?? ""
                        });
                    }
                }

                // Now assign to ObservableCollection (triggers UI update once)
                foreach (var row in tempList)
                    _rows.Add(row);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Failed to load testplan: {ex.Message}");
                Close();
            }
        }

        private void ContinueButton_Click(object sender, RoutedEventArgs e)
        {
            // Group rows by testcaseid
            var grouped = new Dictionary<string, Dictionary<string, string>>();

            foreach (var row in _rows)
            {
                if (!grouped.ContainsKey(row.TestCaseId))
                    grouped[row.TestCaseId] = new Dictionary<string, string>();

                grouped[row.TestCaseId][row.Step] = row.TestData ?? "";
            }

            // Identify current_testid (assumed last group)
            string currentId = _rows[^1].TestCaseId;

            // Rebuild testplan dict
            var pretest = new Dictionary<string, Dictionary<string, string>>();
            var currentSteps = new Dictionary<string, string>();

            foreach (var kv in grouped)
            {
                if (kv.Key == currentId)
                    currentSteps = kv.Value;     // current steps
                else
                    pretest[kv.Key] = kv.Value;  // prereq steps
            }

            var output = new Dictionary<string, object>
            {
                { "pretestid_steps", pretest },
                { "current_testid", currentId },
                { "current_bdd_steps", currentSteps }
            };

            EditedTestPlanJson = JsonSerializer.Serialize(output);
            DialogResult = true;
            Close();
        }

        private void SkipButton_Click(object sender, RoutedEventArgs e)
        {
            EditedTestPlanJson = null;
            DialogResult = true;
            Close();
        }

        private void CancelButton_Click(object sender, RoutedEventArgs e)
        {
            EditedTestPlanJson = null;
            DialogResult = false;
            Close();
        }
    }
}
