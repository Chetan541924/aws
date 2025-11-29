using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;

namespace jpmc_genai
{
    public partial class ReusableMethodViewWindow : Window
    {
        public ReusableMethodViewWindow(List<ReusableMethodDto> methods)
        {
            InitializeComponent();

            // Bind Data
            ReusableMethodsGrid.ItemsSource = methods;
        }

        private void ReusableMethodsGrid_DoubleClick(object sender, MouseButtonEventArgs e)
        {
            if (ReusableMethodsGrid.SelectedItem is not ReusableMethodDto selected)
                return;

            if (string.IsNullOrWhiteSpace(selected.method_code))
            {
                MessageBox.Show("No method code available.", "Info", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            // Create popup window for code display
            var win = new Window
            {
                Title = $"{selected.method_name}() — Full Method Code",
                Width = 800,
                Height = 600,
                Background = Brushes.Black,
                Foreground = Brushes.White
            };

            var box = new TextBox
            {
                Text = selected.method_code,
                AcceptsReturn = true,
                AcceptsTab = true,
                IsReadOnly = true,
                VerticalScrollBarVisibility = ScrollBarVisibility.Auto,
                HorizontalScrollBarVisibility = ScrollBarVisibility.Auto,
                Background = Brushes.Black,
                Foreground = Brushes.Lime,
                FontFamily = new FontFamily("Consolas"),
                FontSize = 14,
                Margin = new Thickness(10)
            };

            win.Content = box;
            win.ShowDialog();
        }
    }
}
