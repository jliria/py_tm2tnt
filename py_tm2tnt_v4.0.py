import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import csv
from scipy.stats import sem, bootstrap, f_oneway, kruskal, shapiro, levene
from itertools import combinations
from collections import defaultdict

class py_tm2tntApp:
    def __init__(self, root):
        self.root = root
        self.root.title("py_tm2tnt 4.0")
        
        self.traditional_measurements = defaultdict(list)
        self.species_count = {}
        self.interval_type = tk.StringVar(value="CI")
        self.confidence_level = tk.DoubleVar(value=0.95)
        self.analysis_results = []
        
        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Load CSV File Button
        self.load_csv_button = ttk.Button(main_frame, text="Load CSV File", command=self.load_csv_file)
        self.load_csv_button.grid(row=0, column=0, padx=5, pady=5)
        
        # Load Specimen Counts Button
        self.load_counts_button = ttk.Button(main_frame, text="Load Specimen Counts", command=self.load_specimen_counts, state="disabled")
        self.load_counts_button.grid(row=1, column=0, padx=5, pady=5)

        # Interval Type Option
        ttk.Label(main_frame, text="Interval Type:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.interval_options = ttk.Combobox(main_frame, textvariable=self.interval_type, values=["CI", "Mean ± SE"], state='readonly')
        self.interval_options.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)

        # Confidence Level Entry
        ttk.Label(main_frame, text="Confidence Level:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.confidence_entry = ttk.Entry(main_frame, textvariable=self.confidence_level)
        self.confidence_entry.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)

        # Calculate Intervals Button
        self.calculate_button = ttk.Button(main_frame, text="Calculate Intervals", command=self.calculate_intervals, state="disabled")
        self.calculate_button.grid(row=4, column=0, padx=5, pady=5)

        # Export to TNT Button
        self.export_button = ttk.Button(main_frame, text="Export to TNT", command=self.export_to_tnt, state="disabled")
        self.export_button.grid(row=5, column=0, padx=5, pady=5)

        # Perform Statistical Analysis Button
        self.analysis_button = ttk.Button(main_frame, text="Perform Statistical Analysis", command=self.perform_statistical_analysis, state="disabled")
        self.analysis_button.grid(row=6, column=0, padx=5, pady=5)

        # Export Analysis Results Button
        self.export_analysis_button = ttk.Button(main_frame, text="Export Analysis Results", command=self.save_statistical_results, state="disabled")
        self.export_analysis_button.grid(row=7, column=0, padx=5, pady=5)

    def load_csv_file(self):
        csv_file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if not csv_file_path:
            return

        self.traditional_measurements.clear()
        try:
            with open(csv_file_path, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    try:
                        species = row[0].strip()
                        measurements = list(map(float, row[1:]))
                        self.traditional_measurements[species].append(measurements)
                    except ValueError:
                        messagebox.showwarning("Warning", f"Invalid data in row: {row}")
                        continue

            messagebox.showinfo("Success", "CSV file loaded successfully.")
            self.load_counts_button.state(['!disabled'])

        except Exception as e:
            messagebox.showerror("Error", f"Could not load CSV file: {e}")

    def load_specimen_counts(self):
        counts_file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if not counts_file_path:
            return

        self.species_count.clear()
        try:
            with open(counts_file_path, 'r') as file:
                for line in file:
                    species, count = line.strip().split()
                    self.species_count[species] = int(count)

            messagebox.showinfo("Success", "Specimen counts loaded successfully.")
            self.calculate_button.state(['!disabled'])
            self.analysis_button.state(['!disabled'])

        except Exception as e:
            messagebox.showerror("Error", f"Could not load specimen counts: {e}")
    
    def calculate_intervals(self):
        try:
            confidence_level = float(self.confidence_level.get())
            self.calculated_intervals = {}

            for species, measurements in self.traditional_measurements.items():
                if species not in self.species_count:
                    continue

                count = self.species_count.get(species, 0)
                measurements = np.array(measurements)[:count]

                if measurements.size == 0:
                    continue

                if self.interval_type.get() == "CI":
                    ci_low, ci_high = [], []
                    for i in range(measurements.shape[1]):
                        res = bootstrap((measurements[:, i],), np.mean, confidence_level=confidence_level)
                        ci_low.append(res.confidence_interval.low)
                        ci_high.append(res.confidence_interval.high)
                    self.calculated_intervals[species] = [f"{low:.6f}-{high:.6f}" for low, high in zip(ci_low, ci_high)]
                else:
                    mean_values = np.mean(measurements, axis=0)
                    se_values = sem(measurements, axis=0)
                    min_values = mean_values - se_values
                    max_values = mean_values + se_values
                    self.calculated_intervals[species] = [f"{min_val:.6f}-{max_val:.6f}" for min_val, max_val in zip(min_values, max_values)]

            messagebox.showinfo("Success", "Intervals calculated successfully.")
            self.export_button.state(['!disabled'])

        except Exception as e:
            messagebox.showerror("Error", f"Could not calculate intervals: {e}")

    def perform_statistical_analysis(self):
        significant_results = []
        num_characters = len(next(iter(self.traditional_measurements.values()))[0])

        for char_index in range(num_characters):
            data_by_species = {species: [m[char_index] for m in measurements]
                               for species, measurements in self.traditional_measurements.items()}
        
            # Normality test
            normal_data = all(shapiro(values)[1] > 0.05 for values in data_by_species.values() if len(values) > 1)
        
            # Variance test
            if normal_data:
                homogeneity_of_variance = levene(*data_by_species.values())[1] > 0.05
            else:
                homogeneity_of_variance = False

            # Statistical test selection
            if normal_data and homogeneity_of_variance:
                f_stat, p_value = f_oneway(*data_by_species.values())
                test_used = "ANOVA"
            else:
                h_stat, p_value = kruskal(*data_by_species.values())
                test_used = "Kruskal-Wallis"

            # Small p values
            if p_value < 0.0001:
                p_value_str = f"{p_value:.2e}"
            else:
                p_value_str = f"{p_value:.6f}"

            species = list(data_by_species.keys())
            num_comparisons = len(list(combinations(species, 2)))  # Num comparisons
            corrected_alpha = 0.05 / num_comparisons  # Bonferroni correction

            # Paired comparisons with Bonferroni corrections
            pairwise_results = []
            for (sp1, sp2) in combinations(species, 2):
                try:
                    _, pair_p = f_oneway(data_by_species[sp1], data_by_species[sp2]) \
                        if test_used == "ANOVA" else kruskal(data_by_species[sp1], data_by_species[sp2])
                    if pair_p < corrected_alpha:
                        pairwise_results.append(f"{sp1}-{sp2}")
                except:
                    continue  # Ignore errors in comparisons

            significant_results.append({
                "Character": char_index + 1,
                "Test": test_used,
                "p-value": p_value_str,
                "Significant Pairs": ", ".join(pairwise_results) if pairwise_results else "None"
            })

        self.analysis_results = significant_results
        self.export_analysis_button.state(['!disabled'])
        messagebox.showinfo("Analysis Completed", "Statistical analysis completed successfully.")

    def save_statistical_results(self):
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if not save_path:
            return
    
        try:
            with open(save_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Character", "Test", "p-value", "Significant Pairs"])
            
                for result in self.analysis_results:
                    character = result["Character"]
                    test = result["Test"]
                    p_value = result["p-value"]  # This is already a string, so no additional formatting needed
                
                    # Convert significant pairs to a string separated by line breaks for better readability
                    if result["Significant Pairs"] != "None":
                        significant_pairs = "\n".join(result["Significant Pairs"].split(", "))
                    else:
                        significant_pairs = "None"

                    # Write the row with the formatted significant pairs
                    writer.writerow([character, test, p_value, significant_pairs])
        
            messagebox.showinfo("Export Completed", "Statistical results exported successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not export results: {e}")

    def export_to_tnt(self):
        save_tnt_path = filedialog.asksaveasfilename(defaultextension=".tnt", filetypes=[("TNT Files", "*.tnt"), ("All Files", "*.*")])
        if not save_tnt_path:
            return

        try:
            num_species = len(self.calculated_intervals)
            num_characters = len(next(iter(self.calculated_intervals.values()))) if num_species > 0 else 0
            
            with open(save_tnt_path, 'w') as file:
                file.write("nstates cont;\n")
                file.write("nstates 32;\n")
                file.write(f"xread 'Traditional Measurements TNT Export'\n")
                file.write(f"{num_characters} {num_species}\n")
                
                file.write("&[cont]\n")
                for species, intervals in self.calculated_intervals.items():
                    species_name = species.replace(" ", "_")
                    intervals_str = " ".join(intervals)
                    file.write(f"{species_name} {intervals_str}\n")
                file.write("\n;\n")
                file.write("proc/;\n")

            messagebox.showinfo("Export Completed", "TNT file exported successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not export to TNT: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = py_tm2tntApp(root)
    root.mainloop()
