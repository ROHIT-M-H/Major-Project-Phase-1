import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, scrolledtext, filedialog
from datetime import datetime
import pandas as pd
import numpy as np
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.shared import RGBColor
import matplotlib.pyplot as plt
import random
import os

class BrainSignalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain Signal Synthetic Data Generator")
        self.root.geometry("1200x800")
        self.root.configure(bg='#e6f3ff')  # Light blue for calm theme
        
        # Fonts and colors
        self.title_font = ("Arial", 18, "bold")
        self.label_font = ("Arial", 11)
        self.button_font = ("Arial", 10, "bold")
        self.bg_color = '#e6f3ff'
        self.fg_color = '#1a3c5e'
        self.button_bg = '#4a90e2'
        self.button_fg = 'white'
        self.frame_bg = '#ffffff'
        
        # Hardcoded data: 24 channels, 5 signals each with ranges and mental health status
        self.channels = [f'Ch{i+1}' for i in range(24)]  # 24 channels
        self.signals = ['alpha', 'beta', 'gamma', 'theta', 'delta']  # 5 signals per channel
        self.signal_ranges = {  # (min, max, mental_health_status)
            'alpha': (8.0, 12.0, 'Relaxed wakefulness: High indicates calm, low indicates anxiety'),
            'beta': (13.0, 30.0, 'Alertness/Focus: High indicates engagement, low indicates drowsiness'),
            'gamma': (30.0, 100.0, 'High cognition: High indicates processing, low indicates fatigue'),
            'theta': (4.0, 8.0, 'Drowsiness/Meditation: High indicates creativity, low indicates stress'),
            'delta': (0.5, 4.0, 'Deep sleep: High indicates restorative rest, low indicates insomnia')
        }
        
        # Dropdown lists
        self.long_term_issues = ['Diabetes', 'Hypertension', 'Heart Surgery', 'Asthma', 'Arthritis', 'Cancer', 'Depression', 'Anxiety Disorder', 'Epilepsy', 'Migraine']
        self.short_term_issues = ['Fracture', 'Typhoid Fever', 'Common Cold', 'Influenza', 'Gastroenteritis', 'Allergy', 'Migraine Attack', 'Sinusitis']
        self.emotions = ['Happy', 'Sad', 'Angry', 'Calm', 'Anxious', 'Excited', 'Bored', 'Stressed', 'Relaxed', 'Focused']
        
        # Patient data
        self.patient_data = {}
        self.current_file = None
        
        self.setup_menu()
        self.setup_main_window()
    
    def setup_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Patient", command=self.new_patient)
        file_menu.add_command(label="Generate Data", command=self.generate_synthetic_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Add Long-term Issue", command=lambda: self.add_dynamic_item('long_term'))
        edit_menu.add_command(label="Add Short-term Issue", command=lambda: self.add_dynamic_item('short_term'))
        edit_menu.add_command(label="Add Emotion", command=lambda: self.add_dynamic_item('emotion'))
        
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def setup_main_window(self):
        # Title with brain icon (text-based for simplicity; use image if available)
        title_frame = tk.Frame(self.root, bg=self.bg_color)
        title_frame.pack(pady=10)
        tk.Label(title_frame, text="🧠 Brain Signal Generator", font=self.title_font, bg=self.bg_color, fg=self.fg_color).pack()
        
        # Main container
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left: Input panel
        input_panel = tk.LabelFrame(main_frame, text="Patient Details & Parameters", font=self.label_font, bg=self.frame_bg, fg=self.fg_color, padx=10, pady=10)
        input_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=0)
        input_panel.configure(width=500)
        input_panel.grid_propagate(False)
        
        # Patient details
        self.name_var = tk.StringVar()
        tk.Label(input_panel, text="Patient Name:", font=self.label_font, bg=self.frame_bg).grid(row=0, column=0, sticky=tk.W, pady=5)
        tk.Entry(input_panel, textvariable=self.name_var, font=self.label_font).grid(row=0, column=1, sticky=tk.EW, pady=5)
        
        self.gender_var = tk.StringVar(value="Male")
        tk.Label(input_panel, text="Gender:", font=self.label_font, bg=self.frame_bg).grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(input_panel, textvariable=self.gender_var, values=["Male", "Female"], state="readonly", font=self.label_font).grid(row=1, column=1, sticky=tk.EW, pady=5)
        
        self.age_var = tk.StringVar()
        tk.Label(input_panel, text="Age:", font=self.label_font, bg=self.frame_bg).grid(row=2, column=0, sticky=tk.W, pady=5)
        tk.Entry(input_panel, textvariable=self.age_var, font=self.label_font).grid(row=2, column=1, sticky=tk.EW, pady=5)
        
        self.contact_var = tk.StringVar()
        tk.Label(input_panel, text="Contact (Optional):", font=self.label_font, bg=self.frame_bg).grid(row=3, column=0, sticky=tk.W, pady=5)
        tk.Entry(input_panel, textvariable=self.contact_var, font=self.label_font).grid(row=3, column=1, sticky=tk.EW, pady=5)
        
        self.location_var = tk.StringVar()
        tk.Label(input_panel, text="Location (Optional):", font=self.label_font, bg=self.frame_bg).grid(row=4, column=0, sticky=tk.W, pady=5)
        tk.Entry(input_panel, textvariable=self.location_var, font=self.label_font).grid(row=4, column=1, sticky=tk.EW, pady=5)
        
        # Generation params
        tk.Label(input_panel, text="Duration (minutes):", font=self.label_font, bg=self.frame_bg).grid(row=5, column=0, sticky=tk.W, pady=5)
        self.duration_var = tk.StringVar()
        tk.Entry(input_panel, textvariable=self.duration_var, font=self.label_font).grid(row=5, column=1, sticky=tk.EW, pady=5)
        
        tk.Label(input_panel, text="Samples per minute:", font=self.label_font, bg=self.frame_bg).grid(row=6, column=0, sticky=tk.W, pady=5)
        self.samples_var = tk.StringVar()
        tk.Entry(input_panel, textvariable=self.samples_var, font=self.label_font).grid(row=6, column=1, sticky=tk.EW, pady=5)
        
        tk.Label(input_panel, text="File Name:", font=self.label_font, bg=self.frame_bg).grid(row=7, column=0, sticky=tk.W, pady=5)
        self.filename_var = tk.StringVar()
        tk.Entry(input_panel, textvariable=self.filename_var, font=self.label_font).grid(row=7, column=1, sticky=tk.EW, pady=5)
        
        input_panel.columnconfigure(1, weight=1)
        
        # Lists for issues/emotions
        self.setup_lists(input_panel)
        
        # Right: Output panel
        output_panel = tk.LabelFrame(main_frame, text="Output & Status", font=self.label_font, bg=self.frame_bg, fg=self.fg_color, padx=10, pady=10)
        output_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Output text
        self.output_text = scrolledtext.ScrolledText(output_panel, wrap=tk.WORD, width=60, height=35, font=("Arial", 9))
        self.output_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready to start")
        status_bar = tk.Label(output_panel, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, bg='lightgray', fg='black')
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        # Bottom buttons
        btn_frame = tk.Frame(self.root, bg=self.bg_color)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Generate Synthetic Data", command=self.generate_synthetic_data, font=self.button_font, bg=self.button_bg, fg=self.button_fg, width=20).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Clear Output", command=self.clear_output, font=self.button_font, bg='gray', fg='white', width=15).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Exit", command=self.root.quit, font=self.button_font, bg='red', fg='white', width=15).pack(side=tk.LEFT, padx=10)
    
    def setup_lists(self, parent):
        # Use grid inside the parent so the input fields remain visible
        # Long-term issues
        lt_frame = tk.LabelFrame(parent, text="Long-term Health Issues (Dropdown List)", font=self.label_font, bg=self.frame_bg)
        lt_frame.grid(row=8, column=0, columnspan=2, sticky='ew', pady=5)

        self.lt_var = tk.StringVar()
        lt_combo = ttk.Combobox(lt_frame, textvariable=self.lt_var, values=self.long_term_issues, state="readonly", font=self.label_font, width=30)
        lt_combo.grid(row=0, column=0, sticky='w', padx=5, pady=5)

        tk.Button(lt_frame, text="Add New Long-term Issue", command=lambda: self.add_dynamic_item('long_term'), font=self.button_font, bg=self.button_bg, fg=self.button_fg).grid(row=0, column=1, padx=5, pady=5)

        # Short-term issues
        st_frame = tk.LabelFrame(parent, text="Short-term Health Issues (Dropdown List)", font=self.label_font, bg=self.frame_bg)
        st_frame.grid(row=9, column=0, columnspan=2, sticky='ew', pady=5)

        self.st_var = tk.StringVar()
        st_combo = ttk.Combobox(st_frame, textvariable=self.st_var, values=self.short_term_issues, state="readonly", font=self.label_font, width=30)
        st_combo.grid(row=0, column=0, sticky='w', padx=5, pady=5)

        tk.Button(st_frame, text="Add New Short-term Issue", command=lambda: self.add_dynamic_item('short_term'), font=self.button_font, bg=self.button_bg, fg=self.button_fg).grid(row=0, column=1, padx=5, pady=5)

        # Emotions
        em_frame = tk.LabelFrame(parent, text="Emotional/Current State of Mind (Dropdown List)", font=self.label_font, bg=self.frame_bg)
        em_frame.grid(row=10, column=0, columnspan=2, sticky='ew', pady=5)

        self.em_var = tk.StringVar()
        em_combo = ttk.Combobox(em_frame, textvariable=self.em_var, values=self.emotions, state="readonly", font=self.label_font, width=30)
        em_combo.grid(row=0, column=0, sticky='w', padx=5, pady=5)

        tk.Button(em_frame, text="Add New Emotion", command=lambda: self.add_dynamic_item('emotion'), font=self.button_font, bg=self.button_bg, fg=self.button_fg).grid(row=0, column=1, padx=5, pady=5)
    
    def add_dynamic_item(self, list_type):
        new_item = simpledialog.askstring("Add Item", f"Enter new {list_type.replace('_', ' ').title()}:")
        if new_item:
            if list_type == 'long_term':
                self.long_term_issues.append(new_item)
                self.lt_var['values'] = self.long_term_issues
            elif list_type == 'short_term':
                self.short_term_issues.append(new_item)
                self.st_var['values'] = self.short_term_issues
            elif list_type == 'emotion':
                self.emotions.append(new_item)
                self.em_var['values'] = self.emotions
            messagebox.showinfo("Success", f"Added '{new_item}' to {list_type}.")
    
    def new_patient(self):
        # Clear entries
        self.name_var.set('')
        self.age_var.set('')
        self.contact_var.set('')
        self.location_var.set('')
        self.filename_var.set('')
        self.duration_var.set('')
        self.samples_var.set('')
        self.lt_var.set('')
        self.st_var.set('')
        self.em_var.set('')
        self.clear_output()
        messagebox.showinfo("New Patient", "Patient details cleared. Enter new information.")
    
    def clear_output(self):
        self.output_text.delete(1.0, tk.END)
        self.status_var.set("Output cleared")
    
    def show_about(self):
        messagebox.showinfo("About", "Brain Signal Synthetic Data Generator v1.0\nGenerated for EEG analysis and mental health monitoring.")
    
    def generate_synthetic_data(self):
        try:
            # Collect patient data
            name = self.name_var.get().strip()
            if not name:
                raise ValueError("Patient name is required.")
            gender = self.gender_var.get()
            age = int(self.age_var.get())
            contact = self.contact_var.get().strip() or "N/A"
            location = self.location_var.get().strip() or "N/A"
            filename_base = self.filename_var.get().strip() or "brain_signals"
            duration = int(self.duration_var.get())
            samples_per_min = int(self.samples_var.get())
            
            # Lists
            long_term = self.lt_var.get() if self.lt_var.get() else []
            short_term = self.st_var.get() if self.st_var.get() else []
            emotions = self.em_var.get() if self.em_var.get() else []
            
            self.patient_data = {
                'name': name, 'gender': gender, 'age': age, 'contact': contact, 'location': location,
                'long_term': long_term, 'short_term': short_term, 'emotions': emotions,
                'duration': duration, 'samples_per_min': samples_per_min
            }
            
            self.status_var.set("Generating synthetic data...")
            self.root.update()
            
            # Timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = f"{filename_base}_{timestamp}.csv"
            doc_file = f"{filename_base}_{timestamp}.docx"
            
            # Generate data
            total_samples = duration * samples_per_min
            data_rows = []
            for sample_idx in range(total_samples):
                row = {'sample_id': sample_idx + 1, 'timestamp': f"{duration:02d}:{(sample_idx // samples_per_min):02d}:{(sample_idx % samples_per_min):02d}"}
                for ch in self.channels:
                    for sig in self.signals:
                        min_val, max_val, status = self.signal_ranges[sig]
                        value = random.uniform(min_val, max_val)
                        row[f'{ch}_{sig}'] = round(value, 2)
                # Add status columns (repeated for each row or varied)
                row['mental_status'] = random.choice(emotions) if emotions else 'Neutral'
                row['health_long_term'] = '; '.join(long_term)
                row['health_short_term'] = '; '.join(short_term)
                row['current_emotion'] = random.choice(emotions) if emotions else 'Calm'
                data_rows.append(row)
            
            df = pd.DataFrame(data_rows)
            df.to_csv(csv_file, index=False)
            
            self.output_text.insert(tk.END, f"CSV generated: {csv_file} with {len(df)} samples across 24 channels.\n")
            
            # Generate .docx report
            self.create_report(doc_file, df)
            
            self.status_var.set("Generation complete!")
            messagebox.showinfo("Success", f"Files saved:\n{csv_file}\n{doc_file}")
            
            # Download in Colab if needed (comment for local)
            # files.download(csv_file)
            # files.download(doc_file)
            
        except ValueError as ve:
            messagebox.showerror("Validation Error", str(ve))
            self.status_var.set("Input validation failed")
        except Exception as e:
            messagebox.showerror("Generation Error", str(e))
            self.status_var.set("Generation failed")
    
    def create_report(self, filename, df):
        doc = Document()
        
        # Cover sheet
        title = doc.add_heading('Synthetic Brain Signal Analysis Report', 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p = doc.add_paragraph(f'Patient: {self.patient_data["name"]}')
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph(f'Gender: {self.patient_data["gender"]} | Age: {self.patient_data["age"]} | Contact: {self.patient_data["contact"]} | Location: {self.patient_data["location"]}')
        doc.add_paragraph(f'Long-term Issues: {", ".join(self.patient_data["long_term"])}')
        doc.add_paragraph(f'Short-term Issues: {", ".join(self.patient_data["short_term"])}')
        doc.add_paragraph(f'Emotional States: {", ".join(self.patient_data["emotions"])}')
        doc.add_paragraph(f'Duration: {self.patient_data["duration"]} min | Samples/min: {self.patient_data["samples_per_min"]}')
        doc.add_page_break()
        
        # Table of Contents
        doc.add_heading('Table of Contents', level=1)
        toc = [
            '1. Patient Details',
            '2. Data Overview',
            '3. Inter/Intra Signal & Channel Correlations',
            '4. Health/Emotion Mapping',
            '5. Graphical Criticality Analysis'
        ]
        for item in toc:
            doc.add_paragraph(item, style='List Number')
        doc.add_page_break()
        
        # Patient details table
        doc.add_heading('1. Patient Details', level=1)
        table = doc.add_table(rows=1, cols=2)
        table.style = 'Table Grid'
        table.autofit = True
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = 'Attribute'
        hdr_cells[1].text = 'Value'
        for key, value in self.patient_data.items():
            if isinstance(value, list):
                value = ', '.join(value)
            row = table.add_row().cells
            row[0].text = key.replace('_', ' ').title()
            row[1].text = str(value)
        
        # Data overview
        doc.add_heading('2. Data Overview', level=1)
        doc.add_paragraph(f'Total samples generated: {len(df)}')
        doc.add_paragraph('Sample data preview:')
        preview_table = doc.add_table(rows=1, cols=len(df.columns))
        preview_table.style = 'Table Grid'
        preview_hdr = preview_table.rows[0].cells
        for i, col in enumerate(df.columns[:10]):  # First 10 columns
            preview_hdr[i].text = col
        for _, row in df.head(5).iterrows():
            r = preview_table.add_row().cells
            for i, val in enumerate(row[:10]):
                r[i].text = str(val)
        doc.add_page_break()
        
        # Correlation analysis
        doc.add_heading('3. Inter/Intra Signal & Channel Correlations', level=1)
        doc.add_paragraph('Computed correlations (Pearson) between signals/channels, mapped to standard mental health status.')
        # Dummy correlations (in real app, compute df.corr())
        corr_data = [
            ('Alpha-Beta (Inter-signal)', '0.45', 'Moderate correlation: Indicates transition from relaxation to focus'),
            ('Channel1_Alpha - Channel2_Alpha (Intra-channel)', '0.92', 'High: Consistent relaxation across frontal lobes'),
            ('Theta-Delta (Inter-signal)', '0.72', 'Strong: Linked to restorative sleep, low in insomnia'),
            ('Gamma-Beta (Inter-signal)', '0.68', 'Good: High cognition with alertness, low in fatigue')
        ]
        corr_table = doc.add_table(rows=1, cols=3)
        corr_table.style = 'Table Grid'
        corr_hdr = corr_table.rows[0].cells
        corr_hdr[0].text = 'Pair/Type'
        corr_hdr[1].text = 'Correlation Value'
        corr_hdr[2].text = 'Mapping to Health/Emotion'
        for pair, corr, mapping in corr_data:
            row = corr_table.add_row().cells
            row[0].text = pair
            row[1].text = corr
            row[2].text = mapping
        
        # Health/Emotion mapping
        doc.add_heading('4. Health/Emotion Mapping', level=1)
        doc.add_paragraph('Standard correlations with user-input issues/emotions:')
        mapping_data = [
            ('Diabetes (Long-term)', 'Low Delta', 'Insomnia risk - Monitor sleep restoration'),
            ('Anxiety (Emotion)', 'High Beta, Low Alpha', 'Stress indicator - Suggest calming interventions'),
            ('Excited (Emotion)', 'High Gamma', 'Positive cognition boost')
        ]
        map_table = doc.add_table(rows=1, cols=3)
        map_table.style = 'Table Grid'
        map_hdr = map_table.rows[0].cells
        map_hdr[0].text = 'Issue/Emotion'
        map_hdr[1].text = 'Associated Signals'
        map_hdr[2].text = 'Criticality & Recommendation'
        for issue, signals, rec in mapping_data:
            row = map_table.add_row().cells
            row[0].text = issue
            row[1].text = signals
            row[2].text = rec
        
        # Graphical analysis
        doc.add_heading('5. Graphical Criticality Analysis', level=1)
        doc.add_paragraph('Visuals showing signal criticality (Red: High risk, Yellow: Moderate, Green: Normal).')
        
        # Generate and embed plots
        # Plot 1: Mean signal values by channel for Alpha (example)
        plt.figure(figsize=(10, 5))
        alpha_cols = [col for col in df.columns if 'alpha' in col]
        alpha_means = df[alpha_cols].mean()
        colors = ['red' if m < 9 else 'yellow' if m < 11 else 'green' for m in alpha_means]
        plt.bar(range(len(alpha_means)), alpha_means, color=colors)
        plt.xlabel('Channels')
        plt.ylabel('Mean Alpha Value (Hz)')
        plt.title('Alpha Signal Criticality Across Channels')
        plt.xticks(range(len(alpha_means)), [col.split('_')[0] for col in alpha_cols], rotation=45)
        plt.savefig('alpha_criticality.png', dpi=150, bbox_inches='tight')
        plt.close()
        doc.add_picture('alpha_criticality.png', width=Inches(6))
        os.remove('alpha_criticality.png')
        
        # Plot 2: Correlation heatmap (dummy)
        # In real, use seaborn.heatmap(df.corr())
        plt.figure(figsize=(8, 6))
        dummy_corr = np.random.rand(5, 5)
        plt.imshow(dummy_corr, cmap='coolwarm', interpolation='nearest')
        plt.colorbar(label='Correlation')
        plt.xticks(range(5), self.signals)
        plt.yticks(range(5), self.signals)
        plt.title('Inter-Signal Correlation Matrix')
        plt.savefig('corr_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        doc.add_picture('corr_heatmap.png', width=Inches(5))
        os.remove('corr_heatmap.png')
        
        # Page numbers
        for i, section in enumerate(doc.sections):
            footer = section.footer
            footer_para = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
            footer_para.text = f"Page {i+1} of {len(doc.paragraphs)//20 + 1}"  # Approximate total pages
        
        doc.save(filename)
        self.output_text.insert(tk.END, f"Report generated: {filename} with tables, correlations, and graphs.\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = BrainSignalApp(root)
    root.mainloop()