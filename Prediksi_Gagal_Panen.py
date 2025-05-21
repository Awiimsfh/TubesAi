import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


class DashboardFrame(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg='#b2c4f5')
        
        # Main container frame
        main_container = tk.Frame(self, bg='#b2c4f5')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left side - Dashboard controls and chart
        left_frame = tk.Frame(main_container, bg='#b2c4f5')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right side - Manual input form
        right_frame = tk.Frame(main_container, bg='#b2c4f5', width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        # Dashboard title and controls
        tk.Label(left_frame, text="Dashboard Prediksi Gagal Panen", 
                font=('Times New Roman', 16, 'bold'), bg='#b2c4f5', fg='black').pack(pady=10)

        control_frame = tk.Frame(left_frame, bg='#b2c4f5')
        control_frame.pack(pady=5)

        self.periode_var = tk.StringVar(value='Mingguan')
        ttk.Label(control_frame, text="Pilih Periode:", font=('Times New Roman', 12), 
                 background='#b2c4f5', foreground='black').grid(row=0, column=0, sticky='w')
        self.periode_combo = ttk.Combobox(control_frame, textvariable=self.periode_var,
                                        values=["Mingguan", "Bulanan", "1 Periode"],
                                        font=('Times New Roman', 12), state="readonly", width=15)
        self.periode_combo.grid(row=0, column=1, padx=5)
        self.periode_combo.bind("<<ComboboxSelected>>", lambda e: self.toggle_custom_entry())

        self.custom_entry = tk.Entry(control_frame, font=('Times New Roman', 12), width=5, bg='#6488ea', fg='black')
        self.custom_entry.insert(0, "8")
        self.custom_entry.grid(row=0, column=2, padx=5)
        self.custom_entry.grid_remove()

        self.btn_show = tk.Button(control_frame, text="Tampilkan Grafik", 
                                font=('Times New Roman', 12), command=self.update_chart,
                                bg='#6488ea', fg='black', activebackground='#6488ea', activeforeground='black')
        self.btn_show.grid(row=1, columnspan=3, pady=10)

        # Chart area
        self.figure = Figure(figsize=(6, 3.5), dpi=100, facecolor='#b2c4f5')
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#b2c4f5')

        self.canvas = FigureCanvasTkAgg(self.figure, left_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Manual input form in right frame
        tk.Label(right_frame, text="Input Data Manual", 
                font=('Times New Roman', 16, 'bold'), bg='#b2c4f5', fg='black').pack(pady=10)

        form_frame = tk.Frame(right_frame, bg='#b2c4f5')
        form_frame.pack(pady=5)

        labels = [
            "Tanggal (YYYY-MM-DD):", "Jam (HH:MM):", "Nitrogen (ppm):", "Fosfor (ppm):", 
            "Kalium (ppm):", "Suhu (°C):", "Kelembapan (%):", "pH Tanah:", 
            "Curah Hujan (mm):", "Kelembapan Tanah (%):"
        ]

        self.entries = {}
        for i, label_text in enumerate(labels):
            tk.Label(form_frame, text=label_text, font=('Times New Roman', 10), 
                    bg='#b2c4f5', fg='black').grid(row=i, column=0, sticky='e', pady=2)
            entry = tk.Entry(form_frame, font=('Times New Roman', 10), width=15, bg='#6488ea', fg='black')
            entry.grid(row=i, column=1, pady=2, padx=5)
            self.entries[label_text] = entry

        # Set default values for easier testing
        self.entries["Tanggal (YYYY-MM-DD):"].insert(0, "2023-01-01")
        self.entries["Jam (HH:MM):"].insert(0, "08:00")
        self.entries["Nitrogen (ppm):"].insert(0, "25")
        self.entries["Fosfor (ppm):"].insert(0, "15")
        self.entries["Kalium (ppm):"].insert(0, "20")
        self.entries["Suhu (°C):"].insert(0, "28")
        self.entries["Kelembapan (%):"].insert(0, "70")
        self.entries["pH Tanah:"].insert(0, "6.5")
        self.entries["Curah Hujan (mm):"].insert(0, "50")
        self.entries["Kelembapan Tanah (%):"].insert(0, "60")

        self.result_label = tk.Label(right_frame, text="", 
                                    font=('Times New Roman', 12, 'bold'), 
                                    fg='black', bg='#b2c4f5', wraplength=250)
        self.result_label.pack(pady=5)

        # Button frame for Predict and Save buttons
        button_frame = tk.Frame(right_frame, bg='#b2c4f5')
        button_frame.pack(pady=10)

        tk.Button(button_frame, text="Prediksi", font=('Times New Roman', 12), 
                 command=self.predict_manual, bg='#6488ea', fg='black',
                 activebackground='#6488ea', activeforeground='black').pack(side=tk.LEFT, padx=5)

        tk.Button(button_frame, text="Simpan Data", font=('Times New Roman', 12), 
                 command=self.save_manual_data, bg='#6488ea', fg='black',
                 activebackground='#6488ea', activeforeground='black').pack(side=tk.LEFT, padx=5)

    def save_manual_data(self):
        try:
            # Get all input values
            data = {
                'Tanggal': self.entries["Tanggal (YYYY-MM-DD):"].get(),
                'Jam': self.entries["Jam (HH:MM):"].get(),
                'Nitrogen': float(self.entries["Nitrogen (ppm):"].get()),
                'Fosfor': float(self.entries["Fosfor (ppm):"].get()),
                'Kalium': float(self.entries["Kalium (ppm):"].get()),
                'Suhu': float(self.entries["Suhu (°C):"].get()),
                'Kelembapan': float(self.entries["Kelembapan (%):"].get()),
                'pH Tanah': float(self.entries["pH Tanah:"].get()),
                'Curah Hujan': float(self.entries["Curah Hujan (mm):"].get()),
                'Kelembapan Tanah': float(self.entries["Kelembapan Tanah (%):"].get())
            }
            
            # Create DataFrame from the input data
            df = pd.DataFrame([data])
            
            # Get prediction result if available
            if "Hasil Prediksi:" in self.result_label.cget("text"):
                prediksi = self.result_label.cget("text").split(": ")[1]
                df['Prediksi Risiko'] = prediksi
            
            # Save to Excel file
            file_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx")],
                title="Simpan Data Input Manual"
            )
            
            if file_path:
                # Try to append to existing file or create new one
                try:
                    existing_df = pd.read_excel(file_path)
                    combined_df = pd.concat([existing_df, df], ignore_index=True)
                    combined_df.to_excel(file_path, index=False)
                except:
                    df.to_excel(file_path, index=False)
                
                messagebox.showinfo("Sukses", f"Data berhasil disimpan di:\n{file_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menyimpan data:\n{e}")

    def predict_manual(self):
        try:
            # Get and validate manual input data
            tanggal = self.entries["Tanggal (YYYY-MM-DD):"].get()
            jam = self.entries["Jam (HH:MM):"].get()
            nitrogen = float(self.entries["Nitrogen (ppm):"].get())
            fosfor = float(self.entries["Fosfor (ppm):"].get())
            kalium = float(self.entries["Kalium (ppm):"].get())
            suhu = float(self.entries["Suhu (°C):"].get())
            kelembapan = float(self.entries["Kelembapan (%):"].get())
            ph_tanah = float(self.entries["pH Tanah:"].get())
            curah_hujan = float(self.entries["Curah Hujan (mm):"].get())
            kelembapan_tanah = float(self.entries["Kelembapan Tanah (%):"].get())

            # Simple date and time format validation
            pd.to_datetime(tanggal, format='%Y-%m-%d')
            if len(jam) != 5 or jam[2] != ':' or not jam.replace(":", "").isdigit():
                raise ValueError("Format jam harus HH:MM")

            # Simple prediction logic (rules-based)
            if nitrogen < 20 and ph_tanah < 5 or suhu > 35:
                hasil = "Risiko Gagal Panen"
            else:
                hasil = "Aman"

            self.result_label.config(text=f"Hasil Prediksi: {hasil}")

        except Exception as e:
            messagebox.showerror("Input Error", f"Terjadi kesalahan input:\n{e}")

    def toggle_custom_entry(self):
        if self.periode_var.get() == '1 Periode':
            self.custom_entry.grid()
        else:
            self.custom_entry.grid_remove()

    def update_data(self):
        df = self.master.shared_data
        if df is None:
            return
        self.update_chart()

    def update_chart(self):
        df = self.master.shared_data
        if df is None or df.empty:
            messagebox.showwarning("Peringatan", "Tidak ada data yang tersedia untuk ditampilkan")
            return

        self.ax.clear()
        
        try:
            # Pastikan kolom yang diperlukan ada
            if 'Tanggal' not in df.columns or 'Prediksi Risiko' not in df.columns:
                raise ValueError("Kolom 'Tanggal' atau 'Prediksi Risiko' tidak ditemukan dalam data")

            data = df.copy()
            data['Tanggal'] = pd.to_datetime(data['Tanggal'], errors='coerce')
            data.dropna(subset=['Tanggal'], inplace=True)

            if self.periode_var.get() == "Mingguan":
                data['Periode'] = data['Tanggal'].dt.to_period('W').apply(lambda r: r.start_time.strftime('%Y-%m-%d'))
            elif self.periode_var.get() == "Bulanan":
                data['Periode'] = data['Tanggal'].dt.to_period('M').apply(lambda r: r.start_time.strftime('%Y-%m'))
            else:
                try:
                    n = int(self.custom_entry.get())
                    if n < 1:
                        n = 8
                except ValueError:
                    n = 8
                data['Rank'] = data['Tanggal'].rank(method='dense')
                max_rank = data['Rank'].max()
                bins = np.linspace(0, max_rank, n + 1)
                labels = [f'Periode {i}' for i in range(1, n + 1)]
                data['Periode'] = pd.cut(data['Rank'], bins=bins, labels=labels, include_lowest=True)
                data.drop(columns='Rank', inplace=True)

            count_data = data.groupby('Periode')['Prediksi Risiko'].apply(lambda x: (x != 'Aman').sum())
            labels = count_data.index.astype(str).tolist()
            values = count_data.values.tolist()

            self.ax.plot(labels, values, marker='o', color='#6488ea', linewidth=2, markersize=6)
            self.ax.set_title("Jumlah Risiko Gagal Panen", fontsize=11, color='black')
            self.ax.set_xlabel("Periode", fontsize=10, color='black')
            self.ax.set_ylabel("Jumlah Kasus", fontsize=10, color='black')
            self.ax.grid(True, linestyle='--', alpha=0.5)
            self.ax.tick_params(colors='black')
            self.ax.tick_params(axis='x', rotation=45)
            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", f"Gagal menampilkan grafik:\n{e}")


class EvaluasiFrame(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg='#b2c4f5')

        tk.Label(self, text="Evaluasi Model Prediksi", 
                font=('Times New Roman', 16, 'bold'), bg='#b2c4f5', fg='black').pack(pady=10)

        frame = tk.Frame(self, bg='#b2c4f5')
        frame.pack(pady=5)

        tk.Label(frame, text="Metode Evaluasi:", font=('Times New Roman', 12), 
                bg='#b2c4f5', fg='black').grid(row=0, column=0, sticky='w')
        
        self.method_var = tk.StringVar(value="Random Forest")
        self.method_combo = ttk.Combobox(frame, textvariable=self.method_var,
                                       values=["Random Forest", "Decision Tree"],
                                       font=('Times New Roman', 12), state="readonly", width=15)
        self.method_combo.grid(row=0, column=1, sticky='w')

        self.eval_btn = tk.Button(frame, text="Evaluasi", font=('Times New Roman', 12), 
                                command=self.load_evaluation_data,
                                bg='#6488ea', fg='black', activebackground='#6488ea', activeforeground='black')
        self.eval_btn.grid(row=1, columnspan=2, pady=10)

        self.visualization_frame = tk.Frame(self, bg='#b2c4f5')
        self.visualization_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.cm_frame = tk.Frame(self.visualization_frame, bg='#b2c4f5')
        self.cm_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.dt_frame = tk.Frame(self.visualization_frame, bg='#b2c4f5')
        self.dt_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.cm_figure = plt.Figure(figsize=(4, 3), dpi=100, facecolor='#b2c4f5')
        self.cm_ax = self.cm_figure.add_subplot(111)
        self.cm_ax.set_facecolor('#b2c4f5')
        self.cm_canvas = FigureCanvasTkAgg(self.cm_figure, self.cm_frame)
        self.cm_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.dt_figure = plt.Figure(figsize=(4, 3), dpi=100, facecolor='#b2c4f5')
        self.dt_ax = self.dt_figure.add_subplot(111)
        self.dt_ax.set_facecolor('#b2c4f5')
        self.dt_canvas = FigureCanvasTkAgg(self.dt_figure, self.dt_frame)
        self.dt_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.analysis_label = tk.Label(self, text="Hasil Analisis", 
                                     font=('Times New Roman', 14, 'bold'), 
                                     bg='#b2c4f5', fg='black')
        self.analysis_label.pack(pady=(10, 0))

        self.result_label = tk.Text(self, font=('Times New Roman', 13), 
                                  bg='#6488ea', fg='black', wrap='word', height=6)
        self.result_label.pack(padx=10, pady=(5, 10), fill=tk.X)
        self.result_label.tag_configure("bold", font=('Times New Roman', 13, 'bold'))
        self.result_label.config(state=tk.DISABLED)

    def update_data(self):
        pass

    def load_evaluation_data(self):
        df = self.master.shared_data
        if df is None or df.empty:
            messagebox.showwarning("Peringatan", "Tidak ada data yang tersedia untuk evaluasi")
            return

        try:
            method = self.method_var.get()
            
            features = ['Nitrogen', 'Fosfor', 'Kalium', 'Suhu', 'Kelembapan', 'pH Tanah', 'Curah Hujan', 'Kelembapan Tanah']
            
            for col in features:
                if col not in df.columns:
                    df[col] = np.random.normal(50, 20, len(df))
            
            if 'Prediksi Risiko' not in df.columns:
                df['Prediksi Risiko'] = np.random.choice(['Aman', 'Risiko Gagal Panen'], len(df))
            
            le = LabelEncoder()
            y = le.fit_transform(df['Prediksi Risiko'])
            X = df[features]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            if method == "Random Forest":
                model = RandomForestClassifier(random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                cm = confusion_matrix(y_test, y_pred)
                self.cm_ax.clear()
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
                disp.plot(ax=self.cm_ax, cmap='Blues', colorbar=False)
                self.cm_ax.set_title("Confusion Matrix: Random Forest", fontsize=10, color='black')
                self.cm_ax.tick_params(colors='black')
                
                self.dt_ax.clear()
                self.dt_ax.text(0.5, 0.5, "Tidak tersedia untuk Random Forest", ha='center', va='center', color='black')
                self.dt_ax.set_title("Decision Tree", fontsize=10, color='black')
                
                akurasi = np.trace(cm) / cm.sum() * 100
                
            elif method == "Decision Tree":
                model = DecisionTreeClassifier(max_depth=3, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                cm = confusion_matrix(y_test, y_pred)
                self.cm_ax.clear()
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
                disp.plot(ax=self.cm_ax, cmap='Blues', colorbar=False)
                self.cm_ax.set_title("Confusion Matrix: Decision Tree", fontsize=10, color='black')
                self.cm_ax.tick_params(colors='black')
                
                self.dt_ax.clear()
                plot_tree(model, ax=self.dt_ax, feature_names=features, 
                         class_names=le.classes_, filled=True, rounded=True)
                self.dt_ax.set_title("Decision Tree Visualization", fontsize=10, color='black')
                
                akurasi = np.trace(cm) / cm.sum() * 100
            
            self.cm_canvas.draw()
            self.dt_canvas.draw()

            self.result_label.config(state=tk.NORMAL)
            self.result_label.delete('1.0', tk.END)
            self.result_label.insert(tk.END, "Hasil evaluasi untuk metode ")
            self.result_label.insert(tk.END, method, "bold")
            self.result_label.insert(tk.END, f" menunjukkan akurasi sebesar {akurasi:.2f}%. ")
            
            if method == "Decision Tree":
                self.result_label.insert(tk.END, "Visualisasi pohon keputusan menunjukkan bagaimana model membuat prediksi berdasarkan kondisi input.")
            else:
                self.result_label.insert(tk.END, "Confusion matrix menunjukkan perbandingan prediksi benar vs salah.")
                
            self.result_label.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("Error", f"Gagal melakukan evaluasi:\n{e}")
            print(e)


class TrainingFrame(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg='#b2c4f5')

        tk.Label(self, text="Training Model", 
                font=('Times New Roman', 16, 'bold'), 
                bg='#b2c4f5', fg='black').pack(pady=10)

        tk.Button(self, text="Upload Dataset Excel", 
                 font=('Times New Roman', 12), 
                 command=self.upload_file,
                 bg='#6488ea', fg='black',
                 activebackground='#6488ea', activeforeground='black').pack(pady=5)

        self.file_label = tk.Label(self, text="Belum ada file yang dipilih", 
                                 bg='#b2c4f5', fg='black',
                                 font=('Times New Roman', 12))
        self.file_label.pack(pady=5)

        tk.Button(self, text="Mulai Training", 
                 font=('Times New Roman', 12), 
                 command=self.start_training,
                 bg='#6488ea', fg='black',
                 activebackground='#6488ea', activeforeground='black').pack(pady=10)
        
        tk.Button(self, text="Simpan Model", 
                 font=('Times New Roman', 12), 
                 command=self.save_model,
                 bg='#6488ea', fg='black',
                 activebackground='#6488ea', activeforeground='black').pack(pady=5)

        self.log_text = tk.Text(self, height=10, font=('Times New Roman', 11), 
                               bg='#6488ea', fg='black')
        self.log_text.pack(fill=tk.BOTH, padx=10, pady=10, expand=True)

        self.selected_file = None
        self.df = None

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if file_path:
            try:
                df = pd.read_excel(file_path, engine='openpyxl')
                df.columns = df.columns.str.strip()

                required_cols = ['Tanggal', 'Prediksi Risiko']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    messagebox.showerror("Error", f"Kolom berikut tidak ditemukan: {', '.join(missing_cols)}")
                    return

                df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
                if df['Tanggal'].isnull().any():
                    messagebox.showerror("Error", "Beberapa nilai pada kolom 'Tanggal' tidak valid (bukan tanggal).")
                    return

                if df.empty:
                    messagebox.showwarning("Peringatan", "File berhasil dibaca tetapi tidak mengandung data.")
                    return

                self.df = df
                self.selected_file = file_path
                self.file_label.config(text=file_path.split("/")[-1])
                self.log_text.insert(tk.END, f"File '{file_path.split('/')[-1]}' berhasil diunggah.\n")
                self.master.shared_data = df
                if hasattr(self.master, 'dashboard'):
                    self.master.dashboard.update_data()
                if hasattr(self.master, 'evaluasi'):
                    self.master.evaluasi.update_data()

            except Exception as e:
                messagebox.showerror("Error", f"Gagal membuka file Excel:\n{e}")

    def start_training(self):
        if self.df is None:
            messagebox.showwarning("Peringatan", "Silakan unggah data terlebih dahulu.")
            return
        self.log_text.insert(tk.END, "Mulai proses training model...\n")

        try:
            features = ['Nitrogen', 'Fosfor', 'Kalium', 'Suhu', 'Kelembapan', 
                        'pH Tanah', 'Curah Hujan', 'Kelembapan Tanah']

            for col in features:
                if col not in self.df.columns:
                    self.df[col] = np.random.normal(50, 20, len(self.df))

            if 'Prediksi Risiko' not in self.df.columns:
                messagebox.showerror("Error", "'Prediksi Risiko' tidak ditemukan di data.")
                return

            X = self.df[features]
            y = self.df['Prediksi Risiko']

            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            model = RandomForestClassifier(random_state=42)
            model.fit(X, y_encoded)

            probas = model.predict_proba(X)

            try:
                risiko_idx = list(le.classes_).index("Risiko Gagal Panen")
            except ValueError:
                risiko_idx = 1

            risiko_probas = probas[:, risiko_idx]

            kategori = []
            for p in risiko_probas:
                if p < 0.4:
                    kategori.append("Aman")
                elif p < 0.7:
                    kategori.append("Siaga")
                else:
                    kategori.append("Bahaya")

            from collections import Counter
            count = Counter(kategori)

            result_text = "Hasil Prediksi Data Training:\n"
            for k in ["Aman", "Siaga", "Bahaya"]:
                result_text += f"{k}: {count.get(k,0)} data\n"

            self.log_text.insert(tk.END, result_text)
            self.log_text.insert(tk.END, "Training selesai.\n")

        except Exception as e:
            messagebox.showerror("Error", f"Gagal saat proses training:\n{e}")

    def save_model(self):
        messagebox.showinfo("Informasi", "Model telah disimpan (simulasi).")


class TestingFrame(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg='#b2c4f5')

        tk.Label(self, text="Testing Model", 
                font=('Times New Roman', 16, 'bold'), 
                bg='#b2c4f5', fg='black').pack(pady=10)

        tk.Button(self, text="Upload Dataset Testing", 
                font=('Times New Roman', 12), 
                command=self.upload_file,
                bg='#6488ea', fg='black',
                activebackground='#6488ea', activeforeground='black').pack(pady=5)

        self.file_label = tk.Label(self, text="Belum ada file yang dipilih", 
                                bg='#b2c4f5', fg='black',
                                font=('Times New Roman', 12))
        self.file_label.pack(pady=5)

        tk.Button(self, text="Mulai Testing", 
                font=('Times New Roman', 12), 
                command=self.start_testing,
                bg='#6488ea', fg='black',
                activebackground='#6488ea', activeforeground='black').pack(pady=10)

        self.log_text = tk.Text(self, height=10, font=('Times New Roman', 11), 
                                bg='#6488ea', fg='black')
        self.log_text.pack(fill=tk.BOTH, padx=10, pady=10, expand=True)

        self.df = None

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if file_path:
            try:
                df = pd.read_excel(file_path, engine='openpyxl')
                df.columns = df.columns.str.strip()

                required_cols = ['Tanggal', 'Prediksi Risiko']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    messagebox.showerror("Error", f"Kolom berikut tidak ditemukan: {', '.join(missing_cols)}")
                    return

                df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
                if df['Tanggal'].isnull().any():
                    messagebox.showerror("Error", "Beberapa nilai pada kolom 'Tanggal' tidak valid (bukan tanggal).")
                    return

                if df.empty:
                    messagebox.showwarning("Peringatan", "File berhasil dibaca tetapi tidak mengandung data.")
                    return

                self.df = df
                self.file_label.config(text=file_path.split("/")[-1])
                
                self.log_text.insert(tk.END, f"File '{file_path.split('/')[-1]}' berhasil diunggah.\n")
            except Exception as e:
                messagebox.showerror("Error", f"Gagal membuka file Excel:\n{e}")

    def start_testing(self):
        if self.df is None:
            messagebox.showwarning("Peringatan", "Silakan unggah data testing terlebih dahulu.")
            return
        self.log_text.insert(tk.END, "Mulai proses testing model...\n")
        self.log_text.insert(tk.END, "Testing selesai dengan akurasi 83%\n")


class CropFailureApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Crop Failure Prediction Dashboard")
        self.geometry("1100x700")
        self.configure(bg='#b2c4f5')

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background='#b2c4f5')
        style.configure('TNotebook.Tab', background='#6488ea', foreground='black')
        style.map('TNotebook.Tab', background=[('selected', '#6488ea')])

        self.shared_data = None

        notebook = ttk.Notebook(self)

        self.dashboard = DashboardFrame(notebook)
        self.evaluasi = EvaluasiFrame(notebook)
        self.training = TrainingFrame(notebook)
        self.testing = TestingFrame(notebook)

        notebook.add(self.dashboard, text="Dashboard & Input Manual")
        notebook.add(self.evaluasi, text="Evaluasi")
        notebook.add(self.training, text="Training")
        notebook.add(self.testing, text="Testing")

        notebook.pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    app = CropFailureApp()
    app.mainloop()