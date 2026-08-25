import tkinter as tk
from tkinter import messagebox, filedialog, ttk

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split


# ============================================================
# GLOBAL DATA
# ============================================================

data = None


# ============================================================
# MAIN WINDOW
# ============================================================

root = tk.Tk()

root.title("AI & Data Science Toolkit")
root.geometry("720x800")
root.resizable(False, False)

BG = "#F4F6F8"
HEADER = "#1F2937"
SECTION = "#E5E7EB"
TEXT = "#111827"

root.configure(bg=BG)


# ============================================================
# HEADER
# ============================================================

header = tk.Frame(
    root,
    bg=HEADER,
    height=100
)

header.pack(fill="x")

title = tk.Label(
    header,
    text="AI & DATA SCIENCE TOOLKIT",
    font=("Arial", 22, "bold"),
    bg=HEADER,
    fg="white"
)

title.pack(pady=(18, 4))

subtitle = tk.Label(
    header,
    text="Data Analysis • Visualization • Machine Learning",
    font=("Arial", 11),
    bg=HEADER,
    fg="white"
)

subtitle.pack()


# ============================================================
# STATUS
# ============================================================

status_frame = tk.Frame(
    root,
    bg=BG
)

status_frame.pack(pady=10)

status_label = tk.Label(
    status_frame,
    text="Dataset: Not Loaded",
    font=("Arial", 11, "bold"),
    bg=BG,
    fg=TEXT
)

status_label.pack()


# ============================================================
# LOAD CSV
# ============================================================

def load_csv():

    global data

    file_path = filedialog.askopenfilename(
        title="Select CSV File",
        filetypes=[
            ("CSV Files", "*.csv"),
            ("All Files", "*.*")
        ]
    )

    if not file_path:
        return

    try:

        data = pd.read_csv(file_path)

        data.columns = data.columns.str.strip()

        status_label.config(
            text="Dataset: Loaded ✓",
            fg="green"
        )

        messagebox.showinfo(
            "Success",
            "CSV file loaded successfully!"
        )

    except Exception as e:

        messagebox.showerror(
            "Error",
            "Could not load CSV file.\n\n" + str(e)
        )


# ============================================================
# CHECK DATA
# ============================================================

def check_data():

    if data is None:

        messagebox.showwarning(
            "No Dataset",
            "Please load a CSV file first."
        )

        return False

    return True


# ============================================================
# STEP 19 - VIEW DATASET
# ============================================================

def view_dataset():

    if not check_data():
        return

    window = tk.Toplevel(root)

    window.title("Dataset Preview")
    window.geometry("800x500")

    window.configure(bg="white")

    title_label = tk.Label(
        window,
        text="Dataset Preview",
        font=("Arial", 16, "bold"),
        bg="white"
    )

    title_label.pack(pady=10)

    table_frame = tk.Frame(
        window,
        bg="white"
    )

    table_frame.pack(
        fill="both",
        expand=True,
        padx=10,
        pady=10
    )

    vertical_scrollbar = ttk.Scrollbar(
        table_frame,
        orient="vertical"
    )

    vertical_scrollbar.pack(
        side="right",
        fill="y"
    )

    horizontal_scrollbar = ttk.Scrollbar(
        table_frame,
        orient="horizontal"
    )

    horizontal_scrollbar.pack(
        side="bottom",
        fill="x"
    )

    table = ttk.Treeview(
        table_frame,
        columns=list(data.columns),
        show="headings",
        yscrollcommand=vertical_scrollbar.set,
        xscrollcommand=horizontal_scrollbar.set
    )

    table.pack(
        fill="both",
        expand=True
    )

    vertical_scrollbar.config(
        command=table.yview
    )

    horizontal_scrollbar.config(
        command=table.xview
    )

    for column in data.columns:

        table.heading(
            column,
            text=column
        )

        table.column(
            column,
            width=130,
            anchor="center"
        )

    for index, row in data.iterrows():

        values = []

        for value in row:

            values.append(
                str(value)
            )

        table.insert(
            "",
            "end",
            values=values
        )

    info_label = tk.Label(
        window,
        text=(
            "Rows: "
            + str(data.shape[0])
            + "     Columns: "
            + str(data.shape[1])
        ),
        font=("Arial", 10, "bold"),
        bg="white"
    )

    info_label.pack(
        pady=8
    )


# ============================================================
# STEP 20 - EXPORT DATASET
# ============================================================

def export_dataset():

    if not check_data():
        return

    file_path = filedialog.asksaveasfilename(
        title="Save Dataset",
        defaultextension=".csv",
        filetypes=[
            ("CSV Files", "*.csv")
        ]
    )

    if not file_path:
        return

    try:

        data.to_csv(
            file_path,
            index=False
        )

        messagebox.showinfo(
            "Export Successful",
            "Dataset exported successfully!\n\n"
            + file_path
        )

    except Exception as e:

        messagebox.showerror(
            "Export Error",
            "Could not export the dataset.\n\n"
            + str(e)
        )


# ============================================================
# DATASET INFORMATION
# ============================================================

def dataset_info():

    if not check_data():
        return

    columns = "\n".join(
        str(column)
        for column in data.columns
    )

    result = (
        "Rows: "
        + str(data.shape[0])
        + "\n"
        + "Columns: "
        + str(data.shape[1])
        + "\n\n"
        + "Column Names:\n"
        + columns
    )

    messagebox.showinfo(
        "Dataset Information",
        result
    )


# ============================================================
# MISSING VALUES
# ============================================================

def missing_values():

    if not check_data():
        return

    missing = data.isnull().sum()

    result = "===== MISSING VALUES =====\n\n"

    for column, value in missing.items():

        result += (
            str(column)
            + ": "
            + str(value)
            + "\n"
        )

    messagebox.showinfo(
        "Missing Values",
        result
    )


# ============================================================
# STATISTICAL SUMMARY
# ============================================================

def statistics():

    if not check_data():
        return

    window = tk.Toplevel(root)

    window.title("Statistical Summary")
    window.geometry("800x500")

    text = tk.Text(
        window,
        font=("Consolas", 10)
    )

    text.pack(
        fill="both",
        expand=True,
        padx=10,
        pady=10
    )

    text.insert(
        "1.0",
        data.describe().to_string()
    )


# ============================================================
# CALCULATE AVERAGE
# ============================================================

def average():

    if not check_data():
        return

    columns = data.select_dtypes(
        include="number"
    ).columns.tolist()

    if not columns:

        messagebox.showwarning(
            "Error",
            "No numerical columns found."
        )

        return

    window = tk.Toplevel(root)

    window.title("Calculate Average")
    window.geometry("400x250")

    tk.Label(
        window,
        text="Select Numerical Column",
        font=("Arial", 11, "bold")
    ).pack(pady=15)

    selected = tk.StringVar(
        value=columns[0]
    )

    dropdown = ttk.Combobox(
        window,
        textvariable=selected,
        values=columns,
        state="readonly",
        width=25
    )

    dropdown.pack()

    def calculate():

        column = selected.get()

        result = data[column].mean()

        messagebox.showinfo(
            "Average",
            column
            + " Average = "
            + str(round(result, 2))
        )

    tk.Button(
        window,
        text="Calculate Average",
        width=20,
        command=calculate
    ).pack(pady=25)


# ============================================================
# HISTOGRAM
# ============================================================

def histogram():

    if not check_data():
        return

    columns = data.select_dtypes(
        include="number"
    ).columns.tolist()

    if not columns:

        messagebox.showwarning(
            "Error",
            "No numerical columns found."
        )

        return

    window = tk.Toplevel(root)

    window.title("Histogram")
    window.geometry("400x250")

    tk.Label(
        window,
        text="Select Column",
        font=("Arial", 11, "bold")
    ).pack(pady=15)

    selected = tk.StringVar(
        value=columns[0]
    )

    ttk.Combobox(
        window,
        textvariable=selected,
        values=columns,
        state="readonly",
        width=25
    ).pack()

    def create_histogram():

        column = selected.get()

        plt.figure(
            figsize=(8, 5)
        )

        plt.hist(
            data[column].dropna(),
            bins=10
        )

        plt.title(
            "Distribution of "
            + column
        )

        plt.xlabel(column)

        plt.ylabel("Frequency")

        plt.grid(True)

        plt.tight_layout()

        plt.show()

    tk.Button(
        window,
        text="Create Histogram",
        width=20,
        command=create_histogram
    ).pack(pady=25)


# ============================================================
# BASIC INSIGHTS
# ============================================================

def insights():

    if not check_data():
        return

    numerical_data = data.select_dtypes(
        include="number"
    )

    result = "===== BASIC INSIGHTS =====\n"

    for column in numerical_data.columns:

        result += (
            "\n"
            + str(column)
            + "\nAverage: "
            + str(round(
                numerical_data[column].mean(),
                2
            ))
            + "\nMinimum: "
            + str(numerical_data[column].min())
            + "\nMaximum: "
            + str(numerical_data[column].max())
            + "\n"
        )

    window = tk.Toplevel(root)

    window.title("Basic Insights")
    window.geometry("550x500")

    text = tk.Text(
        window,
        font=("Consolas", 10)
    )

    text.pack(
        fill="both",
        expand=True,
        padx=10,
        pady=10
    )

    text.insert(
        "1.0",
        result
    )


# ============================================================
# DATA CLEANING
# ============================================================

def data_cleaning():

    if not check_data():
        return

    missing = data.isnull().sum()

    duplicates = data.duplicated().sum()

    result = (
        "===== DATA CLEANING REPORT =====\n\n"
        "Missing Values:\n"
    )

    for column, value in missing.items():

        result += (
            str(column)
            + ": "
            + str(value)
            + "\n"
        )

    result += (
        "\nDuplicate Rows: "
        + str(duplicates)
        + "\n\nTotal Rows: "
        + str(len(data))
        + "\nTotal Columns: "
        + str(len(data.columns))
    )

    window = tk.Toplevel(root)

    window.title("Data Cleaning Report")
    window.geometry("550x450")

    text = tk.Text(
        window,
        font=("Consolas", 10)
    )

    text.pack(
        fill="both",
        expand=True,
        padx=10,
        pady=10
    )

    text.insert(
        "1.0",
        result
    )


# ============================================================
# CORRELATION HEATMAP
# ============================================================

def correlation_heatmap():

    if not check_data():
        return

    numerical_data = data.select_dtypes(
        include="number"
    )

    if numerical_data.shape[1] < 2:

        messagebox.showwarning(
            "Error",
            "At least two numerical columns are required."
        )

        return

    correlation = numerical_data.corr()

    plt.figure(
        figsize=(8, 6)
    )

    plt.imshow(
        correlation,
        cmap="coolwarm"
    )

    plt.colorbar()

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=45
    )

    plt.yticks(
        range(len(correlation.columns)),
        correlation.columns
    )

    plt.title(
        "Correlation Heatmap"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# SCATTER PLOT
# ============================================================

def scatter_plot():

    if not check_data():
        return

    columns = data.select_dtypes(
        include="number"
    ).columns.tolist()

    if len(columns) < 2:

        messagebox.showwarning(
            "Error",
            "At least two numerical columns are required."
        )

        return

    window = tk.Toplevel(root)

    window.title("Scatter Plot")
    window.geometry("450x350")

    tk.Label(
        window,
        text="X-axis",
        font=("Arial", 10, "bold")
    ).pack(pady=(15, 5))

    x_column = tk.StringVar(
        value=columns[0]
    )

    ttk.Combobox(
        window,
        textvariable=x_column,
        values=columns,
        state="readonly",
        width=25
    ).pack()

    tk.Label(
        window,
        text="Y-axis",
        font=("Arial", 10, "bold")
    ).pack(pady=(15, 5))

    y_column = tk.StringVar(
        value=columns[1]
    )

    ttk.Combobox(
        window,
        textvariable=y_column,
        values=columns,
        state="readonly",
        width=25
    ).pack()

    def create_plot():

        x = x_column.get()

        y = y_column.get()

        plt.figure(
            figsize=(8, 5)
        )

        plt.scatter(
            data[x],
            data[y]
        )

        plt.xlabel(x)

        plt.ylabel(y)

        plt.title(
            y
            + " vs "
            + x
        )

        plt.grid(True)

        plt.tight_layout()

        plt.show()

    tk.Button(
        window,
        text="Create Scatter Plot",
        width=22,
        command=create_plot
    ).pack(pady=30)


# ============================================================
# REQUIRED ML DATA
# ============================================================

def get_ml_data():

    required_columns = [
        "Age",
        "Study_Hours",
        "Attendance",
        "Assignments",
        "Marks"
    ]

    if not check_data():
        return None

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:

        messagebox.showerror(
            "Missing Columns",
            "Your CSV is missing:\n\n"
            + "\n".join(missing_columns)
        )

        return None

    model_data = data[
        required_columns
    ].dropna()

    if len(model_data) < 5:

        messagebox.showerror(
            "Not Enough Data",
            "At least 5 complete rows are required."
        )

        return None

    return model_data


# ============================================================
# PREDICT MARKS
# ============================================================

def predict_marks():

    model_data = get_ml_data()

    if model_data is None:
        return

    X = model_data[
        [
            "Age",
            "Study_Hours",
            "Attendance",
            "Assignments"
        ]
    ]

    y = model_data["Marks"]

    model = LinearRegression()

    model.fit(
        X,
        y
    )

    window = tk.Toplevel(root)

    window.title("Marks Prediction")
    window.geometry("430x520")

    tk.Label(
        window,
        text="PREDICT STUDENT MARKS",
        font=("Arial", 15, "bold")
    ).pack(pady=15)

    fields = {}

    field_names = [
        "Age",
        "Study Hours",
        "Attendance (%)",
        "Assignments"
    ]

    for field in field_names:

        tk.Label(
            window,
            text=field,
            font=("Arial", 10, "bold")
        ).pack(pady=(8, 2))

        entry = tk.Entry(
            window,
            width=25
        )

        entry.pack()

        fields[field] = entry

    def predict():

        try:

            age = float(
                fields["Age"].get()
            )

            study_hours = float(
                fields["Study Hours"].get()
            )

            attendance = float(
                fields["Attendance (%)"].get()
            )

            assignments = float(
                fields["Assignments"].get()
            )

            new_student = pd.DataFrame(
                {
                    "Age": [age],
                    "Study_Hours": [study_hours],
                    "Attendance": [attendance],
                    "Assignments": [assignments]
                }
            )

            prediction = model.predict(
                new_student
            )[0]

            messagebox.showinfo(
                "Prediction",
                "Predicted Marks: "
                + str(round(
                    prediction,
                    2
                ))
            )

        except ValueError:

            messagebox.showerror(
                "Invalid Input",
                "Please enter valid numbers."
            )

    tk.Button(
        window,
        text="Predict Marks",
        width=25,
        height=2,
        command=predict
    ).pack(pady=30)


# ============================================================
# MODEL EVALUATION
# ============================================================

def model_evaluation():

    model_data = get_ml_data()

    if model_data is None:
        return

    X = model_data[
        [
            "Age",
            "Study_Hours",
            "Attendance",
            "Assignments"
        ]
    ]

    y = model_data["Marks"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    result = (
        "===== MODEL EVALUATION =====\n\n"
        "Model:\n"
        "Multiple Linear Regression\n\n"
        "Features:\n"
        "• Age\n"
        "• Study Hours\n"
        "• Attendance\n"
        "• Assignments\n\n"
        "Target:\n"
        "Marks\n\n"
        "Training Rows: "
        + str(len(X_train))
        + "\n"
        "Testing Rows: "
        + str(len(X_test))
        + "\n\n"
        "R² Score: "
        + str(round(r2, 3))
        + "\n"
        "Mean Absolute Error: "
        + str(round(mae, 2))
    )

    messagebox.showinfo(
        "Model Evaluation",
        result
    )


# ============================================================
# PREDICTED VS ACTUAL
# ============================================================

def predicted_vs_actual():
    model_data = get_ml_data()

    if model_data is None:
        return

    X = model_data[
        [
            "Age",
            "Study_Hours",
            "Attendance",
            "Assignments"
        ]
    ]

    y = model_data["Marks"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.scatter(
        y_test,
        predictions
    )

    minimum = min(
        y_test.min(),
        predictions.min()
    )

    maximum = max(
        y_test.max(),
        predictions.max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum]
    )

    plt.xlabel(
        "Actual Marks"
    )

    plt.ylabel(
        "Predicted Marks"
    )

    plt.title(
        "Predicted vs Actual Marks"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.show()


# ============================================================
# SECTION TITLE
# ============================================================

def section_title(parent, text):

    frame = tk.Frame(
        parent,
        bg=SECTION
    )

    frame.pack(
        fill="x",
        padx=20,
        pady=(10, 5)
    )

    label = tk.Label(
        frame,
        text=text,
        font=("Arial", 11, "bold"),
        bg=SECTION,
        fg=TEXT
    )

    label.pack(
        anchor="w",
        padx=10,
        pady=6
    )


# ============================================================
# BUTTON CREATOR
# ============================================================

def create_button(
    parent,
    text,
    command
):

    button = tk.Button(
        parent,
        text=text,
        width=30,
        height=1,
        font=("Arial", 10),
        command=command
    )

    button.pack(
        pady=4
    )


# ============================================================
# SCROLLABLE AREA
# ============================================================

container = tk.Frame(
    root,
    bg=BG
)

container.pack(
    fill="both",
    expand=True
)

canvas = tk.Canvas(
    container,
    bg=BG,
    highlightthickness=0
)

scrollbar = ttk.Scrollbar(
    container,
    orient="vertical",
    command=canvas.yview
)

scrollable_frame = tk.Frame(
    canvas,
    bg=BG
)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window(
    (0, 0),
    window=scrollable_frame,
    anchor="nw",
    width=680
)

canvas.configure(
    yscrollcommand=scrollbar.set
)

canvas.pack(
    side="left",
    fill="both",
    expand=True
)

scrollbar.pack(
    side="right",
    fill="y"
)


# ============================================================
# DATASET SECTION
# ============================================================

section_title(
    scrollable_frame,
    "DATASET"
)

create_button(
    scrollable_frame,
    "Load CSV Dataset",
    load_csv
)

create_button(
    scrollable_frame,
    "View Dataset",
    view_dataset
)

create_button(
    scrollable_frame,
    "Export Dataset",
    export_dataset
)

create_button(
    scrollable_frame,
    "Dataset Information",
    dataset_info
)

create_button(
    scrollable_frame,
    "Check Missing Values",
    missing_values
)

create_button(
    scrollable_frame,
    "Data Cleaning Report",
    data_cleaning
)


# ============================================================
# DATA ANALYSIS
# ============================================================

section_title(
    scrollable_frame,
    "DATA ANALYSIS"
)

create_button(
    scrollable_frame,
    "Statistical Summary",
    statistics
)

create_button(
    scrollable_frame,
    "Calculate Average",
    average
)

create_button(
    scrollable_frame,
    "Basic Insights",
    insights
)


# ============================================================
# VISUALIZATION
# ============================================================

section_title(
    scrollable_frame,
    "VISUALIZATION"
)

create_button(
    scrollable_frame,
    "Create Histogram",
    histogram
)

create_button(
    scrollable_frame,
    "Create Scatter Plot",
    scatter_plot
)

create_button(
    scrollable_frame,
    "Correlation Heatmap",
    correlation_heatmap
)


# ============================================================
# MACHINE LEARNING
# ============================================================

section_title(
    scrollable_frame,
    "MACHINE LEARNING"
)

create_button(
    scrollable_frame,
    "Predict Marks",
    predict_marks
)

create_button(
    scrollable_frame,
    "Model Evaluation",
    model_evaluation
)

create_button(
    scrollable_frame,
    "Predicted vs Actual",
    predicted_vs_actual
)


# ============================================================
# EXIT
# ============================================================

tk.Button(
    scrollable_frame,
    text="EXIT",
    width=30,
    height=2,
    font=("Arial", 10, "bold"),
    command=root.destroy
).pack(
    pady=15
)


# ============================================================
# MOUSE WHEEL
# ============================================================

def mouse_wheel(event):

    canvas.yview_scroll(
        int(-1 * (event.delta / 120)),
        "units"
    )


canvas.bind_all(
    "<MouseWheel>",
    mouse_wheel
)


# ============================================================
# START
# ============================================================

root.mainloop()
