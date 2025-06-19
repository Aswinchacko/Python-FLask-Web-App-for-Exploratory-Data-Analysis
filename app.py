from flask import Flask, render_template, request, session, redirect
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
import base64
import os 
from fpdf import FPDF
from flask import send_file
import datetime
import numpy as np

def plot_to_img():
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.clf()
    return img

def save_plot_to_file(plot_name):
    """Save plot directly to file and return filename"""
    import os
    os.makedirs('static/eda_plots', exist_ok=True)
    filepath = f'static/eda_plots/{plot_name}.png'
    plt.tight_layout()
    plt.savefig(filepath, format='png', bbox_inches='tight', dpi=100)
    plt.clf()
    return plot_name  # Return just the filename, not full path



app = Flask(__name__)
app.secret_key = 'abc123'


@app.route("/")
def upload_file():
    return render_template("index.html")

@app.route("/health")
def health_check():
    return {"status": "healthy"}, 200

@app.route("/drop_rows", methods=["GET"])
def drop_rows():
    if "df" not in session:
        return "No data uploaded. Please upload a CSV file first.", 400
    
    col = request.args.get("col")
    df = pd.read_json(session["df"])
    df = df[df[col].notna()] # to drop the rows with missing values
    session["df"] = df.to_json()    
    return redirect("/upload")

@app.route("/upload", methods=["POST", "GET"])
def upload_file_func():
    if request.method == "POST":
        f = request.files["file"]
        if f.filename.endswith(".csv"):
            f.save(os.path.join(app.root_path, "uploads", f.filename))
            df = pd.read_csv('uploads/'+f.filename)
            session["df"] = df.to_json()
            dataframe = pd.read_json(session["df"])
            # to get the shape of the data
            shape = dataframe.shape
            # to get the columns of the data
            columns = dataframe.columns.tolist()
            # to get the null counts of the data
            null_counts = dataframe.isnull().sum().to_dict()

            # to get the preview of the data
            preview = dataframe.head(10).to_html(classes='data', header="true", index=False)

            # to get the missing columns of the data
            missing_cols = {}
            for col, count in null_counts.items():
                if count > 0:
                    missing_cols[col] = count

            describe_table = dataframe.describe().to_html(classes="data", header="true")
            value_counts_dict = {}
            for col in dataframe.select_dtypes(include='object').columns:
                value_counts_dict[col] = dataframe[col].value_counts().to_dict()
            skewness = dataframe.skew(numeric_only=True).to_dict()
            kurtosis = dataframe.kurtosis(numeric_only=True).to_dict()


            analytics = []
            numeric_cols = dataframe.select_dtypes(include='number')
            for col in numeric_cols.columns:
                col_data = dataframe[col]
                analytics.append(f"📊 '{col}' → mean: {col_data.mean():.2f}, min: {col_data.min()}, max: {col_data.max()}")
                skew = col_data.skew()
                if abs(skew) > 1:
                    analytics.append(f"⚠️ Column '{col}' is highly skewed (skewness = {skew:.2f}).")
            categoricals = dataframe.select_dtypes(include='object')
            for col in categoricals.columns:
                val_counts = dataframe[col].value_counts(normalize=True)
                top_val = val_counts.index[0]
                percent = val_counts.iloc[0] * 100
                analytics.append(f"🧠 Column '{col}' → Most frequent: '{top_val}' ({percent:.1f}%)")    

                if percent > 80:
                     analytics.append(f"⚠️ Column '{col}' is imbalanced — '{top_val}' dominates.")

            dtypes = {}
            for col in dataframe.columns:
                dtypes[col] = str(dataframe[col].dtype)
            type_suggestions = []
            for col in dataframe.select_dtypes(include='object').columns:
                try:
                    pd.to_datetime(dataframe[col])
                    type_suggestions.append(f"📅 '{col}' looks like a date. Convert to datetime?")
                except:
                    pass
            for col in dataframe.select_dtypes(include='float').columns:
                if (dataframe[col] % 1 == 0).all():
                    type_suggestions.append(f"🔢 '{col}' has only whole numbers. Convert to int?")

            # Store both file-based and base64 plots for template display
            univariate_plots = {}
            plot_files = {}  # Store filenames for PDF generation

            for col in dataframe.select_dtypes(include='object').columns:
                counts = dataframe[col].value_counts()
                
                # Bar chart
                plt.figure()
                counts.plot(kind='bar')
                plt.title(f'Bar Chart of {col}')
                univariate_plots[f'{col}_bar'] = plot_to_img()
                
                # Also save to file for PDF
                plt.figure()
                counts.plot(kind='bar')
                plt.title(f'Bar Chart of {col}')
                plot_files[f'{col}_bar'] = save_plot_to_file(f'{col}_bar')

                # Pie chart
                plt.figure()
                counts.plot(kind='pie', autopct='%1.1f%%')
                plt.title(f'Pie Chart of {col}')
                univariate_plots[f'{col}_pie'] = plot_to_img()
                
                # Also save to file for PDF
                plt.figure()
                counts.plot(kind='pie', autopct='%1.1f%%')
                plt.title(f'Pie Chart of {col}')
                plot_files[f'{col}_pie'] = save_plot_to_file(f'{col}_pie')

            for col in dataframe.select_dtypes(include='number').columns:
                # Histogram
                plt.figure()
                plt.hist(dataframe[col].dropna(), bins=10)
                plt.title(f'Histogram of {col}')
                univariate_plots[f'{col}_hist'] = plot_to_img()
                
                # Also save to file for PDF
                plt.figure()
                plt.hist(dataframe[col].dropna(), bins=10)
                plt.title(f'Histogram of {col}')
                plot_files[f'{col}_hist'] = save_plot_to_file(f'{col}_hist')

                # Boxplot
                plt.figure()
                plt.boxplot(dataframe[col].dropna())
                plt.title(f'Boxplot of {col}')
                plt.xticks([1], [col])
                univariate_plots[f'{col}_box'] = plot_to_img()
                
                # Also save to file for PDF
                plt.figure()
                plt.boxplot(dataframe[col].dropna())
                plt.title(f'Boxplot of {col}')
                plt.xticks([1], [col])
                plot_files[f'{col}_box'] = save_plot_to_file(f'{col}_box')



            bivariate_plots = {}
            numeric_cols = dataframe.select_dtypes(include='number').columns

            for i in range(len(numeric_cols)-1):
                x = numeric_cols[i]
                y = numeric_cols[i+1]
                
                # Scatter plot for template
                plt.figure()
                plt.scatter(dataframe[x], dataframe[y])
                plt.xlabel(x)
                plt.ylabel(y)
                plt.title(f'{x} vs {y}')
                bivariate_plots[f'{x}_vs_{y}'] = plot_to_img()
                
                # Also save to file for PDF
                plt.figure()
                plt.scatter(dataframe[x], dataframe[y])
                plt.xlabel(x)
                plt.ylabel(y)
                plt.title(f'{x} vs {y}')
                plot_files[f'{x}_vs_{y}'] = save_plot_to_file(f'{x}_vs_{y}')

            # Correlation heatmap for template
            plt.figure()
            corr = dataframe[numeric_cols].corr()
            plt.imshow(corr, cmap='coolwarm', interpolation='none')
            plt.colorbar()
            plt.xticks(range(len(corr)), corr.columns, rotation=90)
            plt.yticks(range(len(corr)), corr.columns)
            plt.title('Correlation Heatmap')
            bivariate_plots['heatmap'] = plot_to_img()
            
            # Also save to file for PDF
            plt.figure()
            corr = dataframe[numeric_cols].corr()
            plt.imshow(corr, cmap='coolwarm', interpolation='none')
            plt.colorbar()
            plt.xticks(range(len(corr)), corr.columns, rotation=90)
            plt.yticks(range(len(corr)), corr.columns)
            plt.title('Correlation Heatmap')
            plot_files['heatmap'] = save_plot_to_file('heatmap')


            # Store only plot filenames in session (much smaller)
            session['plot_files'] = plot_files



            # to render the data to the html page
            return render_template('eda.html',
                               shape=shape,
                               columns=columns,
                               nulls=null_counts,
                               preview=preview,
                               missing_cols=missing_cols,
                               describe_table=describe_table,
                               value_counts_dict=value_counts_dict,
                               skewness=skewness,
                               kurtosis=kurtosis,
                               analytics=analytics,
                               dtypes=dtypes,
                               type_suggestions=type_suggestions,
                               univariate_plots=univariate_plots,
                               bivariate_plots=bivariate_plots)
        else:
            return "Invalid file type"
    if "df" in session:
        df = pd.read_json(session["df"])

        shape = df.shape
        columns = df.columns.tolist()
        null_counts = df.isnull().sum().to_dict()
        preview = df.head(10).to_html(classes='data', header="true", index=False)
        describe_table = df.describe().to_html(classes="data", header="true")
        value_counts_dict = {}
        for col in df.select_dtypes(include='object').columns:
            value_counts_dict[col] = df[col].value_counts().to_dict()
        skewness = df.skew(numeric_only=True).to_dict()
        kurtosis = df.kurtosis(numeric_only=True).to_dict()

        missing_cols = {}
        for col, count in null_counts.items():
            if count > 0:
                missing_cols[col] = count

        # Add missing variable definitions for GET method
        analytics = []
        numeric_cols = df.select_dtypes(include='number')
        for col in numeric_cols.columns:
            col_data = df[col]
            analytics.append(f"📊 '{col}' → mean: {col_data.mean():.2f}, min: {col_data.min()}, max: {col_data.max()}")
            skew = col_data.skew()
            if abs(skew) > 1:
                analytics.append(f"⚠️ Column '{col}' is highly skewed (skewness = {skew:.2f}).")
        categoricals = df.select_dtypes(include='object')
        for col in categoricals.columns:
            val_counts = df[col].value_counts(normalize=True)
            top_val = val_counts.index[0]
            percent = val_counts.iloc[0] * 100
            analytics.append(f"🧠 Column '{col}' → Most frequent: '{top_val}' ({percent:.1f}%)")    

            if percent > 80:
                 analytics.append(f"⚠️ Column '{col}' is imbalanced — '{top_val}' dominates.")

        dtypes = {}
        for col in df.columns:
            dtypes[col] = str(df[col].dtype)
        type_suggestions = []
        for col in df.select_dtypes(include='object').columns:
            try:
                pd.to_datetime(df[col])
                type_suggestions.append(f"📅 '{col}' looks like a date. Convert to datetime?")
            except:
                pass
        for col in df.select_dtypes(include='float').columns:
            if (df[col] % 1 == 0).all():
                type_suggestions.append(f"🔢 '{col}' has only whole numbers. Convert to int?")

        # Generate plots for GET method (use existing plots if available)
        if 'plot_files' not in session:
            # Only generate if not already generated
            plot_files = {}
            univariate_plots = {}
            
            for col in df.select_dtypes(include='object').columns:
                counts = df[col].value_counts()
                
                # Bar chart
                plt.figure()
                counts.plot(kind='bar')
                plt.title(f'Bar Chart of {col}')
                univariate_plots[f'{col}_bar'] = plot_to_img()
                plot_files[f'{col}_bar'] = save_plot_to_file(f'{col}_bar')

                # Pie chart
                plt.figure()
                counts.plot(kind='pie', autopct='%1.1f%%')
                plt.title(f'Pie Chart of {col}')
                univariate_plots[f'{col}_pie'] = plot_to_img()
                plot_files[f'{col}_pie'] = save_plot_to_file(f'{col}_pie')

            for col in df.select_dtypes(include='number').columns:
                # Histogram
                plt.figure()
                plt.hist(df[col].dropna(), bins=10)
                plt.title(f'Histogram of {col}')
                univariate_plots[f'{col}_hist'] = plot_to_img()
                plot_files[f'{col}_hist'] = save_plot_to_file(f'{col}_hist')

                # Boxplot
                plt.figure()
                plt.boxplot(df[col].dropna())
                plt.title(f'Boxplot of {col}')
                plt.xticks([1], [col])
                univariate_plots[f'{col}_box'] = plot_to_img()
                plot_files[f'{col}_box'] = save_plot_to_file(f'{col}_box')

            bivariate_plots = {}
            numeric_cols = df.select_dtypes(include='number').columns

            for i in range(len(numeric_cols)-1):
                x = numeric_cols[i]
                y = numeric_cols[i+1]
                
                plt.figure()
                plt.scatter(df[x], df[y])
                plt.xlabel(x)
                plt.ylabel(y)
                plt.title(f'{x} vs {y}')
                bivariate_plots[f'{x}_vs_{y}'] = plot_to_img()
                plot_files[f'{x}_vs_{y}'] = save_plot_to_file(f'{x}_vs_{y}')

            # Correlation heatmap
            plt.figure()
            corr = df[numeric_cols].corr()
            plt.imshow(corr, cmap='coolwarm', interpolation='none')
            plt.colorbar()
            plt.xticks(range(len(corr)), corr.columns, rotation=90)
            plt.yticks(range(len(corr)), corr.columns)
            plt.title('Correlation Heatmap')
            bivariate_plots['heatmap'] = plot_to_img()
            plot_files['heatmap'] = save_plot_to_file('heatmap')
            
            # Store only plot filenames in session
            session['plot_files'] = plot_files
        else:
            # Load existing plots from files for display
            plot_files = session['plot_files']
            univariate_plots = {}
            bivariate_plots = {}
            
            # Read base64 data from files for template display
            for plot_name, filename in plot_files.items():
                try:
                    with open(f'static/eda_plots/{filename}.png', 'rb') as f:
                        img_data = f.read()
                        base64_data = base64.b64encode(img_data).decode('utf-8')
                        
                        if any(x in plot_name for x in ['bar', 'pie', 'hist', 'box']):
                            univariate_plots[plot_name] = base64_data
                        else:
                            bivariate_plots[plot_name] = base64_data
                except:
                    continue

        return render_template('eda.html',
                               shape=shape,
                               columns=columns,
                               nulls=null_counts,
                               preview=preview,
                               missing_cols=missing_cols,
                               describe_table=describe_table,
                               value_counts_dict=value_counts_dict,
                               skewness=skewness,
                               kurtosis=kurtosis,
                               analytics=analytics,
                               dtypes=dtypes,
                               type_suggestions=type_suggestions,
                               univariate_plots=univariate_plots,
                               bivariate_plots=bivariate_plots)
    else:
        return render_template("upload.html")  # if no file in session, show upload form





@app.route("/generate_pdf")
def download_report():
    if "df" not in session:
        return "No data uploaded. Please upload a CSV file first.", 400
    
    df = pd.read_json(session["df"])
    
    # Use plot files from session
    plot_files = session.get('plot_files', {})
    
    shape = df.shape
    dtypes = df.dtypes.astype(str).to_dict()
    nulls = df.isnull().sum()
    describe = df.describe().round(2)

    # Create PDF with professional styling
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Set up fonts
    pdf.add_font('Arial', '', 'C:\\Windows\\Fonts\\arial.ttf', uni=True)
    pdf.add_font('Arial', 'B', 'C:\\Windows\\Fonts\\arialbd.ttf', uni=True)
    pdf.add_font('Arial', 'I', 'C:\\Windows\\Fonts\\ariali.ttf', uni=True)
    pdf.add_font('Arial', 'BI', 'C:\\Windows\\Fonts\\arialbi.ttf', uni=True)

    # ========================================
    # COVER PAGE
    # ========================================
    pdf.set_font("Arial", 'B', 24)
    pdf.cell(0, 40, "", ln=True)  # Top margin
    pdf.cell(0, 20, "EXPLORATORY DATA ANALYSIS", ln=True, align='C')
    pdf.cell(0, 10, "REPORT", ln=True, align='C')
    
    pdf.set_font("Arial", '', 14)
    pdf.cell(0, 30, "", ln=True)
    pdf.cell(0, 10, f"Dataset: {df.columns[0] if len(df.columns) > 0 else 'Uploaded Data'}", ln=True, align='C')
    pdf.cell(0, 10, f"Generated: {datetime.datetime.now().strftime('%B %d, %Y at %I:%M %p')}", ln=True, align='C')
    
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 40, "", ln=True)
    pdf.cell(0, 10, "Prepared by: Mini EDA Tool", ln=True, align='C')
    pdf.cell(0, 10, "Data Science Project", ln=True, align='C')

    # ========================================
    # TABLE OF CONTENTS
    # ========================================
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "TABLE OF CONTENTS", ln=True)
    pdf.ln(10)
    
    pdf.set_font("Arial", '', 12)
    toc_items = [
        "1. Executive Summary",
        "2. Dataset Overview", 
        "3. Data Quality Assessment",
        "4. Statistical Analysis",
        "5. Data Distribution Analysis",
        "6. Correlation Analysis",
        "7. Key Insights & Recommendations",
        "8. Appendices"
    ]
    
    for item in toc_items:
        pdf.cell(0, 8, item, ln=True)

    # ========================================
    # EXECUTIVE SUMMARY
    # ========================================
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "1. EXECUTIVE SUMMARY", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 6, "This report presents a comprehensive exploratory data analysis (EDA) of the uploaded dataset. The analysis covers data quality assessment, statistical summaries, distribution analysis, and correlation studies to provide actionable insights for further data processing and modeling.")
    pdf.ln(5)
    
    # Key metrics box
    pdf.set_fill_color(240, 240, 240)
    pdf.rect(10, pdf.get_y(), 190, 25, 'F')
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "KEY METRICS:", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 6, f"• Total Records: {shape[0]:,}", ln=True)
    pdf.cell(0, 6, f"• Total Features: {shape[1]}", ln=True)
    pdf.cell(0, 6, f"• Missing Values: {df.isnull().sum().sum():,}", ln=True)

    # ========================================
    # DATASET OVERVIEW
    # ========================================
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "2. DATASET OVERVIEW", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "2.1 Dataset Information", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 6, f"Dataset Shape: {shape[0]} rows × {shape[1]} columns", ln=True)
    pdf.cell(0, 6, f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB", ln=True)
    pdf.ln(5)
    
    # Feature list
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "2.2 Feature List", ln=True)
    pdf.set_font("Arial", '', 10)
    for i, col in enumerate(df.columns, 1):
        pdf.cell(0, 6, f"{i}. {col}", ln=True)

    # ========================================
    # DATA QUALITY ASSESSMENT
    # ========================================
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "3. DATA QUALITY ASSESSMENT", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "3.1 Data Types", ln=True)
    pdf.set_font("Arial", '', 10)
    
    # Data types summary
    type_counts = df.dtypes.value_counts()
    for dtype, count in type_counts.items():
        pdf.cell(0, 6, f"• {dtype}: {count} columns", ln=True)
    pdf.ln(5)
    
    # Missing values analysis
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "3.2 Missing Values Analysis", ln=True)
    pdf.set_font("Arial", '', 10)
    
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    if missing_data.sum() > 0:
        pdf.cell(0, 6, "Columns with missing values:", ln=True)
        for col in missing_data[missing_data > 0].index:
            pdf.cell(0, 6, f"• {col}: {missing_data[col]:,} ({missing_percent[col]:.1f}%)", ln=True)
    else:
        pdf.cell(0, 6, "✓ No missing values detected in the dataset.", ln=True)

    # ========================================
    # STATISTICAL ANALYSIS
    # ========================================
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "4. STATISTICAL ANALYSIS", ln=True)
    pdf.ln(5)
    
    # Summary statistics for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, "4.1 Summary Statistics", ln=True)
        pdf.set_font("Arial", '', 9)
        
        for col in numeric_cols:
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(0, 6, f"{col}:", ln=True)
            pdf.set_font("Arial", '', 9)
            stats = df[col].describe()
            pdf.cell(0, 5, f"  Mean: {stats['mean']:.2f} | Std: {stats['std']:.2f} | Min: {stats['min']:.2f} | Max: {stats['max']:.2f}", ln=True)
            pdf.cell(0, 5, f"  Q1: {stats['25%']:.2f} | Median: {stats['50%']:.2f} | Q3: {stats['75%']:.2f}", ln=True)
            pdf.ln(2)

    # ========================================
    # DATA DISTRIBUTION ANALYSIS
    # ========================================
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "5. DATA DISTRIBUTION ANALYSIS", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", '', 10)
    pdf.multi_cell(0, 6, "The following visualizations provide insights into the distribution patterns of individual variables and their relationships. This analysis helps identify outliers, skewness, and potential data quality issues.")

    # Use cached plots from session if available
    import os
    import base64
    import tempfile
    from io import BytesIO
    graph_dir = "static/eda_plots"
    os.makedirs(graph_dir, exist_ok=True)

    # Add visualizations section
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "5. DATA VISUALIZATIONS", ln=True)
    pdf.ln(5)

    # Function to decode base64 plot and save temporarily
    def save_plot_from_session(plot_key, filename):
        if plot_key in cached_plots:
            try:
                # Decode base64 image
                img_data = base64.b64decode(cached_plots[plot_key])
                img_path = os.path.join(graph_dir, filename)
                with open(img_path, 'wb') as f:
                    f.write(img_data)
                return img_path
            except:
                return None
        return None

    # Add plots from files
    if plot_files:
        for plot_name, filename in plot_files.items():
            try:
                # Use existing plot file
                img_path = os.path.join(graph_dir, f"{filename}.png")
                
                # Add to PDF
                pdf.add_page()
                pdf.set_font("Arial", 'B', 14)
                title = plot_name.replace('_', ' ').title()
                pdf.cell(0, 10, title, ln=True)
                pdf.ln(5)
                
                # Add image to PDF
                pdf.image(img_path, x=15, y=pdf.get_y(), w=180)
                
                # Add some analysis text
                pdf.set_y(pdf.get_y() + 120)  # Move below image
                pdf.set_font("Arial", '', 10)
                
                # Add insights based on plot type
                if "hist" in plot_name.lower():
                    col_name = plot_name.replace('_hist', '').replace('_', ' ')
                    if col_name in df.columns:
                        col_data = df[col_name]
                        if col_data.dtype in ['float64', 'int64']:
                            skew = col_data.skew()
                            pdf.cell(0, 6, f"• Distribution shows {'right skew' if skew > 0.5 else 'left skew' if skew < -0.5 else 'symmetric pattern'}", ln=True)
                            pdf.cell(0, 6, f"• Mean: {col_data.mean():.2f}, Median: {col_data.median():.2f}", ln=True)
                elif "box" in plot_name.lower():
                    col_name = plot_name.replace('_box', '').replace('_', ' ')
                    if col_name in df.columns:
                        col_data = df[col_name]
                        if col_data.dtype in ['float64', 'int64']:
                            Q1 = col_data.quantile(0.25)
                            Q3 = col_data.quantile(0.75)
                            IQR = Q3 - Q1
                            outliers = df[(col_data < Q1 - 1.5*IQR) | (col_data > Q3 + 1.5*IQR)]
                            pdf.cell(0, 6, f"• Outliers detected: {len(outliers)} ({len(outliers)/len(df)*100:.1f}% of data)", ln=True)
                            pdf.cell(0, 6, f"• IQR: {IQR:.2f} (Q1: {Q1:.2f}, Q3: {Q3:.2f})", ln=True)
                elif "bar" in plot_name.lower() or "pie" in plot_name.lower():
                    col_name = plot_name.replace('_bar', '').replace('_pie', '').replace('_', ' ')
                    if col_name in df.columns:
                        counts = df[col_name].value_counts()
                        top_val = counts.index[0]
                        top_percent = (counts.iloc[0] / len(df)) * 100
                        pdf.cell(0, 6, f"• Most frequent value: '{top_val}' ({top_percent:.1f}%)", ln=True)
                        pdf.cell(0, 6, f"• Unique values: {len(counts)}", ln=True)
                        if top_percent > 80:
                            pdf.cell(0, 6, "• ⚠️ High class imbalance detected", ln=True)
                elif "_vs_" in plot_name:
                    vars = plot_name.replace('_vs_', '|').split('|')
                    if len(vars) == 2:
                        var1, var2 = vars[0].replace('_', ' '), vars[1].replace('_', ' ')
                        if var1 in df.columns and var2 in df.columns:
                            if df[var1].dtype in ['float64', 'int64'] and df[var2].dtype in ['float64', 'int64']:
                                corr = df[var1].corr(df[var2])
                                pdf.cell(0, 6, f"• Correlation coefficient: {corr:.3f}", ln=True)
                                if abs(corr) > 0.7:
                                    pdf.cell(0, 6, f"• Strong {'positive' if corr > 0 else 'negative'} correlation", ln=True)
                                elif abs(corr) > 0.3:
                                    pdf.cell(0, 6, f"• Moderate {'positive' if corr > 0 else 'negative'} correlation", ln=True)
                                else:
                                    pdf.cell(0, 6, "• Weak correlation", ln=True)
                elif "heatmap" in plot_name:
                    pdf.cell(0, 6, "• Correlation matrix showing relationships between all numeric variables", ln=True)
                    pdf.cell(0, 6, "• Darker colors indicate stronger correlations", ln=True)
                
            except Exception as e:
                print(f"Error adding plot {plot_name}: {e}")
                continue

    # ========================================
    # CORRELATION ANALYSIS (Simplified)
    # ========================================
    if len(numeric_cols) > 1:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "6. CORRELATION ANALYSIS", ln=True)
        pdf.ln(5)
        
        pdf.set_font("Arial", '', 10)
        pdf.multi_cell(0, 6, "Correlation analysis reveals relationships between numeric variables. Full visualizations are available in your analysis dashboard.")
        pdf.ln(5)
        
        # Correlation summary (without generating plots)
        corr = df[numeric_cols].corr()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, "6.1 Strong Correlations (|r| > 0.7):", ln=True)
        pdf.set_font("Arial", '', 10)
        
        strong_corrs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                corr_val = corr.iloc[i, j]
                if abs(corr_val) > 0.7:
                    var1, var2 = corr.columns[i], corr.columns[j]
                    strong_corrs.append((var1, var2, corr_val))
        
        if strong_corrs:
            for var1, var2, corr_val in strong_corrs:
                pdf.cell(0, 6, f"• {var1} ↔ {var2}: {corr_val:.3f}", ln=True)
        else:
            pdf.cell(0, 6, "• No strong correlations found (|r| > 0.7)", ln=True)
        
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, "6.2 Moderate Correlations (0.3 < |r| < 0.7):", ln=True)
        pdf.set_font("Arial", '', 10)
        
        moderate_corrs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                corr_val = corr.iloc[i, j]
                if 0.3 < abs(corr_val) <= 0.7:
                    var1, var2 = corr.columns[i], corr.columns[j]
                    moderate_corrs.append((var1, var2, corr_val))
        
        if moderate_corrs:
            for var1, var2, corr_val in moderate_corrs[:5]:  # Show only top 5
                pdf.cell(0, 6, f"• {var1} ↔ {var2}: {corr_val:.3f}", ln=True)
            if len(moderate_corrs) > 5:
                pdf.cell(0, 6, f"• ... and {len(moderate_corrs)-5} more (see dashboard)", ln=True)
        else:
            pdf.cell(0, 6, "• No moderate correlations found", ln=True)

    # ========================================
    # KEY INSIGHTS & RECOMMENDATIONS
    # ========================================
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "7. KEY INSIGHTS & RECOMMENDATIONS", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "7.1 Data Quality Issues", ln=True)
    pdf.set_font("Arial", '', 10)
    
    issues = []
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            issues.append(f"• {col}: {df[col].isnull().sum()} missing values")
        
        if df[col].dtype in ['float64', 'int64']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
            if len(outliers) > 0:
                issues.append(f"• {col}: {len(outliers)} potential outliers")
    
    if issues:
        for issue in issues:
            pdf.cell(0, 6, issue, ln=True)
    else:
        pdf.cell(0, 6, "✓ No major data quality issues detected.", ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "7.2 Recommendations", ln=True)
    pdf.set_font("Arial", '', 10)
    
    recommendations = [
        "• Handle missing values through imputation or removal based on business context",
        "• Investigate and treat outliers appropriately",
        "• Consider feature engineering for highly correlated variables",
        "• Apply appropriate scaling/normalization for skewed variables",
        "• Validate data quality with domain experts",
        "• Consider sampling strategies if dealing with imbalanced classes"
    ]
    
    for rec in recommendations:
        pdf.cell(0, 6, rec, ln=True)

    # ========================================
    # APPENDICES
    # ========================================
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "8. APPENDICES", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "8.1 Detailed Statistics", ln=True)
    pdf.set_font("Arial", '', 9)
    
    # Detailed describe table
    if len(numeric_cols) > 0:
        describe_table = df[numeric_cols].describe().round(3)
        for col in describe_table.columns:
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(0, 6, f"{col}:", ln=True)
            pdf.set_font("Arial", '', 8)
            for stat in describe_table.index:
                pdf.cell(0, 5, f"  {stat}: {describe_table[col][stat]}", ln=True)
            pdf.ln(2)

    # Save final report
    report_path = "static/final_eda_report.pdf"
    pdf.output(report_path)

    # Set completion cookie if download_id provided
    download_id = request.args.get('download_id')
    response = send_file(report_path, as_attachment=True, download_name='eda_report.pdf')
    
    if download_id:
        response.set_cookie(f'download_{download_id}', 'completed', max_age=60)
    
    return response



if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0', port=5000)