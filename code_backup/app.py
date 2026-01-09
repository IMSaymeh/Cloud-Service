import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import os

# بنجيب مكتبات السبارك (Spark) اللي بدنا نشتغل عليها
from pyspark.sql import functions as F
from pyspark.sql.functions import col, split, array_distinct
from process import DataProcessor


def build_perf_plot(perf_results):
    cores = [c for c, t, s, e, times in perf_results]
    exec_times = [t for c, t, s, e, times in perf_results]

    # بنعمل الرسم البياني شفاف عشان يتناسب مع الخلفية
    fig, ax = plt.subplots(figsize=(6, 3.6), facecolor='none')
    ax.set_facecolor('none')
    
    # بنختار لون الخط - أزرق
    line_color = "#5859f6" 
    ax.plot(cores, exec_times, marker="o", color=line_color, linewidth=2)
    
    ax.set_title("Execution Time vs Cores", color="gray")
    ax.set_xlabel("Cores", color="gray")
    ax.set_ylabel("Time (s)", color="gray")
    ax.tick_params(colors='gray')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def run_full_analytics(file, use_cloud_data):
    processor = DataProcessor()

    if use_cloud_data:
        # بنحمل الداتاسيت الكبير من موقع كاجل (Kaggle)
        processor.download_dataset()
        processor.input_path = os.path.join(processor.base_path, 'input', 'data.csv')
    elif file is not None:
        # بننسخ الملف اللي رفعه المستخدم لمجلد آمن على الدرايف (Drive)
        staged = processor.stage_uploaded_file(file.name)
        processor.input_path = staged
    else:
        return [None] * 8

    try:
        df = processor.load_data()

        # ------ الخطوة الأولى: حساب الإحصائيات الوصفية ------
        stats_data = processor.compute_descriptive_stats(df, processor.input_path)

        # ------ الخطوة الثانية: تدريب نموذج الانحدار الخطي (Linear Regression) ------
        clean_df = df.dropna(subset=["rating_num", "reviews_num"])
        from pyspark.ml.feature import VectorAssembler
        from pyspark.ml.regression import LinearRegression

        # بنحول عمود المراجعات لفيكتور عشان نقدر ندربه
        reg_assembler = VectorAssembler(inputCols=["reviews_num"], outputCol="features")
        reg_df = reg_assembler.transform(clean_df)
        lr = LinearRegression(featuresCol="features", labelCol="rating_num", regParam=0.01)
        lr_model = lr.fit(reg_df)

        regression_res = f"Coeff: {lr_model.coefficients[0]:.4f} | Intercept: {lr_model.intercept:.4f}"

        # ------ الخطوة الثالثة: تدريب نموذج التجميع (KMeans) ------
        from pyspark.ml.clustering import KMeans
        # بنحول عمودين (التقييم والمراجعات) لفيكتور عشان نجمعهم
        km_assembler = VectorAssembler(inputCols=["rating_num", "reviews_num"], outputCol="features")
        km_df = km_assembler.transform(clean_df)
        # بنقسم البيانات لـ3 مجموعات
        kmeans = KMeans(k=3, seed=1)
        km_model = kmeans.fit(km_df)
        centers = km_model.clusterCenters()

        clusters_data = pd.DataFrame({
            "Cluster": [0, 1, 2],
            "Avg Rating (Center)": [round(float(c[0]), 2) for c in centers],
            "Avg Reviews (Center)": [round(float(c[1]), 1) for c in centers]
        })

        # ------ الخطوة الرابعة: البحث عن الأنماط المتكررة (FPGrowth) ------
        from pyspark.ml.fpm import FPGrowth
        # بنستخدم عمود المطورين (developer) بدل عمود السعر عشان أكثر فايدة
        df_fp = df.withColumn("items", array_distinct(split(col("developer"), " "))).filter(
            F.col("developer").isNotNull() & (F.length(F.col("developer")) > 0)
        )
        
        if df_fp.count() > 0:
            # بنشغل موديل البحث عن الأنماط
            fp = FPGrowth(itemsCol="items", minSupport=0.01, minConfidence=0.5)
            fp_model = fp.fit(df_fp)
            # بنجيب أكثر 3 أنماط متكررة
            top_patterns = fp_model.freqItemsets.orderBy(F.col("freq").desc()).limit(3).toPandas()
            
            if not top_patterns.empty:
                top_item = top_patterns.iloc[0]
                fp_res = f"Top Pattern: {' '.join(top_item['items'][:3])} (Freq: {top_item['freq']})"
            else:
                fp_res = "No significant patterns found"
        else:
            fp_res = "Insufficient data for pattern mining"

        # ------ الخطوة الخامسة: تدريب نموذج الانحدار المتساوي (Isotonic Regression) ------
        rmse_val = processor.isotonic_regression_job(df)
        isotonic_res = f"{rmse_val:.4f}"

        # ------ الخطوة السادسة: اختبار الأداء على عدد كورات مختلفة ------
        cores = [1, 2, 4, 8]
        perf_results = processor.performance_test(cores)

        # بنحول نتائج الأداء لجدول منظم
        perf_df = pd.DataFrame([
            {
                "Cores": c,
                "Time (s)": f"{t:.3f}",
                "Speedup": f"{s:.2f}x",
                "Efficiency": f"{e*100:.1f}%"
            } for c, t, s, e, times in perf_results
        ])

        # بنرسم الرسم البياني للأداء
        perf_plot = build_perf_plot(perf_results)

        # بنحفظ كل النتائج في ملف نصي
        save_msg = processor.save_all_results_to_file(
            stats_data, regression_res, clusters_data, fp_res, rmse_val, perf_df
        )

        # بنرفع النتائج على قاعدة البيانات (Firebase)
        firebase_status = processor.save_to_firebase(
            stats_data, regression_res, clusters_data, fp_res, rmse_val, perf_df
        )

        return stats_data, regression_res, clusters_data, fp_res, isotonic_res, perf_df, perf_plot, " Analysis Completed Successfully!"

    except Exception as e:
        return None, str(e), None, None, None, None, None, f" Error: {str(e)}"


# ------ كود التنسيق (CSS) للواجهة - يدعم الوضع الليلي والنهاري ------
CUSTOM_CSS = """
/* المتغيرات الأساسية للألوان */
:root {
    --bg-color: #f6f7fb;
    --card-bg: #ffffff;
    --text-main: #1f2a44;
    --border-color: rgba(0,0,0,0.06);
    --table-header-bg: #f2f4ff;
}

/* تعديل الألوان عند تفعيل الوضع الليلي في Gradio */
.dark {
    --bg-color: #0b0f19;
    --card-bg: #151c2c;
    --text-main: #e2e8f0;
    --border-color: rgba(255,255,255,0.1);
    --table-header-bg: #1e293b;
}

.gradio-container {
    background: var(--bg-color) !important;
    font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
}

.title { text-align:center; font-size: 2.15rem; font-weight: 900; color: var(--text-main); }
.subtitle { text-align:center; font-size: 1.02rem; opacity: 0.75; color: var(--text-main); }

#left-card, #right-card {
    background: var(--card-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 18px;
    padding: 16px;
    box-shadow: none !important;
}

#run-btn {
    border-radius: 16px !important;
    font-weight: 850 !important;
    box-shadow: none !important;
}

/* Tab Styling */
.tab-nav button.selected {
    background: rgba(88, 89, 246, 0.20) !important;
    color: #5859f6 !important;
}

#right-card .tabitem {
    background: var(--card-bg) !important;
    border: none !important;
}

/* Table Styling */
.pretty-table {
    border: 1px solid var(--border-color) !important;
    border-radius: 14px;
    background: var(--card-bg) !important;
}

.pretty-table thead th {
    background: var(--table-header-bg) !important;
    color: var(--text-main) !important;
}

.pretty-table tbody td {
    color: var(--text-main) !important;
    border-bottom: 1px solid var(--border-color) !important;
}

.soft-card, #perf-card {
    background: var(--card-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 16px;
    padding: 12px;
}

/* إخفاء الليبلات الرمادية */
#right-card .label-wrap, #right-card .block-label { display: none !important; }
"""

# ------ بناء الواجهة الرسومية باستخدام مكتبة جراديو (Gradio) ------
with gr.Blocks(theme=gr.themes.Soft(), title="Cloud Spark Analytics", css=CUSTOM_CSS) as demo:
    gr.Markdown("<h1 class='title'>Cloud-Based Distributed Data Processing</h1>")
    gr.Markdown("<h3 class='subtitle'>IUG - Cloud Computing Final Project</h3>")

    with gr.Row():
        with gr.Column(scale=1, elem_id="left-card"):
            input_file = gr.File(label="Upload Dataset")
            cloud_opt = gr.Checkbox(label="Use Large Cloud Dataset (Kaggle)", value=True)
            run_btn = gr.Button(" Run Full Analysis", variant="primary", elem_id="run-btn")
            status_msg = gr.Markdown("Status: Ready")

        with gr.Column(scale=2, elem_id="right-card"):
            with gr.Tabs():
                with gr.TabItem(" Data Insights"):
                    gr.Markdown("### Descriptive Statistics")
                    out_stats = gr.DataFrame(show_label=False, elem_classes=["pretty-table"])

                with gr.TabItem(" Machine Learning (4 Jobs)"):
                    gr.Markdown("### Machine Learning Results")
                    with gr.Group(elem_classes=["soft-card"]):
                        gr.Markdown("#### 1) Linear Regression")
                        out_reg = gr.Textbox(show_label=False)
                    with gr.Group(elem_classes=["soft-card"]):
                        gr.Markdown("#### 2) KMeans")
                        out_kmeans = gr.DataFrame(show_label=False, elem_classes=["pretty-table"])
                    with gr.Group(elem_classes=["soft-card"]):
                        gr.Markdown("#### 3) FPGrowth")
                        out_fp = gr.Textbox(show_label=False)
                    with gr.Group(elem_classes=["soft-card"]):
                        gr.Markdown("#### 4) Isotonic Regression")
                        out_ir = gr.Textbox(show_label=False)

                with gr.TabItem(" Performance & Scalability"):
                    gr.Markdown("### Performance & Scalability")
                    with gr.Group(elem_id="perf-card"):
                        gr.Markdown("#### Multi-Core Execution Benchmarks")
                        out_perf = gr.DataFrame(show_label=False, elem_classes=["pretty-table"])
                        gr.Markdown("#### Execution Time vs Cores")
                        out_perf_plot = gr.Plot(show_label=False)

    run_btn.click(
        fn=run_full_analytics,
        inputs=[input_file, cloud_opt],
        outputs=[out_stats, out_reg, out_kmeans, out_fp, out_ir, out_perf, out_perf_plot, status_msg]
    )

demo.launch(share=True)
